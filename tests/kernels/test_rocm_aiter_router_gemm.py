# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm AITER router-gate GEMM: bf16 x bf16 -> out_dtype.

This is the ``GateLinear`` tier used on gfx950, where the MoE router gate is a
skinny GEMM (M=num_tokens, N=num_experts, K=hidden_size). Correctness baseline
is a float64 matmul; what ultimately matters is that numeric error never flips
the top-k expert selection.
"""

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

# Register torch.ops.vllm.rocm_aiter_router_gemm.
import vllm.model_executor.layers.fused_moe.router.gate_linear  # noqa: F401  isort: skip

# (hidden_size, num_experts): GLM-5.2 / DeepSeek-V3 style routers.
SHAPES = [(6144, 160), (6144, 256), (7168, 256)]
NUM_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]
ATOL_BF16 = 2e-2


def _requires_aiter_tgemm():
    if not current_platform.is_rocm():
        pytest.skip("AITER router GEMM requires ROCm")
    if not rocm_aiter_ops.is_tgemm_enabled():
        pytest.skip("AITER tuned GEMM not enabled (needs AITER linear + gfx950)")


def _run(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype):
    return torch.ops.vllm.rocm_aiter_router_gemm(x, weight, out_dtype)


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_matches_reference(
    num_tokens: int, hidden_dim: int, num_experts: int, out_dtype: torch.dtype
):
    """bf16 activation x bf16 weight should match a float64 reference."""
    _requires_aiter_tgemm()
    torch.manual_seed(42)
    device = torch.device("cuda")
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16, device=device)

    out = _run(x, weight, out_dtype)
    ref = (x.double() @ weight.double().t()).to(out_dtype)

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == out_dtype
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL_BF16, rtol=0)


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_topk_routing_consistency(num_tokens: int, hidden_dim: int, num_experts: int):
    """The gate feeds top-k expert selection, so numeric error only matters if
    it changes the selected experts. Ties around the k-th value are tolerated."""
    _requires_aiter_tgemm()
    top_k = 8
    device = torch.device("cuda")
    for seed in range(5):
        torch.manual_seed(1000 + seed)
        x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
        weight = torch.randn(
            num_experts, hidden_dim, dtype=torch.bfloat16, device=device
        )

        out = _run(x, weight, torch.float32)
        ref = x.double() @ weight.double().t()
        kernel_idx = out.topk(top_k, dim=-1).indices
        ref_vals, ref_idx = ref.topk(top_k, dim=-1)
        for t in range(num_tokens):
            got = set(kernel_idx[t].tolist())
            want = set(ref_idx[t].tolist())
            if got == want:
                continue
            kth = ref_vals[t, -1].item()
            for e in got.symmetric_difference(want):
                gap = abs(ref[t, e].item() - kth)
                assert gap < 1e-2, (
                    f"top-{top_k} mismatch beyond tie tolerance: token {t}, "
                    f"expert {e}, gap {gap:.3e}"
                )


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
def test_matches_gate_linear_fallback(hidden_dim: int, num_experts: int):
    """The AITER tier must agree with the F.linear fallback it replaces."""
    _requires_aiter_tgemm()
    torch.manual_seed(7)
    device = torch.device("cuda")
    x = torch.randn(16, hidden_dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16, device=device)

    out = _run(x, weight, torch.float32)
    fallback = torch.nn.functional.linear(x, weight).to(torch.float32)

    torch.testing.assert_close(out, fallback, atol=ATOL_BF16, rtol=0)
