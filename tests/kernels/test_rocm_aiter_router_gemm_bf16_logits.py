# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the bf16-logits variant of the ROCm AITER router-gate GEMM.

The default ``GateLinear`` AITER tier runs AITER's tuned bf16 GEMM and then
casts the tiny ``[num_tokens, num_experts]`` logits up to fp32, one copy per MoE
layer per forward. Because that tuned kernel already produces bf16, the fp32
tensor only ever holds bf16-representable values; widening it and handing it to
the grouped top-k buys no precision. ``VLLM_ROCM_ROUTER_GEMM_BF16_LOGITS=1``
keeps the logits in bf16 and lets the ROCm grouped top-k consume them directly,
which drops the per-layer copy (the top-k still produces fp32 routing weights).

These tests check two things:

1. On ROCm, the op returns bf16 when asked, and the resulting top-k expert
   selection matches the fp32 path (the bf16 logits carry the same values that
   the fp32 tensor would have held).
2. Portably, feeding the router the bf16 logits *and* the bf16-rounded
   correction bias -- the only real numeric change vs. the fp32 path -- only
   ever reorders experts that are genuinely tied at the routing-score level.
   The grouped masking on top of this is validated on-device via the accuracy
   evals.
"""

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

# Register torch.ops.vllm.rocm_aiter_router_gemm.
import vllm.model_executor.layers.fused_moe.router.gate_linear  # noqa: F401  isort: skip

# (hidden_size, num_experts): GLM-5/5.2, DeepSeek-V3 and Kimi-K2 routers.
SHAPES = [(6144, 256), (7168, 256), (7168, 384)]
NUM_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]

TOP_K = 8

# bf16 keeps ~8 mantissa bits, so its worst-case relative rounding is 2**-8.
# A top-k membership change needs the promoted and demoted values to straddle
# the boundary, so the tie window is twice the single-value rounding.
BF16_REL = 2.0**-8
TIE_REL_TO_PEAK = 2 * BF16_REL


def _requires_aiter_tgemm():
    if not current_platform.is_rocm():
        pytest.skip("AITER router GEMM requires ROCm")
    if not rocm_aiter_ops.is_tgemm_enabled():
        pytest.skip("AITER tuned GEMM not enabled (needs AITER linear + gfx950)")


def _run(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype):
    return torch.ops.vllm.rocm_aiter_router_gemm(x, weight, out_dtype)


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_bf16_logits_match_fp32_routing(
    num_tokens: int, hidden_dim: int, num_experts: int
):
    """Asking the op for bf16 (what the flag does) must return bf16 and pick the
    same experts as the fp32 request it replaces. The tuned kernel emits bf16
    either way, so the two outputs should be bitwise-equal after upcast; the
    top-k check is the property the router actually depends on."""
    _requires_aiter_tgemm()
    torch.manual_seed(42)
    device = torch.device("cuda")
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16, device=device)

    out_fp32 = _run(x, weight, torch.float32)
    out_bf16 = _run(x, weight, torch.bfloat16)

    assert out_fp32.dtype == torch.float32
    assert out_bf16.dtype == torch.bfloat16
    assert out_bf16.shape == (num_tokens, num_experts)

    # The fp32 tier only widens the bf16 kernel output, so keeping bf16 must not
    # move any expert past the fp32 result.
    fp32_idx = out_fp32.topk(TOP_K, dim=-1).indices
    bf16_idx = out_bf16.float().topk(TOP_K, dim=-1).indices
    for t in range(num_tokens):
        assert set(fp32_idx[t].tolist()) == set(bf16_idx[t].tolist())


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
def test_bf16_router_scores_preserve_topk(
    hidden_dim: int, num_experts: int, num_tokens: int
):
    """Portable check of the only real numeric change the flag introduces.

    GLM-5.2 routes with ``noaux_tc``: score = sigmoid(logits) + correction_bias,
    then grouped top-k. With the flag on, both the logits and the (otherwise
    fp32) correction bias reach the kernel in bf16. Compare the fp32-scored
    selection against the bf16-scored one and require every disagreement to sit
    inside the bf16 tie window -- i.e. the flag only reshuffles true ties."""
    torch.manual_seed(7)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    weight = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16)
    bias = torch.randn(num_experts, dtype=torch.float32)

    logits = (x.double() @ weight.double().t()).float()

    scores_fp32 = torch.sigmoid(logits) + bias
    scores_bf16 = (
        torch.sigmoid(logits.bfloat16().float()) + bias.bfloat16().float()
    )

    # Scores live in ~[0, 2] for sigmoid + unit-ish bias; scale the tie window
    # by the peak score so it tracks the actual magnitude the kernel sees.
    tie_window = TIE_REL_TO_PEAK * scores_fp32.abs().max().item()

    fp32_vals, fp32_idx = scores_fp32.topk(TOP_K, dim=-1)
    bf16_idx = scores_bf16.topk(TOP_K, dim=-1).indices
    for t in range(num_tokens):
        got = set(bf16_idx[t].tolist())
        want = set(fp32_idx[t].tolist())
        if got == want:
            continue
        kth = fp32_vals[t, -1].item()
        for e in got.symmetric_difference(want):
            gap = abs(scores_fp32[t, e].item() - kth)
            assert gap < tie_window, (
                f"top-{TOP_K} selection changed beyond the bf16 tie window: "
                f"token {t}, expert {e}, gap {gap:.3e} > {tie_window:.3e}"
            )
