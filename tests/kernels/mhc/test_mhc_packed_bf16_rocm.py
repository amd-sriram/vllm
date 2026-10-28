# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER packed BF16 hi/lo mHC weights (VLLM_ROCM_MHC_PACKED_BF16_MIN_TOKENS)."""

import pytest
import torch

import vllm.model_executor.kernels.mhc.aiter as mhc_aiter
import vllm.model_executor.layers.mhc as mhc_layers
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

HIDDEN_SIZE = 4096
HC_MULT = 4
MIX = HC_MULT * (2 + HC_MULT)
EPS = 1e-6
SINKHORN = 20
POST_MULT = 2.0


@pytest.mark.parametrize(
    "num_tokens,has_packed,min_tokens,expected",
    [
        (8192, True, 1024, True),
        (1024, True, 1024, True),
        (1023, True, 1024, False),
        (8192, False, 1024, False),
        (8192, True, 0, False),
    ],
)
def test_packed_gating(num_tokens, has_packed, min_tokens, expected):
    fn_packed = torch.empty(1) if has_packed else None
    assert mhc_aiter.use_mhc_packed_bf16(num_tokens, fn_packed, min_tokens) is expected


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("env", ["0", "1024"])
def test_min_tokens_needs_env_and_support(monkeypatch, supported, env):
    monkeypatch.setenv("VLLM_ROCM_MHC_PACKED_BF16_MIN_TOKENS", env)
    monkeypatch.setattr(mhc_layers, "HAS_AITER_MHC_PACKED_BF16", supported)
    expected = 1024 if supported and env == "1024" else 0
    assert mhc_layers.mhc_packed_bf16_min_tokens() == expected


@pytest.mark.parametrize("num_tokens,packed", [(4, False), (8, True)])
def test_aiter_ops_get_packed_fn_only_above_threshold(monkeypatch, num_tokens, packed):
    calls = []

    def fake_mhc_pre(residual, fn, *args, w_preshuffle_bf16=False):
        calls.append((fn, w_preshuffle_bf16))

    def fake_fused(x, residual, post, comb, fn, *args, w_preshuffle_bf16=False):
        calls.append((fn, w_preshuffle_bf16))

    monkeypatch.setattr(rocm_aiter_ops, "mhc_pre", fake_mhc_pre)
    monkeypatch.setattr(rocm_aiter_ops, "mhc_fused_post_pre", fake_fused)
    residual = torch.zeros(num_tokens, HC_MULT, 256, dtype=torch.bfloat16)
    x = torch.zeros(num_tokens, 256, dtype=torch.bfloat16)
    fn = torch.zeros(MIX, HC_MULT * 256)
    fn_packed = torch.zeros(MIX, HC_MULT * 256, dtype=torch.int32)
    scalars = (EPS, EPS, EPS, POST_MULT, SINKHORN)
    mhc_aiter.mhc_pre_aiter(
        residual, fn, None, None, *scalars, 1, None, 0.0, fn_packed, 8
    )
    mhc_aiter.mhc_fused_post_pre_aiter(
        x, residual, None, None, fn, None, None, *scalars, 1, 1, None, 0.0,
        fn_packed, 8,
    )  # fmt: skip
    assert len(calls) == 2
    for got_fn, got_packed in calls:
        assert got_packed is packed
        assert got_fn is (fn_packed if packed else fn)


def _inputs(m: int):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(m, HIDDEN_SIZE, device=dev, dtype=torch.bfloat16) * 0.02
    residual = (
        torch.randn(m, HC_MULT, HIDDEN_SIZE, device=dev, dtype=torch.bfloat16) * 0.02
    )
    post = torch.sigmoid(torch.randn(m, HC_MULT, 1, device=dev))
    comb = torch.softmax(torch.randn(m, HC_MULT, HC_MULT, device=dev), dim=-1)
    fn = torch.randn(MIX, HC_MULT * HIDDEN_SIZE, device=dev) * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device=dev)
    base = torch.zeros(MIX, device=dev)
    norm_weight = torch.linspace(
        0.75, 1.25, HIDDEN_SIZE, device=dev, dtype=torch.bfloat16
    )
    return x, residual, post, comb, fn, scale, base, norm_weight


def _assert_mixes_close(packed, ref):
    # post/comb mixes: FP32 vs BF16 hi/lo GEMM (~16 mantissa bits) + Sinkhorn.
    torch.testing.assert_close(packed[0], ref[0], atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(packed[1], ref[1], atol=1e-4, rtol=1e-3)
    # BF16 layer input.
    torch.testing.assert_close(packed[2], ref[2], atol=6.25e-2, rtol=3e-2)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and mhc_layers.HAS_AITER_MHC_PACKED_BF16),
    reason="AITER packed BF16 mHC (gfx950) required",
)
@pytest.mark.parametrize("m,min_tokens", [(8192, 1024), (64, 1024)])
def test_packed_matches_fp32(m, min_tokens):
    x, residual, post, comb, fn, scale, base, norm_weight = _inputs(m)
    fn_packed = mhc_aiter.pack_mhc_fn_bf16(fn)
    assert fn_packed.dtype == torch.int32 and fn_packed.shape == fn.shape
    scalars = (EPS, EPS, EPS, POST_MULT, SINKHORN, 1)
    norm = (norm_weight, EPS)

    ref = torch.ops.vllm.mhc_pre_aiter(residual, fn, scale, base, *scalars, *norm)
    out = torch.ops.vllm.mhc_pre_aiter(
        residual, fn, scale, base, *scalars, *norm, fn_packed, min_tokens
    )
    ref_f = torch.ops.vllm.mhc_fused_post_pre_aiter(
        x, residual, post, comb, fn, scale, base, *scalars, 1, *norm
    )
    out_f = torch.ops.vllm.mhc_fused_post_pre_aiter(
        x, residual, post, comb, fn, scale, base, *scalars, 1, *norm,
        fn_packed, min_tokens,
    )  # fmt: skip
    torch.cuda.synchronize()

    if m < min_tokens:
        # Below the threshold the FP32 path runs unchanged.
        for a, b in zip((*out, *out_f), (*ref, *ref_f)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        return
    _assert_mixes_close(out, ref)
    # Fused returns (residual, post_mix, comb_mix, layer_input); the post step
    # does not read fn, so the new residual is bit-exact.
    torch.testing.assert_close(out_f[0], ref_f[0], atol=0, rtol=0)
    _assert_mixes_close(out_f[1:], ref_f[1:])
    for t in (*out, *out_f):
        assert torch.isfinite(t).all()
