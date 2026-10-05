# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash fused KDA decode (VLLM_ROCM_USE_FUSED_KDA_DECODE).

The fused AITER kernel must match the three-launch decode path (conv1d update,
recurrent step, gated RMSNorm) on outputs and on both cached states, including
CUDA-graph padded batches whose padding rows point at the null slot.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("fused KDA decode is ROCm-only", allow_module_level=True)

fused_kda_decode = pytest.importorskip(
    "aiter.ops.triton.gated_delta_net.fused_kda_decode"
).fused_kda_decode

from vllm.models.glm5next.common import kda  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.kda import (  # noqa: E402
    FusedRMSNormGated,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata  # noqa: E402

pytestmark = pytest.mark.usefixtures("default_vllm_config")

H, D, W = 16, 128, 4
LP = H * D
PREFIX = "layer.0"


def make_layer(dim_first: bool, num_slots: int, device: torch.device):
    layer = object.__new__(kda.Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = PREFIX
    layer.local_num_heads = H
    layer.head_dim = D
    layer.conv_size = W
    layer.local_projection_size = LP
    layer.kda_safe_gate = True
    layer.kda_lower_bound = -5.0
    layer.kda_prefill_backend = "triton"
    layer._conv_state_dim_first = dim_first
    layer._merged_conv_weight = torch.randn(3 * LP, W, device=device) * 0.3
    layer.q_conv1d = SimpleNamespace(bias=None)
    layer.A_log = torch.randn(1, 1, H, 1, device=device) * 0.5
    layer.dt_bias = torch.randn(LP, device=device) * 0.1
    layer.o_norm = FusedRMSNormGated(D, activation="sigmoid").to(device)
    layer.o_norm.weight.data = 1 + 0.1 * torch.randn(
        D, device=device, dtype=torch.bfloat16
    )
    layer._fused_decode = fused_kda_decode

    # Padded slot strides, as the hybrid KV-cache allocator lays them out.
    conv_shape = (3 * LP, W - 1) if dim_first else (W - 1, 3 * LP)
    conv = torch.randn(
        num_slots, 3 * LP * (W - 1) + 8, device=device, dtype=torch.bfloat16
    )[:, : 3 * LP * (W - 1)].view(num_slots, *conv_shape)
    recurrent = torch.randn(num_slots, H * D * D + 8, device=device) * 0.1
    recurrent = recurrent[:, : H * D * D].view(num_slots, H, D, D)
    layer.kv_cache = (conv, recurrent)
    return layer


def decode_metadata(num_decodes: int, batch: int, device: torch.device):
    """Decode-only metadata padded to ``batch`` as the CUDA-graph path builds
    it: padding requests are empty and point at the null slot 0."""
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32, device=device)
    query_start_loc[num_decodes + 1 :] = num_decodes
    slots = torch.zeros(batch, dtype=torch.int32, device=device)
    slots[:num_decodes] = torch.randperm(num_decodes, device=device) + 1
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_decodes,
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=slots,
    )


def use_metadata(monkeypatch, metadata, mode=CUDAGraphMode.NONE):
    monkeypatch.setattr(
        kda,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={PREFIX: metadata}, cudagraph_runtime_mode=mode
        ),
    )


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize(("num_decodes", "batch"), [(1, 1), (7, 7), (5, 8), (64, 64)])
@torch.inference_mode()
def test_fused_decode_matches_unfused(monkeypatch, dim_first, num_decodes, batch):
    torch.manual_seed(0)
    device = torch.device("cuda")
    layer = make_layer(dim_first, batch + 1, device)
    metadata = decode_metadata(num_decodes, batch, device)
    use_metadata(monkeypatch, metadata)

    projected = torch.randn(
        batch, 3 * LP + H + 2 * D, device=device, dtype=torch.bfloat16
    )
    g1 = torch.randn(1, batch, H, D, device=device, dtype=torch.bfloat16)
    g_proj_states = torch.randn(batch, LP, device=device, dtype=torch.bfloat16)

    def split(projected):
        """Column slices of the fused projection, as forward() produces them."""
        return projected[:, : 3 * LP], projected[:, 3 * LP : 3 * LP + H].unsqueeze(0)

    conv, recurrent = layer.kv_cache
    conv_ref, recurrent_ref = conv.clone(), recurrent.clone()
    conv_before, recurrent_before = conv.clone(), recurrent.clone()

    # causal_conv1d_update writes its output over qkv, so each path gets a copy.
    qkv, beta = split(projected.clone())
    layer.kv_cache = (conv_ref, recurrent_ref)
    core_attn_out = torch.empty(1, batch, H, D, device=device, dtype=torch.bfloat16)
    layer._forward(qkv, g1, beta, core_attn_out)
    expected = layer.o_norm.forward_native(
        core_attn_out, g_proj_states.view(batch, H, D)
    ).reshape(batch, LP)

    qkv, beta = split(projected)
    layer.kv_cache = (conv, recurrent)
    assert layer._plain_decode_metadata() is metadata
    actual = layer._fused_decode_attn(qkv, g1, beta, g_proj_states, metadata)

    n = num_decodes
    torch.testing.assert_close(actual[:n], expected[:n], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(conv, conv_ref, atol=0, rtol=0)
    # The unfused path rounds the conv output to bf16 before the recurrence.
    torch.testing.assert_close(recurrent, recurrent_ref, atol=1e-2, rtol=1e-2)
    # The null slot that padding rows point at is left untouched.
    torch.testing.assert_close(conv[0], conv_before[0], atol=0, rtol=0)
    torch.testing.assert_close(recurrent[0], recurrent_before[0], atol=0, rtol=0)


@torch.inference_mode()
def test_fused_decode_only_on_plain_eager_or_full_graph_decode(monkeypatch):
    device = torch.device("cuda")
    layer = make_layer(False, 3, device)
    metadata = decode_metadata(2, 2, device)

    for mode in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
        use_metadata(monkeypatch, metadata, mode)
        assert layer._plain_decode_metadata() is metadata

    use_metadata(monkeypatch, metadata, CUDAGraphMode.PIECEWISE)
    assert layer._plain_decode_metadata() is None

    for other in (replace(metadata, num_prefills=1), replace(metadata, num_decodes=0)):
        use_metadata(monkeypatch, other)
        assert layer._plain_decode_metadata() is None

    use_metadata(monkeypatch, None)
    assert layer._plain_decode_metadata() is None
