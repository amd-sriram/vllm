# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.platforms import current_platform

if not current_platform.is_rocm() or not is_aiter_found_and_supported():
    pytest.skip("Requires ROCm with AITER.", allow_module_level=True)

from vllm._aiter_ops import rocm_aiter_ops
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_mod


@pytest.mark.parametrize("on_gfx950", [True, False])
@pytest.mark.parametrize("with_per_req", [True, False])
def test_rocm_fp8_paged_mqa_logits_uses_per_req_context_lens(
    monkeypatch, on_gfx950, with_per_req
):
    batch_size, next_n, heads, max_model_len = 2, 3, 32, 256
    seen = {}

    def fake_kernel(*args, **kwargs):
        seen["context_lens"] = args[4]

    workspace = SimpleNamespace(
        get_simultaneous=lambda *specs: tuple(
            torch.empty(shape, dtype=dtype) for shape, dtype in specs
        )
    )
    monkeypatch.setattr(sparse_mod, "_ON_GFX942", False)
    monkeypatch.setattr(sparse_mod, "_ON_GFX950", on_gfx950)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        sparse_mod,
        "paged_mqa_logits_module",
        lambda: SimpleNamespace(
            deepgemm_fp8_paged_mqa_logits=fake_kernel,
            deepgemm_fp8_paged_mqa_logits_stage1=fake_kernel,
        ),
    )
    monkeypatch.setattr(sparse_mod, "current_workspace_manager", lambda: workspace)

    seq_lens = torch.tensor([[198, 199, 200], [88, 89, 90]], dtype=torch.int32)
    per_req = seq_lens[:, -1].contiguous()
    sparse_mod.rocm_fp8_paged_mqa_logits(
        torch.empty((batch_size, next_n, heads, 128)),
        torch.empty((8, 64, 1, 132), dtype=torch.uint8),
        torch.empty((batch_size * next_n, heads)),
        seq_lens,
        torch.zeros((batch_size, 4), dtype=torch.int32),
        None,
        max_model_len,
        per_req_context_lens=per_req if with_per_req else None,
    )

    assert seen["context_lens"] is (per_req if with_per_req else seq_lens)
