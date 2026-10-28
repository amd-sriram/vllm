# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collectives of the GLM-5.x token-sharded prefill on ROCm, at TP4.

The gather is exact. The reduce-scatter matches the fp32 sum of the ranks'
partials, exactly for the bf16 backends and within the INT4 codec's error for
the QuickReduce backend.
"""

import os

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group
from vllm.platforms import current_platform

from ...utils import init_test_distributed_environment, multi_process_parallel

TP = 4
HIDDEN = 4096


@ray.remote(num_gpus=1, max_calls=1)
def _collectives(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    rs = os.environ["VLLM_ROCM_GLM5_PREFILL_SP_RS"]
    with monkeypatch.context() as m:
        for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
            m.delenv(var, raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        from vllm.models.glm5next.amd import prefill_sp

        assert prefill_sp.prefill_sp_min_tokens() == 1024
        prefill_sp.init_collectives()
        if rs == "int4":
            assert prefill_sp._qr_comm is not None
        group = get_tp_group().device_group

        for num_tokens in (8192, 8190, 1029):
            gen = torch.Generator(device=device).manual_seed(1000 + rank)
            partial = torch.randn(num_tokens, HIDDEN, generator=gen, device=device).to(
                torch.bfloat16
            )
            full = partial.float()
            dist.all_reduce(full, group=group)
            ref = prefill_sp.shard(full)

            out = prefill_sp.scatter(partial)
            assert out.shape == ref.shape and out.dtype == torch.bfloat16
            if rs == "int4":
                err = (out.float() - ref).abs().max().item()
                assert err <= 0.05 * ref.abs().max().item(), err
            else:
                torch.testing.assert_close(out.float(), ref, atol=0.05, rtol=0.02)

            # The gather is a copy: shard then gather returns the input rows.
            rows = full.to(torch.bfloat16)
            back = prefill_sp.gather(prefill_sp.shard(rows), num_tokens)
            assert torch.equal(back, rows)

        assert prefill_sp.should_shard(8192, 1024)
        assert not prefill_sp.should_shard(512, 1024)
        assert not prefill_sp.should_shard(8192, 0)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
@pytest.mark.parametrize("rs", ["int4", "aiter", "nccl"])
def test_prefill_sp_collectives(monkeypatch: pytest.MonkeyPatch, rs: str):
    if torch.accelerator.device_count() < TP:
        pytest.skip("needs 4 GPUs")
    monkeypatch.setenv("VLLM_ROCM_GLM5_PREFILL_SP_MIN_TOKENS", "1024")
    monkeypatch.setenv("VLLM_ROCM_GLM5_PREFILL_SP_RS", rs)
    monkeypatch.setenv("AITER_QUICK_REDUCE_QUANTIZATION", "INT4")
    multi_process_parallel(monkeypatch, TP, 1, _collectives)
