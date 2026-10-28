# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-sharded mHC residual stream for GLM-5.x prefill on ROCm (pure TP).

With TP, every rank repeats the same per-token mHC work (``hc_pre``,
``hc_post``, the Sinkhorn mix and its RMSNorm) on all tokens of a step. For
eager steps with at least ``VLLM_ROCM_GLM5_PREFILL_SP_MIN_TOKENS`` tokens the
residual stream is sharded over tokens instead, padded to a multiple of TP:

- each attention / MLP output all-reduce becomes a reduce-scatter, and the
  block input is all-gathered right before the next attention / MLP;
- the mHC ops run on this rank's token shard only.

The bytes on the wire are about those of the all-reduce (RS + AG), but the
reduce-scatter can use AITER's INT4 QuickReduce codec, as the all-reduce does.

It stays off during CUDA graph capture, for MonoKernel decode steps, with PP,
with expert-parallel sequence parallelism, and when aux hidden states are
requested.
"""

import os

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# AITER's HIP QuickReduce communicator, whose IPC state the INT4 reduce-scatter
# kernel uses. Created once at model init (before memory profiling).
_qr_comm = None
_qr_rs = None


def prefill_sp_min_tokens() -> int:
    """The token threshold, or 0 when this path is unavailable."""
    min_tokens = envs.VLLM_ROCM_GLM5_PREFILL_SP_MIN_TOKENS
    if (
        min_tokens <= 0
        or not current_platform.is_rocm()
        or get_tensor_model_parallel_world_size() == 1
        or get_pp_group().world_size != 1
    ):
        return 0
    return min_tokens


def init_collectives() -> None:
    """Create the INT4 reduce-scatter state on every TP rank."""
    global _qr_comm, _qr_rs
    if _qr_comm is not None or envs.VLLM_ROCM_GLM5_PREFILL_SP_RS != "int4":
        return
    if get_tensor_model_parallel_world_size() != 4:
        logger.warning(
            "GLM-5 prefill SP: the INT4 reduce-scatter needs TP4; using the "
            "AITER bf16 reduce-scatter instead."
        )
        return
    try:
        from aiter.dist.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )
        from aiter.ops.quick_reduce_scatter import qr_reduce_scatter
    except ImportError:
        logger.warning(
            "GLM-5 prefill SP: AITER has no INT4 reduce-scatter; using the "
            "AITER bf16 reduce-scatter instead."
        )
        return
    # The legacy default sizes the IPC buffers for 2 GiB messages (4 GiB per
    # rank); a prefill step's reduce-scatter input is at most
    # max_num_batched_tokens x hidden in bf16.
    os.environ.setdefault("AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB", "128")
    tp = get_tp_group()
    comm = QuickAllReduce(group=tp.cpu_group, device=tp.device)
    if comm.disabled:
        logger.warning(
            "GLM-5 prefill SP: AITER QuickReduce is disabled (set "
            "AITER_QUICK_REDUCE_QUANTIZATION=INT4); using the AITER bf16 "
            "reduce-scatter instead."
        )
        return
    _qr_comm, _qr_rs = comm, qr_reduce_scatter
    logger.info(
        "GLM-5 prefill SP: INT4 reduce-scatter on, max %d MiB",
        comm.qr_max_size >> 20,
    )


def should_shard(num_tokens: int, min_tokens: int) -> bool:
    """Whether this step runs with a token-sharded residual stream.

    Every TP rank sees the same ``num_tokens``, so they all agree.
    """
    return (
        min_tokens > 0
        and num_tokens >= min_tokens
        and not torch.cuda.is_current_stream_capturing()
    )


def shard(x: torch.Tensor) -> torch.Tensor:
    """This rank's token rows, after padding to a multiple of TP."""
    tp_size = get_tensor_model_parallel_world_size()
    pad = (-x.shape[0]) % tp_size
    if pad:
        x = F.pad(x, (0, 0) * (x.dim() - 1) + (0, pad))
    rows = x.shape[0] // tp_size
    rank = get_tp_group().rank_in_group
    return x[rank * rows : (rank + 1) * rows]


def gather(x: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """All-gather the token shards and drop the TP padding rows."""
    x = x.contiguous()
    ca = get_tp_group().device_communicator.aiter_ar_comm
    if ca is not None and not ca.disabled and ca.should_custom_ag(x):
        out = ca.custom_all_gather(x, dim=0)
        if out is not None:
            return out[:num_tokens]
    return tensor_model_parallel_all_gather(x, 0)[:num_tokens]


def scatter(partial: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter per-token partial sums over tokens (pads to TP)."""
    tp_size = get_tensor_model_parallel_world_size()
    pad = (-partial.shape[0]) % tp_size
    if pad:
        partial = F.pad(partial, (0, 0, 0, pad))
    partial = partial.contiguous()
    out = torch.empty(
        (partial.shape[0] // tp_size, *partial.shape[1:]),
        dtype=partial.dtype,
        device=partial.device,
    )
    if (
        _qr_comm is not None
        and partial.numel() * partial.element_size() <= _qr_comm.qr_max_size
        and (out.numel() * out.element_size()) % 16 == 0
    ):
        _qr_rs(_qr_comm._ptr, partial, out)
        return out
    if envs.VLLM_ROCM_GLM5_PREFILL_SP_RS != "nccl":
        ca = get_tp_group().device_communicator.aiter_ar_comm
        if ca is not None and not ca.disabled:
            res = ca.custom_reduce_scatter(partial, out, dim=0)
            if res is not None:
                return res
    return tensor_model_parallel_reduce_scatter(partial, 0)


def all_reduce(partial: torch.Tensor) -> torch.Tensor:
    """The unsharded steps' all-reduce, deferred out of the output layer."""
    return tensor_model_parallel_all_reduce(partial)
