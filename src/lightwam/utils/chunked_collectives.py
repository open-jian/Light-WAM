from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


logger = logging.getLogger(__name__)

_PATCHED = False
_ORIG_BROADCAST = None
_ORIG_ALL_REDUCE = None
_ORIG_REDUCE = None
_ORIG_ALL_REDUCE_COALESCED = None
_ORIG_ALL_GATHER = None
_ORIG_ALL_GATHER_INTO_TENSOR = None
_ORIG_ALL_GATHER_BASE = None
_ORIG_REDUCE_SCATTER = None
_ORIG_REDUCE_SCATTER_TENSOR = None
_ORIG_REDUCE_SCATTER_BASE = None
_MAX_BYTES = 64 * 1024


def _should_chunk(tensor: Any, async_op: bool, max_bytes: int) -> bool:
    if async_op:
        return False
    if not isinstance(tensor, torch.Tensor):
        return False
    if not tensor.is_cuda:
        return False
    if tensor.numel() <= 0:
        return False
    if not tensor.is_contiguous():
        return False
    return tensor.numel() * tensor.element_size() > int(max_bytes)


def _chunk_numel(tensor: torch.Tensor, max_bytes: int) -> int:
    return max(int(max_bytes) // max(int(tensor.element_size()), 1), 1)


def _flat_chunks(tensor: torch.Tensor, max_bytes: int):
    flat = tensor.view(-1)
    chunk_numel = _chunk_numel(tensor, max_bytes=max_bytes)
    for start in range(0, flat.numel(), chunk_numel):
        yield flat[start : start + chunk_numel]


def _patch_deepspeed_cached_collectives():
    try:
        import deepspeed.comm.comm as ds_comm
    except Exception:
        return

    cdb = getattr(ds_comm, "cdb", None)
    if cdb is None:
        return

    all_gather_into_tensor = getattr(dist, "all_gather_into_tensor", None)
    if all_gather_into_tensor is not None and hasattr(cdb, "all_gather_function"):
        cdb.all_gather_function = all_gather_into_tensor

    reduce_scatter_tensor = getattr(dist, "reduce_scatter_tensor", None)
    if reduce_scatter_tensor is not None and hasattr(cdb, "reduce_scatter_function"):
        cdb.reduce_scatter_function = reduce_scatter_tensor


def install_chunked_collectives(max_bytes: int = 64 * 1024) -> bool:
    """Patch large synchronous CUDA collectives into smaller NCCL calls.

    Some RTX PRO 6000 Blackwell + NCCL 2.26 environments hit illegal memory
    accesses on collectives larger than about 64-512 KiB. This keeps the public
    torch.distributed API intact for the sync calls DeepSpeed uses during model
    initialization, while leaving async/non-CUDA/non-contiguous calls untouched.
    """

    global _PATCHED, _ORIG_BROADCAST, _ORIG_ALL_REDUCE, _ORIG_REDUCE
    global _ORIG_ALL_REDUCE_COALESCED, _ORIG_ALL_GATHER, _ORIG_ALL_GATHER_INTO_TENSOR
    global _ORIG_ALL_GATHER_BASE, _ORIG_REDUCE_SCATTER, _ORIG_REDUCE_SCATTER_TENSOR
    global _ORIG_REDUCE_SCATTER_BASE, _MAX_BYTES
    _MAX_BYTES = int(max_bytes)
    if _PATCHED:
        _patch_deepspeed_cached_collectives()
        return False

    _ORIG_BROADCAST = dist.broadcast
    _ORIG_ALL_REDUCE = dist.all_reduce
    _ORIG_REDUCE = dist.reduce
    _ORIG_ALL_REDUCE_COALESCED = getattr(dist, "all_reduce_coalesced", None)
    _ORIG_ALL_GATHER = dist.all_gather
    _ORIG_ALL_GATHER_INTO_TENSOR = getattr(dist, "all_gather_into_tensor", None)
    _ORIG_ALL_GATHER_BASE = getattr(c10d, "_all_gather_base", None)
    _ORIG_REDUCE_SCATTER = dist.reduce_scatter
    _ORIG_REDUCE_SCATTER_TENSOR = getattr(dist, "reduce_scatter_tensor", None)
    _ORIG_REDUCE_SCATTER_BASE = getattr(c10d, "_reduce_scatter_base", None)

    def chunked_broadcast(tensor, src, group=None, async_op=False):
        if not _should_chunk(tensor, async_op=bool(async_op), max_bytes=_MAX_BYTES):
            return _ORIG_BROADCAST(tensor=tensor, src=src, group=group, async_op=async_op)
        for chunk in _flat_chunks(tensor, max_bytes=_MAX_BYTES):
            _ORIG_BROADCAST(tensor=chunk, src=src, group=group, async_op=False)
        return None

    def chunked_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if not _should_chunk(tensor, async_op=bool(async_op), max_bytes=_MAX_BYTES):
            return _ORIG_ALL_REDUCE(tensor=tensor, op=op, group=group, async_op=async_op)
        for chunk in _flat_chunks(tensor, max_bytes=_MAX_BYTES):
            _ORIG_ALL_REDUCE(tensor=chunk, op=op, group=group, async_op=False)
        return None

    def chunked_reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if not _should_chunk(tensor, async_op=bool(async_op), max_bytes=_MAX_BYTES):
            return _ORIG_REDUCE(tensor=tensor, dst=dst, op=op, group=group, async_op=async_op)
        for chunk in _flat_chunks(tensor, max_bytes=_MAX_BYTES):
            _ORIG_REDUCE(tensor=chunk, dst=dst, op=op, group=group, async_op=False)
        return None

    def chunked_all_reduce_coalesced(tensors, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if _ORIG_ALL_REDUCE_COALESCED is None:
            raise RuntimeError("torch.distributed.all_reduce_coalesced is not available.")
        if async_op:
            return _ORIG_ALL_REDUCE_COALESCED(tensors=tensors, op=op, group=group, async_op=async_op)
        if not any(_should_chunk(tensor, async_op=False, max_bytes=_MAX_BYTES) for tensor in tensors):
            return _ORIG_ALL_REDUCE_COALESCED(tensors=tensors, op=op, group=group, async_op=False)
        for tensor in tensors:
            chunked_all_reduce(tensor=tensor, op=op, group=group, async_op=False)
        return None

    def chunked_all_gather(tensor_list, tensor, group=None, async_op=False):
        if not _should_chunk(tensor, async_op=bool(async_op), max_bytes=_MAX_BYTES):
            return _ORIG_ALL_GATHER(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)
        if any(
            not isinstance(out, torch.Tensor)
            or not out.is_cuda
            or not out.is_contiguous()
            or out.numel() != tensor.numel()
            for out in tensor_list
        ):
            return _ORIG_ALL_GATHER(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)

        flat_in = tensor.view(-1)
        flat_outs = [out.view(-1) for out in tensor_list]
        chunk_numel = _chunk_numel(tensor, max_bytes=_MAX_BYTES)
        for start in range(0, flat_in.numel(), chunk_numel):
            end = min(start + chunk_numel, flat_in.numel())
            _ORIG_ALL_GATHER(
                tensor_list=[out[start:end] for out in flat_outs],
                tensor=flat_in[start:end],
                group=group,
                async_op=False,
            )
        return None

    def chunked_all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
        if _ORIG_ALL_GATHER_INTO_TENSOR is None:
            raise RuntimeError("torch.distributed.all_gather_into_tensor is not available.")
        if not _should_chunk(input_tensor, async_op=bool(async_op), max_bytes=_MAX_BYTES):
            return _ORIG_ALL_GATHER_INTO_TENSOR(
                output_tensor=output_tensor,
                input_tensor=input_tensor,
                group=group,
                async_op=async_op,
            )
        if not output_tensor.is_cuda or not output_tensor.is_contiguous():
            return _ORIG_ALL_GATHER_INTO_TENSOR(
                output_tensor=output_tensor,
                input_tensor=input_tensor,
                group=group,
                async_op=async_op,
            )

        world_size = dist.get_world_size(group=group)
        flat_in = input_tensor.view(-1)
        flat_out = output_tensor.view(-1)
        input_numel = flat_in.numel()
        expected_numel = input_numel * world_size
        if flat_out.numel() != expected_numel:
            return _ORIG_ALL_GATHER_INTO_TENSOR(
                output_tensor=output_tensor,
                input_tensor=input_tensor,
                group=group,
                async_op=async_op,
            )

        chunk_numel = _chunk_numel(input_tensor, max_bytes=_MAX_BYTES)
        for start in range(0, input_numel, chunk_numel):
            end = min(start + chunk_numel, input_numel)
            gathered = torch.empty(
                (world_size, end - start),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
            _ORIG_ALL_GATHER_INTO_TENSOR(
                output_tensor=gathered.view(-1),
                input_tensor=flat_in[start:end],
                group=group,
                async_op=False,
            )
            for rank in range(world_size):
                flat_out[rank * input_numel + start : rank * input_numel + end].copy_(gathered[rank])
        return None

    def chunked_reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if not _should_chunk(output, async_op=bool(async_op), max_bytes=_MAX_BYTES):
            return _ORIG_REDUCE_SCATTER(
                output=output,
                input_list=input_list,
                op=op,
                group=group,
                async_op=async_op,
            )
        if any(
            not isinstance(inp, torch.Tensor)
            or not inp.is_cuda
            or not inp.is_contiguous()
            or inp.numel() != output.numel()
            for inp in input_list
        ):
            return _ORIG_REDUCE_SCATTER(
                output=output,
                input_list=input_list,
                op=op,
                group=group,
                async_op=async_op,
            )

        flat_out = output.view(-1)
        flat_inputs = [inp.view(-1) for inp in input_list]
        chunk_numel = _chunk_numel(output, max_bytes=_MAX_BYTES)
        for start in range(0, flat_out.numel(), chunk_numel):
            end = min(start + chunk_numel, flat_out.numel())
            _ORIG_REDUCE_SCATTER(
                output=flat_out[start:end],
                input_list=[inp[start:end] for inp in flat_inputs],
                op=op,
                group=group,
                async_op=False,
            )
        return None

    def chunked_reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if _ORIG_REDUCE_SCATTER_TENSOR is None:
            raise RuntimeError("torch.distributed.reduce_scatter_tensor is not available.")
        if not _should_chunk(output_tensor, async_op=bool(async_op), max_bytes=_MAX_BYTES):
            return _ORIG_REDUCE_SCATTER_TENSOR(
                output_tensor=output_tensor,
                input_tensor=input_tensor,
                op=op,
                group=group,
                async_op=async_op,
            )
        if not input_tensor.is_cuda or not input_tensor.is_contiguous():
            return _ORIG_REDUCE_SCATTER_TENSOR(
                output_tensor=output_tensor,
                input_tensor=input_tensor,
                op=op,
                group=group,
                async_op=async_op,
            )

        world_size = dist.get_world_size(group=group)
        flat_out = output_tensor.view(-1)
        flat_in = input_tensor.view(-1)
        output_numel = flat_out.numel()
        expected_numel = output_numel * world_size
        if flat_in.numel() != expected_numel:
            return _ORIG_REDUCE_SCATTER_TENSOR(
                output_tensor=output_tensor,
                input_tensor=input_tensor,
                op=op,
                group=group,
                async_op=async_op,
            )

        chunk_numel = _chunk_numel(output_tensor, max_bytes=_MAX_BYTES)
        for start in range(0, output_numel, chunk_numel):
            end = min(start + chunk_numel, output_numel)
            chunk_len = end - start
            scatter_input = torch.empty(
                (world_size, chunk_len),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
            for rank in range(world_size):
                scatter_input[rank].copy_(flat_in[rank * output_numel + start : rank * output_numel + end])
            _ORIG_REDUCE_SCATTER_TENSOR(
                output_tensor=flat_out[start:end],
                input_tensor=scatter_input.view(-1),
                op=op,
                group=group,
                async_op=False,
            )
        return None

    dist.broadcast = chunked_broadcast
    dist.all_reduce = chunked_all_reduce
    dist.reduce = chunked_reduce
    if _ORIG_ALL_REDUCE_COALESCED is not None:
        dist.all_reduce_coalesced = chunked_all_reduce_coalesced
    dist.all_gather = chunked_all_gather
    if _ORIG_ALL_GATHER_INTO_TENSOR is not None:
        dist.all_gather_into_tensor = chunked_all_gather_into_tensor
        c10d.all_gather_into_tensor = chunked_all_gather_into_tensor
    if _ORIG_ALL_GATHER_BASE is not None:
        c10d._all_gather_base = chunked_all_gather_into_tensor
        if hasattr(dist, "_all_gather_base"):
            dist._all_gather_base = chunked_all_gather_into_tensor
    dist.reduce_scatter = chunked_reduce_scatter
    if _ORIG_REDUCE_SCATTER_TENSOR is not None:
        dist.reduce_scatter_tensor = chunked_reduce_scatter_tensor
        c10d.reduce_scatter_tensor = chunked_reduce_scatter_tensor
    if _ORIG_REDUCE_SCATTER_BASE is not None:
        c10d._reduce_scatter_base = chunked_reduce_scatter_tensor
        if hasattr(dist, "_reduce_scatter_base"):
            dist._reduce_scatter_base = chunked_reduce_scatter_tensor
    _patch_deepspeed_cached_collectives()
    _PATCHED = True
    logger.warning(
        "Installed chunked torch.distributed CUDA collectives for tensors larger than %d bytes.",
        _MAX_BYTES,
    )
    return True


def chunked_collectives_installed() -> bool:
    return _PATCHED
