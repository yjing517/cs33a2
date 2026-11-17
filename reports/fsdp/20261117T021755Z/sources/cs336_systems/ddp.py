"""Synchronous per-parameter DDP, as specified in section 5.2."""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn


def setup_process_group(rank: int, world_size: int, init_method: str, backend: str = "gloo") -> None:
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size,
                            init_method=init_method, timeout=timedelta(seconds=60))


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class DDP(nn.Module):
    """Average gradients after backward for equal-sized data shards."""

    def __init__(self, module: nn.Module):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Initialize the process group before constructing DDP")
        self.module = module
        self.world_size = dist.get_world_size()
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)
            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for parameter in self.module.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(self.world_size)


class FlatDDP(DDP):
    """Synchronize dense gradients with one synchronous, contiguous all-reduce."""

    def finish_gradient_synchronization(self) -> None:
        gradients = [parameter.grad for parameter in self.module.parameters()
                     if parameter.requires_grad and parameter.grad is not None]
        if not gradients:
            return
        if any(gradient.layout != torch.strided for gradient in gradients):
            raise ValueError("FlatDDP requires dense gradients")
        if any(gradient.dtype != gradients[0].dtype or gradient.device != gradients[0].device
               for gradient in gradients):
            raise ValueError("FlatDDP requires gradients with the same dtype and device")
        flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)
        offset = 0
        for gradient in gradients:
            gradient.copy_(flat.narrow(0, offset, gradient.numel()).view_as(gradient))
            offset += gradient.numel()


class OverlapDDP(DDP):
    """Launch ordered per-parameter reductions from post-accumulate hooks."""

    def __init__(self, module: nn.Module):
        super().__init__(module)
        self._gradient_parameters = tuple(parameter for parameter in reversed(list(module.parameters())) if parameter.requires_grad)
        self._ready = {}
        self._pending = []
        self._next_parameter = 0
        self._hook_handles = [parameter.register_post_accumulate_grad_hook(
            lambda parameter, index=index: self._on_gradient_ready(index, parameter)
        ) for index, parameter in enumerate(self._gradient_parameters)]

    def forward(self, *args, **kwargs):
        if self._ready or self._pending:
            raise RuntimeError("Finish gradient synchronization before starting another forward")
        return super().forward(*args, **kwargs)

    def _on_gradient_ready(self, index, parameter):
        if index in self._ready:
            raise RuntimeError("Only one backward is supported before gradient synchronization")
        if parameter.grad.layout != torch.strided:
            raise ValueError("OverlapDDP requires dense gradients")
        self._ready[index] = parameter.grad.detach().clone(memory_format=torch.contiguous_format)
        while self._next_parameter in self._ready:
            gradient = self._ready[self._next_parameter]
            work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)
            self._pending.append((self._next_parameter, work, gradient))
            self._next_parameter += 1

    def finish_gradient_synchronization(self) -> None:
        if not self._ready and not self._pending:
            return
        try:
            for index, work, gradient in self._pending:
                work.wait()
            if len(self._ready) != len(self._gradient_parameters):
                raise RuntimeError("Not all trainable gradients are ready; unused parameters are unsupported")
            for index, work, gradient in self._pending:
                gradient.div_(self.world_size)
                self._gradient_parameters[index].grad.copy_(gradient)
        finally:
            self._ready.clear()
            self._pending.clear()
            self._next_parameter = 0
