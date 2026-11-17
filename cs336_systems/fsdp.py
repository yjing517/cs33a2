"""Layer-wise weight/gradient sharding with FP32 masters and bounded prefetch."""
from types import MethodType
import weakref

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from cs336_basics.model import Linear, Embedding


class _Weight:
    def __init__(self, parameter, rank, world):
        self.parameter = parameter
        self.shape = tuple(parameter.shape)
        self.numel = parameter.numel()
        self.chunk = (self.numel + world - 1) // world
        flat = F.pad(parameter.detach().reshape(-1), (0, self.chunk * world - self.numel))
        parameter.data = flat.narrow(0, rank * self.chunk, self.chunk).clone()

    def gather(self, dtype):
        source = self.parameter.detach().to(dtype).contiguous()
        output = torch.empty(self.chunk * dist.get_world_size(), device=source.device, dtype=dtype)
        work = dist.all_gather_into_tensor(output, source, async_op=True)
        return output, work, source


class _Layer:
    def __init__(self, owner, weight, module):
        self.owner = weakref.ref(owner)
        self.weight = weight
        self.embedding = isinstance(module, (Embedding, nn.Embedding))
        self.padding_idx = getattr(module, 'padding_idx', None)
        self.scale_grad_by_freq = getattr(module, 'scale_grad_by_freq', False)
        if getattr(module, 'max_norm', None) is not None or getattr(module, 'sparse', False):
            raise ValueError('FSDP requires dense embeddings without max_norm')
        self.pending = None

    def prefetch(self):
        if self.pending is None:
            dtype = self.owner().compute_dtype or self.weight.parameter.dtype
            self.pending = self.weight.gather(dtype)

    def consume(self):
        self.prefetch()
        output, work, source = self.pending
        work.wait()
        self.pending = None
        return output[:self.weight.numel].view(self.weight.shape)

    def discard(self):
        if self.pending is not None:
            self.pending[1].wait()
            self.pending = None


class _Compute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, shard, bias, layer):
        owner = layer.owner()
        weight = layer.consume()
        ctx.layer = layer
        ctx.has_bias = bias is not None
        ctx.save_for_backward(inputs)
        if layer.embedding:
            result = F.embedding(inputs, weight, padding_idx=layer.padding_idx, scale_grad_by_freq=layer.scale_grad_by_freq)
        else:
            result = F.linear(inputs.to(weight.dtype), weight, bias.to(weight.dtype) if bias is not None else None)
        del weight
        owner._completed(layer, backward=False)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        layer = ctx.layer
        owner = layer.owner()
        owner._start_backward()
        weight = layer.consume()
        inputs, = ctx.saved_tensors
        grad_output = grad_output.to(weight.dtype)
        if layer.embedding:
            grad_input = None
            gradient = torch.ops.aten.embedding_dense_backward(grad_output, inputs, layer.weight.shape[0], layer.padding_idx if layer.padding_idx is not None else -1, layer.scale_grad_by_freq)
            grad_bias = None
        else:
            flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
            flat_input = inputs.to(weight.dtype).reshape(-1, inputs.shape[-1])
            grad_input = grad_output.matmul(weight).to(inputs.dtype) if ctx.needs_input_grad[0] else None
            gradient = flat_grad.t().matmul(flat_input)
            grad_bias = flat_grad.sum(0).float() if ctx.has_bias else None
        del weight
        full_gradient = F.pad(gradient.float().reshape(-1), (0, layer.weight.chunk * owner.world_size - layer.weight.numel))
        local_gradient = torch.empty_like(layer.weight.parameter)
        work = dist.reduce_scatter_tensor(local_gradient, full_gradient, async_op=True)
        # NCCL wait adds a stream dependency; it does not synchronize the CPU/device.
        # Consume each full gradient here rather than retain every layer's full buffer.
        work.wait()
        local_gradient.div_(owner.world_size)
        del gradient, full_gradient
        owner._completed(layer, backward=True)
        return grad_input, local_gradient if ctx.needs_input_grad[1] else None, grad_bias, None


def _forward(layer):
    def call(module, inputs):
        return _Compute.apply(inputs, module.weight, getattr(module, 'bias', None), layer)
    return call


class FSDP(nn.Module):
    """Shard Linear/Embedding weights; replicate other parameters and average gradients.

    The first iteration discovers execution order. Later iterations prefetch at most
    two layers: gathering layer i starts only after layer i-2 has completed compute.
    One forward/backward per synchronization is supported, as in the handout.
    """
    def __init__(self, module, compute_dtype=None):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError('Initialize the process group before FSDP')
        self.module = module.float()
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        for value in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(value.detach(), src=0)
        self._weights = {}
        self._layers = []
        for child in module.modules():
            if isinstance(child, (Linear, Embedding, nn.Linear, nn.Embedding)):
                parameter = child.weight
                if parameter not in self._weights:
                    self._weights[parameter] = _Weight(parameter, self.rank, self.world_size)
                layer = _Layer(self, self._weights[parameter], child)
                child.forward = MethodType(_forward(layer), child)
                self._layers.append(layer)
        self._replicated = [p for p in module.parameters() if p not in self._weights]
        self._forward_order = []
        self._backward_order = []
        self._current_forward = []
        self._current_backward = []
        self._backward_started = False

    def _completed(self, layer, backward):
        current = self._current_backward if backward else self._current_forward
        order = self._backward_order if backward else self._forward_order
        current.append(layer)
        position = len(current) - 1
        if position < len(order) and order[position] is layer and position + 2 < len(order):
            order[position + 2].prefetch()

    def _start_backward(self):
        if not self._backward_started:
            self._backward_started = True
            for layer in self._backward_order[:2]:
                layer.prefetch()

    def forward(self, *args, **kwargs):
        self._current_forward = []
        self._current_backward = []
        self._backward_started = False
        for layer in self._forward_order[:2]:
            layer.prefetch()
        device_type = next(self.module.parameters()).device.type
        with torch.autocast(device_type, dtype=self.compute_dtype, enabled=self.compute_dtype in (torch.float16, torch.bfloat16)):
            result = self.module(*args, **kwargs)
        for layer in self._layers:
            layer.discard()
        self._forward_order = list(self._current_forward)
        return result

    def finish_gradient_synchronization(self):
        for layer in self._layers:
            layer.discard()
        for parameter in self._replicated:
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(self.world_size)
        self._backward_order = list(self._current_backward)

    @torch.no_grad()
    def gather_full_params(self):
        result = {}
        for name, parameter in self.module.named_parameters():
            if parameter in self._weights:
                record = self._weights[parameter]
                full, work, source = record.gather(parameter.dtype)
                work.wait()
                result[name] = full[:record.numel].view(record.shape).clone()
            else:
                result[name] = parameter.detach().clone()
        return result
