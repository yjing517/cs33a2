"""Replicated parameters/gradients with rank-local optimizer states (ZeRO-1 style)."""

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """Largest-first whole-tensor shards; local updates followed by broadcasts.

    All ranks must supply the same ordered parameter groups and call step/add_param_group
    collectively. Gradient averaging remains the responsibility of the DDP wrapper.
    state_dict/load_state_dict are rank-local checkpoints, with global parameter indices.
    """

    def __init__(self, params, optimizer_cls, **kwargs):
        if not dist.is_initialized():
            raise RuntimeError('Initialize the process group before constructing ShardedOptimizer')
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.parameter_owners = {}
        self.shard_numel = [0] * self.world_size
        self._ordered_parameters = []
        self.local_optimizer = None
        super().__init__(params, dict(kwargs))
        # Largest-first packing avoids a late embedding/projection unbalancing shards.
        # Stable sort preserves registration order for equally sized tensors.
        self.shard_numel = [0] * self.world_size
        for parameter in sorted(self._ordered_parameters, key=lambda p: -p.numel()):
            owner = min(range(self.world_size), key=self.shard_numel.__getitem__)
            self.parameter_owners[parameter] = owner
            self.shard_numel[owner] += parameter.numel()
        groups = [self._local_group(group) for group in self.param_groups]
        # Keep empty groups: a rank may own no parameters but must still participate.
        self.local_optimizer = optimizer_cls(groups, **kwargs)
        self.defaults.update(self.local_optimizer.defaults)
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)
        self.state = self.local_optimizer.state

    def _local_group(self, group):
        return {**group, 'params': [p for p in group['params'] if self.parameter_owners[p] == self.rank]}

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        for parameter in group['params']:
            if parameter not in self.parameter_owners:
                owner = min(range(self.world_size), key=self.shard_numel.__getitem__)
                self.parameter_owners[parameter] = owner
                self.shard_numel[owner] += parameter.numel()
                self._ordered_parameters.append(parameter)
        if self.local_optimizer is not None:
            self.local_optimizer.add_param_group(self._local_group(group))

    def _sync_group_options(self):
        # Schedulers and callers update the public global groups, not inner groups.
        for group, local_group in zip(self.param_groups, self.local_optimizer.param_groups):
            for key, value in group.items():
                if key != 'params':
                    local_group[key] = value

    def step(self, closure=None, **kwargs):
        self._sync_group_options()
        loss = self.local_optimizer.step(closure=closure, **kwargs)
        with torch.no_grad():
            for parameter in self._ordered_parameters:
                dist.broadcast(parameter, src=self.parameter_owners[parameter])
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.local_optimizer.state = self.state
        self._sync_group_options()
