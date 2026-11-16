"""Direct gradient and update checks against a global-batch CPU reference."""

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.ddp import DDP, cleanup_process_group, setup_process_group
from .common import ToyModel, ToyModelWithTiedWeights


def _check_worker(rank, world_size, rendezvous, model_class):
    torch.set_num_threads(1)
    setup_process_group(rank, world_size, rendezvous)
    try:
        torch.manual_seed(100 + rank)
        module = model_class().double()
        module.register_buffer("initial_buffer", torch.tensor(float(rank)))
        distributed = DDP(module)
        assert module.initial_buffer.item() == 0
        reference = deepcopy(module)
        optimizer = torch.optim.SGD(distributed.parameters(), lr=0.1, momentum=0.9)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
        generator = torch.Generator().manual_seed(42)
        max_gradient_error = 0.0
        max_parameter_error = 0.0
        for step in range(5):
            inputs = torch.randn(8, 10, generator=generator, dtype=torch.float64)
            targets = torch.randn(8, 10, generator=generator, dtype=torch.float64)
            local_inputs = inputs.chunk(world_size)[rank]
            local_targets = targets.chunk(world_size)[rank]
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(reference(inputs), targets).backward()
            torch.nn.functional.mse_loss(distributed(local_inputs), local_targets).backward()
            distributed.finish_gradient_synchronization()
            for actual, expected in zip(distributed.parameters(), reference.parameters()):
                if expected.grad is None:
                    assert actual.grad is None
                else:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-10, atol=1e-12)
                    max_gradient_error = max(max_gradient_error, (actual.grad - expected.grad).abs().max().item())
            optimizer.step()
            reference_optimizer.step()
            for actual, expected in zip(distributed.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
                max_parameter_error = max(max_parameter_error, (actual - expected).abs().max().item())
                replicas = [torch.empty_like(actual) for _ in range(world_size)]
                dist.all_gather(replicas, actual.detach())
                for replica in replicas:
                    torch.testing.assert_close(actual, replica, rtol=0, atol=0)
        print(f"{model_class.__name__} rank={rank} steps=5 max_gradient_error={max_gradient_error:.3e} "
              f"max_parameter_error={max_parameter_error:.3e}", flush=True)
    finally:
        cleanup_process_group()


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_cpu_gradients_and_updates(tmp_path, model_class):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_check_worker, args=(2, rendezvous, model_class), nprocs=2, join=True)
