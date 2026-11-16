"""Flat DDP checks, including the unmodified official correctness assertions."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.ddp import DDP, FlatDDP, cleanup_process_group, setup_process_group
from . import test_ddp as official
from .common import ToyModel, ToyModelWithTiedWeights


def _official_worker(rank, world_size, model_class):
    with patch.object(official, "get_ddp", FlatDDP):
        try:
            official._test_DistributedDataParallel(rank, world_size, model_class)
        finally:
            cleanup_process_group()


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_flat_official_assertions(model_class):
    mp.spawn(_official_worker, args=(2, model_class), nprocs=2, join=True)


def _compare_worker(rank, rendezvous, model_class, dtype):
    torch.set_num_threads(1)
    setup_process_group(rank, 2, rendezvous)
    try:
        torch.manual_seed(100 + rank)
        baseline = DDP(model_class().to(dtype=dtype))
        flat = FlatDDP(deepcopy(baseline.module))
        reference = deepcopy(baseline.module)
        models = [baseline, flat, reference]
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) for model in models]
        generator = torch.Generator().manual_seed(42)
        tolerance = dict(rtol=1e-10, atol=1e-12) if dtype == torch.float64 else dict(rtol=1e-4, atol=1e-7)
        errors = dict(basic_gradient=0.0, flat_gradient=0.0, basic_parameter=0.0, flat_parameter=0.0,
                      flat_vs_basic_gradient=0.0, flat_vs_basic_parameter=0.0)
        for step in range(5):
            inputs = torch.randn(8, 10, generator=generator, dtype=dtype)
            targets = torch.randn(8, 10, generator=generator, dtype=dtype)
            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad(set_to_none=step % 2 == 0)
                local_inputs = inputs if model is reference else inputs.chunk(2)[rank]
                local_targets = targets if model is reference else targets.chunk(2)[rank]
                torch.nn.functional.mse_loss(model(local_inputs), local_targets).backward()
            with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as calls:
                baseline.finish_gradient_synchronization()
                assert calls.call_count == sum(parameter.grad is not None for parameter in baseline.parameters())
            with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as calls:
                flat.finish_gradient_synchronization()
                assert calls.call_count == 1
                communicated = calls.call_args.args[0]
                assert communicated.is_contiguous() and communicated.ndim == 1
                assert communicated.numel() == sum(parameter.numel() for parameter in flat.parameters() if parameter.grad is not None)
            for basic_parameter, flat_parameter, expected in zip(*(model.parameters() for model in models)):
                if expected.grad is None:
                    assert basic_parameter.grad is None and flat_parameter.grad is None
                    continue
                for label, parameter in [("basic", basic_parameter), ("flat", flat_parameter)]:
                    torch.testing.assert_close(parameter.grad, expected.grad, **tolerance)
                    errors[f"{label}_gradient"] = max(errors[f"{label}_gradient"], (parameter.grad - expected.grad).abs().max().item())
                torch.testing.assert_close(flat_parameter.grad, basic_parameter.grad, **tolerance)
                errors["flat_vs_basic_gradient"] = max(errors["flat_vs_basic_gradient"], (flat_parameter.grad - basic_parameter.grad).abs().max().item())
            for optimizer in optimizers:
                optimizer.step()
            for basic_parameter, flat_parameter, expected in zip(*(model.parameters() for model in models)):
                for label, parameter in [("basic", basic_parameter), ("flat", flat_parameter)]:
                    torch.testing.assert_close(parameter, expected, **tolerance)
                    errors[f"{label}_parameter"] = max(errors[f"{label}_parameter"], (parameter - expected).abs().max().item())
                torch.testing.assert_close(flat_parameter, basic_parameter, **tolerance)
                errors["flat_vs_basic_parameter"] = max(errors["flat_vs_basic_parameter"], (flat_parameter - basic_parameter).abs().max().item())
                replicas = [torch.empty_like(flat_parameter) for _ in range(2)]
                dist.all_gather(replicas, flat_parameter.detach())
                torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
        for parameter in flat.parameters():
            parameter.grad = None
        with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as calls:
            flat.finish_gradient_synchronization()
            assert calls.call_count == 0
        print(f"{model_class.__name__} dtype={dtype} rank={rank} steps=5 errors={errors}", flush=True)
    finally:
        cleanup_process_group()


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flat_basic_reference(tmp_path, model_class, dtype):
    mp.spawn(_compare_worker, args=((tmp_path / "rendezvous").as_uri(), model_class, dtype), nprocs=2, join=True)
