"""Four-way numerical and asynchronous lifecycle checks on CPU/Gloo."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.ddp import DDP, FlatDDP, OverlapDDP, cleanup_process_group, setup_process_group
from .common import ToyModel, ToyModelWithTiedWeights


class _TrackedWork:
    def __init__(self, work):
        self.work = work
        self.waited = False

    def wait(self):
        result = self.work.wait()
        self.waited = True
        return result


def _worker(rank, rendezvous, model_class, dtype):
    torch.set_num_threads(1)
    setup_process_group(rank, 2, rendezvous)
    try:
        torch.manual_seed(100 + rank)
        baseline = DDP(model_class().to(dtype=dtype))
        flat = FlatDDP(deepcopy(baseline.module))
        overlap = OverlapDDP(deepcopy(baseline.module))
        reference = deepcopy(baseline.module)
        models = [baseline, flat, overlap, reference]
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) for model in models]
        generator = torch.Generator().manual_seed(42)
        tolerance = dict(rtol=1e-10, atol=1e-12) if dtype == torch.float64 else dict(rtol=1e-4, atol=1e-7)
        errors = dict(loss=0.0, gradient=0.0, parameter=0.0, overlap_vs_basic_gradient=0.0, overlap_vs_flat_parameter=0.0)
        real_all_reduce = dist.all_reduce
        for step in range(5):
            inputs = torch.randn(8, 10, generator=generator, dtype=dtype)
            targets = torch.randn(8, 10, generator=generator, dtype=dtype)
            losses = []
            requests = []
            readiness = []

            def launch(tensor, op, async_op):
                assert async_op is True
                readiness.append(len(overlap._ready))
                request = _TrackedWork(real_all_reduce(tensor, op=op, async_op=async_op))
                requests.append(request)
                return request

            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad(set_to_none=step % 2 == 0)
                local_inputs = inputs if model is reference else inputs.chunk(2)[rank]
                local_targets = targets if model is reference else targets.chunk(2)[rank]
                loss = torch.nn.functional.mse_loss(model(local_inputs), local_targets)
                losses.append(loss.detach())
                if model is overlap:
                    with patch.object(dist, "all_reduce", side_effect=launch):
                        loss.backward()
                    assert len(requests) == len(overlap._gradient_parameters)
                    assert min(readiness) < len(overlap._gradient_parameters)
                    assert not any(request.waited for request in requests)
                    with pytest.raises(RuntimeError, match="Finish gradient synchronization"):
                        overlap(local_inputs)
                else:
                    loss.backward()
            for model in [baseline, flat, overlap]:
                model.finish_gradient_synchronization()
            assert all(request.waited for request in requests)
            assert not overlap._ready and not overlap._pending and overlap._next_parameter == 0
            overlap.finish_gradient_synchronization()
            for loss in losses[:-1]:
                dist.all_reduce(loss)
                loss.div_(2)
                torch.testing.assert_close(loss, losses[-1], **tolerance)
                errors["loss"] = max(errors["loss"], (loss - losses[-1]).abs().item())
            for parameters in zip(*(model.parameters() for model in models)):
                expected = parameters[-1]
                for parameter in parameters[:-1]:
                    if expected.grad is None:
                        assert parameter.grad is None
                    else:
                        torch.testing.assert_close(parameter.grad, expected.grad, **tolerance)
                        errors["gradient"] = max(errors["gradient"], (parameter.grad - expected.grad).abs().max().item())
                if expected.grad is not None:
                    torch.testing.assert_close(parameters[2].grad, parameters[0].grad, **tolerance)
                    errors["overlap_vs_basic_gradient"] = max(errors["overlap_vs_basic_gradient"], (parameters[2].grad - parameters[0].grad).abs().max().item())
            for optimizer in optimizers:
                optimizer.step()
            for parameters in zip(*(model.parameters() for model in models)):
                for parameter in parameters[:-1]:
                    torch.testing.assert_close(parameter, parameters[-1], **tolerance)
                    errors["parameter"] = max(errors["parameter"], (parameter - parameters[-1]).abs().max().item())
                torch.testing.assert_close(parameters[2], parameters[1], **tolerance)
                errors["overlap_vs_flat_parameter"] = max(errors["overlap_vs_flat_parameter"], (parameters[2] - parameters[1]).abs().max().item())
                replicas = [torch.empty_like(parameters[2]) for _ in range(2)]
                dist.all_gather(replicas, parameters[2].detach())
                torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
        print(f"{model_class.__name__} dtype={dtype} rank={rank} steps=5 errors={errors}", flush=True)
        module = torch.nn.Linear(2, 2)
        guarded = OverlapDDP(module)
        module.weight.sum().backward()
        with pytest.raises(RuntimeError, match="Not all trainable gradients"):
            guarded.finish_gradient_synchronization()
        module.zero_grad(set_to_none=True)
        guarded(torch.ones(1, 2)).sum().backward()
        guarded.finish_gradient_synchronization()
        assert not guarded._pending
    finally:
        cleanup_process_group()


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_overlap_four_way(tmp_path, model_class, dtype):
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(), model_class, dtype), nprocs=2, join=True)
