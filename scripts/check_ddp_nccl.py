"""Small two-GPU correctness preflight; run with torchrun, never a benchmark."""

import argparse
from copy import deepcopy
import json
import os

import torch
import torch.distributed as dist

from cs336_systems.ddp import DDP, FlatDDP, OverlapDDP, cleanup_process_group, setup_process_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This preflight requires two visible CUDA GPUs")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("Launch exactly two workers")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    setup_process_group(rank, world_size, "env://", backend="nccl")
    try:
        torch.manual_seed(100 + rank)
        module = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4)).to(device=device, dtype=dtype)
        basic = DDP(module)
        flat = FlatDDP(deepcopy(module))
        overlap = OverlapDDP(deepcopy(module))
        reference = deepcopy(module)
        models = [basic, flat, overlap, reference]
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9) for model in models]
        generator = torch.Generator().manual_seed(42)
        tolerance = dict(rtol=1e-4, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-12)
        errors = torch.zeros(3, device=device, dtype=torch.float64)
        for step in range(args.steps):
            inputs = torch.randn(8, 8, generator=generator, dtype=dtype).to(device)
            targets = torch.randn(8, 4, generator=generator, dtype=dtype).to(device)
            losses = []
            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad(set_to_none=step % 2 == 0)
                local_inputs = inputs if model is reference else inputs.chunk(world_size)[rank]
                local_targets = targets if model is reference else targets.chunk(world_size)[rank]
                loss = torch.nn.functional.mse_loss(model(local_inputs), local_targets)
                loss.backward()
                losses.append(loss.detach())
                if model is not reference:
                    model.finish_gradient_synchronization()
            for loss in losses[:-1]:
                dist.all_reduce(loss)
                loss.div_(world_size)
                torch.testing.assert_close(loss, losses[-1], **tolerance)
                errors[0] = torch.maximum(errors[0], (loss - losses[-1]).abs().double())
            for parameters in zip(*(model.parameters() for model in models)):
                for parameter in parameters[:-1]:
                    torch.testing.assert_close(parameter.grad, parameters[-1].grad, **tolerance)
                    errors[1] = torch.maximum(errors[1], (parameter.grad - parameters[-1].grad).abs().max().double())
            for optimizer in optimizers:
                optimizer.step()
            for parameters in zip(*(model.parameters() for model in models)):
                for parameter in parameters[:-1]:
                    torch.testing.assert_close(parameter, parameters[-1], **tolerance)
                    errors[2] = torch.maximum(errors[2], (parameter - parameters[-1]).abs().max().double())
                replicas = [torch.empty_like(parameters[2]) for _ in range(world_size)]
                dist.all_gather(replicas, parameters[2].detach())
                torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(json.dumps(dict(status="passed", backend=dist.get_backend(), world_size=world_size,
                                  dtype=args.dtype, steps=args.steps, torch_version=torch.__version__,
                                  gpu=torch.cuda.get_device_name(local_rank),
                                  max_errors=dict(zip(["loss", "gradient", "parameter"], errors.tolist())))), flush=True)
    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
