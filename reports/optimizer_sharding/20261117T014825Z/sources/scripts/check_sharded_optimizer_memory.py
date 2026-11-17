"""One training step per fresh process pair for uncontaminated allocator snapshots."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp import DDP, setup_process_group, cleanup_process_group
from cs336_systems.sharded_optimizer import ShardedOptimizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sharded', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    setup_process_group(rank, 2, 'env://', backend='nccl')
    try:
        def memory():
            torch.cuda.synchronize(device)
            return dict(allocated_bytes=torch.cuda.memory_allocated(device), peak_allocated_bytes=torch.cuda.max_memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        model = DDP(BasicsTransformerLM(10000, 128, 256, 4, 4, 1024).to(device))
        points = {'after_model_init': memory()}
        options = dict(lr=.001, weight_decay=.03, betas=(.9, .99))
        opt = ShardedOptimizer(model.parameters(), torch.optim.AdamW, **options) if args.sharded else torch.optim.AdamW(model.parameters(), **options)
        torch.manual_seed(1234 + rank)
        x = torch.randint(10000, (2, 128), device=device)
        y = torch.randint(10000, (2, 128), device=device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        model.finish_gradient_synchronization()
        points['before_optimizer_step'] = memory()
        opt.step()
        points['after_optimizer_step'] = memory()
        assert torch.isfinite(loss).item()
        local = dict(rank=rank, mode='sharded' if args.sharded else 'baseline', memory=points,
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for p in model.parameters()),
            state_bytes=sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if isinstance(v, torch.Tensor)),
            state_cuda_bytes=sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if isinstance(v, torch.Tensor) and v.is_cuda))
        if args.sharded:
            assert all(opt.parameter_owners[p] == rank for p in opt.state)
        ranks = [None, None]
        dist.all_gather_object(ranks, local)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(status='passed', fresh_processes=True, torch_version=torch.__version__, ranks=ranks), indent=2)+'\n')
            print(json.dumps(ranks), flush=True)
    finally:
        cleanup_process_group()


if __name__ == '__main__':
    main()
