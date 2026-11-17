"""Compare ordinary DDP/AdamW and optimizer state sharding, without benchmarking."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from cs336_systems.ddp import DDP, setup_process_group, cleanup_process_group
from cs336_systems.sharded_optimizer import ShardedOptimizer


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    return deepcopy(value)


def state_bytes(optimizer, cuda_only=False):
    return sum(v.numel() * v.element_size() for state in optimizer.state.values() for v in state.values()
               if isinstance(v, torch.Tensor) and (not cuda_only or v.is_cuda))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['gloo', 'nccl'], required=True)
    parser.add_argument('--model', choices=['mlp', 'transformer'], default='mlp')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 2:
        parser.error('Requires exactly two ranks')
    torch.set_num_threads(1)
    dtype = getattr(torch, args.dtype)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if args.backend == 'nccl' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    if args.model == 'transformer':
        from cs336_basics.model import BasicsTransformerLM
        initial = BasicsTransformerLM(10000, 128, 256, 4, 4, 1024).to(dtype=dtype)
    else:
        initial = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 4)).to(dtype=dtype)
    setup_process_group(rank, world, 'env://', backend=args.backend)
    expected = []
    reference_states = {}
    errors = dict(loss=0., gradient=0., parameter=0., optimizer_state=0.)
    tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-12)

    def memory():
        if device.type != 'cuda':
            return None
        torch.cuda.synchronize(device)
        return dict(allocated_bytes=torch.cuda.memory_allocated(device), peak_allocated_bytes=torch.cuda.max_memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device))

    def assert_close(actual, target, kind):
        actual = actual.detach().cpu()
        torch.testing.assert_close(actual, target, **tolerance)
        errors[kind] = max(errors[kind], (actual - target).abs().max().item())

    def train(sharded):
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = DDP(deepcopy(initial).to(device))
        options = dict(lr=.001, weight_decay=.03, betas=(.9, .99))
        optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, **options) if sharded else torch.optim.AdamW(model.parameters(), **options)
        names = dict(model.module.named_parameters())
        points = {'after_model_init': memory()}
        generator = torch.Generator().manual_seed(1234 + rank)
        losses = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=step % 2 == 0)
            if args.model == 'transformer':
                x = torch.randint(10000, (2, 128), generator=generator).to(device)
                y = torch.randint(10000, (2, 128), generator=generator).to(device)
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
            else:
                x = torch.randn(8, 8, generator=generator, dtype=dtype).to(device)
                y = torch.randn(8, 4, generator=generator, dtype=dtype).to(device)
                logits = model(x)
                loss = torch.nn.functional.mse_loss(logits, y)
            loss.backward()
            model.finish_gradient_synchronization()
            if sharded:
                assert_close(loss, expected[step]['loss'], 'loss')
                for name, parameter in names.items():
                    assert_close(parameter.grad, expected[step]['gradients'][name], 'gradient')
            else:
                expected.append(dict(loss=cpu_copy(loss), gradients={n: cpu_copy(p.grad) for n, p in names.items()}))
            if step == 0:
                points['before_optimizer_step'] = memory()
            optimizer.step()
            if step == 0:
                points['after_optimizer_step'] = memory()
            for name, parameter in names.items():
                if sharded:
                    assert_close(parameter, expected[step]['parameters'][name], 'parameter')
                replicas = [torch.empty_like(parameter) for _ in range(world)]
                dist.all_gather(replicas, parameter.detach())
                torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
                del replicas
            if sharded:
                assert all(optimizer.parameter_owners[p] == rank for p in optimizer.state)
            else:
                expected[step]['parameters'] = {n: cpu_copy(p) for n, p in names.items()}
            losses.append(loss.item())
            del x, y, logits, loss
        state_names = [n for n, p in names.items() if p in optimizer.state]
        if sharded:
            assert set(optimizer.state) == {p for p in names.values() if optimizer.parameter_owners[p] == rank}
            for name, parameter in names.items():
                if parameter not in optimizer.state:
                    continue
                for key, value in optimizer.state[parameter].items():
                    if isinstance(value, torch.Tensor):
                        assert_close(value, reference_states[name][key], 'optimizer_state')
                    else:
                        assert value == reference_states[name][key]
        else:
            reference_states.update({n: cpu_copy(optimizer.state[p]) for n, p in names.items()})
        result = dict(rank=rank, device=str(device), losses=losses, state_parameter_names=state_names,
            state_parameter_numel=sum(names[n].numel() for n in state_names), state_bytes=state_bytes(optimizer),
            state_cuda_bytes=state_bytes(optimizer, cuda_only=True), memory_points_first_step=points,
            parameter_bytes=sum(p.numel()*p.element_size() for p in names.values()), gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for p in names.values()),
            ownership={n: {'owner': optimizer.parameter_owners[p], 'numel': p.numel()} for n, p in names.items()} if sharded else None)
        del model, optimizer
        return result

    try:
        baseline = train(False)
        sharded = train(True)
        local = dict(rank=rank, baseline=baseline, sharded=sharded, max_errors=errors)
        ranks = [None] * world
        dist.all_gather_object(ranks, local)
        ownership = [r['sharded']['ownership'] for r in ranks]
        assert ownership[0] == ownership[1]
        a, b = (set(r['sharded']['state_parameter_names']) for r in ranks)
        assert not a.intersection(b)
        assert a.union(b) == set(ownership[0])
        assert sum(r['sharded']['state_bytes'] for r in ranks) == ranks[0]['baseline']['state_bytes']
        if rank == 0:
            result = dict(status='passed', arguments={**vars(args), 'output': str(args.output)}, backend=dist.get_backend(), world_size=world,
                torch_version=torch.__version__, gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                ownership_policy='Initial largest-first assignment of unique whole tensors to least-loaded rank; stable registration-order ties and lower-rank load ties; dynamic groups preserve old assignments',
                replicas_exactly_equal_every_step=True, partition_disjoint_and_complete=True, states_equal_to_unsharded=True,
                memory_definition='Separate sequential baseline/sharded phases; first-step CUDA allocation/peak snapshots before replica-validation collectives; CPU references do not occupy GPU memory; NCCL context memory is not counted by PyTorch allocator', ranks=ranks)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(dict(status='passed', backend=args.backend, model=args.model, dtype=args.dtype, state_bytes=[r['sharded']['state_bytes'] for r in ranks], baseline_state_bytes=[r['baseline']['state_bytes'] for r in ranks], max_errors=errors)), flush=True)
    finally:
        cleanup_process_group()


if __name__ == '__main__':
    main()
