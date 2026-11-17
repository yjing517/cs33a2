"""Distributed API regression checks beyond the unmodified official tests."""
from copy import deepcopy
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_systems.sharded_optimizer import ShardedOptimizer


def _worker(rank, rendezvous, optimizer_cls):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        parameters = [torch.nn.Parameter(torch.tensor([1., 2.])), torch.nn.Parameter(torch.tensor([3., 4., 5.]))]
        reference = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
        opt = ShardedOptimizer([{'params': [parameters[0]], 'lr': .01}], optimizer_cls, lr=.1)
        plain = optimizer_cls([{'params': [reference[0]], 'lr': .01}], lr=.1)
        assert opt.shard_numel == [2, 0]
        assert not opt.local_optimizer.param_groups[0]['params'] if rank == 1 else True
        for step in range(5):
            if step == 2:
                opt.add_param_group({'params': [parameters[1]], 'lr': .03})
                plain.add_param_group({'params': [reference[1]], 'lr': .03})
                assert opt.shard_numel == [2, 3]
            if step == 3:
                opt.param_groups[0]['lr'] = plain.param_groups[0]['lr'] = .005
                # Rank-local roundtrip must reconnect the wrapped state dictionary.
                opt.load_state_dict(deepcopy(opt.state_dict()))
                assert opt.state is opt.local_optimizer.state
            count = 1 if step < 2 else 2
            plain.zero_grad(set_to_none=step % 2 == 0)
            expected = sum(p.square().sum() for p in reference[:count])
            expected.backward()
            plain.step()
            calls = []
            def closure():
                calls.append(True)
                opt.zero_grad(set_to_none=step % 2 == 0)
                loss = sum(p.square().sum() for p in parameters[:count])
                loss.backward()
                return loss
            actual = opt.step(closure)
            assert len(calls) == 1
            torch.testing.assert_close(actual, expected)
            for actual_p, expected_p in zip(parameters, reference):
                torch.testing.assert_close(actual_p, expected_p, rtol=0, atol=0)
            assert all(opt.parameter_owners[p] == rank for p in opt.state)
            assert len({id(p) for g in opt.local_optimizer.param_groups for p in g['params']}) == sum(opt.parameter_owners[p] == rank for p in parameters[:count])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('optimizer_cls', [torch.optim.AdamW, torch.optim.SGD])
def test_groups_closure_checkpoint_and_empty_shard(tmp_path, optimizer_cls):
    mp.spawn(_worker, args=(str(tmp_path / 'rendezvous'), optimizer_cls), nprocs=2, join=True)
