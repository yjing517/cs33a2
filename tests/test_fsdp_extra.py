"""CPU/Gloo coverage for padded shards and replicated biases in native layers."""
from copy import deepcopy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_systems.fsdp import FSDP


def _worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        torch.manual_seed(7)
        reference=torch.nn.Sequential(torch.nn.Embedding(7,5),torch.nn.Tanh(),torch.nn.Linear(5,3),torch.nn.ReLU(),torch.nn.Linear(3,7))
        model=FSDP(deepcopy(reference))
        plain=torch.optim.SGD(reference.parameters(),lr=.02)
        opt=torch.optim.SGD(model.parameters(),lr=.02)
        x=torch.randint(7,(8,4))
        for _ in range(3):
            plain.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
            reference(x).square().mean().backward();plain.step()
            model(x.chunk(2)[rank]).square().mean().backward();model.finish_gradient_synchronization();opt.step()
            full=model.gather_full_params()
            for name,param in reference.named_parameters():torch.testing.assert_close(full[name],param,atol=1e-6,rtol=1e-4)
            for param,record in model._weights.items():
                assert param.numel()==(record.numel+1)//2
                assert param.grad.shape==param.shape
            assert all(layer.pending is None for layer in model._layers)
    finally:dist.destroy_process_group()


def test_padded_shards_native_linear_embedding_and_bias(tmp_path):
    mp.spawn(_worker,args=(str(tmp_path/'init'),),nprocs=2,join=True)
