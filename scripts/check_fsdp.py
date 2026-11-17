"""Fixed-data FSDP correctness on Gloo/NCCL against a full-batch reference."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from tests.test_fsdp import ToyFSDPModel, _apply_mixed_precision_hooks
from cs336_basics.optimizer import AdamW
from cs336_systems.fsdp import FSDP


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['gloo','nccl'],required=True)
    p.add_argument('--model',choices=['toy','transformer'],default='toy')
    p.add_argument('--dtype',choices=['float32','float16'],default='float32')
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--optimizer',choices=['sgd','reference_adamw'],default='sgd')
    p.add_argument('--reference',choices=['full_batch','ddp'],default='full_batch')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==2 and a.steps>0
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda',int(os.environ['LOCAL_RANK'])) if a.backend=='nccl' else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    dist.init_process_group(a.backend,rank=rank,world_size=world,**({'device_id':device} if a.backend=='nccl' else {}))
    try:
        torch.manual_seed(42)
        if a.model=='toy':initial=ToyFSDPModel().to(device)
        else:
            from cs336_basics.model import BasicsTransformerLM
            initial=BasicsTransformerLM(100,16,64,2,4,128).to(device)
        reference=deepcopy(initial)
        compute_dtype=torch.float16 if a.dtype=='float16' else None
        if compute_dtype:_apply_mixed_precision_hooks(reference,compute_dtype)
        model=FSDP(deepcopy(initial),compute_dtype)
        optimizer_cls=torch.optim.SGD if a.optimizer=='sgd' else AdamW
        options=dict(lr=.01) if a.optimizer=='sgd' else dict(lr=.001,weight_decay=.01)
        ref_opt=optimizer_cls(reference.parameters(),**options)
        opt=optimizer_cls(model.parameters(),**options)
        torch.manual_seed(123)
        x=torch.randint(0,100,(20,8),device=device)
        y=torch.randint(0,100,(20,),device=device)
        errors=dict(loss=0.,gradient=0.,parameter=0.)
        tol=dict(atol=1e-6,rtol=1e-4) if compute_dtype is None else dict(atol=1e-4,rtol=1e-4)
        gradient_tol=dict(atol=1e-6,rtol=1e-4) if compute_dtype is None else dict(atol=5e-3,rtol=2e-2)
        losses=[]
        for step in range(a.steps):
            ref_opt.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
            ref_x=x.chunk(world)[rank] if a.reference=='ddp' else x
            ref_y=y.chunk(world)[rank] if a.reference=='ddp' else y
            with torch.autocast(device.type,dtype=compute_dtype,enabled=compute_dtype is not None and a.model=='transformer'):
                ref_out=reference(ref_x)
            ref_loss=torch.nn.functional.cross_entropy(ref_out[:,-1].float(),ref_y)
            ref_loss.backward()
            if a.reference=='ddp':
                for parameter in reference.parameters():
                    dist.all_reduce(parameter.grad);parameter.grad.div_(world)
                dist.all_reduce(ref_loss.detach());ref_loss.data.div_(world)
            out=model(x.chunk(world)[rank])
            loss=torch.nn.functional.cross_entropy(out[:,-1].float(),y.chunk(world)[rank])
            loss.backward();model.finish_gradient_synchronization()
            mean_loss=loss.detach().clone();dist.all_reduce(mean_loss);mean_loss.div_(world)
            errors['loss']=max(errors['loss'],abs(mean_loss.item()-ref_loss.item()))
            torch.testing.assert_close(mean_loss,ref_loss,atol=1e-3 if compute_dtype else 1e-6,rtol=1e-4)
            for name,param in model.module.named_parameters():
                assert param.dtype==torch.float32 and param.grad.dtype==torch.float32
                assert param.grad.shape==param.shape
                if param in model._weights:
                    record=model._weights[param]
                    gathered=[torch.empty_like(param.grad) for _ in range(world)]
                    dist.all_gather(gathered,param.grad)
                    grad=torch.cat(gathered)[:record.numel].view(record.shape)
                else:grad=param.grad
                target=dict(reference.named_parameters())[name].grad
                errors['gradient']=max(errors['gradient'],(grad-target).abs().max().item())
                torch.testing.assert_close(grad,target,**gradient_tol)
            opt.step();ref_opt.step()
            full=model.gather_full_params()
            for name,target in reference.named_parameters():
                errors['parameter']=max(errors['parameter'],(full[name]-target).abs().max().item())
                torch.testing.assert_close(full[name],target,**tol)
                replicas=[torch.empty_like(full[name]) for _ in range(world)]
                dist.all_gather(replicas,full[name])
                torch.testing.assert_close(replicas[0],replicas[1],rtol=0,atol=0)
            assert all(layer.pending is None for layer in model._layers)
            losses.append(dict(reference=ref_loss.item(),fsdp_global=mean_loss.item()))
            torch.manual_seed(42+step);perm=torch.randperm(20)
            x=x[perm];y=y[perm]
        local=dict(rank=rank,max_errors=errors,losses=losses,master_bytes=sum(t.numel()*t.element_size() for t in model.parameters()),
            reference_parameter_bytes=sum(t.numel()*t.element_size() for t in reference.parameters()),
            forward_order_length=len(model._forward_order),backward_order_length=len(model._backward_order))
        ranks=[None]*world;dist.all_gather_object(ranks,local)
        if rank==0:
            result=dict(status='passed',backend=a.backend,model=a.model,compute_dtype=a.dtype,optimizer=a.optimizer,reference=a.reference,steps=a.steps,torch=torch.__version__,ranks=ranks,
                tolerances=dict(parameter=tol,gradient=gradient_tol),full_model_replicas_exactly_equal=True)
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
