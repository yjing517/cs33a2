"""Independent-process FSDP memory, bounded timing, and short Nsight experiments."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.fsdp import FSDP
from cs336_systems.ddp import DDP
from tests.test_fsdp import _apply_mixed_precision_hooks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['fsdp','reference'],default='fsdp')
    p.add_argument('--dtype',choices=['float32','float16'],default='float32')
    p.add_argument('--model-size',choices=['small','surrogate'],default='small')
    p.add_argument('--warmup',type=int,default=0)
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert a.steps>0 and a.warmup>=0
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    torch.set_num_threads(1);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',rank=rank,world_size=world,device_id=device)
    try:
        def memory():
            torch.cuda.synchronize(device)
            return dict(allocated_bytes=torch.cuda.memory_allocated(device),max_memory_allocated=torch.cuda.max_memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device),max_memory_reserved=torch.cuda.max_memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        d,ff,layers,heads=(768,3072,12,12) if a.model_size=='small' else (256,1024,4,4)
        module=BasicsTransformerLM(10000,128,d,layers,heads,ff).to(device)
        dtype=torch.float16 if a.dtype=='float16' else None
        if a.mode=='reference':
            if dtype:_apply_mixed_precision_hooks(module,dtype)
            model=DDP(module)
        else:model=FSDP(module,dtype)
        points={'after_model_init':memory()}
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.03,betas=(.9,.99))
        torch.manual_seed(1234+rank)
        x=torch.randint(10000,(2,128),device=device);y=torch.randint(10000,(2,128),device=device)
        counters={'all_gather':{},'reduce_scatter':{},'all_reduce':{}}
        originals={name:getattr(dist,name) for name in ('all_gather_into_tensor','reduce_scatter_tensor','all_reduce')}
        def instrument(name,kind):
            def call(*args,**kwargs):
                tensor=args[0] if kind=='all_reduce' else args[1]
                bucket=counters[kind].setdefault(str(tensor.dtype),dict(calls=0,input_bytes=0))
                bucket['calls']+=1;bucket['input_bytes']+=tensor.numel()*tensor.element_size()
                return originals[name](*args,**kwargs)
            return call
        dist.all_gather_into_tensor=instrument('all_gather_into_tensor','all_gather')
        dist.reduce_scatter_tensor=instrument('reduce_scatter_tensor','reduce_scatter')
        dist.all_reduce=instrument('all_reduce','all_reduce')
        def region(name):return torch.cuda.nvtx.range(name) if a.profile else nullcontext()
        def step(index):
            dist.barrier();torch.cuda.synchronize(device);start=time.perf_counter_ns()
            with region(f'rank{rank}/step{index}'):
                opt.zero_grad(set_to_none=True)
                with region('forward'):
                    with torch.autocast('cuda',dtype=dtype,enabled=dtype is not None):
                        logits=model(x)
                    loss=torch.nn.functional.cross_entropy(logits.float().flatten(0,1),y.flatten())
                with region('backward'):loss.backward()
                with region('gradient_sync'):model.finish_gradient_synchronization()
                if index==0:points['before_optimizer_step']=memory()
                with region('optimizer'):opt.step()
                torch.cuda.synchronize(device)
                if index==0:points['after_optimizer_step']=memory()
            end=time.perf_counter_ns();value=loss.item();del logits,loss
            return dict(start_ns=start,end_ns=end,local_ms=(end-start)/1e6,loss=value)
        for index in range(a.warmup):step(-a.warmup+index)
        counters={k:{} for k in counters}
        dist.barrier();torch.cuda.synchronize(device)
        if a.warmup:torch.cuda.reset_peak_memory_stats(device)
        if a.profile:torch.cuda.cudart().cudaProfilerStart()
        rows=[]
        with torch.autograd.profiler.emit_nvtx() if a.profile else nullcontext():
            for index in range(a.steps):rows.append(step(index))
        dist.barrier();torch.cuda.synchronize(device)
        if a.profile:torch.cuda.cudart().cudaProfilerStop()
        assert all(torch.isfinite(torch.tensor(r['loss'])).item() for r in rows)
        local=dict(rank=rank,pid=os.getpid(),device=device.index,rows=rows,memory_points=points,measurement_peak=memory(),
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for p in model.parameters()),
            state_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor)),communication=counters)
        ranks=[None]*world;dist.all_gather_object(ranks,local)
        if rank==0:
            durations=[(max(r['rows'][i]['end_ns'] for r in ranks)-min(r['rows'][i]['start_ns'] for r in ranks))/1e6 for i in range(a.steps)]
            config=dict(vocab_size=10000,context_length=128,d_model=d,d_ff=ff,num_layers=layers,num_heads=heads,batch_size_per_rank=2,optimizer='torch.optim.AdamW',lr=.001,weight_decay=.03,betas=[.9,.99],master_dtype='float32',compute_dtype=a.dtype,warmup=a.warmup,steps=a.steps,mode=a.mode,profile=a.profile)
            result=dict(status='passed',configuration=config,torch=torch.__version__,ranks=ranks,global_step_ms=durations,median_ms=statistics.median(durations),min_ms=min(durations),max_ms=max(durations),global_tokens_per_s=512*a.steps*1000/sum(durations),timing='max rank end minus min rank start; synchronized CUDA, barrier excluded; first step includes three memory snapshot calls; performance median has 10 measured steps',communication_definition='Python collective input tensor payload bytes, not physical link traffic; all-gather input is a shard, reduce-scatter input is full FP32 gradient; weight traffic casts before collective')
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(status='passed',mode=a.mode,dtype=a.dtype,median_ms=result['median_ms'],global_tokens_per_s=result['global_tokens_per_s'],ranks=[dict(rank=r['rank'],state_bytes=r['state_bytes'],peak=r['measurement_peak']) for r in ranks])),flush=True)
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
