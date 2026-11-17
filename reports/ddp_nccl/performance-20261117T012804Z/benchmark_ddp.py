"""Two-rank Transformer DDP benchmark and short CUDA-profiler capture."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp import DDP, FlatDDP, OverlapDDP, setup_process_group, cleanup_process_group
from cs336_systems.gpu_experiments import SIZES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation',choices=['DDP','FlatDDP','OverlapDDP'],required=True)
    p.add_argument('--model-size',choices=['small','surrogate','xl'],default='small')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--context-length',type=int,default=512)
    p.add_argument('--vocab-size',type=int,default=10000)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.warmup<0 or min(a.steps,a.batch_size,a.context_length,a.vocab_size)<1:p.error('Invalid dimensions or steps')
    rank,local_rank,world=(int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    if world!=2:p.error('Requires two ranks on one host')
    torch.cuda.set_device(local_rank)
    device=torch.device('cuda',local_rank)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    setup_process_group(rank,world,'env://',backend='nccl')
    try:
        d,ff,layers,heads=SIZES[a.model_size]
        module=BasicsTransformerLM(a.vocab_size,a.context_length,d,layers,heads,ff).to(device)
        model={'DDP':DDP,'FlatDDP':FlatDDP,'OverlapDDP':OverlapDDP}[a.implementation](module)
        model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        torch.manual_seed(1234+rank)
        x=torch.randint(a.vocab_size,(a.batch_size,a.context_length),device=device)
        y=torch.randint(a.vocab_size,x.shape,device=device)
        def region(name,capture):return torch.cuda.nvtx.range(name) if capture else nullcontext()
        def step(index,capture=False):
            dist.barrier()
            torch.cuda.synchronize(device)
            start=time.perf_counter_ns()
            with region(f'rank{rank}/step{index}',capture):
                optimizer.zero_grad(set_to_none=True)
                with region('forward',capture):
                    logits=model(x)
                    loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
                with region('backward',capture):loss.backward()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                with region('gradient_sync',capture):model.finish_gradient_synchronization()
                end.record()
                with region('optimizer',capture):optimizer.step()
                torch.cuda.synchronize(device)
            finish=time.perf_counter_ns()
            row=dict(rank=rank,step=index,start_ns=start,end_ns=finish,local_ms=(finish-start)/1e6,gradient_sync_tail_ms=begin.elapsed_time(end))
            del logits
            return row,loss.detach()
        for i in range(a.warmup):_,loss=step(i-a.warmup)
        dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        if a.profile:torch.cuda.cudart().cudaProfilerStart()
        records=[]
        with torch.autograd.profiler.emit_nvtx() if a.profile else nullcontext():
            for i in range(a.steps):
                row,loss=step(i,a.profile)
                records.append(row)
        dist.barrier()
        torch.cuda.synchronize(device)
        if a.profile:torch.cuda.cudart().cudaProfilerStop()
        assert torch.isfinite(loss).item(),'nonfinite loss'
        local=dict(rank=rank,pid=os.getpid(),device=local_rank,records=records,final_loss=loss.item(),peak_memory_bytes=torch.cuda.max_memory_allocated(device))
        ranks=[None]*world
        dist.all_gather_object(ranks,local)
        if rank==0:
            # perf_counter_ns uses the same monotonic clock in both local processes.
            durations=[(max(r['records'][i]['end_ns'] for r in ranks)-min(r['records'][i]['start_ns'] for r in ranks))/1e6 for i in range(a.steps)]
            assert all(t>0 for t in durations)
            tokens=world*a.batch_size*a.context_length
            result=dict(status='ok',arguments=vars(a),dtype='float32',optimizer='torch.optim.AdamW',lr=1e-4,seed=42,tf32=False,
                torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device),world_size=world,
                parameters=sum(p.numel() for p in module.parameters()),shape=dict(d_model=d,d_ff=ff,num_layers=layers,num_heads=heads),
                ranks=ranks,global_step_ms=durations,median_ms=statistics.median(durations),min_ms=min(durations),max_ms=max(durations),
                mean_ms=statistics.mean(durations),std_ms=statistics.stdev(durations) if len(durations)>1 else 0.,
                global_tokens_per_step=tokens,global_tokens_per_s=tokens*a.steps*1000/sum(durations),
                gradient_sync_tail_ms=[max(r['records'][i]['gradient_sync_tail_ms'] for r in ranks) for i in range(a.steps)],
                timing_definition='max(rank end_ns)-min(rank start_ns), shared host monotonic clock; zero_grad+forward+loss+backward+gradient sync+optimizer+CUDA synchronize; barrier excluded',
                sync_tail_definition='CUDA events around finish_gradient_synchronization including pack/copy/wait; OverlapDDP excludes communication already executed in backward')
            out=Path(a.output)
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:result[k] for k in ('status','median_ms','global_tokens_per_s','parameters')}),flush=True)
    finally:cleanup_process_group()
if __name__=='__main__':main()
