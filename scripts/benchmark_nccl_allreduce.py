"""Handout p32: synchronized two-rank FP32 all-reduce latency."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--sizes', type=int, nargs='+', default=[1_000_000,10_000_000,100_000_000,1_000_000_000])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert a.warmup >= 0 and a.steps > 0
    assert all(s > 0 and s % 4 == 0 for s in a.sizes)
    rank,world = int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    assert world == 2
    device = torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group('nccl',device_id=device)
    try:
        results=[]
        for size in a.sizes:
            tensor=torch.empty(size//4,dtype=torch.float32,device=device)
            rows=[]
            for i in range(-a.warmup,a.steps):
                tensor.fill_(rank+1)
                dist.barrier()
                torch.cuda.synchronize(device)
                start=time.perf_counter_ns()
                dist.all_reduce(tensor)
                torch.cuda.synchronize(device)
                end=time.perf_counter_ns()
                if i>=0: rows.append(dict(start_ns=start,end_ns=end,local_ms=(end-start)/1e6))
            incorrect_elements=torch.count_nonzero(tensor != 3).item()
            assert incorrect_elements==0, f'rank {rank}: incorrect reduction'
            correct=dict(checked_elements=tensor.numel(),incorrect_elements=incorrect_elements,expected_value=3.0)
            ranks=[None]*world
            dist.all_gather_object(ranks,dict(rank=rank,pid=os.getpid(),correctness=correct,rows=rows))
            if rank==0:
                durations=[(max(r['rows'][i]['end_ns'] for r in ranks)-min(r['rows'][i]['start_ns'] for r in ranks))/1e6 for i in range(a.steps)]
                assert all(t>0 for t in durations)
                median=statistics.median(durations)
                results.append(dict(bytes=size,ranks=ranks,global_ms=durations,median_ms=median,
                    mean_ms=statistics.mean(durations),std_ms=statistics.stdev(durations) if a.steps>1 else 0.,
                    algorithm_GB_s=size/(median*1e6),bus_GB_s=size/(median*1e6)*2*(world-1)/world))
            del tensor
        if rank==0:
            out=dict(status='passed',world_size=world,dtype='float32',warmup=a.warmup,steps=a.steps,
                torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
                gpu=torch.cuda.get_device_name(device),size_units='decimal MB/GB',
                timing='CUDA-synchronized max(rank end)-min(rank start), shared host clock; barrier and payload reset excluded',
                bandwidth='algorithm=message_bytes/median_seconds; bus=algorithm*2*(N-1)/N; GB=1e9 bytes; not a link counter',
                results=results)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(out,indent=2)+'\n')
            print(json.dumps([dict(bytes=r['bytes'],median_ms=r['median_ms'],algorithm_GB_s=r['algorithm_GB_s']) for r in results]),flush=True)
    finally: dist.destroy_process_group()

if __name__=='__main__': main()

