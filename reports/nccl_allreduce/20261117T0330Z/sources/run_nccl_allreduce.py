"""Only the missing two-GPU NCCL All-Reduce benchmark."""
import datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/cs336-assignment2')
OUT=ROOT/'reports/nccl_allreduce/20261117T0330Z'
OUT.mkdir(parents=True,exist_ok=True)
os.chdir(ROOT)
os.environ.update(CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='1',NCCL_DEBUG='WARN')
def progress(message):
    print(message,flush=True)
    with (OUT/'progress.log').open('a') as stream: stream.write(message+'\n')
progress('NCCL ONLY: driver started')
import torch
env=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.executable,torch=torch.__version__,
    cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),gpus=[torch.cuda.get_device_name(i) for i in range(2)],
    memory_bytes=[torch.cuda.get_device_properties(i).total_memory for i in range(2)],
    environment={k:os.environ[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','NCCL_DEBUG']},
    scope='Only two-GPU All-Reduce; no DDP/sharding/FSDP/single-GPU experiments',
    warmup=5,measured_iterations=20,independent_repetitions=3,units='decimal MB/GB')
(OUT/'environment.json').write_text(json.dumps(env,indent=2))
for name,args in [('topology',['nvidia-smi','topo','-m']),('gpu-before',['nvidia-smi']),('processes-before',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'])]:
    (OUT/(name+'.txt')).write_text(subprocess.check_output(args,text=True))
for name in ['benchmark_nccl_allreduce.py','run_nccl_allreduce.py']:
    dest=OUT/'sources'/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes((ROOT/'scripts'/name).read_bytes())
start=time.monotonic()
for name,warmup,steps in [('smoke',1,2),('allreduce-1',5,20),('allreduce-2',5,20),('allreduce-3',5,20)]:
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2','--module','scripts.benchmark_nccl_allreduce',
        '--warmup',str(warmup),'--steps',str(steps),'--output',str(OUT/(name+'.json'))]
    meta=dict(argv=command,cwd=str(ROOT),environment=env['environment'],start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT/(name+'.command.json')).write_text(json.dumps(meta,indent=2))
    progress('START '+name)
    tick=time.monotonic()
    with (OUT/(name+'.stdout.log')).open('w') as stdout,(OUT/(name+'.stderr.log')).open('w') as stderr:
        process=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
        try: code=process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    meta.update(exit_code=code,elapsed_s=time.monotonic()-tick)
    (OUT/(name+'.command.json')).write_text(json.dumps(meta,indent=2))
    (OUT/(name+'.exit_code.txt')).write_text(str(code)+'\n')
    progress('END '+name+' code='+str(code)+' seconds='+str(round(meta['elapsed_s'],2)))
    if code:
        progress((OUT/(name+'.stderr.log')).read_text()[-5000:])
        raise SystemExit(code)
    result=json.loads((OUT/(name+'.json')).read_text())
    assert result['status']=='passed'
    assert all(len(r['ranks'])==2 and all(k['correctness']['incorrect_elements']==0 for k in r['ranks']) for r in result['results'])
for name,args in [('gpu-after',['nvidia-smi']),('processes-after',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'])]:
    (OUT/(name+'.txt')).write_text(subprocess.check_output(args,text=True))
(OUT/'run-status.json').write_text(json.dumps(dict(status='passed',elapsed_s=time.monotonic()-start,instance_action='none; left running'),indent=2))
progress('ALL NCCL EXPERIMENT PROCESSES EXITED; INSTANCE LEFT RUNNING')
hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'}
(OUT/'SHA256SUMS').write_text(''.join(h+'  '+name+'\n' for name,h in hashes.items()))

