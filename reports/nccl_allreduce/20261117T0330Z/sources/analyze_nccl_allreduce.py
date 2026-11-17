"""Offline NCCL benchmark aggregation; does not initialize CUDA."""
import hashlib,json,statistics,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(sys.argv[1])
load=lambda name:json.loads((root/(name+'.json')).read_text(encoding='utf-8-sig'))
runs=[load('allreduce-'+str(i)) for i in range(1,4)]
expected=[1_000_000,10_000_000,100_000_000,1_000_000_000]
for i,run in enumerate(runs,1):
    assert run['world_size']==2 and run['dtype']=='float32' and run['warmup']==5 and run['steps']==20
    assert [r['bytes'] for r in run['results']]==expected
    assert (root/('allreduce-'+str(i)+'.exit_code.txt')).read_text().strip()=='0'
    for item in run['results']:
        assert {r['rank'] for r in item['ranks']}=={0,1}
        for rank in item['ranks']:
            assert rank['correctness']['incorrect_elements']==0
            assert rank['correctness']['checked_elements']==item['bytes']//4
        reconstructed=[(max(r['rows'][j]['end_ns'] for r in item['ranks'])-min(r['rows'][j]['start_ns'] for r in item['ranks']))/1e6 for j in range(20)]
        assert reconstructed==item['global_ms']
assert (root/'smoke.exit_code.txt').read_text().strip()=='0'
rows=[]
for index,size in enumerate(expected):
    times=[t for run in runs for t in run['results'][index]['global_ms']]
    median=statistics.median(times)
    rows.append(dict(bytes=size,samples=len(times),warmup_per_run=5,repetitions=3,median_ms=median,
        mean_ms=statistics.mean(times),std_ms=statistics.stdev(times),min_ms=min(times),max_ms=max(times),
        p5_ms=float(np.percentile(times,5)),p95_ms=float(np.percentile(times,95)),
        algorithm_GB_s=size/(median*1e6),bus_GB_s=size/(median*1e6),
        per_run_median_ms=[r['results'][index]['median_ms'] for r in runs]))
summary=dict(status='passed',scope='two GPU NCCL only',results=rows,correctness='Both ranks, all elements, all sizes, smoke plus three independent runs',
    bandwidth='algorithm GB/s = bytes / (median_ms * 1e6); bus = algorithm * 2*(N-1)/N, identical at N=2; decimal units',
    statistical_unit='one shared-clock, CUDA-synchronized collective completed by both ranks; 60 observations per size, pooled median')
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
with (root/'results.csv').open('w',encoding='utf-8') as stream:
    stream.write('bytes,median_ms,mean_ms,std_ms,min_ms,max_ms,p5_ms,p95_ms,algorithm_GB_s,samples\n')
    for r in rows: stream.write(','.join(str(r[k]) for k in ['bytes','median_ms','mean_ms','std_ms','min_ms','max_ms','p5_ms','p95_ms','algorithm_GB_s','samples'])+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
x=np.array(expected)/1e6
for ax in axes:
    ax.set_xscale('log');ax.set_xticks(x,['1 MB','10 MB','100 MB','1 GB'])
    ax.grid(alpha=.25);ax.set_xlabel('FP32 tensor size (decimal bytes)')
lat=np.array([r['median_ms'] for r in rows])
axes[0].errorbar(x,lat,yerr=[lat-np.array([r['min_ms'] for r in rows]),np.array([r['max_ms'] for r in rows])-lat],fmt='o-',capsize=4,color='#1f77b4',label='Median; min-max')
axes[0].set_yscale('log');axes[0].set_ylabel('Completed all-reduce latency (ms)');axes[0].legend()
band=np.array([r['algorithm_GB_s'] for r in rows])
axes[1].errorbar(x,band,yerr=[band-np.array([r['bytes']/(r['max_ms']*1e6) for r in rows]),np.array([r['bytes']/(r['min_ms']*1e6) for r in rows])-band],fmt='s-',capsize=4,color='#d55e00',label='Effective bandwidth')
axes[1].set_ylim(bottom=0);axes[1].set_ylabel('Algorithm bandwidth (GB/s)');axes[1].legend()
fig.suptitle('2 x RTX 4090 | NCCL FP32 All-Reduce | 3 runs x 20 samples')
fig.savefig(root/'allreduce-latency-bandwidth.png',dpi=180)
fig.savefig(root/'allreduce-latency-bandwidth.svg')
plt.close(fig)
env=load('environment')
commands=[load('allreduce-'+str(i)+'.command') for i in range(1,4)]
lines=['# 双 RTX 4090 NCCL All-Reduce Benchmark','',
'仅运行 NCCL All-Reduce；DDP、Optimizer Sharding、FSDP、单卡实验、4/6 GPU 和 B200 均未运行。','',
'## 讲义与配置','',
'handout 第 32 页 distributed_communication_single_node 要求 FP32 的 1MB、10MB、100MB、1GB，比较 2/4/6 GPU，每次运行少于 5 分钟；建议 5 次 warmup、CUDA synchronize、汇总各 Rank。讲义没有固定测量次数或统计量。本轮只完成实际可用的 2 GPU 档，采用十进制 MB/GB（1MB=1,000,000 bytes），每档 5 次 warmup + 20 次测量，独立重启 torchrun 重复 3 轮，共 60 个正式样本；正式测试前四档消息均通过 1 warmup + 2 次测量的冒烟。','',
f"环境：Python={env['python']}，PyTorch={env['torch']}，CUDA={env['cuda']}，NCCL={env['nccl']}；CUDA_VISIBLE_DEVICES=0,1，OMP_NUM_THREADS=1。没有安装或升级远端依赖。实际命令、环境、UTC 时间、退出码和 stdout/stderr 均见 *.command.json 与对应日志。三轮正式进程总时长分别为 "+', '.join(f"{c['elapsed_s']:.2f}s" for c in commands)+'，均小于 5 分钟。','',
'## 计时和带宽口径','',
'每次通信前重新填充 Rank 0=1、Rank 1=2；barrier 后同步 CUDA，再记录 perf_counter_ns；dist.all_reduce(SUM) 返回后再次 CUDA synchronize，最后记录结束时间。每步耗时取 max(两个 Rank 的结束时间)-min(两个 Rank 的开始时间)，使用同节点共享 monotonic clock；包括通信完成等待和 API/同步开销，排除 barrier、填充和结果验证。原始双 Rank 起止时间全部保存，并由分析脚本逐项重算验证。','',
'有效带宽 algorithm GB/s = 消息字节数 / (延迟中位数 ms × 10^6)。NCCL 常用归一化 bus GB/s = algorithm GB/s × 2(N-1)/N；N=2 时二者相同。这是由 collective 耗时计算的有效吞吐，不是物理链路计数器读数。波动范围为 60 个双 Rank 完成时间的 min–max；JSON 另提供 mean、样本标准差、P5/P95 和三次独立中位数。','',
'## 实测结果','',
'| 消息 | 延迟中位数 ms | min–max ms | 标准差 ms | 有效带宽 GB/s | 样本数 |','|---|---:|---:|---:|---:|---:|']
for name,r in zip(['1MB','10MB','100MB','1GB'],rows):
    lines.append(f"| {name} | {r['median_ms']:.4f} | {r['min_ms']:.4f}–{r['max_ms']:.4f} | {r['std_ms']:.4f} | {r['algorithm_GB_s']:.3f} | {r['samples']} |")
lines+=['','![延迟与有效带宽](allreduce-latency-bandwidth.png)','',
'## 性能趋势与限制','',
f"从 1MB 到 1GB，延迟由 {rows[0]['median_ms']:.4f}ms 增至 {rows[-1]['median_ms']:.4f}ms，有效带宽由 {rows[0]['algorithm_GB_s']:.3f} 增至 {rows[-1]['algorithm_GB_s']:.3f} GB/s。小消息更容易受 collective 启动、进程调度和 CUDA 同步的固定开销影响；大消息主要反映实际可用通信路径的吞吐。", '',
'本机 GPU 间 topology 为 NODE，跨同 NUMA 节点的 PCIe host bridges，无 NVLink。该拓扑及大消息有效带宽与通信路径限制相容；仅凭此测试不能区分 PCIe、host staging、CPU 内存或 NCCL 算法各自贡献，未把有效吞吐当作某条物理链路的带宽。4/6 GPU 的扩展曲线未测，双 4090 数据不替代其他卡数或 B200。','',
'## 正确性与退出','',
'每种消息在每轮最后一次 All-Reduce 后，两个 Rank 都检查整个 FP32 张量是否逐元素等于 3；检查 250,000 / 2,500,000 / 25,000,000 / 250,000,000 个元素，错误元素数全部为 0。冒烟及三轮正式运行所有 worker 均通过 destroy_process_group 正常退出，torchrun 退出码均为 0。实验结束不执行任何实例关机、释放或重启操作；最终 GPU 空闲状态见 idle-after.txt。','',
'## 原始文件与同步','',
'*.json 保留配置、原始时间、逐 Rank 正确性与版本；*.command.json / *.stdout.log / *.stderr.log / *.exit_code.txt 保存实际执行记录，sources/ 为运行源码快照。远端原始文件的 SHA256SUMS 在 WSL 下载后逐文件核验，结果见 SYNC_VERIFICATION.json；本地生成的报告与图表单独列入 LOCAL_SHA256SUMS。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))

