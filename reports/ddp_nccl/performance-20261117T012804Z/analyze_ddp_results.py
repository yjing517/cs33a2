"""Summarize paired-rank timings and actual same-GPU Nsight kernel intersections."""
import argparse
import json
from pathlib import Path
import re
import sqlite3
import statistics


def union_ns(intervals):
    total=0
    end=None
    for a,b in sorted(intervals):
        if end is None or a>end:
            total+=b-a
            end=b
        elif b>end:
            total+=b-end
            end=b
    return total


def intersections(left,right):
    result=[]
    for a in left:
        for b in right:
            start=max(a['start'],b['start'])
            end=min(a['end'],b['end'])
            if start<end:result.append((start,end,a,b))
    return result


def trace_evidence(root,implementation):
    db=root/f'trace-{implementation}.sqlite'
    connection=sqlite3.connect(db)
    connection.row_factory=sqlite3.Row
    rows=[dict(r) for r in connection.execute('''
        SELECT k.start,k.end,k.deviceId,k.globalPid,k.streamId,k.correlationId,
               s.value AS name,r.start AS launch_start,r.globalTid
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
             AND (r.globalTid & -16777216)=k.globalPid
    ''')]
    raw_count=connection.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    assert len(rows)==raw_count,('Kernel launch association missing or duplicated',len(rows),raw_count)
    ranges=[dict(r) for r in connection.execute('''
        SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) AS name
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL
    ''')]
    processes={r['globalPid']:r['pid'] for r in connection.execute('SELECT globalPid,pid FROM PROCESSES')}
    profile=json.loads((root/f'trace-{implementation}.json').read_text())
    rank_by_pid={r['pid']:r['rank'] for r in profile['ranks']}
    result=[]
    for global_pid,device in sorted({(r['globalPid'],r['deviceId']) for r in rows}):
        process_rows=[r for r in rows if r['globalPid']==global_pid and r['deviceId']==device]
        backward=[r for r in ranges if r['name']=='backward' and (r['globalTid'] & -16777216)==global_pid]
        steps=[r for r in ranges if re.fullmatch(r'rank\d+/step\d+',r['name'] or '') and (r['globalTid'] & -16777216)==global_pid]
        assert len(backward)==3 and len(steps)==3,(device,len(backward),len(steps))
        def launched_within(kernel,regions):return any(r['start']<=kernel['launch_start']<r['end'] for r in regions)
        communication=[k for k in process_rows if 'nccl' in k['name'].lower() and launched_within(k,steps)]
        computation=[k for k in process_rows if 'nccl' not in k['name'].lower() and launched_within(k,backward)]
        gemms=[k for k in computation if re.search(r'gemm|gemv|mma|matmul',k['name'],re.I)]
        assert communication and computation and gemms
        all_overlap=intersections(communication,computation)
        gemm_overlap=intersections(communication,gemms)
        examples=[]
        for start,end,nccl,compute in sorted(gemm_overlap,key=lambda r:r[1]-r[0],reverse=True)[:3]:
            examples.append(dict(overlap_start_ns=start,overlap_end_ns=end,overlap_us=(end-start)/1000,
                nccl={k:nccl[k] for k in ('start','end','name','streamId','correlationId')},
                backward_gemm={k:compute[k] for k in ('start','end','name','streamId','correlationId','launch_start')}))
        verdict='observed_backward_gemm_overlap' if gemm_overlap else ('no_backward_kernel_overlap_observed' if not all_overlap else 'inconclusive_only_non_gemm_overlap')
        result.append(dict(rank=rank_by_pid[processes[global_pid]],pid=processes[global_pid],gpu=device,
            backward_kernel_count=len(computation),backward_gemm_count=len(gemms),nccl_kernel_count=len(communication),
            nccl_kernel_sum_ms=sum(k['end']-k['start'] for k in communication)/1e6,
            backward_all_overlap_union_ms=union_ns([(a,b) for a,b,_,_ in all_overlap])/1e6,
            backward_gemm_overlap_union_ms=union_ns([(a,b) for a,b,_,_ in gemm_overlap])/1e6,
            examples=examples,verdict=verdict))
    connection.close()
    assert {r['rank'] for r in result}=={0,1}
    return dict(database=db.name,kernel_association='correlationId + process identity; backward attribution uses launch timestamp inside same-process backward NVTX range, including worker threads; intersection uses GPU kernel execution start/end on same device',timebase='Nanoseconds relative to Nsight report origin; each report has its own origin',ranks=result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    root=args.directory
    assert union_ns([(1,4),(3,5),(7,8)])==5
    assert union_ns([])==0
    implementations=('DDP','FlatDDP','OverlapDDP')
    timing={}
    reference=None
    for implementation in implementations:
        trials=[]
        for repeat in range(1,4):
            stem=root/f'timing-{implementation}-{repeat}'
            assert Path(str(stem)+'.exit_code.txt').read_text().strip()=='0'
            data=json.loads(Path(str(stem)+'.json').read_text())
            assert data['status']=='ok' and not data['arguments']['profile']
            assert {r['rank'] for r in data['ranks']}=={0,1}
            configuration={k:v for k,v in data['arguments'].items() if k not in ('implementation','output')}
            if reference is None:reference=configuration
            assert configuration==reference
            for i,value in enumerate(data['global_step_ms']):
                expected=(max(r['records'][i]['end_ns'] for r in data['ranks'])-min(r['records'][i]['start_ns'] for r in data['ranks']))/1e6
                assert value==expected
            trials.append(data)
        samples=[t for trial in trials for t in trial['global_step_ms']]
        tokens=trials[0]['global_tokens_per_step']
        timing[implementation]=dict(steps=len(samples),median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples),
            mean_ms=statistics.mean(samples),std_ms=statistics.stdev(samples),repeat_median_ms=[t['median_ms'] for t in trials],
            global_tokens_per_s=tokens*len(samples)*1000/sum(samples),
            sync_tail_median_ms=statistics.median([t for trial in trials for t in trial['gradient_sync_tail_ms']]),
            peak_memory_gib=max(r['peak_memory_bytes'] for trial in trials for r in trial['ranks'])/2**30)
    traces={implementation:trace_evidence(root,implementation) for implementation in implementations}
    summary=dict(configuration=reference,timing=timing,traces=traces)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 双 RTX 4090 DDP 短测结果','',
        '## 配置与范围','',
        '单节点双卡；handout small：d_model=768、d_ff=3072、12 层、12 heads、vocab=10000、sequence=512。FP32、TF32 关闭；每卡 batch=4，全局每步 4096 tokens。AdamW(lr=1e-4)，每组 warmup=5、测量=10 步，各重复 3 次，轮换实现顺序。正式计时完全关闭 profiler。',
        '',
        'handout 第 33–36 页要求 xl（2560/10240、32 层、32 heads，序列长度默认 512）。当前 xl 约 34.07 亿参数，仅 FP32 权重+梯度就需约 25.38 GiB/卡，超过 4090 容量；AdamW 四份持久张量约 50.77 GiB，尚未计激活及 Flat/Overlap 缓冲。因此本轮结果是 small 替代规模数据，不能视作完成 handout xl 基准；原实现无法在本双卡配置上直接运行 xl，不能给出可信运行时长。',
        '',
        '## 无 profiler 计时','',
        '两个 rank 均在每步前 barrier 并同步 CUDA；各自记录同机 perf_counter_ns 起止时间。每步取 max(rank end)-min(rank start)，涵盖 zero_grad、forward、loss、backward、梯度同步、optimizer 和 CUDA 完成等待，排除起始 barrier。不是 rank 0 局部耗时。中位数及范围来自 3×10=30 个全局步；Tokens/s=4096×30/这些全局步耗时之和。单独提供三次中位数和全部原始 rank 时间戳。',
        '',
        '| 实现 | Step 中位数 ms | 30 步 min–max ms | 三次中位数 ms | 双卡 Tokens/s | sync tail 中位数 ms | 峰值 GiB/卡 |',
        '|---|---:|---:|---|---:|---:|---:|']
    for name,row in timing.items():
        repeats=', '.join(f'{t:.3f}' for t in row['repeat_median_ms'])
        lines.append(f"| {name} | {row['median_ms']:.3f} | {row['min_ms']:.3f}–{row['max_ms']:.3f} | {repeats} | {row['global_tokens_per_s']:.0f} | {row['sync_tail_median_ms']:.3f} | {row['peak_memory_gib']:.2f} |")
    speedup=timing['DDP']['median_ms']/timing['OverlapDDP']['median_ms']
    lines.extend(['',f'OverlapDDP 相对基础 DDP 的 Step 中位数加速约 {speedup:.3f}×。FlatDDP 在此配置下未显示明显收益；其打包与回写成本包含在测量中，不能仅由调用次数推断加速。',
        '', 'sync tail 为 finish_gradient_synchronization 前后的 CUDA Event 间隔：基础/Flat 包括通信与相关打包、回写和等待；Overlap 只反映 backward 之后的尾部等待与回写，不能当作它的总通信时间。Trace 中另存 NCCL kernel 时间总和，该和也不是端到端墙钟耗时。',
        '', '## Nsight 实际执行区间证据','',
        '每种实现 5 步 warmup 后，仅采集 3 个稳定训练步；原始 .nsys-rep、导出 SQLite 及 profile 命令均保留。通过 correlationId 和进程身份关联 CUDA launch；用 launch 时间定位同进程 backward NVTX，允许 Autograd worker 线程；最后求同一 GPU 上 NCCL 与 backward GPU kernel 实际 start/end 区间交集。不会用 CPU NVTX 区间重叠替代 GPU 执行重叠。时间均为对应 Nsight 报告相对起点的 ms，各 Trace 起点独立。',''])
    for name,evidence in traces.items():
        lines.append(f'### {name}')
        lines.append('')
        for rank in evidence['ranks']:
            lines.append(f"- rank {rank['rank']} / GPU {rank['gpu']}：NCCL kernel {rank['nccl_kernel_count']} 个，backward kernel {rank['backward_kernel_count']} 个，其中 GEMM {rank['backward_gemm_count']} 个；与全部 backward kernel 的交集并集为 {rank['backward_all_overlap_union_ms']:.3f} ms，与 backward GEMM 的交集并集为 {rank['backward_gemm_overlap_union_ms']:.3f} ms。结论：{rank['verdict']}。")
            for ex in rank['examples'][:1]:
                n,g=ex['nccl'],ex['backward_gemm']
                lines.append(f"  NCCL [{n['start']/1e6:.6f}, {n['end']/1e6:.6f}] ms（stream {n['streamId']}）；backward GEMM [{g['start']/1e6:.6f}, {g['end']/1e6:.6f}] ms（stream {g['streamId']}）；实际交集 [{ex['overlap_start_ns']/1e6:.6f}, {ex['overlap_end_ns']/1e6:.6f}] ms，共 {ex['overlap_us']:.3f} us。完整 kernel 名称及 correlationId 见 summary.json。")
        lines.append('')
    lines.extend(['## 文件与环境','',
        'dependencies.json 记录实际版本与 torch 路径；依赖安装于项目外 /root/autodl-tmp/cs336-ddp-deps-20261117，仅本轮命令设置 PYTHONPATH，原 Conda/CUDA/PyTorch/NCCL 栈不变。numpy 已按锁文件调整为 2.4.4。',
        '', 'timing-*.json 保留两个 rank 的每步起止时间和耗时；*.command.txt、*.log、*.exit_code.txt 保留完整执行记录。benchmark_ddp.py 和 analyze_ddp_results.py 为运行源码快照。',
        '', '此前 FP32/FP64 各 5 次正确性日志直接复用，未重跑。未启动 optimizer state sharding 或 FSDP。'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'timing':timing,'trace_verdicts':{k:[r['verdict'] for r in v['ranks']] for k,v in traces.items()}},indent=2))


if __name__=='__main__':main()
