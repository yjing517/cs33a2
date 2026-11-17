# 双 RTX 4090 DDP 短测结果

## 配置与范围

单节点双卡；handout small：d_model=768、d_ff=3072、12 层、12 heads、vocab=10000、sequence=512。FP32、TF32 关闭；每卡 batch=4，全局每步 4096 tokens。AdamW(lr=1e-4)，每组 warmup=5、测量=10 步，各重复 3 次，轮换实现顺序。正式计时完全关闭 profiler。

handout 第 33–36 页要求 xl（2560/10240、32 层、32 heads，序列长度默认 512）。当前 xl 约 34.07 亿参数，仅 FP32 权重+梯度就需约 25.38 GiB/卡，超过 4090 容量；AdamW 四份持久张量约 50.77 GiB，尚未计激活及 Flat/Overlap 缓冲。因此本轮结果是 small 替代规模数据，不能视作完成 handout xl 基准；原实现无法在本双卡配置上直接运行 xl，不能给出可信运行时长。

## 无 profiler 计时

两个 rank 均在每步前 barrier 并同步 CUDA；各自记录同机 perf_counter_ns 起止时间。每步取 max(rank end)-min(rank start)，涵盖 zero_grad、forward、loss、backward、梯度同步、optimizer 和 CUDA 完成等待，排除起始 barrier。不是 rank 0 局部耗时。中位数及范围来自 3×10=30 个全局步；Tokens/s=4096×30/这些全局步耗时之和。单独提供三次中位数和全部原始 rank 时间戳。

| 实现 | Step 中位数 ms | 30 步 min–max ms | 三次中位数 ms | 双卡 Tokens/s | sync tail 中位数 ms | 峰值 GiB/卡 |
|---|---:|---:|---|---:|---:|---:|
| DDP | 128.186 | 127.702–132.074 | 128.096, 128.266, 128.101 | 31866 | 37.053 | 5.02 |
| FlatDDP | 128.929 | 127.216–132.533 | 128.753, 128.376, 129.275 | 31759 | 37.611 | 5.02 |
| OverlapDDP | 105.866 | 105.462–110.128 | 105.940, 105.865, 105.866 | 38624 | 6.199 | 5.03 |

OverlapDDP 相对基础 DDP 的 Step 中位数加速约 1.211×。FlatDDP 在此配置下未显示明显收益；其打包与回写成本包含在测量中，不能仅由调用次数推断加速。

sync tail 为 finish_gradient_synchronization 前后的 CUDA Event 间隔：基础/Flat 包括通信与相关打包、回写和等待；Overlap 只反映 backward 之后的尾部等待与回写，不能当作它的总通信时间。Trace 中另存 NCCL kernel 时间总和，该和也不是端到端墙钟耗时。

## Nsight 实际执行区间证据

每种实现 5 步 warmup 后，仅采集 3 个稳定训练步；原始 .nsys-rep、导出 SQLite 及 profile 命令均保留。通过 correlationId 和进程身份关联 CUDA launch；用 launch 时间定位同进程 backward NVTX，允许 Autograd worker 线程；最后求同一 GPU 上 NCCL 与 backward GPU kernel 实际 start/end 区间交集。不会用 CPU NVTX 区间重叠替代 GPU 执行重叠。时间均为对应 Nsight 报告相对起点的 ms，各 Trace 起点独立。

### DDP

- rank 0 / GPU 0：NCCL kernel 333 个，backward kernel 3684 个，其中 GEMM 654 个；与全部 backward kernel 的交集并集为 0.000 ms，与 backward GEMM 的交集并集为 0.000 ms。结论：no_backward_kernel_overlap_observed。
- rank 1 / GPU 1：NCCL kernel 333 个，backward kernel 3684 个，其中 GEMM 654 个；与全部 backward kernel 的交集并集为 0.000 ms，与 backward GEMM 的交集并集为 0.000 ms。结论：no_backward_kernel_overlap_observed。

### FlatDDP

- rank 0 / GPU 0：NCCL kernel 3 个，backward kernel 3684 个，其中 GEMM 654 个；与全部 backward kernel 的交集并集为 0.000 ms，与 backward GEMM 的交集并集为 0.000 ms。结论：no_backward_kernel_overlap_observed。
- rank 1 / GPU 1：NCCL kernel 3 个，backward kernel 3684 个，其中 GEMM 654 个；与全部 backward kernel 的交集并集为 0.000 ms，与 backward GEMM 的交集并集为 0.000 ms。结论：no_backward_kernel_overlap_observed。

### OverlapDDP

- rank 0 / GPU 0：NCCL kernel 333 个，backward kernel 3684 个，其中 GEMM 654 个；与全部 backward kernel 的交集并集为 97.156 ms，与 backward GEMM 的交集并集为 52.389 ms。结论：observed_backward_gemm_overlap。
  NCCL [222.306458, 224.712767] ms（stream 17）；backward GEMM [223.929862, 224.221580] ms（stream 7）；实际交集 [223.929862, 224.221580] ms，共 291.718 us。完整 kernel 名称及 correlationId 见 summary.json。
- rank 1 / GPU 1：NCCL kernel 333 个，backward kernel 3684 个，其中 GEMM 654 个；与全部 backward kernel 的交集并集为 168.020 ms，与 backward GEMM 的交集并集为 83.054 ms。结论：observed_backward_gemm_overlap。
  NCCL [88.239133, 96.091864] ms（stream 17）；backward GEMM [94.720564, 95.030456] ms（stream 7）；实际交集 [94.720564, 95.030456] ms，共 309.892 us。完整 kernel 名称及 correlationId 见 summary.json。

## 文件与环境

dependencies.json 记录实际版本与 torch 路径；依赖安装于项目外 /root/autodl-tmp/cs336-ddp-deps-20261117，仅本轮命令设置 PYTHONPATH，原 Conda/CUDA/PyTorch/NCCL 栈不变。numpy 已按锁文件调整为 2.4.4。

timing-*.json 保留两个 rank 的每步起止时间和耗时；*.command.txt、*.log、*.exit_code.txt 保留完整执行记录。benchmark_ddp.py 和 analyze_ddp_results.py 为运行源码快照。

此前 FP32/FP64 各 5 次正确性日志直接复用，未重跑。未启动 optimizer state sharding 或 FSDP。
