# FSDP 阶段验证报告

## 实现范围与正确性

实现 cs336_systems/fsdp.py，接通 get_fsdp、fsdp_on_after_backward、fsdp_gather_full_params。Linear/Embedding 每个权重 Tensor 使用连续带 padding 的 FP32 master shard，norm/bias 等小参数复制；forward/backward 按层 all-gather，用后释放完整权重，自定义 Autograd 仅保存输入；完整 FP32 梯度 reduce-scatter 并平均后进入局部 optimizer。复制参数梯度 all-reduce。

首步按需 gather 并记录实际执行顺序；后续最多提前两层 prefetch，layer i 仅在 i-2 完成计算后开始 gather。梯度 reduce-scatter 异步提交后 wait 建立当前 stream 依赖，不宣称梯度通信完全隐藏。compute_dtype=FP16 时权重在通信前转换，master/梯度/optimizer state 保持 FP32，forward 用 autocast 处理 RoPE/attention 等非分片运算的 dtype。未修改既有 Transformer 或官方断言。

WSL：官方四项 CPU/Gloo + 一项奇数尺寸 padding/原生 Linear/Embedding/bias 回归，最终 5/5 通过。远端原样官方测试使用 CUDA Tensor、官方固定的 Gloo backend，最终连续 5 轮×4 项全部通过；另外独立运行 NCCL 冒烟和多步数值验证，避免把 Gloo 结果写成 NCCL。

固定 seed=42 初始化、seed=123 数据、global batch=20、sequence=8，训练 3 步；参考模型和 FSDP 使用同样数据、optimizer 配置。对完整参数逐步比较，两个 rank gather 后的完整模型逐元素相同。小 Transformer 对照结果：

| compute_dtype | 最大 Loss 误差 | 最大梯度误差 | 最大参数误差 | 参数 atol / rtol |
|---|---:|---:|---:|---|
| float32 | 4.77e-07 | 6.75e-08 | 1.19e-07 | 1e-06 / 0.0001 |
| float16 | 7.63e-06 | 0.00019 | 2.91e-06 | 0.0001 / 0.0001 |

FP16 梯度对照 atol=5e-3、rtol=2e-2，观察到的误差远低于此阈值；参数使用与官方相同的容差。assignment-1 AdamW 默认 eps=1e-8 会放大全局 batch 与分批归约的舍入差异：初次全局 batch 对照记录到单参数 1.7e-5 差异；改用相同局部数据及归约顺序的完整模型参考后，FP32/FP16 的 Loss、梯度和更新参数误差均为 0，未放宽参数容差。初次失败日志保留。

真实 Transformer FP16 另发现 RoPE FP32 缓存使 Q/K 与 V dtype 不一致；最终 forward autocast 修复后，重新通过 CPU 和五轮 GPU 官方测试。

## 独立进程显存

本次新的统一 small 对照：768/3072、12 层、12 heads、vocab=10000、context=128、每卡 batch=2；torch.optim.AdamW(lr=0.001, weight_decay=0.03, betas=(0.9,0.99))。每个方案/精度各启动新的双 rank 进程，测量一个完整训练步，包含首次惰性 optimizer state 建立；不能把该首步耗时当作稳定性能。两 rank 结果均保留，表中取每卡最大值。旧 DDP 与状态分片的正确性/性能结果没有重跑；此表是新 FSDP 的统一显存对照。

| 方案 | compute | 初始化 allocated / peak GiB | step 前 allocated / peak GiB | step 后 allocated / peak GiB | max_memory_reserved GiB | optimizer state MiB/卡 |
|---|---|---:|---:|---:|---:|---:|
| reference | float32 | 0.481 / 0.481 | 1.039 / 1.069 | 2.001 / 2.481 | 2.852 | 981.33 |
| reference | float16 | 0.481 / 0.481 | 1.033 / 1.048 | 1.996 / 2.476 | 2.785 | 981.33 |
| fsdp | float32 | 0.259 / 0.524 | 0.588 / 0.717 | 1.092 / 1.342 | 1.469 | 490.74 |
| fsdp | float16 | 0.259 / 0.524 | 0.580 / 0.639 | 1.078 / 1.328 | 1.414 | 490.74 |

FP32 的 persistent 参数、梯度和两个 Adam moment 大体从每卡四份完整模型变为四份半模型，仅 norm/bias 复制；FP16 只降低通信/计算缓存与激活，不降低 FP32 master/state 大小。峰值还包括初始化完整模型的短暂驻留、最多两层 gather buffer、输出、激活、梯度临时 Tensor 等，因此不恰好减半。allocated 和 reserved 分别报告，不能互换；PyTorch allocator 不包含全部驱动/CUDA context 显存。

## 短性能与通信

仅对 FSDP 新实现计时：5 步 warmup 后测量 10 步，无 profiler。每步用同机 monotonic clock 的 max(rank end)-min(rank start)，包含 forward/loss/backward/sync/AdamW/CUDA 完成，排除起始 barrier；首个测量步含内存读数调用，10 步中位数主要反映其余稳定步。Tokens/s=每步双卡 512 tokens×10/全部全局步耗时之和。

| compute | Step 中位数 ms | 10 步范围 ms | 双卡全局 Tokens/s |
|---|---:|---:|---:|
| float32 | 103.644 | 99.184–116.640 | 4878 |
| float16 | 107.408 | 97.678–127.429 | 4733 |

FP16 在此短测中没有加速，不将权重通信字节减半误写成端到端加速。计时仅一次 10 步短测，不能代替大规模 benchmark。communication 字段记录实际 collective 输入 Tensor 的 dtype、调用数和 payload，不能当作实际链路流量；权重 all-gather 为 FP16/FP32 对应精度，梯度 reduce-scatter 为 FP32。

Nsight 仅采 small FP16 的两个稳定训练步，原始 .nsys-rep 与 SQLite 均保留。按 correlationId+进程身份关联 launch，forward 归属由同进程 forward NVTX 内 launch 时间确定；实际重叠由同一 GPU 的 kernel start/end 区间交集判定。时间为此 Nsight 报告的相对起点。

- rank 0 / GPU 0：forward AllGather kernel 172 个，ReduceScatter kernel 172 个，forward GEMM 218 个；observed_same_GPU_forward_all_gather_overlap。
  AllGather [147.523660, 148.355746] ms；forward GEMM [147.523820, 147.557554] ms；交集 [147.523820, 147.557554] ms，共 33.734 us，完整名称/stream/correlationId 见 summary.json。
- rank 1 / GPU 1：forward AllGather kernel 172 个，ReduceScatter kernel 172 个，forward GEMM 218 个；observed_same_GPU_forward_all_gather_overlap。
  AllGather [349.119362, 349.790171] ms；forward GEMM [349.119522, 349.154889] ms；交集 [349.119522, 349.154889] ms，共 35.367 us，完整名称/stream/correlationId 见 summary.json。

此 Trace 能证明部分权重 gather 与 forward 计算同时执行；没有逐层反事实的 GPU consumer-ready 时间，不能由“消费 kernel 在 gather 后启动”推断所有通信都隐藏或没有等待。是否每一层 gather 都及时完成，本轮不能作完整肯定结论；xl 的截图/时序要求仍待合适硬件执行。

## 未执行的官方配置

- FSDP accounting（handout 第 39 页）：xl=2560/10240、32 层、32 heads、默认 sequence=512、双 GPU，检查 all-gather 是否赶上 forward 并提供 Nsight 截图。本实现 FP32 master+gradient+AdamW moments 分片后持久下界约 25.38 GiB/卡，已超过 4090 的容量，尚未计激活及 gather buffer；未尝试会 OOM 的正式 xl AdamW 运行。待采集：分阶段 allocated/reserved、稳定 step、每层 gather/consumer 时序与 GUI 截图。small 数据不替代 xl。
- 双 B200 leaderboard（第 46–48 页）：BF16、causal、batch=2、sequence=32768、vocab=151936、d_model=4096、d_ff=11008、layers=34、heads=32；完整 forward/loss/backward/AdamW。官方示例 do_bench rep=30000 ms、warmup=10000 ms，冷缓存运行须在 10 分钟内完成。没有 B200，4090 单卡 24GB 不足以容纳该模型状态和长序列激活；未运行，不提供虚构耗时。待采集：正确性误差、完整 step wall-clock、峰值 allocated/reserved、冷缓存总时长及必要时通信时序。

## 文件与停止状态

final-* 为最终实现正确性结果；旧的成功/诊断失败过程日志保留，不计入最终通过结论。每次执行保留 command.json、stdout/stderr、exit_code；sources/ 保存最终实际运行源码。本轮没有升级 CUDA/PyTorch，仅在项目外安装了锁定版本 pytest 及轻量测试依赖。

实验进程已结束，实例不关机、不释放、不重启；全部文件同步 WSL 并以 SHA-256 清单核验。
