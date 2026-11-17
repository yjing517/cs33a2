# GPU Benchmark 与 Profiling：交付说明

## A. 本次目标
测量 Transformer 运行速度，定位 Forward、Backward、Optimizer 的耗时，比较 BF16 与显存。本次按收敛要求结束实验，不再扩展 GPU 任务。

## B. 完成情况
- 39 个主 Benchmark case：21 成功、4 实测 OOM、14 容量下界超限。可运行 case 为 10 次测量，比较 0/1/2/5 次 warmup。
- 3 份有效 Nsight 报告、2 份 PyTorch Profiler trace、6 份 CUDA 显存快照、10 张分析图。
- 累加精度和 ToyModel FP16/BF16 dtype 实验完成。
- 新增测试 7 passed、0 failed、0 skipped，Ruff 通过；原测试断言未修改。
- 未完成：全部官方配置、两模型各三长度的完整 Nsight 矩阵、XL 显存图和 GUI 截图、单层残差分析。小模型图不能替代 XL 官方结果。
- checkpoint/compile GPU 对照及 DDP/FSDP/多卡实验没有启动；专用 Triton Backward 未开发。

## C. 文件修改
原 benchmark.py、FlashAttention 和原 adapters.py 均未修改。本次新增：

|文件|关键函数/类|用途|
|---|---|---|
|cs336_systems/gpu_experiments.py|parameter_count, experiment, stage, main|容量预检、独立模式计时、BF16、Profiler 和快照|
|同上|CheckpointGroup|后续支持，仅做了 CPU 梯度一致性检查|
|scripts/run_logged.py|main, gpu_state|stdout/stderr、退出码、耗时和 GPU 状态|
|scripts/run_gpu_study.py|case, logged, main|新进程运行每个 case|
|scripts/precision_probe.py|ToyModel, main|累加与 dtype 实测|
|scripts/summarize_gpu_study.py|main, cell, table|JSON/快照生成 CSV、图表与报告|
|scripts/analyze_nsys.py|main, summarize|CUDA launch 与 NVTX 归因|
|tests/test_gpu_experiments.py|两个测试函数，共七个参数化实例|容量公式与 checkpoint 正确性|
|scripts/residual_probe.py|main|已准备但未执行，不提供实验结论|

## D. 关键代码讲解
```python
model.zero_grad(set_to_none=True)
with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
loss.backward()
optimizer.step()
```
1. 清除上一轮梯度，避免累积；此操作在计时外。
2. autocast 为部分运算选择 BF16，模型参数没有被永久改成 BF16。
3. x 为 [batch,tokens]，输出 logits 为 [batch,tokens,vocab]。
4. CrossEntropy 把 [4,512,10000] 展成 [2048,10000]：每一行对应一个 token 的词表预测。
5. backward 在 autocast 外求梯度；FP32 参数对应 FP32 梯度。
6. 只有 train_step 更新参数；forward_backward 不更新，forward 不构建反向图。

计时先 CUDA synchronize，再 perf_counter；执行本轮完整操作后再次同步，读取耗时。CUDA 是异步的，缺少同步会只量到发起任务的时间。Profiler 会扰动执行，所以其数据与纯计时分开报告。

Nsight kernel 用 correlationId 找到发起它的 CUDA API，再按 API 的发起时间归类。Backward 可能从 Autograd 工作线程发起；初版分析仅匹配主线程，已修复并重新分析原文件，没有重跑 GPU。

## E. 执行记录与实际耗时
完整 argv、stdout、stderr、退出码与 GPU 前后状态：../logs/gpu_study/。综合终端日志：../logs/gpu-study-terminal-delivery.log。日志不包含 SSH 密码。

|批次|实际时间|外层退出码|
|---|---:|---:|
|timings-suite|282.232 秒，4分42秒|0|
|profiles-suite|105.774 秒，1分46秒|0|
|memory-suite|334.515 秒，5分35秒|0|

合计 722.521 秒，约 12分03秒，不含编码、传输和整理。外层退出 0 表示调度结束，不代表子 case 全成功。退出 42 是实测 OOM，43 是容量下界超限。

已执行的关键命令如下，收敛后没有重跑前三条：
```bash
python scripts/run_gpu_study.py timings
python scripts/run_gpu_study.py profiles
python scripts/run_gpu_study.py memory
python scripts/precision_probe.py
python -m pytest tests/test_gpu_experiments.py -q
# 只分析既有数据：
python scripts/summarize_gpu_study.py
python scripts/analyze_nsys.py
```

## F. 结果与作业可用性
完整表格见 [RESULTS.md](RESULTS.md)，kernel 名称和计数见 [NSIGHT_ANALYSIS.md](NSIGHT_ANALYSIS.md)。
- 官方 small、context=512：FP32/BF16 的 Forward 为 47.519/27.990 ms，Forward+Backward 为 145.316/84.850 ms，Train Step 为 156.181/96.518 ms；训练加速 1.62 倍，峰值 allocated 从 5.024 降至 4.115 GiB。
- BF16 不保证推理峰值更低：small Forward 从 0.712 增至 0.888 GiB，涉及 FP32 主参数与低精度转换/缓存共存；需以实测为准。
- small Train Step：warmup=0 时为 196.979±129.077 ms，warmup=5 时为 156.181±0.324 ms。首次库与优化器初始化抬高均值和方差；1/2 次已明显改善，但不保证所有环境都充分预热。
- small N=512 Nsight GPU kernel sum：Forward 44.734、Backward 94.488、Optimizer 22.650 ms。这不是独立模式端到端延迟。
- Forward 最耗时 kernel 为 ampere_sgemm_128x64_tn，85 次、24.892 ms；GEMM 时间比例从 Forward 的约63.2%降至训练的约51.4%。分类按 kernel 名称，完整明细可复核。
- Attention 的 mask+Softmax 相关范围为7.180 ms，QK/PV 两个矩阵乘相关范围合计5.081 ms。Softmax FLOPs 少，但包含多次显存读写与归约。
- FP16 连加1000次0.01得到9.953125；FP32累加FP16项得到10.002136。高精度累加减少舍入，不能恢复输入低精度表示时已丢失的精度。
- ToyModel 的两种 autocast：Linear/logits 为低精度；LayerNorm、Loss、参数和梯度为FP32。BF16范围大，但尾数较短，仍需高精度统计。

### OOM
medium 完整训练在 FP32/BF16 均实测 OOM；large 的反向实测 OOM，训练持久内存下界已超限。XL 单是 FP32 参数约12.69 GiB，超过本机约11.63 GiB可见容量；BF16 autocast保留FP32参数，不能解决这个下界。

XL 残差流理论大小：4×context×2560×4/1024² MiB；context=128/512/2048时为5/20/80 MiB。这是形状推导，不是峰值实测。

已测官方形状、warmup、精度与实际kernel统计可写入报告，但必须注明RTX3080Ti平台。small显存图只是替代实验；缺失的XL、完整Nsight矩阵和GUI截图不能宣称完成。10张PNG是数据分析图，不是GUI截图。memory图横轴为分配事件编号，不是秒。

## G. 本地同步
仅同步本次源码、报告、逐轮JSON、日志、图和可复核的kernel摘要。不包含权重、.venv、缓存。原始快照、Nsight二进制、HTML、Chrome trace和可再生SQLite因下载较慢留在远程，详见UNSYNCED.md。已有本地内容不一致则停止，不覆盖；最终结果见SYNC_VERIFICATION.json。

VS Code 用 Remote - WSL 打开 ~/projects/cs336/assignment2-systems，从本文件和RESULTS.md开始。

## H. 五个概念
1. 独立测量：Forward-only是另一次运行，不是训练中的一个片段。
2. Warmup：把首次初始化费用与稳态每步费用分开。
3. Mixed Precision：主账本保留FP32，部分矩阵乘用BF16。
4. 峰值显存：结束占用小不代表中途峰值小；reserved还包含缓存池空闲块。
5. 异步与归因：CPU发起、GPU执行是两个时间轴，要用关联ID连接。

## I. 下一步
不自动继续实验。下一次控制在10～15分钟，优先做checkpoint必做项的小规模正确性与显存对照；超过15分钟先说明工作量并等待确认。XL官方对照需要更大显存。compile、DDP、Sharding、FSDP、多卡All-Reduce、最终writeup和提交包仍未完成。
