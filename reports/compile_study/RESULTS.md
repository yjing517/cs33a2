# torch.compile GPU 对照：small FP32 Train Step

## A. 目标与作业范围
PDF 第16–17页 torch_compile 要求 attention 的编译/未编译 Forward、Backward 对照，以及完整 Transformer 的 Forward 与训练对照。本次按用户限定，只完成 small 的 Train Step 子集；未完成完整 attention 网格及独立 Forward 对照，不能将本表当作整题完成。

## B. 环境与方法
RTX 3080 Ti 12GB；PyTorch 2.8.0+cu128、CUDA runtime 12.8、Triton 3.4.0；未升级依赖。small：128625408参数、12层、d_model=768、d_ff=3072、12 heads、vocab=10000；batch=4、context=512、FP32、torch.optim.Adam(lr=1e-3)，seed=42，TF32关闭。
两组独立进程，5次warmup（包含首次Step）、10次测量，统计样本标准差；计时两端CUDA同步，用time.perf_counter。每轮set_to_none清梯度。未开启checkpoint、BF16、profile或NVTX。
编译设置：backend=inductor，mode=default，dynamic=None，fullgraph=False；只编译model及其生成的Backward，Loss和Adam调用保持eager。dynamic=None允许编译器自动决定动态shape策略，本次输入shape固定。OMP/MKL/Inductor编译线程均为1。
Inductor使用运行前新建的 /root/autodl-tmp/cs336-compile-study-cache；未清除已有全局Triton/CUDA缓存，因此这是新Inductor目录的实际首次调用，不能声称所有层级缓存完全冷启动。

## C. 修改文件
cs336_systems/gpu_experiments.py：experiment明确compile设置、记录wrapper调用时间、每次warmup耗时和首次Step峰值；增加测量后可选verification回调。原CLI保持兼容。
scripts/compile_study.py：只调度eager/compiled两组，复用experiment和run_logged；单次420秒上限；失败停止不重试；同权重输出、Loss和所有参数梯度断言。未修改任何原有测试断言。

## D. 关键代码与原理
```python
model = torch.compile(model, backend="inductor", mode="default", dynamic=None)
```
torch.compile通常先捕获PyTorch计算图，由AOTAutograd生成反向图，再由Inductor生成、融合和调度计算（GPU上可生成Triton kernel）。融合可减少小kernel启动、临时张量写回及读取；矩阵乘仍可调用已有高效库。这里没有再做profiling，具体kernel收益属于机制解释，不能据此断言某个kernel是实测瓶颈。
包装调用并不完成全部编译；首次Forward/Backward触发延迟编译，因此首次Step通常很慢。优化范围是模型，未把整个Python optimizer.step编译。

## E. 实际命令与日志
```bash
export PYTHONPATH=.:cs336-basics OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/cs336-compile-study-cache
python scripts/run_logged.py --directory reports/logs/compile_study --name suite -- python scripts/compile_study.py run
```
每个case复用gpu_experiments.parser：--model-size small --mode train_step --precision fp32 --optimizer adam --batch-size 4 --context-length 512 --warmup 5 --steps 10；compiled额外--compile。完整子命令argv见日志JSON。suite耗时62.457秒，eager进程7.520秒，compiled进程51.441秒，全部exit_code=0。
本地回归：PYTHONPATH=.:cs336-basics uv run --no-sync pytest tests/test_gpu_experiments.py -q；真实结果7 passed in 4.97s，exit=0。Ruff与py_compile均通过。这里只运行因实验器修改相关的小型CPU回归，没有重跑历史GPU实验。
编译仅出现TF32未启用的性能建议警告；为保持FP32对照一致，未修改TF32设置。

## F. 实测结果
|模式|wrapper调用ms|首次完整Step ms|预热后平均ms|标准差ms|峰值allocated GiB|OOM|
|---|---:|---:|---:|---:|---:|---|
|eager|0.000|557.217|155.538|0.098|5.024|否|
|compile|1379.875|39129.898|120.253|0.224|4.311|否|

稳定态加速 1.293x，Step时间减少 22.7%，allocated峰值减少 14.2%。峰值在warmup后reset，表中不包含后续正确性检查；各次原始耗时及首次Step峰值均在JSON。
首次Step包括延迟编译、实际训练、CUDA库初始化和Adam状态初始化；wrapper是torch.compile调用本身耗时，二者不能称为纯编译耗时。
如果仅用本次首次Step差值加wrapper耗时估计，额外启动成本约 39.953秒，按每Step节省35.285ms估计约需1132个后续Step摊平。这是线性估算，不是额外测量。
数值验证通过：相同性能运行结束后的模型权重、输入和目标，先eager再compiled，均不更新参数；输出最大绝对误差4.76837158e-06，Loss误差1.1920929e-07，全部111个参数梯度最大绝对误差2.32830644e-09。
验证Loss：eager=1.540560007，compiled=1.540560126。容差为输出atol=1e-5/rtol=1e-4，Loss 1e-5/1e-5，梯度1e-5/1e-3。验证3类全部通过，失败0。
独立训练运行最终Loss：eager=2.271644592、compiled=2.270823002。浮点融合与运算次序变化经多次Adam更新可造成轨迹差异，不能用不同轨迹的Loss代替同权重正确性检查。

## G. 本地同步
只交付本次两个源文件、reports/compile_study/和reports/logs/compile_study/。本地源文件是此次修改的原件，同步时先对比SHA256；新增报告目录才解压，目录已存在则停止。同步后的实证状态见SYNC_VERIFICATION.json及SHA256SUMS。不同步编译缓存、环境或数据。

## H. 核心知识
计算图：把一串Tensor运算描述成依赖关系；融合：例如多个逐元素操作合并执行，减少中间读写；延迟编译：包装时少做工作，首次调用才生成代码；摊销：首次慢，只有运行足够多Step才可能赚回启动成本。

## I. 未完成与停止
未执行attention网格、独立Forward对照、多尺寸/长序列对照、DDP/FSDP；不自动继续。10个样本及单卡单形状的结果仅适用于当前环境，不能外推到B200或其他配置。
