# Gradient Checkpointing：GPU 对照与教学说明


## A. 目标与范围

只研究保存激活与反向重计算的取舍，不重跑历史矩阵，不使用 BF16、compile、Nsight 或 DDP。

## B. 配置与完成情况

RTX 3080 Ti 12GB；PyTorch 2.8.0+cu128；官方 small：d_model=768、d_ff=3072、12层、12 heads；batch=4、context=512、vocab=10000；FP32、Adam(lr=1e-3)，固定seed=42。每种分组独立新进程，5次warmup、10次测量，样本标准差。
这是讲义 XL、batch=4、context=2048 实验的替代，不是 XL 正式结果；本次未尝试该已知超容量配置。

## C. 修改文件

仅新增 scripts/checkpoint_study.py：verify 检查输出/梯度；logged 复用日志器；report 生成本表；main 限定只运行3组。已有 CheckpointGroup、gpu_experiments.py 与原有测试断言未修改。

## D. 关键代码

```python
layers = list(model.layers)
model.layers = torch.nn.ModuleList([
    CheckpointGroup(layers[i:i + group])
    for i in range(0, 12, group)
])
```
1. 取出12层，顺序保持不变。2. 切成连续的group层。3. 每组套用现有CheckpointGroup；group=1得到12组，group=2得到6组。4. 内部使用 checkpoint(self.run, x, use_reentrant=False)，保存组输入，在反向时重算组内激活。
每个group保存的输入shape为[4,512,768]，FP32大小为6 MiB；这不是全部显存，因为还包括参数、梯度、Adam状态和临时buffer。

## E. 实际执行命令与日志

```bash
export PYTHONPATH=.:cs336-basics OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
python scripts/run_logged.py --directory reports/logs/checkpoint_study --name checkpoint-suite -- python scripts/checkpoint_study.py run
```
验证子命令是 python scripts/checkpoint_study.py verify；三个性能子命令完整argv、stdout、stderr、退出码、进程耗时与GPU状态均在 reports/logs/checkpoint_study/。

## F. 实测结果

|Checkpoint组大小|训练Step平均ms|样本标准差ms|峰值allocated GiB|显存减少|时间增加|OOM|
|---|---:|---:|---:|---:|---:|---|
|关闭|156.102|0.134|5.024|0.0%|0.0%|否|
|每1层|195.620|0.431|2.510|50.0%|25.3%|否|
|每2层|199.409|0.319|2.555|49.2%|27.7%|否|

正确性检查：2 passed、0 failed、0 skipped；每次比较输出与全部111个参数张量的梯度。
最大输出绝对误差：0.000e+00；最大梯度绝对误差：0.000e+00。

## G. 同步

本地 WSL 项目为 ~/projects/cs336/assignment2-systems。只同步新增脚本、本目录结果和本次日志；SHA256清单为 SHA256SUMS。已有源码不同则停止，不覆盖；最终核验见 SYNC_VERIFICATION.json。

## H. 为什么节省显存、增加时间

普通Autograd保留大量层内中间结果；checkpoint只保留组输入，反向时重新计算需要的中间结果。因此以额外Forward计算换取更少长期保存的激活。
分组越大，组输入数量越少，但反向重计算时临时激活越多，峰值不保证单调下降。参数、梯度和Adam状态仍占空间，所以总显存不会按层数等比例缩小。
例如输入6 MiB、组内残差远大于6 MiB时，保存输入通常有利；但这不表示整个训练只占6 MiB。use_reentrant=False允许按反向需求提前停止重计算，因此时间开销也不等于精确多跑一次完整Forward。

## I. 未完成与下一步

已完成本次三组GPU对照。尚未完成讲义XL规模、最优粒度的更细子层搜索及递归checkpoint理论题；不根据这三个case宣称全局最优。本次结束后停止；下一阶段需单独授权，不自动开始torch.compile或DDP。