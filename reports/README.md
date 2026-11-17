# 实验归档

记录 CS336 Assignment 2 的现有实测结果。下列复算命令从仓库根目录运行，只读取已归档数据；生成报告或图片的脚本请在临时克隆中运行，以免改写归档。各实验的 `*.command.json` 或 `*.command.txt` 保留原始运行命令。

| 实验 | 报告 | 原始数据与日志 | 曲线或 trace |
| --- | --- | --- | --- |
| NCCL All-Reduce，双 RTX 4090 | [报告](nccl_allreduce/20261117T0330Z/REPORT.md) | [第 1 轮](nccl_allreduce/20261117T0330Z/allreduce-1.json)、[第 2 轮](nccl_allreduce/20261117T0330Z/allreduce-2.json)、[第 3 轮](nccl_allreduce/20261117T0330Z/allreduce-3.json)、[CSV](nccl_allreduce/20261117T0330Z/results.csv)、[日志](nccl_allreduce/20261117T0330Z/) | [PNG](nccl_allreduce/20261117T0330Z/allreduce-latency-bandwidth.png)、[SVG](nccl_allreduce/20261117T0330Z/allreduce-latency-bandwidth.svg) |
| 单卡 GPU benchmark | [结果](gpu_study/RESULTS.md)、[Nsight 分析](gpu_study/NSIGHT_ANALYSIS.md)、[交付说明](gpu_study/DELIVERY.md) | [JSON](gpu_study/data/)、[CSV](gpu_study/measurements.csv)、[日志](logs/gpu_study/) | [曲线与显存图](gpu_study/figures/)、[profiling 产物](gpu_study/artifacts/) |
| Gradient Checkpointing | [报告](checkpoint_study/RESULTS.md) | [基线](checkpoint_study/group0.json)、[每层](checkpoint_study/group1.json)、[每两层](checkpoint_study/group2.json)、[正确性](checkpoint_study/verification.json)、[日志](logs/checkpoint_study/) | — |
| torch.compile | [报告](compile_study/RESULTS.md) | [Eager](compile_study/eager.json)、[Compiled](compile_study/compiled.json)、[日志](logs/compile_study/) | — |
| DDP | [NCCL 性能](ddp_nccl/performance-20261117T012804Z/REPORT.md)、[CPU 基线](ddp_cpu/README.md)、[Flat](ddp_flat_cpu/README.md)、[Overlap](ddp_overlap_cpu/README.md) | [NCCL 原始数据与日志](ddp_nccl/)、[CPU 预检](ddp_preflight_cpu/) | [Nsight trace](ddp_nccl/performance-20261117T012804Z/trace-OverlapDDP.nsys-rep) |
| Optimizer Sharding | [报告](optimizer_sharding/20261117T014825Z/REPORT.md) | [原始数据与日志](optimizer_sharding/20261117T014825Z/)、[显存对照](optimizer_sharding/20261117T014825Z/balanced/)、[参数归属 CSV](optimizer_sharding/20261117T014825Z/parameter-ownership.csv) | — |
| FSDP | [报告](fsdp/20261117T021755Z/REPORT.md) | [原始数据与日志](fsdp/20261117T021755Z/) | [Nsight trace](fsdp/20261117T021755Z/trace-small-fsdp-float16.nsys-rep)、[SQLite](fsdp/20261117T021755Z/trace-small-fsdp-float16.sqlite) |

## 完成状态

- **XL 未完成。** 单卡 GPU benchmark 记录了容量下界超限，显存图来自 small。Checkpointing 完成 small 三组对照。官方 XL DDP 性能、Sharding accounting 和 FSDP accounting 未完成，目录中保留较小配置的实测记录。
- **双 B200 leaderboard 未执行。** 归档中的双 GPU 实测使用 RTX 4090。双 B200 的正确性、完整 step 耗时和峰值显存均无实测结果。

## 离线复算

All-Reduce 使用每轮 20 个双 rank collective 的完成时间，合并 3 轮取中位数。100 MB 使用十进制定义。

```bash
python3 - <<'PY'
import json
from pathlib import Path
from statistics import median

root = Path("reports/nccl_allreduce/20261117T0330Z")
times = []
for index in range(1, 4):
    run = json.loads((root / f"allreduce-{index}.json").read_text(encoding="utf-8-sig"))
    item = next(result for result in run["results"] if result["bytes"] == 100_000_000)
    starts, ends = zip(*(
        (min(row["start_ns"] for row in rows), max(row["end_ns"] for row in rows))
        for rows in zip(*(rank["rows"] for rank in item["ranks"]), strict=True)
    ))
    reconstructed = [(end - start) / 1e6 for start, end in zip(starts, ends, strict=True)]
    assert len(reconstructed) == 20 and reconstructed == item["global_ms"]
    times.extend(reconstructed)

result = f"{median(times):.4f}"
assert len(times) == 60 and result == "6.5804"
print(f"100 MB: {result} ms ({len(times)} samples)")
PY

# All-Reduce 全部尺寸的表格和曲线（需 NumPy、Matplotlib）
python3 scripts/analyze_nccl_allreduce.py reports/nccl_allreduce/20261117T0330Z

# 单卡时间和显存汇总（需 Matplotlib）
python3 scripts/summarize_gpu_study.py

# DDP 与 FSDP 结果及 trace 摘要
python3 scripts/analyze_ddp_results.py reports/ddp_nccl/performance-20261117T012804Z
python3 reports/fsdp/20261117T021755Z/analyze_results.py reports/fsdp/20261117T021755Z
```

Checkpointing 和 torch.compile 的均值、样本标准差直接从 JSON 的 `per_step_ms` 复算。Sharding 的显存对照使用每个 rank 的已保存记录：

```bash
python3 - <<'PY'
import json
from pathlib import Path
from statistics import mean, stdev

root = Path("reports")
files = [
    *(root / "checkpoint_study").glob("group[012].json"),
    root / "compile_study/eager.json",
    root / "compile_study/compiled.json",
]
for path in sorted(files):
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    steps = result["per_step_ms"]
    print(f"{path}: {mean(steps):.3f} ± {stdev(steps):.3f} ms")

for path in sorted((root / "optimizer_sharding").glob("*/balanced/memory-*.json")):
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if "ranks" in result:
        print(path, json.dumps(result["ranks"], ensure_ascii=False))
PY
```

## 完整性校验

从仓库根目录运行 `sha256sum --check reports/ARCHIVE_SHA256SUMS`。归档清单覆盖 `reports/` 中除清单自身外的全部文件。各实验原有的 SHA-256 清单也保留，便于追溯各自的同步范围。
