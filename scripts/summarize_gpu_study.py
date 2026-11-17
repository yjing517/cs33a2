"""Build tables and plots from measured JSON and self-generated CUDA snapshots."""

import csv
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("reports/gpu_study")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def cell(result):
    if result["status"] == "ok":
        return f"{result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms / {result['peak_allocated_bytes']/2**30:.3f} GiB"
    return {"observed_oom": "实测 OOM (exit 42)", "blocked_capacity_lower_bound": "容量下界超限 (exit 43)"}.get(result["status"], result["status"])


def main():
    records = []
    for path in sorted((ROOT / "data").glob("*.json")):
        result = json.loads(path.read_text())
        if "arguments" in result:
            result["file"] = path.name
            records.append(result)
    columns = ["file", "status", "model_size", "mode", "precision", "context_length", "warmup", "steps", "mean_ms", "std_ms", "peak_allocated_bytes", "peak_reserved_bytes"]
    with (ROOT / "measurements.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in records:
            writer.writerow({key: result.get(key, result["arguments"].get(key, "")) for key in columns})
    normal = [r for r in records if r["file"].endswith("-n512.json") and r["arguments"]["steps"] == 10]
    sections = ["# RTX 3080 Ti：GPU Benchmark、Profiling 与显存实验\n",
                "这些数据来自实际 RTX 3080 Ti 12GB，不代表 B200。官方模型形状与硬件平台是两件事；表中 small/medium/large 使用讲义模型尺寸、batch=4、vocab=10,000。surrogate 明确属于替代实验。计时使用独立进程、CUDA 同步和 perf_counter，标准差为样本标准差；所有 JSON 保留每轮数据。\n",
                "## 官方尺寸 FP32 / BF16 对照\n",
                "单元格为平均值 ± 标准差 / PyTorch peak allocated；warmup=5，测量=10。参数和梯度保持 FP32，BF16 只通过 autocast 使用。计时表使用 torch.optim.Adam，与原 benchmark 一致；Profiler/快照另使用作业参考 AdamW，不混算。\n"]
    rows = []
    for size in ("small", "medium", "large", "xl", "10B"):
        for mode in ("forward", "forward_backward", "train_step"):
            pair = [next((r for r in normal if r["arguments"]["model_size"] == size and r["arguments"]["mode"] == mode and
                          r["arguments"]["precision"] == precision and r["arguments"]["warmup"] == 5), None) for precision in ("fp32", "bf16")]
            if all(pair):
                speedup = f"{pair[0]['mean_ms']/pair[1]['mean_ms']:.2f}×" if all(r["status"] == "ok" for r in pair) else "N/A"
                rows.append([size, mode, cell(pair[0]), cell(pair[1]), speedup])
    sections.append(table(["模型", "模式", "FP32", "BF16 autocast", "加速比"], rows))
    sections.append("\n## Warmup 对照\n所有 case 在新进程运行，w=0 包含首次内核/library/optimizer 初始化。不能把不同模式的均值相减当成直接测得的阶段耗时。\n")
    warm_rows = []
    for mode in ("forward", "forward_backward", "train_step"):
        warm_rows.append([mode, *[next((cell(r) for r in normal if r["arguments"]["model_size"] == "small" and r["arguments"]["mode"] == mode and
                                      r["arguments"]["precision"] == "fp32" and r["arguments"]["warmup"] == w), "missing") for w in (0, 1, 2, 5)]])
    sections.append(table(["small FP32", "w=0", "w=1", "w=2", "w=5"], warm_rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode in ("forward", "forward_backward", "train_step"):
        values = [r for r in normal if r["arguments"]["model_size"] == "small" and r["arguments"]["mode"] == mode and r["arguments"]["precision"] == "fp32" and r["status"] == "ok"]
        values.sort(key=lambda r: r["arguments"]["warmup"])
        axes[0].plot([r["arguments"]["warmup"] for r in values], [r["mean_ms"] for r in values], "o-", label=mode)
        axes[1].plot([r["arguments"]["warmup"] for r in values], [r["std_ms"] for r in values], "o-", label=mode)
    for ax, label in zip(axes, ("Mean latency (ms)", "Sample std (ms)")):
        ax.set(xlabel="Warmup steps", ylabel=label)
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "warmup.png", dpi=160)
    plt.close(fig)
    sections.append("\n![warmup](figures/warmup.png)\n\n## 显存快照\n")
    memory = [r for r in records if r["file"].endswith("-memory.json")]
    sections.append(table(["模型", "context", "模式", "精度", "结果"], [[r["arguments"]["model_size"], r["arguments"]["context_length"], r["arguments"]["mode"], r["arguments"]["precision"], cell(r)] for r in memory]))
    from torch.cuda import _memory_viz
    for path in (ROOT / "artifacts").glob("*.pickle"):
        with path.open("rb") as file:
            snapshot = pickle.load(file)
        path.with_suffix(".memory.html").write_text(_memory_viz.trace_plot(snapshot))
        events = snapshot["device_traces"][0]
        delta = [(e["size"] if e["action"] == "alloc" else -e["size"] if e["action"] == "free_completed" else 0) for e in events]
        final = sum(block["size"] for segment in snapshot["segments"] for block in segment["blocks"] if block["state"].startswith("active"))
        active = (final - sum(delta) + np.cumsum(delta)) / 2**20
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(active)
        ax.set(title=path.stem, xlabel="Allocator event index (not wall time)", ylabel="Active memory (MiB)")
        ax.grid(alpha=.3)
        fig.tight_layout()
        figure = ROOT / "figures" / f"{path.stem}-memory.png"
        fig.savefig(figure, dpi=160)
        plt.close(fig)
        sections.append(f"\n![{path.stem}](figures/{figure.name})\n")
    sections.append("\n上述 PNG 由真实快照重建，横轴是分配事件编号，不是秒。交互式 `.memory.html` 使用 PyTorch memory_viz；这些是 small 替代图，不是讲义要求的 XL 官方截图。peak allocated、peak reserved 与 nvidia-smi 的进程占用含义不同，不可互换。\n")
    sections.append("\n## Profiler 阶段计时\n开启 Profiler 会扰动运行；以下诊断结果不替代上面的纯计时结果。CUDA event 阶段时长包含该流上的调度间隙，不等于 kernel 时长之和。\n")
    diagnostic = [r for r in records if r["status"] == "ok" and r.get("phase_ms")]
    sections.append(table(["case", "Forward", "Loss", "Backward", "Optimizer"], [[r["file"], *[f"{r['phase_ms'].get(k,0):.3f}" for k in ("forward", "loss", "backward", "optimizer")]] for r in diagnostic]))
    precision = ROOT / "data" / "precision-probe.json"
    if precision.exists():
        probe = json.loads(precision.read_text())
        sections.append("\n## 累加与 ToyModel dtype 实测\n```json\n" + json.dumps(probe, indent=2, ensure_ascii=False) + "\n```\n")
    variants = [r for r in records if "-checkpoint" in r["file"] or "-compile" in r["file"]]
    if variants:
        sections.append("\n## Checkpoint / compile 替代实验\n" + table(["case", "结果"], [[r["file"], cell(r)] for r in variants]))
    sections.append("\n## 数据与可复现性\n原始 JSON：`data/`；命令/stdout/stderr/exit code/GPU telemetry：`../logs/gpu_study/`；Nsight 报告、Chrome trace、显存快照：`artifacts/`。退出码 42 是实测 OOM；43 是容量下界阻断；不能把它们当作成功 benchmark。所有当前无法完成的官方结果须在最终 writeup 标为缺失。\n")
    (ROOT / "RESULTS.md").write_text("\n\n".join(sections))
    statuses = {status: sum(r["status"] == status for r in records) for status in sorted({r["status"] for r in records})}
    print(json.dumps({"cases": len(records), "statuses": statuses, "report": str(ROOT / "RESULTS.md")}, indent=2))


if __name__ == "__main__":
    main()
