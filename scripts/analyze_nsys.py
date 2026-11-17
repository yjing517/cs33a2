"""Attribute Nsight CUDA kernels using launch correlation IDs and NVTX CPU ranges."""

from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("reports/gpu_study")
NSYS = "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys"
PHASES = ("forward", "loss", "backward", "optimizer", "attention_scores_matmul", "attention_mask_softmax", "attention_output_matmul")


def summarize(kernels):
    names = defaultdict(lambda: [0, 0])
    for kernel in kernels:
        names[kernel["name"]][0] += 1
        names[kernel["name"]][1] += kernel["end"] - kernel["start"]
    rows = sorted([{"kernel": name, "count": count, "ms": duration / 1e6} for name, (count, duration) in names.items()], key=lambda x: -x["ms"])
    total = sum(r["ms"] for r in rows)
    gemm = sum(r["ms"] for r in rows if re.search("gemm|matmul|mma|cutlass", r["kernel"], re.I))
    return {"kernel_sum_ms": total, "gemm_kernel_sum_ms": gemm, "gemm_fraction": gemm / total if total else 0., "kernels": rows}


def main():
    results = {}
    for report in sorted((ROOT / "artifacts").glob("*.nsys-rep")):
        database = report.with_suffix(".sqlite")
        if not database.exists():
            command = [NSYS, "export", "--type=sqlite", "--output", str(database), str(report)]
            code = subprocess.run([sys.executable, "scripts/run_logged.py", "--name", "export-" + report.stem, "--", *command]).returncode
            if code:
                raise RuntimeError(f"Nsight export failed: {code}")
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        # A GPU kernel can execute after the CPU range has closed. Attribute it by
        # the timestamp of its correlated launch API, not its GPU start timestamp.
        # Backward launches may originate on Autograd worker threads. Each report
        # contains one model process, and the named top-level phases do not overlap.
        kernels = [dict(row) for row in connection.execute("""
            SELECT k.start, k.end, s.value AS name, r.start AS launch_start, r.globalTid
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
            JOIN StringIds s ON k.demangledName = s.id
        """)]
        ranges = [dict(row) for row in connection.execute("""
            SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value) AS name
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
            WHERE n.end IS NOT NULL
        """)]
        phases = {}
        for name in PHASES:
            matching = [r for r in ranges if r["name"] == name]
            if not matching:
                continue
            selected = [k for k in kernels if any(r["start"] <= k["launch_start"] < r["end"] for r in matching)]
            phases[name] = {**summarize(selected), "nvtx_cpu_range_ms": sum(r["end"] - r["start"] for r in matching) / 1e6, "nvtx_ranges": len(matching)}
        results[report.stem] = {"all": summarize(kernels), "phases": phases}
        print(report.stem, {name: round(value["kernel_sum_ms"], 3) for name, value in phases.items()}, flush=True)
        connection.close()
    (ROOT / "nsight-analysis.json").write_text(json.dumps(results, indent=2))
    sections = ["# Nsight Systems 实测分析\n", "所有统计来自真实 `.nsys-rep` 导出的 SQLite。warmup 被 NVTX capture range 排除。GPU kernel 通过 correlationId 关联到发起它的 CUDA API，再映射到 NVTX CPU 范围；没有把 GPU 执行时间直接与 CPU 范围求交。kernel sum 是 GPU 工作时间之和，不等于端到端墙钟时间。GEMM 比例按 kernel 名称识别，属于可复核的分类估计。\n"]
    for name, result in results.items():
        sections.append(f"\n## {name}\n")
        sections.append("| 阶段 | NVTX CPU 范围 ms | GPU kernel sum ms | GEMM 比例 |\n|---|---:|---:|---:|")
        for phase, value in result["phases"].items():
            sections.append(f"| {phase} | {value['nvtx_cpu_range_ms']:.3f} | {value['kernel_sum_ms']:.3f} | {value['gemm_fraction']:.1%} |")
        for phase in ("forward", "backward", "optimizer"):
            value = result["phases"].get(phase)
            if value:
                sections.append(f"\n{phase} 最耗时的三个 kernel（完整名称与所有计数见 JSON）：\n")
                for row in value["kernels"][:3]:
                    sections.append(f"- `{row['kernel'][:150]}`：{row['count']} 次，{row['ms']:.3f} ms")
        forward = result["phases"].get("forward")
        if forward:
            softmax = result["phases"].get("attention_mask_softmax", {}).get("kernel_sum_ms", 0.)
            matmul = sum(result["phases"].get(key, {}).get("kernel_sum_ms", 0.) for key in ("attention_scores_matmul", "attention_output_matmul"))
            sections.append(f"\nAttention 的 mask+Softmax GPU kernel sum={softmax:.3f} ms；QK/PV 两个矩阵乘相关范围合计={matmul:.3f} ms。Softmax FLOPs 较少，但包含多次读写和归约，FLOPs 比例不能预测相同的耗时比例。\n")
        phases = [p for p in ("forward", "loss", "backward", "optimizer") if p in result["phases"]]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(phases, [result["phases"][p]["kernel_sum_ms"] for p in phases])
        ax.set(title=name, ylabel="GPU kernel sum (ms)")
        fig.tight_layout()
        fig.savefig(ROOT / "figures" / f"{name}-stages.png", dpi=160)
        plt.close(fig)
    sections.append("\n这些柱状图是从采集数据生成的分析图，不是 Nsight GUI 截图。完整版报告可由用户在 Nsight Systems GUI 打开 `.nsys-rep` 检查。\n")
    (ROOT / "NSIGHT_ANALYSIS.md").write_text("\n".join(sections))


if __name__ == "__main__":
    main()
