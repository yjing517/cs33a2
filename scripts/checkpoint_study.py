"""Bounded checkpoint-only GPU correctness and three-case training study."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.gpu_experiments import CheckpointGroup


ROOT = Path("reports/checkpoint_study")
LOGS = "reports/logs/checkpoint_study"


def verify():
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = BasicsTransformerLM(10000, 512, 768, 12, 12, 3072).cuda()
    x = torch.randint(10000, (4, 512), device="cuda")
    target = torch.randint(10000, x.shape, device="cuda")
    logits = reference(x)
    expected_output = logits.detach().cpu()
    loss = F.cross_entropy(logits.reshape(-1, 10000), target.reshape(-1))
    loss.backward()
    expected_gradients = [(name, p.grad.detach().cpu()) for name, p in reference.named_parameters()]
    reference.zero_grad(set_to_none=True)
    del logits, loss
    results = []
    for group in (1, 2):
        model = copy.deepcopy(reference)
        layers = list(model.layers)
        model.layers = torch.nn.ModuleList([CheckpointGroup(layers[i:i + group]) for i in range(0, 12, group)])
        actual = model(x)
        torch.testing.assert_close(actual.detach().cpu(), expected_output, atol=1e-6, rtol=1e-5)
        output_error = (actual.detach().cpu() - expected_output).abs().max().item()
        F.cross_entropy(actual.reshape(-1, 10000), target.reshape(-1)).backward()
        parameters = list(model.parameters())
        assert len(parameters) == len(expected_gradients)
        max_gradient_error = 0.
        for p, (name, expected) in zip(parameters, expected_gradients):
            assert p.grad is not None and torch.isfinite(p.grad).all().item(), name
            observed = p.grad.detach().cpu()
            torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-5, msg=name)
            max_gradient_error = max(max_gradient_error, (observed - expected).abs().max().item())
        result = {"group": group, "status": "passed", "parameter_tensors_checked": len(parameters),
                  "max_output_absolute_error": output_error, "max_gradient_absolute_error": max_gradient_error,
                  "atol": 1e-6, "rtol": 1e-5}
        results.append(result)
        print(json.dumps(result), flush=True)
        del actual, parameters, model, layers
    with (ROOT / "verification.json").open("x") as file:
        json.dump(results, file, indent=2)


def logged(name, command):
    code = subprocess.run([sys.executable, "scripts/run_logged.py", "--directory", LOGS,
                           "--name", name, "--", *command]).returncode
    if code:
        raise RuntimeError(f"{name} exited {code}; stopping checkpoint study")


def report():
    checks = json.loads((ROOT / "verification.json").read_text())
    data = [json.loads((ROOT / f"group{group}.json").read_text()) for group in (0, 1, 2)]
    baseline = data[0]
    lines = ["# Gradient Checkpointing：GPU 对照与教学说明", "",
             "## A. 目标与范围", "只研究保存激活与反向重计算的取舍，不重跑历史矩阵，不使用 BF16、compile、Nsight 或 DDP。",
             "## B. 配置与完成情况", "RTX 3080 Ti 12GB；PyTorch 2.8.0+cu128；官方 small：d_model=768、d_ff=3072、12层、12 heads；batch=4、context=512、vocab=10000；FP32、Adam(lr=1e-3)，固定seed=42。每种分组独立新进程，5次warmup、10次测量，样本标准差。",
             "这是讲义 XL、batch=4、context=2048 实验的替代，不是 XL 正式结果；本次未尝试该已知超容量配置。",
             "## C. 修改文件", "仅新增 scripts/checkpoint_study.py：verify 检查输出/梯度；logged 复用日志器；report 生成本表；main 限定只运行3组。已有 CheckpointGroup、gpu_experiments.py 与原有测试断言未修改。",
             "## D. 关键代码", "```python", "layers = list(model.layers)",
             "model.layers = torch.nn.ModuleList([", "    CheckpointGroup(layers[i:i + group])",
             "    for i in range(0, 12, group)", "])", "```",
             "1. 取出12层，顺序保持不变。2. 切成连续的group层。3. 每组套用现有CheckpointGroup；group=1得到12组，group=2得到6组。4. 内部使用 checkpoint(self.run, x, use_reentrant=False)，保存组输入，在反向时重算组内激活。",
             "每个group保存的输入shape为[4,512,768]，FP32大小为6 MiB；这不是全部显存，因为还包括参数、梯度、Adam状态和临时buffer。",
             "## E. 实际执行命令与日志", "```bash",
             "export PYTHONPATH=.:cs336-basics OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1",
             "python scripts/run_logged.py --directory reports/logs/checkpoint_study --name checkpoint-suite -- python scripts/checkpoint_study.py run",
             "```", "验证子命令是 python scripts/checkpoint_study.py verify；三个性能子命令完整argv、stdout、stderr、退出码、进程耗时与GPU状态均在 reports/logs/checkpoint_study/。",
             "## F. 实测结果", "|Checkpoint组大小|训练Step平均ms|样本标准差ms|峰值allocated GiB|显存减少|时间增加|OOM|",
             "|---|---:|---:|---:|---:|---:|---|"]
    for group, result in zip((0, 1, 2), data):
        assert result["status"] == "ok"
        memory_saving = 1 - result["peak_allocated_bytes"] / baseline["peak_allocated_bytes"]
        slowdown = result["mean_ms"] / baseline["mean_ms"] - 1
        label = "关闭" if group == 0 else f"每{group}层"
        lines.append(f"|{label}|{result['mean_ms']:.3f}|{result['std_ms']:.3f}|{result['peak_allocated_bytes']/2**30:.3f}|{memory_saving:.1%}|{slowdown:.1%}|否|")
    lines += ["", f"正确性检查：{len(checks)} passed、0 failed、0 skipped；每次比较输出与全部{checks[0]['parameter_tensors_checked']}个参数张量的梯度。",
              f"最大输出绝对误差：{max(r['max_output_absolute_error'] for r in checks):.3e}；最大梯度绝对误差：{max(r['max_gradient_absolute_error'] for r in checks):.3e}。",
              "## G. 同步", "本地 WSL 项目为 ~/projects/cs336/assignment2-systems。只同步新增脚本、本目录结果和本次日志；SHA256清单为 SHA256SUMS。已有源码不同则停止，不覆盖；最终核验见 SYNC_VERIFICATION.json。",
              "## H. 为什么节省显存、增加时间", "普通Autograd保留大量层内中间结果；checkpoint只保留组输入，反向时重新计算需要的中间结果。因此以额外Forward计算换取更少长期保存的激活。",
              "分组越大，组输入数量越少，但反向重计算时临时激活越多，峰值不保证单调下降。参数、梯度和Adam状态仍占空间，所以总显存不会按层数等比例缩小。",
              "例如输入6 MiB、组内残差远大于6 MiB时，保存输入通常有利；但这不表示整个训练只占6 MiB。use_reentrant=False允许按反向需求提前停止重计算，因此时间开销也不等于精确多跑一次完整Forward。",
              "## I. 未完成与下一步", "已完成本次三组GPU对照。尚未完成讲义XL规模、最优粒度的更细子层搜索及递归checkpoint理论题；不根据这三个case宣称全局最优。本次结束后停止；下一阶段需单独授权，不自动开始torch.compile或DDP。"]
    (ROOT / "RESULTS.md").write_text("\n".join("\n" + line + "\n" if line.startswith("## ") else line for line in lines))
    print("\n".join(line for line in lines if line.startswith("|")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "verify", "report"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.action == "verify":
        verify()
    elif args.action == "report":
        report()
    else:
        start = time.perf_counter()
        assert not any((ROOT / name).exists() for name in ("verification.json", "group0.json", "group1.json", "group2.json")), "Existing results; do not rerun"
        logged("checkpoint-verification", [sys.executable, "scripts/checkpoint_study.py", "verify"])
        for group in (0, 1, 2):
            logged(f"checkpoint-group{group}", [sys.executable, "-m", "cs336_systems.gpu_experiments",
                   "--model-size", "small", "--mode", "train_step", "--precision", "fp32", "--optimizer", "adam",
                   "--batch-size", "4", "--context-length", "512", "--warmup", "5", "--steps", "10",
                   "--checkpoint-group", str(group), "--output", str(ROOT / f"group{group}.json")])
        report()
        (ROOT / "suite.json").write_text(json.dumps({"status": "ok", "elapsed_s": time.perf_counter() - start,
                                                    "correctness_passed": 2, "performance_cases": 3}, indent=2))


if __name__ == "__main__":
    main()
