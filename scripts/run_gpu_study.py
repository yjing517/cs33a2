"""Run bounded GPU study cases in fresh processes; all commands go through run_logged."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path("reports/gpu_study")
NSYS = "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys"


def logged(name, command):
    return subprocess.run([sys.executable, "scripts/run_logged.py", "--name", name, "--", *command]).returncode


def case(size="small", mode="train_step", precision="fp32", warmup=5, context=512, steps=10,
         suffix="", extra=(), nsys=False):
    name = f"{size}-{mode}-{precision}-w{warmup}-n{context}{suffix}"
    output = ROOT / "data" / f"{name}.json"
    if output.exists():
        print("SKIP existing result:", output, flush=True)
        return 0
    command = [sys.executable, "-m", "cs336_systems.gpu_experiments", "--model-size", size, "--mode", mode,
               "--precision", precision, "--warmup", str(warmup), "--steps", str(steps),
               "--context-length", str(context), "--output", str(output), *extra]
    if nsys:
        prefix = ROOT / "artifacts" / name
        command = [NSYS, "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                   "--cuda-memory-usage=true", "--capture-range=nvtx", "--nvtx-capture=measurement",
                   "--capture-range-end=stop", "--output", str(prefix), *command, "--nvtx"]
    return logged(name, command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("timings", "profiles", "memory", "variants"))
    args = parser.parse_args()
    for directory in ("data", "artifacts", "figures"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    if args.phase == "timings":
        for warmup in (0, 1, 2):
            for mode in ("forward", "forward_backward", "train_step"):
                case(mode=mode, warmup=warmup)
        for size in ("small", "medium", "large", "xl", "10B"):
            for precision in ("fp32", "bf16"):
                for mode in ("forward", "forward_backward", "train_step"):
                    case(size=size, mode=mode, precision=precision)
    elif args.phase == "profiles":
        for size in ("small", "medium"):
            for context in (256, 512, 1024):
                case(size=size, context=context, steps=1, suffix="-nsys", nsys=True,
                     extra=("--optimizer", "reference_adamw"))
        for mode in ("forward", "train_step"):
            prefix = ROOT / "artifacts" / f"small-{mode}-torch"
            case(mode=mode, steps=2, suffix="-torchprofile",
                 extra=("--optimizer", "reference_adamw", "--profile", str(prefix), "--stage-timing"))
        # Bound the longest feasible power-of-two training context without changing the model.
        for context in (2048,):
            case(context=context, steps=1, suffix="-capacity-probe")
    elif args.phase == "memory":
        # XL is preflighted and explicitly recorded, never relabeled as a smaller model.
        for size in ("xl", "small"):
            for context in (128, 2048):
                for precision in ("fp32", "bf16"):
                    for mode in ("forward", "train_step"):
                        snapshot = ROOT / "artifacts" / f"{size}-{mode}-{precision}-n{context}.pickle"
                        case(size=size, context=context, mode=mode, precision=precision, steps=1, suffix="-memory",
                             extra=("--snapshot", str(snapshot), "--stage-timing", "--optimizer", "reference_adamw"))
    elif args.phase == "variants":
        for group in (0, 1, 2, 4):
            case(size="surrogate", context=512, suffix=f"-checkpoint{group}",
                 extra=("--checkpoint-group", str(group)))
        for mode in ("forward", "forward_backward", "train_step"):
            for compiled in (False, True):
                case(size="surrogate", mode=mode, suffix=f"-compile{compiled}", extra=("--compile",) if compiled else ())


if __name__ == "__main__":
    main()
