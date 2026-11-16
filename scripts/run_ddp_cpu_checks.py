"""Run bounded CPU-only DDP checks and preserve the exact subprocess results."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    directory = args.directory or Path("reports/ddp_overlap_cpu" if args.overlap else "reports/ddp_flat_cpu" if args.flat else "reports/ddp_cpu")
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    results = []
    checks = [("official", "tests/test_ddp.py"), ("gradient_reference", "tests/test_ddp_cpu.py")]
    if args.flat or args.overlap:
        checks.append(("flat", "tests/test_ddp_flat.py"))
    if args.overlap:
        checks.extend((f"official_repeat_{repeat}", "tests/test_ddp.py") for repeat in range(2, 6))
        checks.append(("overlap", "tests/test_ddp_overlap.py"))
    for name, test in checks:
        command = [sys.executable, "-m", "pytest", "-q", test, "--timeout=120"]
        start = time.perf_counter()
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
        (directory / f"{name}.stdout.log").write_text(result.stdout)
        (directory / f"{name}.stderr.log").write_text(result.stderr)
        metadata = dict(command=command, cwd=str(Path.cwd()), exit_code=result.returncode,
                        elapsed_s=time.perf_counter() - start, timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                        environment={key: environment[key] for key in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]})
        (directory / f"{name}.json").write_text(json.dumps(metadata, indent=2))
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        print(f"{name}: exit_code={result.returncode}", flush=True)
        results.append(result.returncode)
    return int(any(results))


if __name__ == "__main__":
    sys.exit(main())
