"""Run one command with separate stdout/stderr, exit status and GPU telemetry."""

import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time


def gpu_state():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,pstate", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--directory", default="reports/logs/gpu_study")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%f")
    stem = directory / f"{stamp}-{args.name}"
    metadata = {"command": command, "cwd": str(Path.cwd()), "start_utc": stamp, "gpu_before": gpu_state()}
    print("COMMAND:", shlex.join(command), flush=True)
    start = time.perf_counter()
    with stem.with_suffix(".stdout.log").open("w") as stdout, stem.with_suffix(".stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        def copy(stream, file, console):
            for line in stream:
                file.write(line)
                file.flush()
                console.write(line)
                console.flush()

        threads = [threading.Thread(target=copy, args=(process.stdout, stdout, sys.stdout)),
                   threading.Thread(target=copy, args=(process.stderr, stderr, sys.stderr))]
        for thread in threads:
            thread.start()
        returncode = process.wait()
        for thread in threads:
            thread.join()
    metadata.update(exit_code=returncode, elapsed_s=time.perf_counter() - start, gpu_after=gpu_state())
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"EXIT_CODE={returncode} ELAPSED_S={metadata['elapsed_s']:.3f} LOG={stem}", flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
