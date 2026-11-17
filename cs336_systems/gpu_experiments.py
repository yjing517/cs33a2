"""Reproducible GPU studies; keeps the original benchmark API unchanged."""

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
from pathlib import Path
import statistics
import time
import traceback

from einops import einsum
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import cs336_basics.model as basics
from cs336_basics.optimizer import AdamW


SIZES = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10B": (4608, 12288, 50, 36),
    "surrogate": (256, 1024, 4, 4),
}


def parameter_count(size, vocab_size=10000):
    d, ff, layers, _ = SIZES[size]
    return 2 * vocab_size * d + layers * (4 * d * d + 3 * d * ff + 2 * d) + d


class CheckpointGroup(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def run(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        return checkpoint(self.run, x, use_reentrant=False) if torch.is_grad_enabled() else self.run(x)


@contextmanager
def stage(name, args, event_pairs):
    if not (args.profile or args.nvtx or args.stage_timing):
        yield
        return
    with torch.profiler.record_function(name), torch.cuda.nvtx.range(name):
        if args.stage_timing:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        yield
        if args.stage_timing:
            end.record()
            event_pairs.append((name, start, end))


def annotate_attention(args):
    def attention(Q, K, V, mask=None):
        with stage("attention_scores_matmul", args, []):
            scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(K.shape[-1])
        with stage("attention_mask_softmax", args, []):
            if mask is not None:
                scores = torch.where(mask, scores, float("-inf"))
            weights = basics.softmax(scores, dim=-1)
        with stage("attention_output_matmul", args, []):
            return einsum(weights, V, "... q k, ... k d -> ... q d")
    basics.scaled_dot_product_attention = attention


def experiment(args, verification=None):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("GPU experiments require CUDA")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    d, ff, layers, heads = SIZES[args.model_size]
    count = parameter_count(args.model_size)
    free, total = torch.cuda.mem_get_info(device)
    # FP32 weights/master gradients and two Adam moments: a strict lower bound.
    lower_bound = count * 4 * (4 if args.mode == "train_step" else 2 if args.mode == "forward_backward" else 1)
    result = {"arguments": vars(args), "parameters": count, "parameter_bytes": count * 4,
              "persistent_lower_bound_bytes": lower_bound, "free_bytes": free, "total_bytes": total,
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(device), "tf32": False},
              "official_shape": args.model_size != "surrogate" and args.batch_size == 4 and args.vocab_size == 10000}
    if lower_bound > free:
        result.update(status="blocked_capacity_lower_bound", reason="Persistent tensor lower bound already exceeds free VRAM; not an observed OOM")
        return result, 43
    torch.cuda.reset_peak_memory_stats(device)
    model = basics.BasicsTransformerLM(args.vocab_size, args.context_length, d, layers, heads, ff).to(device)
    assert sum(p.numel() for p in model.parameters()) == count
    if args.checkpoint_group:
        original = list(model.layers)
        model.layers = torch.nn.ModuleList([CheckpointGroup(original[i:i + args.checkpoint_group])
                                          for i in range(0, len(original), args.checkpoint_group)])
    if args.compile:
        result["compile_settings"] = {"backend": "inductor", "mode": "default", "dynamic": None,
                                      "fullgraph": False, "scope": "model forward and generated backward; optimizer eager"}
        compile_start = time.perf_counter()
        model = torch.compile(model, backend="inductor", mode="default", dynamic=None)
        result["compile_wrapper_ms"] = (time.perf_counter() - compile_start) * 1000
    if args.nvtx or args.profile:
        annotate_attention(args)
    optimizer = None
    if args.mode == "train_step":
        optimizer = (AdamW if args.optimizer == "reference_adamw" else torch.optim.Adam)(model.parameters(), lr=1e-3)
    x = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(args.vocab_size, x.shape, device=device)
    timings, phases, memory_points = [], {}, []
    last_loss = None

    def step(measured):
        nonlocal last_loss
        model.zero_grad(set_to_none=True)
        pairs = []
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.set_grad_enabled(args.mode != "forward"):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                with stage("forward", args, pairs):
                    logits = model(x)
                if args.mode != "forward":
                    with stage("loss", args, pairs):
                        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))
            if args.mode != "forward":
                with stage("backward", args, pairs):
                    loss.backward()
            if optimizer is not None:
                with stage("optimizer", args, pairs):
                    optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000
        if measured:
            timings.append(elapsed)
            for name, begin, end in pairs:
                phases.setdefault(name, []).append(begin.elapsed_time(end))
            memory_points.append({"step": len(timings), "allocated_bytes": torch.cuda.memory_allocated(device),
                                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
        if args.mode != "forward":
            last_loss = loss.detach()
        del logits
        return elapsed

    result["warmup_step_ms"] = []
    for index in range(args.warmup):
        elapsed = step(False)
        result["warmup_step_ms"].append(elapsed)
        if index == 0:
            result["first_step_ms"] = elapsed
            result["first_step_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            print(json.dumps({"first_step_ms": elapsed, "compiled": args.compile}), flush=True)
    torch.cuda.synchronize(device)
    result["warmup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result["measurement_start_allocated_bytes"] = torch.cuda.memory_allocated(device)
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                     record_shapes=True, profile_memory=True, with_stack=True) if args.profile else nullcontext()
    nvtx = torch.autograd.profiler.emit_nvtx(record_shapes=True) if args.nvtx else nullcontext()
    if args.snapshot:
        torch.cuda.memory._record_memory_history(max_entries=100000)
    try:
        with profiler as prof, nvtx, torch.cuda.nvtx.range("measurement"):
            for _ in range(args.steps):
                step(True)
                if args.profile:
                    prof.step()
        if args.snapshot:
            Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(args.snapshot)
    finally:
        if args.snapshot:
            torch.cuda.memory._record_memory_history(enabled=None)
    result.update(status="ok", per_step_ms=timings, mean_ms=statistics.mean(timings),
                  std_ms=statistics.stdev(timings) if len(timings) > 1 else 0.,
                  phase_ms={k: statistics.mean(v) for k, v in phases.items()}, memory_points=memory_points,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  final_loss=last_loss.item() if last_loss is not None else None,
                  parameter_dtype=str(next(model.parameters()).dtype))
    if last_loss is not None:
        assert torch.isfinite(last_loss), "nonfinite loss"
        assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "nonfinite gradients"
    if args.profile:
        prefix = Path(args.profile)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(prefix.with_suffix(".trace.json")))
        rows = [{"operator": e.key, "count": e.count, "self_cuda_us": e.self_device_time_total,
                 "total_cuda_us": e.device_time_total, "self_cuda_memory_bytes": e.self_device_memory_usage}
                for e in prof.key_averages()]
        prefix.with_suffix(".operators.json").write_text(json.dumps(sorted(rows, key=lambda r: -r["self_cuda_us"]), indent=2))
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    if verification is not None:
        result["verification"] = verification(model, x, y)
    return result, 0


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", choices=SIZES, default="small")
    p.add_argument("--mode", choices=("forward", "forward_backward", "train_step"), default="train_step")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--optimizer", choices=("adam", "reference_adamw"), default="adam")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--profile", help="PyTorch profiler output prefix (separate from timing-only runs)")
    p.add_argument("--snapshot", help="CUDA memory history pickle output")
    p.add_argument("--nvtx", action="store_true")
    p.add_argument("--stage-timing", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--checkpoint-group", type=int, default=0)
    p.add_argument("--output", required=True)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if min(args.batch_size, args.vocab_size, args.context_length, args.steps) < 1 or args.warmup < 0 or args.checkpoint_group < 0:
        p.error("dimensions/steps must be positive; warmup/checkpoint-group must be nonnegative")
    output = Path(args.output)
    if output.exists():
        p.error(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        result, code = experiment(args)
    except torch.cuda.OutOfMemoryError as exc:
        result, code = {"status": "observed_oom", "arguments": vars(args), "error": str(exc)}, 42
        traceback.print_exc()
    except Exception as exc:
        result, code = {"status": "error", "arguments": vars(args), "error": repr(exc)}, 1
        traceback.print_exc()
    result["process_work_seconds"] = time.perf_counter() - start
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_step_ms", "memory_points")}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
