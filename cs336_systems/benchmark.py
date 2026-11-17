import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM


MODES = ("forward", "forward_backward", "train_step")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    device,
    warmup=5,
    steps=10,
    mode="train_step",
    vocab_size=50257,
    context_length=64,
    d_model=128,
    num_layers=2,
    num_heads=4,
    d_ff=256,
    batch_size=1,
):
    device = torch.device(device)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if warmup < 0 or steps < 1:
        raise ValueError("warmup must be nonnegative and steps must be positive")
    if min(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, batch_size) < 1:
        raise ValueError("model dimensions and batch_size must be positive")
    if d_model % num_heads or (d_model // num_heads) % 2:
        raise ValueError("d_model must be divisible by num_heads with an even head dimension for RoPE")
    torch.manual_seed(42)
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) if mode == "train_step" else None
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    timings = []

    with torch.set_grad_enabled(mode != "forward"):
        for step in range(warmup + steps):
            model.zero_grad(set_to_none=True)
            synchronize(device)
            start = time.perf_counter()
            logits = model(x)
            if mode != "forward":
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                loss.backward()
            if optimizer is not None:
                optimizer.step()
            synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if step >= warmup:
                timings.append(elapsed_ms)
            # Release outputs before the next iteration, outside the timed region.
            del logits
            if mode != "forward":
                del loss
    model.zero_grad(set_to_none=True)

    print(f"mode={mode}, device={device}, warmup={warmup}, steps={steps}")
    print(f"mean={statistics.mean(timings):.3f} ms, std={statistics.stdev(timings) if len(timings) > 1 else 0:.3f} ms")
    print("per_step_ms=[" + ", ".join(f"{value:.3f}" for value in timings) + "]")
    return timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="train_step")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    try:
        benchmark(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
