"""Inspect unique saved activation storages and actual parameter gradient bytes."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from cs336_basics.model import RotaryEmbedding, TransformerBlock
from cs336_systems.gpu_experiments import SIZES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("small", "xl"), default="small")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    d, ff, _, heads = SIZES[args.size]
    block = TransformerBlock(d, heads, ff, RotaryEmbedding(args.context, d // heads)).cuda()
    parameter_storages = {p.untyped_storage().data_ptr() for p in block.parameters()}
    if args.compile:
        block = torch.compile(block, fullgraph=True)
    x = torch.randn(4, args.context, d, device="cuda", requires_grad=True)
    seen, records = set(), []

    def pack(tensor):
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer not in parameter_storages and pointer not in seen and tensor.numel():
            seen.add(pointer)
            records.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype), "storage_bytes": storage.nbytes(),
                            "logical_bytes": tensor.numel() * tensor.element_size(),
                            "producer": type(tensor.grad_fn).__name__ if tensor.grad_fn else "leaf_or_constant"})
        return tensor

    torch.cuda.reset_peak_memory_stats()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        y = block(x)
    torch.cuda.synchronize()
    forward_live = torch.cuda.memory_allocated()
    forward_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y.sum().backward()
    torch.cuda.synchronize()
    groups = defaultdict(int)
    for row in records:
        groups[row["producer"]] += row["storage_bytes"]
    activation_bytes = sum(groups.values())
    result = {"arguments": vars(args), "scope": "one isolated TransformerBlock, not the full XL model",
              "saved_unique_activation_bytes": activation_bytes, "saved_storages": records,
              "top_producers": [{"producer": key, "bytes": value, "fraction": value / activation_bytes}
                                for key, value in sorted(groups.items(), key=lambda pair: -pair[1])[:5]],
              "parameter_bytes": sum(p.numel() * p.element_size() for p in block.parameters()),
              "parameter_gradient_bytes": sum(p.grad.numel() * p.grad.element_size() for p in block.parameters()),
              "forward_live_bytes": forward_live, "forward_peak_bytes": forward_peak,
              "backward_peak_bytes": torch.cuda.max_memory_allocated(),
              "input_gradient_bytes": x.grad.numel() * x.grad.element_size()}
    assert result["parameter_bytes"] == result["parameter_gradient_bytes"]
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    path = Path(args.output)
    with path.open("x") as file:
        json.dump(result, file, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "saved_storages"}, indent=2))


if __name__ == "__main__":
    main()
