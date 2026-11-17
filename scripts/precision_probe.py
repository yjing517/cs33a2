"""Assignment mixed-precision accumulation and measured ToyModel dtype probe."""

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, 32, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.ln(self.relu(self.fc1(x))))


def main():
    accumulation = []
    for accumulator, term, explicit_cast in ((torch.float32, torch.float32, False),
                                             (torch.float16, torch.float16, False),
                                             (torch.float32, torch.float16, False),
                                             (torch.float32, torch.float16, True)):
        total = torch.tensor(0, dtype=accumulator)
        for _ in range(1000):
            value = torch.tensor(0.01, dtype=term)
            total += value.float() if explicit_cast else value
        accumulation.append({"accumulator": str(accumulator), "term": str(term), "explicit_cast": explicit_cast,
                             "actual": total.item(), "absolute_error": abs(total.item() - 10.)})
    dtypes = {}
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(42)
        model = ToyModel().cuda()
        observed = {}
        handles = []
        for name in ("fc1", "ln", "fc2", "relu"):
            def hook(module, args, output, name=name):
                observed[name] = str(output.dtype)
            handles.append(getattr(model, name).register_forward_hook(hook))
        with torch.autocast("cuda", dtype=dtype):
            observed["parameters_inside_autocast"] = sorted({str(p.dtype) for p in model.parameters()})
            logits = model(torch.randn(4, 64, device="cuda"))
            loss = F.cross_entropy(logits, torch.randint(32, (4,), device="cuda"))
        loss.backward()
        observed.update(logits=str(logits.dtype), loss=str(loss.dtype),
                        gradients=sorted({str(p.grad.dtype) for p in model.parameters()}))
        for handle in handles:
            handle.remove()
        dtypes[str(dtype)] = observed
    result = {"accumulation": accumulation, "toy_dtypes": dtypes, "torch": torch.__version__}
    output = Path("reports/gpu_study/data/precision-probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
