import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from cs336.basic.model import BasicTransformerLM

def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
def benchmark(device,warmup=2,steps=5):
    torch.manuak_seed(42)

    model=BasicTransformerLM(
        vocab_size=50257,
        context_length=64,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        device=device,    
    )
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

    x=torch.randint(0,50257,(1,64),device=device)
    y=torch.randint(0,50257,(1,64),device=device)

    results={"fotward":[],"backward:[]",optimize:[]}

    for step in range(steps+warmup):

        oprimizer.zero_grad(set_to_none=True)

        synchronize(device)

        start=time.perf_counter()

        logits=model(x)

        loss=F.cross_entropy(logits.reshape(-1,1000),y.reshape(-1))

        synchronize(device)

        forward_ms=(time.perf_counter()-start)*1000

        start=time.perf_counter()

        loss.backward()

        synchronize(device)
        backward_ms=(time.perf_counter()-start)*1000

        start=time.perf_counter()

        optimizer.step()
        synchronize(device)
        optimize_ms=(time.perf_counter()-start)*1000
        if step>=warmup:
            results["forward"].append(forward_ms)
            results["backward"].append(backward_ms)
            results["optimize"].append(optimize_ms)
            
    for name, values in results.items():
        print(
            f"{name}: "
            f"mean={statistics.mean(values):.2f} ms, "
            f"std={statistics.stdev(values) if len(values) > 1 else 0:.2f} ms"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    benchmark(
        torch.device(args.device),
        warmup=args.warmup,
        steps=args.steps,
    )


