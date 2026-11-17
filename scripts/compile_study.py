"""Bounded eager/compile study; verify after measuring to preserve first-run timing."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from cs336_systems.gpu_experiments import experiment, parser

ROOT = Path('reports/compile_study')
LOGS = 'reports/logs/compile_study'


def verify(model, x, target):
    eager = model._orig_mod
    eager.zero_grad(set_to_none=True)
    output = eager(x)
    expected = output.detach().cpu()
    loss = F.cross_entropy(output.reshape(-1, 10000), target.reshape(-1))
    expected_loss = loss.detach().cpu()
    loss.backward()
    gradients = [(n, p.grad.detach().cpu()) for n, p in eager.named_parameters()]
    del output, loss
    eager.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=False):
        output = model(x)
        loss = F.cross_entropy(output.reshape(-1, 10000), target.reshape(-1))
    loss.backward()
    actual = output.detach().cpu()
    actual_loss = loss.detach().cpu()
    failures = []

    def check(a, b, atol, rtol, name):
        try:
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)
        except AssertionError as error:
            failures.append(str(error))

    check(actual, expected, 1e-5, 1e-4, 'output')
    check(actual_loss, expected_loss, 1e-5, 1e-5, 'loss')
    gradient_error = 0.
    parameters = list(eager.named_parameters())
    assert len(parameters) == len(gradients)
    for (name, p), (expected_name, expected_gradient) in zip(parameters, gradients):
        assert name == expected_name and p.grad is not None
        observed = p.grad.detach().cpu()
        gradient_error = max(gradient_error, (observed - expected_gradient).abs().max().item())
        check(observed, expected_gradient, 1e-5, 1e-3, name)
    return {'status': 'passed' if not failures else 'failed', 'parameter_tensors_checked': len(parameters),
            'max_output_absolute_error': (actual-expected).abs().max().item(),
            'max_gradient_absolute_error': gradient_error,
            'eager_loss': expected_loss.item(), 'compiled_loss': actual_loss.item(),
            'loss_absolute_error': (actual_loss-expected_loss).abs().item(),
            'tolerances': {'output': [1e-5, 1e-4], 'loss': [1e-5, 1e-5], 'gradient': [1e-5, 1e-3]},
            'failures': failures}


def case(name):
    output = ROOT / f'{name}.json'
    if output.exists():
        raise FileExistsError(output)
    args = parser().parse_args(['--model-size', 'small', '--mode', 'train_step', '--precision', 'fp32',
                                '--optimizer', 'adam', '--batch-size', '4', '--context-length', '512',
                                '--warmup', '5', '--steps', '10', '--output', str(output)] +
                               (['--compile'] if name == 'compiled' else []))
    started = time.perf_counter()
    try:
        result, code = experiment(args, verification=verify if args.compile else None)
        if result.get('verification', {}).get('status') == 'failed':
            code = 1
    except Exception as error:
        traceback.print_exc()
        result, code = {'status': 'error', 'error': repr(error), 'arguments': vars(args)}, 1
    result.update(process_work_seconds=time.perf_counter()-started, exit_code=code,
                  cache_directory=os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
                  compile_threads=os.environ.get('TORCHINDUCTOR_COMPILE_THREADS'))
    with output.open('x') as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return code


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    action = sys.argv[1]
    if action in ('eager', 'compiled'):
        return case(action)
    assert action == 'run'
    if any((ROOT / f'{name}.json').exists() for name in ('eager', 'compiled')):
        raise FileExistsError('Refusing to repeat existing cases')
    for name in ('eager', 'compiled'):
        command = [sys.executable, 'scripts/run_logged.py', '--directory', LOGS, '--name', name, '--',
                   'timeout', '--signal=TERM', '--kill-after=10s', '420s', sys.executable, __file__, name]
        code = subprocess.run(command).returncode
        if code:
            print(f'{name} failed with exit {code}; no retry', flush=True)
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
