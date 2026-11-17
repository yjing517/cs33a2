import copy

import pytest
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.gpu_experiments import CheckpointGroup, SIZES, parameter_count


@pytest.mark.parametrize("size", SIZES)
def test_parameter_capacity_formula(size):
    d, ff, layers, heads = SIZES[size]
    with torch.device("meta"):
        model = BasicsTransformerLM(10000, 512, d, layers, heads, ff)
    assert sum(p.numel() for p in model.parameters()) == parameter_count(size)


def test_checkpoint_preserves_output_and_gradients():
    torch.manual_seed(42)
    layers = [torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8)]
    reference = torch.nn.Sequential(*copy.deepcopy(layers)).double()
    actual = CheckpointGroup(layers).double()
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    y = actual(x)
    expected = reference(x)
    torch.testing.assert_close(y, expected)
    y.square().sum().backward()
    expected.square().sum().backward()
    for p, q in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
