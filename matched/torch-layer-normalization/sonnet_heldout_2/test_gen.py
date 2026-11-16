import torch
import pytest

import torch_layer_normalization
from torch_layer_normalization import LayerNormalization


def _reference(x, gamma=None, beta=None, eps=1e-10):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    y = (x - mean) / torch.sqrt(var + eps)
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y = y + beta
    return y


def test_version_defined():
    assert torch_layer_normalization.__version__ == '0.11.0'


def test_default_parameters_initialized():
    layer = LayerNormalization(5)
    assert layer.gamma is not None
    assert layer.beta is not None
    assert tuple(layer.gamma.shape) == (5,)
    assert tuple(layer.beta.shape) == (5,)
    assert torch.allclose(layer.gamma, torch.ones(5))
    assert torch.allclose(layer.beta, torch.zeros(5))


def test_parameter_count_default():
    layer = LayerNormalization(7)
    params = list(layer.parameters())
    assert len(params) == 2
    assert sum(p.numel() for p in params) == 14


def test_no_gamma():
    layer = LayerNormalization(4, gamma=False)
    assert layer.gamma is None
    assert layer.beta is not None
    assert len(list(layer.parameters())) == 1


def test_no_beta():
    layer = LayerNormalization(4, beta=False)
    assert layer.beta is None
    assert layer.gamma is not None
    assert len(list(layer.parameters())) == 1


def test_no_gamma_no_beta_has_no_parameters():
    layer = LayerNormalization(4, gamma=False, beta=False)
    assert len(list(layer.parameters())) == 0
    x = torch.randn(3, 4)
    assert torch.allclose(layer(x), _reference(x), atol=1e-5)


def test_epsilon_stored():
    layer = LayerNormalization(3, epsilon=1e-3)
    assert layer.epsilon == 1e-3


def test_output_shape_preserved_2d_and_3d():
    layer = LayerNormalization(6)
    assert layer(torch.randn(4, 6)).shape == (4, 6)
    assert layer(torch.randn(2, 3, 6)).shape == (2, 3, 6)


def test_output_zero_mean_unit_variance():
    torch.manual_seed(0)
    layer = LayerNormalization(16)
    x = torch.randn(8, 16) * 5 + 3
    y = layer(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(8), atol=1e-5)
    var = ((y - y.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
    assert torch.allclose(var, torch.ones(8), atol=1e-4)


def test_matches_reference_with_custom_gamma_beta():
    torch.manual_seed(1)
    layer = LayerNormalization(5)
    with torch.no_grad():
        layer.gamma.copy_(torch.tensor([1.0, 2.0, 3.0, 0.5, -1.0]))
        layer.beta.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
    x = torch.randn(3, 4, 5)
    expected = _reference(x, layer.gamma.detach(), layer.beta.detach())
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_matches_torch_layer_norm():
    torch.manual_seed(2)
    eps = 1e-5
    layer = LayerNormalization(10, epsilon=eps)
    x = torch.randn(6, 10)
    expected = torch.nn.functional.layer_norm(x, (10,), eps=eps)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_constant_input_gives_beta():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.beta.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.full((2, 4), 7.0)
    y = layer(x)
    assert torch.allclose(y, layer.beta.detach().expand(2, 4), atol=1e-5)


def test_shift_invariance():
    torch.manual_seed(3)
    layer = LayerNormalization(8)
    x = torch.randn(5, 8)
    assert torch.allclose(layer(x), layer(x + 10.0), atol=1e-4)


def test_gradients_flow_to_parameters_and_input():
    torch.manual_seed(4)
    layer = LayerNormalization(6)
    x = torch.randn(3, 6, requires_grad=True)
    layer(x).pow(2).sum().backward()
    assert x.grad is not None
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None
    assert layer.gamma.grad.shape == (6,)


def test_tuple_normal_shape_uses_last_dimension():
    torch.manual_seed(5)
    layer = LayerNormalization((3, 5))
    x = torch.randn(2, 3, 5)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.allclose(y, _reference(x), atol=1e-5)


def test_reset_parameters_restores_defaults():
    layer = LayerNormalization(3)
    with torch.no_grad():
        layer.gamma.fill_(5.0)
        layer.beta.fill_(-2.0)
    layer.reset_parameters()
    assert torch.allclose(layer.gamma, torch.ones(3))
    assert torch.allclose(layer.beta, torch.zeros(3))
