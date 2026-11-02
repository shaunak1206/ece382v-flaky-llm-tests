import pytest
import torch
import torch.nn as nn

import torch_layer_normalization
from torch_layer_normalization import LayerNormalization


def test_version_defined():
    assert isinstance(torch_layer_normalization.__version__, str)
    assert torch_layer_normalization.__version__


def test_int_normal_shape():
    layer = LayerNormalization(5)
    assert layer.normal_shape == torch.Size((5,))


def test_tuple_normal_shape_uses_last_dim():
    layer = LayerNormalization((3, 4, 7))
    assert layer.normal_shape == torch.Size((7,))
    assert tuple(layer.gamma.shape) == (7,)
    assert tuple(layer.beta.shape) == (7,)


def test_torch_size_normal_shape():
    layer = LayerNormalization(torch.Size([2, 6]))
    assert layer.normal_shape == torch.Size((6,))


def test_default_parameter_initialization():
    layer = LayerNormalization(8)
    assert isinstance(layer.gamma, nn.Parameter)
    assert isinstance(layer.beta, nn.Parameter)
    assert torch.equal(layer.gamma.data, torch.ones(8))
    assert torch.equal(layer.beta.data, torch.zeros(8))


def test_no_gamma():
    layer = LayerNormalization(4, gamma=False)
    assert layer.gamma is None
    assert layer.beta is not None
    names = [n for n, _ in layer.named_parameters()]
    assert names == ['beta']


def test_no_beta():
    layer = LayerNormalization(4, beta=False)
    assert layer.beta is None
    assert layer.gamma is not None
    names = [n for n, _ in layer.named_parameters()]
    assert names == ['gamma']


def test_no_gamma_no_beta_has_no_parameters():
    layer = LayerNormalization(4, gamma=False, beta=False)
    assert list(layer.parameters()) == []


def test_epsilon_stored():
    layer = LayerNormalization(4, epsilon=1e-3)
    assert layer.epsilon == 1e-3


def test_output_shape_preserved():
    torch.manual_seed(0)
    layer = LayerNormalization(6)
    x = torch.randn(2, 3, 6)
    y = layer(x)
    assert y.shape == x.shape


def test_output_zero_mean_unit_variance():
    torch.manual_seed(1)
    layer = LayerNormalization(16)
    x = torch.randn(4, 5, 16) * 3.0 + 2.0
    y = layer(x)
    mean = y.mean(dim=-1)
    var = ((y - y.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_known_values():
    layer = LayerNormalization(4, epsilon=0.0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = layer(x)
    # mean 2.5, variance 1.25
    expected = (x - 2.5) / (1.25 ** 0.5)
    assert torch.allclose(y, expected, atol=1e-5)


def test_matches_torch_layer_norm():
    torch.manual_seed(2)
    x = torch.randn(3, 4, 10)
    layer = LayerNormalization(10, epsilon=1e-5)
    ref = nn.LayerNorm(10, eps=1e-5)
    assert torch.allclose(layer(x), ref(x), atol=1e-4)


def test_gamma_and_beta_applied():
    layer = LayerNormalization(4, epsilon=0.0)
    with torch.no_grad():
        layer.gamma.fill_(2.0)
        layer.beta.fill_(3.0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = (x - 2.5) / (1.25 ** 0.5) * 2.0 + 3.0
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_without_affine_params_is_pure_normalization():
    layer = LayerNormalization(4, gamma=False, beta=False, epsilon=0.0)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = (x - 2.5) / (1.25 ** 0.5)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_constant_input_gives_beta():
    layer = LayerNormalization(5)
    with torch.no_grad():
        layer.beta.copy_(torch.arange(5, dtype=torch.float32))
    x = torch.full((2, 5), 7.0)
    y = layer(x)
    assert torch.allclose(y, layer.beta.data.expand(2, 5), atol=1e-6)


def test_shift_invariance():
    torch.manual_seed(3)
    layer = LayerNormalization(8, epsilon=1e-10)
    x = torch.randn(3, 8)
    assert torch.allclose(layer(x), layer(x + 100.0), atol=1e-3)


def test_backward_produces_gradients():
    torch.manual_seed(4)
    layer = LayerNormalization(6)
    x = torch.randn(2, 6, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None
    assert torch.allclose(layer.beta.grad, torch.full((6,), 2.0))


def test_reset_parameters_restores_defaults():
    layer = LayerNormalization(3)
    with torch.no_grad():
        layer.gamma.fill_(5.0)
        layer.beta.fill_(-1.0)
    layer.reset_parameters()
    assert torch.equal(layer.gamma.data, torch.ones(3))
    assert torch.equal(layer.beta.data, torch.zeros(3))


def test_extra_repr_and_repr():
    layer = LayerNormalization(4, gamma=True, beta=False, epsilon=1e-5)
    text = layer.extra_repr()
    assert 'gamma=True' in text
    assert 'beta=False' in text
    assert '1e-05' in text
    assert 'LayerNormalization' in repr(layer)
