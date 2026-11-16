import pytest
import torch

import torch_layer_normalization
from torch_layer_normalization import LayerNormalization


def _manual_norm(x, eps):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def test_version_exposed():
    assert isinstance(torch_layer_normalization.__version__, str)


def test_default_parameters_initialized():
    layer = LayerNormalization(5)
    assert layer.gamma is not None
    assert layer.beta is not None
    assert layer.gamma.shape == (5,)
    assert layer.beta.shape == (5,)
    assert torch.all(layer.gamma == 1)
    assert torch.all(layer.beta == 0)


def test_parameters_are_registered():
    layer = LayerNormalization(4)
    names = {n for n, _ in layer.named_parameters()}
    assert names == {'gamma', 'beta'}
    for p in layer.parameters():
        assert p.requires_grad


def test_no_gamma_no_beta_has_no_parameters():
    layer = LayerNormalization(4, gamma=False, beta=False)
    assert layer.gamma is None
    assert layer.beta is None
    assert len(list(layer.parameters())) == 0


def test_only_gamma():
    layer = LayerNormalization(4, gamma=True, beta=False)
    assert layer.gamma is not None
    assert layer.beta is None
    assert [n for n, _ in layer.named_parameters()] == ['gamma']


def test_only_beta():
    layer = LayerNormalization(4, gamma=False, beta=True)
    assert layer.gamma is None
    assert layer.beta is not None
    assert [n for n, _ in layer.named_parameters()] == ['beta']


def test_tuple_normal_shape_uses_last_dimension():
    layer = LayerNormalization((3, 6))
    assert layer.gamma.shape == (6,)
    assert layer.beta.shape == (6,)


def test_output_shape_preserved():
    layer = LayerNormalization(8)
    x = torch.randn(2, 3, 8)
    y = layer(x)
    assert y.shape == x.shape


def test_output_has_zero_mean_and_unit_variance():
    torch.manual_seed(0)
    layer = LayerNormalization(16)
    x = torch.randn(4, 5, 16) * 3.0 + 7.0
    y = layer(x)
    mean = y.mean(dim=-1)
    var = ((y - y.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_matches_manual_computation_without_affine():
    torch.manual_seed(1)
    layer = LayerNormalization(10, gamma=False, beta=False, epsilon=1e-5)
    x = torch.randn(3, 10)
    expected = _manual_norm(x, 1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_matches_torch_functional_layer_norm():
    torch.manual_seed(2)
    eps = 1e-5
    layer = LayerNormalization(12, epsilon=eps)
    x = torch.randn(2, 7, 12)
    expected = torch.nn.functional.layer_norm(x, (12,), eps=eps)
    assert torch.allclose(layer(x), expected, atol=1e-4)


def test_gamma_and_beta_applied():
    torch.manual_seed(3)
    layer = LayerNormalization(6, epsilon=1e-5)
    with torch.no_grad():
        layer.gamma.copy_(torch.arange(1, 7, dtype=torch.float32))
        layer.beta.copy_(torch.arange(6, dtype=torch.float32) * 0.5)
    x = torch.randn(4, 6)
    expected = _manual_norm(x, 1e-5) * layer.gamma.detach() + layer.beta.detach()
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_constant_input_gives_beta():
    layer = LayerNormalization(5)
    x = torch.full((2, 5), 3.5)
    y = layer(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-4)

    with torch.no_grad():
        layer.beta.fill_(2.0)
    y = layer(x)
    assert torch.allclose(y, torch.full_like(y, 2.0), atol=1e-4)


def test_reset_parameters_restores_defaults():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.gamma.fill_(5.0)
        layer.beta.fill_(-3.0)
    layer.reset_parameters()
    assert torch.all(layer.gamma == 1)
    assert torch.all(layer.beta == 0)


def test_backward_produces_gradients():
    torch.manual_seed(4)
    layer = LayerNormalization(6)
    x = torch.randn(3, 6, requires_grad=True)
    y = layer(x)
    (y ** 2 * torch.arange(6, dtype=torch.float32)).sum().backward()
    assert x.grad is not None
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None
    assert layer.gamma.grad.shape == (6,)
    assert layer.beta.grad.shape == (6,)


def test_epsilon_stored_and_in_extra_repr():
    layer = LayerNormalization(3, gamma=True, beta=False, epsilon=1e-3)
    assert layer.epsilon == 1e-3
    text = repr(layer)
    assert 'LayerNormalization' in text
    assert 'epsilon' in text
    assert 'gamma=True' in text
    assert 'beta=False' in text


def test_larger_epsilon_shrinks_output():
    torch.manual_seed(5)
    x = torch.randn(4, 8)
    small = LayerNormalization(8, gamma=False, beta=False, epsilon=1e-10)(x)
    large = LayerNormalization(8, gamma=False, beta=False, epsilon=10.0)(x)
    assert large.abs().max() < small.abs().max()
