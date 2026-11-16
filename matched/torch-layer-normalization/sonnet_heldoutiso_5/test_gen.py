import pytest
import torch

import torch_layer_normalization
from torch_layer_normalization import LayerNormalization


def reference(x, gamma=None, beta=None, eps=1e-10):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    y = (x - mean) / torch.sqrt(var + eps)
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y = y + beta
    return y


def test_version_exposed():
    assert isinstance(torch_layer_normalization.__version__, str)


def test_is_torch_module():
    assert isinstance(LayerNormalization(4), torch.nn.Module)


def test_default_parameters_initialized():
    layer = LayerNormalization(5)
    assert layer.gamma is not None
    assert layer.beta is not None
    assert tuple(layer.gamma.shape) == (5,)
    assert tuple(layer.beta.shape) == (5,)
    assert torch.all(layer.gamma == 1)
    assert torch.all(layer.beta == 0)


def test_parameters_registered():
    layer = LayerNormalization(5)
    names = {n for n, _ in layer.named_parameters()}
    assert names == {'gamma', 'beta'}


def test_no_gamma():
    layer = LayerNormalization(5, gamma=False)
    assert layer.gamma is None
    names = {n for n, _ in layer.named_parameters()}
    assert names == {'beta'}


def test_no_beta():
    layer = LayerNormalization(5, beta=False)
    assert layer.beta is None
    names = {n for n, _ in layer.named_parameters()}
    assert names == {'gamma'}


def test_no_gamma_no_beta_has_no_parameters():
    layer = LayerNormalization(5, gamma=False, beta=False)
    assert list(layer.parameters()) == []


def test_tuple_normal_shape_uses_last_dim():
    layer = LayerNormalization((3, 6))
    assert tuple(layer.gamma.shape) == (6,)
    assert tuple(layer.beta.shape) == (6,)
    out = layer(torch.randn(2, 3, 6))
    assert out.shape == (2, 3, 6)


def test_output_shape_preserved():
    layer = LayerNormalization(8)
    for shape in [(8,), (4, 8), (2, 3, 8), (2, 3, 4, 8)]:
        assert layer(torch.randn(*shape)).shape == shape


def test_output_zero_mean_unit_variance():
    torch.manual_seed(0)
    layer = LayerNormalization(64)
    x = torch.randn(10, 64) * 5 + 3
    y = layer(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(10), atol=1e-4)
    var = ((y - y.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
    assert torch.allclose(var, torch.ones(10), atol=1e-3)


def test_matches_reference_default_params():
    torch.manual_seed(1)
    layer = LayerNormalization(7, epsilon=1e-5)
    x = torch.randn(3, 4, 7)
    expected = reference(x, layer.gamma.detach(), layer.beta.detach(), eps=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_matches_torch_functional_layer_norm():
    torch.manual_seed(2)
    layer = LayerNormalization(9, epsilon=1e-5)
    x = torch.randn(5, 9)
    expected = torch.nn.functional.layer_norm(x, (9,), eps=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_custom_gamma_beta_applied():
    torch.manual_seed(3)
    layer = LayerNormalization(6, epsilon=1e-8)
    with torch.no_grad():
        layer.gamma.copy_(torch.arange(1, 7, dtype=torch.float32))
        layer.beta.copy_(torch.linspace(-1, 1, 6))
    x = torch.randn(4, 6)
    expected = reference(x, layer.gamma.detach(), layer.beta.detach(), eps=1e-8)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_beta_only_shifts_output():
    layer = LayerNormalization(4, gamma=False, epsilon=1e-8)
    with torch.no_grad():
        layer.beta.fill_(2.0)
    x = torch.randn(3, 4)
    expected = reference(x, None, torch.full((4,), 2.0), eps=1e-8)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_constant_input_gives_beta_without_nan():
    layer = LayerNormalization(4)
    x = torch.full((2, 4), 3.0)
    y = layer(x)
    assert not torch.isnan(y).any()
    assert torch.allclose(y, torch.zeros(2, 4), atol=1e-4)


def test_backward_produces_gradients():
    torch.manual_seed(4)
    layer = LayerNormalization(5)
    x = torch.randn(3, 5, requires_grad=True)
    layer(x).pow(2).sum().backward()
    assert x.grad is not None
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None
    assert layer.gamma.grad.shape == (5,)
    assert layer.beta.grad.shape == (5,)


def test_input_not_modified():
    layer = LayerNormalization(5)
    x = torch.randn(3, 5)
    x_copy = x.clone()
    layer(x)
    assert torch.equal(x, x_copy)


def test_reset_parameters_restores_defaults():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.gamma.fill_(5.0)
        layer.beta.fill_(-2.0)
    layer.reset_parameters()
    assert torch.all(layer.gamma == 1)
    assert torch.all(layer.beta == 0)


def test_scale_invariance_of_normalization():
    torch.manual_seed(5)
    layer = LayerNormalization(10, epsilon=1e-10)
    x = torch.randn(4, 10)
    assert torch.allclose(layer(x), layer(x * 10), atol=1e-3)
