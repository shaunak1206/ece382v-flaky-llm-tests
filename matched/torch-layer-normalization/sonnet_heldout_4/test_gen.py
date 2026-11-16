import pytest
import torch
import torch.nn.functional as F

import torch_layer_normalization
from torch_layer_normalization import LayerNormalization


def test_version_defined():
    assert isinstance(torch_layer_normalization.__version__, str)
    assert torch_layer_normalization.__version__


def test_init_with_int_shape():
    layer = LayerNormalization(5)
    assert tuple(layer.gamma.shape) == (5,)
    assert tuple(layer.beta.shape) == (5,)


def test_init_with_tuple_uses_last_dimension():
    layer = LayerNormalization((3, 7))
    assert tuple(layer.gamma.shape) == (7,)
    assert tuple(layer.beta.shape) == (7,)


def test_default_parameter_values():
    layer = LayerNormalization(4)
    assert torch.allclose(layer.gamma, torch.ones(4))
    assert torch.allclose(layer.beta, torch.zeros(4))


def test_parameters_registered_and_trainable():
    layer = LayerNormalization(4)
    params = list(layer.parameters())
    assert len(params) == 2
    assert all(p.requires_grad for p in params)


def test_no_gamma():
    layer = LayerNormalization(4, gamma=False)
    assert layer.gamma is None
    names = [n for n, _ in layer.named_parameters()]
    assert names == ['beta']


def test_no_beta():
    layer = LayerNormalization(4, beta=False)
    assert layer.beta is None
    names = [n for n, _ in layer.named_parameters()]
    assert names == ['gamma']


def test_no_gamma_no_beta_has_no_parameters():
    layer = LayerNormalization(4, gamma=False, beta=False)
    assert len(list(layer.parameters())) == 0
    x = torch.randn(2, 4)
    y = layer(x)
    assert y.shape == x.shape


def test_epsilon_stored():
    layer = LayerNormalization(4, epsilon=1e-3)
    assert layer.epsilon == 1e-3


def test_output_shape_preserved_2d_and_3d():
    layer = LayerNormalization(6)
    for shape in [(3, 6), (2, 5, 6), (2, 3, 4, 6)]:
        x = torch.randn(*shape)
        assert layer(x).shape == x.shape


def test_output_is_normalized_with_default_params():
    torch.manual_seed(0)
    layer = LayerNormalization(16)
    x = torch.randn(4, 10, 16) * 3 + 5
    y = layer(x)
    mean = y.mean(dim=-1)
    var = ((y - mean.unsqueeze(-1)) ** 2).mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_matches_torch_layer_norm():
    torch.manual_seed(1)
    eps = 1e-5
    layer = LayerNormalization(8, epsilon=eps)
    x = torch.randn(3, 5, 8)
    expected = F.layer_norm(x, (8,), eps=eps)
    assert torch.allclose(layer(x), expected, atol=1e-4)


def test_gamma_and_beta_are_applied():
    torch.manual_seed(2)
    eps = 1e-5
    layer = LayerNormalization(6, epsilon=eps)
    gamma = torch.randn(6)
    beta = torch.randn(6)
    with torch.no_grad():
        layer.gamma.copy_(gamma)
        layer.beta.copy_(beta)
    x = torch.randn(4, 6)
    expected = F.layer_norm(x, (6,), weight=gamma, bias=beta, eps=eps)
    assert torch.allclose(layer(x), expected, atol=1e-4)


def test_constant_input_gives_beta():
    layer = LayerNormalization(5)
    with torch.no_grad():
        layer.beta.copy_(torch.arange(5, dtype=torch.float32))
    x = torch.full((2, 5), 3.0)
    y = layer(x)
    expected = torch.arange(5, dtype=torch.float32).expand(2, 5)
    assert torch.allclose(y, expected, atol=1e-4)


def test_backward_produces_gradients():
    torch.manual_seed(3)
    layer = LayerNormalization(5)
    x = torch.randn(3, 5, requires_grad=True)
    layer(x).sum().backward()
    assert layer.beta.grad is not None
    assert torch.allclose(layer.beta.grad, torch.full((5,), 3.0))
    assert layer.gamma.grad is not None
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_does_not_modify_input():
    layer = LayerNormalization(4)
    x = torch.randn(3, 4)
    x_copy = x.clone()
    layer(x)
    assert torch.equal(x, x_copy)
