import pytest
import torch

from torch_layer_normalization import LayerNormalization


def test_output_shape_2d():
    layer = LayerNormalization(8)
    x = torch.randn(4, 8)
    assert layer(x).shape == x.shape


def test_output_shape_3d():
    layer = LayerNormalization(6)
    x = torch.randn(2, 5, 6)
    assert layer(x).shape == x.shape


def test_default_parameters_initialization():
    layer = LayerNormalization(5)
    assert layer.gamma is not None
    assert layer.beta is not None
    assert layer.gamma.shape == (5,)
    assert layer.beta.shape == (5,)
    assert torch.all(layer.gamma == 1)
    assert torch.all(layer.beta == 0)


def test_parameters_are_registered():
    layer = LayerNormalization(5)
    params = list(layer.parameters())
    assert len(params) == 2
    assert all(p.requires_grad for p in params)


def test_no_gamma():
    layer = LayerNormalization(5, gamma=False)
    assert layer.gamma is None
    assert layer.beta is not None
    assert len(list(layer.parameters())) == 1
    x = torch.randn(3, 5)
    assert layer(x).shape == x.shape


def test_no_beta():
    layer = LayerNormalization(5, beta=False)
    assert layer.beta is None
    assert layer.gamma is not None
    assert len(list(layer.parameters())) == 1
    x = torch.randn(3, 5)
    assert layer(x).shape == x.shape


def test_no_gamma_no_beta():
    layer = LayerNormalization(5, gamma=False, beta=False)
    assert len(list(layer.parameters())) == 0
    x = torch.randn(3, 5)
    assert layer(x).shape == x.shape


def test_output_is_normalized():
    torch.manual_seed(0)
    layer = LayerNormalization(32)
    x = torch.randn(10, 32) * 5 + 3
    y = layer(x)
    mean = y.mean(dim=-1)
    var = ((y - mean.unsqueeze(-1)) ** 2).mean(dim=-1)
    assert torch.allclose(mean, torch.zeros(10), atol=1e-4)
    assert torch.allclose(var, torch.ones(10), atol=1e-3)


def test_matches_torch_functional_layer_norm():
    torch.manual_seed(1)
    eps = 1e-5
    layer = LayerNormalization(7, epsilon=eps)
    x = torch.randn(3, 4, 7)
    expected = torch.nn.functional.layer_norm(x, (7,), eps=eps)
    assert torch.allclose(layer(x), expected, atol=1e-4)


def test_gamma_and_beta_applied():
    torch.manual_seed(2)
    layer = LayerNormalization(6)
    with torch.no_grad():
        layer.gamma.fill_(2.0)
        layer.beta.fill_(0.5)
    x = torch.randn(4, 6)
    base = LayerNormalization(6, gamma=False, beta=False)(x)
    assert torch.allclose(layer(x), base * 2.0 + 0.5, atol=1e-5)


def test_constant_input_returns_beta():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.beta.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.full((3, 4), 7.0)
    y = layer(x)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(3, 4)
    assert torch.allclose(y, expected, atol=1e-5)
    assert torch.isfinite(y).all()


def test_shift_and_scale_invariance():
    torch.manual_seed(3)
    layer = LayerNormalization(16, epsilon=1e-10)
    x = torch.randn(5, 16)
    y1 = layer(x)
    y2 = layer(x * 10 + 100)
    assert torch.allclose(y1, y2, atol=1e-3)


def test_backward_produces_gradients():
    layer = LayerNormalization(5)
    x = torch.randn(3, 5, requires_grad=True)
    layer(x).pow(2).sum().backward()
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None
    assert x.grad is not None
    assert layer.gamma.grad.shape == (5,)
    assert layer.beta.grad.shape == (5,)


def test_beta_gradient_equals_number_of_rows():
    layer = LayerNormalization(4)
    x = torch.randn(6, 4)
    layer(x).sum().backward()
    assert torch.allclose(layer.beta.grad, torch.full((4,), 6.0))


def test_reset_parameters_restores_defaults():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.gamma.fill_(3.0)
        layer.beta.fill_(-2.0)
    layer.reset_parameters()
    assert torch.all(layer.gamma == 1)
    assert torch.all(layer.beta == 0)


def test_larger_epsilon_shrinks_output():
    torch.manual_seed(4)
    x = torch.randn(4, 8)
    small = LayerNormalization(8, epsilon=1e-10)(x)
    large = LayerNormalization(8, epsilon=10.0)(x)
    assert large.abs().max() < small.abs().max()
