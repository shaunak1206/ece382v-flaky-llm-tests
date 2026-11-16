import pytest
import torch

from torch_layer_normalization import LayerNormalization


def test_init_int_shape_parameters():
    layer = LayerNormalization(8)
    assert tuple(layer.gamma.shape) == (8,)
    assert tuple(layer.beta.shape) == (8,)
    assert isinstance(layer.gamma, torch.nn.Parameter)
    assert isinstance(layer.beta, torch.nn.Parameter)


def test_init_tuple_shape_uses_last_dimension():
    layer = LayerNormalization((4, 6))
    assert tuple(layer.gamma.shape) == (6,)
    assert tuple(layer.beta.shape) == (6,)


def test_default_parameter_values():
    layer = LayerNormalization(5)
    assert torch.all(layer.gamma == 1)
    assert torch.all(layer.beta == 0)


def test_no_gamma():
    layer = LayerNormalization(5, gamma=False)
    assert layer.gamma is None
    assert layer.beta is not None
    names = [n for n, _ in layer.named_parameters()]
    assert names == ["beta"]


def test_no_beta():
    layer = LayerNormalization(5, beta=False)
    assert layer.beta is None
    assert layer.gamma is not None
    names = [n for n, _ in layer.named_parameters()]
    assert names == ["gamma"]


def test_no_gamma_no_beta_has_no_parameters():
    layer = LayerNormalization(5, gamma=False, beta=False)
    assert len(list(layer.parameters())) == 0


def test_output_shape_preserved():
    layer = LayerNormalization(7)
    x = torch.randn(3, 4, 7)
    y = layer(x)
    assert y.shape == x.shape


def test_output_zero_mean_unit_variance():
    torch.manual_seed(0)
    layer = LayerNormalization(16)
    x = torch.randn(10, 16) * 5 + 3
    y = layer(x)
    mean = y.mean(dim=-1)
    var = ((y - y.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_matches_torch_functional_layer_norm():
    torch.manual_seed(1)
    layer = LayerNormalization(12, epsilon=1e-5)
    x = torch.randn(4, 5, 12)
    expected = torch.nn.functional.layer_norm(x, (12,), eps=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-4)


def test_gamma_and_beta_applied():
    torch.manual_seed(2)
    layer = LayerNormalization(6)
    with torch.no_grad():
        layer.gamma.fill_(2.0)
        layer.beta.fill_(0.5)
    base = LayerNormalization(6, gamma=False, beta=False)
    x = torch.randn(3, 6)
    expected = base(x) * 2.0 + 0.5
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_per_feature_gamma_beta():
    layer = LayerNormalization(3)
    with torch.no_grad():
        layer.gamma.copy_(torch.tensor([1.0, 2.0, 3.0]))
        layer.beta.copy_(torch.tensor([0.0, 1.0, -1.0]))
    base = LayerNormalization(3, gamma=False, beta=False)
    x = torch.tensor([[1.0, 2.0, 4.0], [0.0, -1.0, 5.0]])
    expected = base(x) * torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([0.0, 1.0, -1.0])
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_constant_input_gives_beta():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.beta.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.full((2, 4), 7.0)
    y = layer(x)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(2, 4)
    assert torch.allclose(y, expected, atol=1e-5)
    assert torch.isfinite(y).all()


def test_epsilon_stored_and_affects_output():
    small = LayerNormalization(4, gamma=False, beta=False, epsilon=1e-10)
    large = LayerNormalization(4, gamma=False, beta=False, epsilon=10.0)
    assert small.epsilon == 1e-10
    assert large.epsilon == 10.0
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y_small = small(x)
    y_large = large(x)
    assert y_large.abs().max() < y_small.abs().max()


def test_backward_produces_gradients():
    torch.manual_seed(3)
    layer = LayerNormalization(5)
    x = torch.randn(4, 5, requires_grad=True)
    weights = torch.randn(4, 5)
    loss = (layer(x) * weights).sum()
    loss.backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert layer.gamma.grad is not None and layer.gamma.grad.shape == (5,)
    assert layer.beta.grad is not None and layer.beta.grad.shape == (5,)
    assert torch.isfinite(x.grad).all()


def test_input_not_modified():
    torch.manual_seed(4)
    layer = LayerNormalization(6)
    x = torch.randn(3, 6)
    x_copy = x.clone()
    layer(x)
    assert torch.equal(x, x_copy)


def test_reset_parameters_restores_defaults():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.gamma.fill_(5.0)
        layer.beta.fill_(-3.0)
    layer.reset_parameters()
    assert torch.all(layer.gamma == 1)
    assert torch.all(layer.beta == 0)


def test_extra_repr_contains_configuration():
    layer = LayerNormalization(4, gamma=True, beta=False, epsilon=1e-5)
    text = repr(layer)
    assert "LayerNormalization" in text
    assert "gamma=True" in text
    assert "beta=False" in text
    assert "epsilon" in text
