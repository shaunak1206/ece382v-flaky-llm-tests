import pytest
import torch

from torch_layer_normalization import LayerNormalization


def reference(x, gamma=None, beta=None, eps=1e-10):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    y = (x - mean) / (var + eps).sqrt()
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y = y + beta
    return y


def test_default_parameters_initialized():
    layer = LayerNormalization(5)
    assert torch.equal(layer.gamma.data, torch.ones(5))
    assert torch.equal(layer.beta.data, torch.zeros(5))
    assert isinstance(layer.gamma, torch.nn.Parameter)
    assert isinstance(layer.beta, torch.nn.Parameter)


def test_int_normal_shape():
    layer = LayerNormalization(7)
    assert layer.normal_shape == torch.Size([7])
    assert layer.gamma.shape == (7,)
    assert layer.beta.shape == (7,)


def test_tuple_normal_shape_uses_last_dim():
    layer = LayerNormalization((3, 4))
    assert layer.normal_shape == torch.Size([4])
    assert layer.gamma.shape == (4,)


def test_output_shape_matches_input_2d():
    layer = LayerNormalization(6)
    x = torch.randn(4, 6)
    assert layer(x).shape == x.shape


def test_output_shape_matches_input_3d():
    layer = LayerNormalization(6)
    x = torch.randn(2, 3, 6)
    assert layer(x).shape == x.shape


def test_output_zero_mean_unit_variance():
    torch.manual_seed(0)
    layer = LayerNormalization(16)
    x = torch.randn(8, 16) * 5 + 3
    y = layer(x)
    mean = y.mean(dim=-1)
    var = ((y - mean.unsqueeze(-1)) ** 2).mean(dim=-1)
    assert torch.allclose(mean, torch.zeros(8), atol=1e-5)
    assert torch.allclose(var, torch.ones(8), atol=1e-3)


def test_matches_reference_default():
    torch.manual_seed(1)
    layer = LayerNormalization(10)
    x = torch.randn(3, 4, 10)
    expected = reference(x, layer.gamma.data, layer.beta.data)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_custom_gamma_and_beta():
    torch.manual_seed(2)
    layer = LayerNormalization(4)
    layer.gamma.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    layer.beta.data = torch.tensor([0.5, -0.5, 1.0, -1.0])
    x = torch.randn(5, 4)
    expected = reference(x, layer.gamma.data, layer.beta.data)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_no_gamma():
    layer = LayerNormalization(4, gamma=False)
    assert layer.gamma is None
    names = [n for n, _ in layer.named_parameters()]
    assert 'gamma' not in names
    assert 'beta' in names
    x = torch.randn(3, 4)
    expected = reference(x, None, layer.beta.data)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_no_beta():
    layer = LayerNormalization(4, beta=False)
    assert layer.beta is None
    names = [n for n, _ in layer.named_parameters()]
    assert 'beta' not in names
    assert 'gamma' in names
    x = torch.randn(3, 4)
    expected = reference(x, layer.gamma.data, None)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_no_gamma_no_beta_has_no_parameters():
    layer = LayerNormalization(4, gamma=False, beta=False)
    assert len(list(layer.parameters())) == 0
    x = torch.randn(3, 4)
    expected = reference(x)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_constant_input_gives_beta():
    layer = LayerNormalization(5)
    layer.beta.data = torch.arange(5, dtype=torch.float32)
    x = torch.full((2, 5), 3.0)
    y = layer(x)
    assert torch.isfinite(y).all()
    assert torch.allclose(y, layer.beta.data.expand(2, 5), atol=1e-5)


def test_epsilon_effect():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    small = LayerNormalization(4, epsilon=1e-10)
    large = LayerNormalization(4, epsilon=10.0)
    assert large.epsilon == 10.0
    assert torch.allclose(small(x), reference(x, eps=1e-10), atol=1e-5)
    assert torch.allclose(large(x), reference(x, eps=10.0), atol=1e-5)
    assert large(x).abs().max() < small(x).abs().max()


def test_shift_and_scale_invariance():
    torch.manual_seed(3)
    layer = LayerNormalization(8)
    x = torch.randn(4, 8)
    y1 = layer(x)
    y2 = layer(x * 10 + 100)
    assert torch.allclose(y1, y2, atol=1e-3)


def test_backward_produces_gradients():
    torch.manual_seed(4)
    layer = LayerNormalization(6)
    x = torch.randn(3, 6, requires_grad=True)
    layer(x).pow(2).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None
    assert layer.gamma.grad.shape == (6,)


def test_reset_parameters():
    layer = LayerNormalization(3)
    layer.gamma.data.fill_(5)
    layer.beta.data.fill_(7)
    layer.reset_parameters()
    assert torch.equal(layer.gamma.data, torch.ones(3))
    assert torch.equal(layer.beta.data, torch.zeros(3))


def test_extra_repr_contains_info():
    layer = LayerNormalization(4, gamma=False)
    text = repr(layer)
    assert 'LayerNormalization' in text
    assert 'normal_shape' in text
