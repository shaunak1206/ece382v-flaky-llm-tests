import pytest
import torch
import torch.nn.functional as F

from torch_layer_normalization import LayerNormalization


def test_default_parameters_initialized():
    layer = LayerNormalization(5)
    assert layer.gamma is not None
    assert layer.beta is not None
    assert tuple(layer.gamma.shape) == (5,)
    assert tuple(layer.beta.shape) == (5,)
    assert torch.all(layer.gamma == 1)
    assert torch.all(layer.beta == 0)


def test_parameters_are_trainable():
    layer = LayerNormalization(4)
    params = list(layer.parameters())
    assert len(params) == 2
    assert all(p.requires_grad for p in params)


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
    assert layer(x).shape == x.shape


def test_output_shape_2d_and_3d():
    layer = LayerNormalization(6)
    assert layer(torch.randn(3, 6)).shape == (3, 6)
    assert layer(torch.randn(2, 5, 6)).shape == (2, 5, 6)


def test_output_zero_mean_unit_variance():
    torch.manual_seed(0)
    layer = LayerNormalization(16)
    x = torch.randn(8, 16) * 3 + 5
    y = layer(x)
    mean = y.mean(dim=-1)
    var = ((y - y.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
    assert torch.allclose(mean, torch.zeros(8), atol=1e-4)
    assert torch.allclose(var, torch.ones(8), atol=1e-3)


def test_matches_torch_functional_layer_norm():
    torch.manual_seed(1)
    layer = LayerNormalization(10, epsilon=1e-5)
    x = torch.randn(4, 7, 10)
    expected = F.layer_norm(x, (10,), eps=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-4)


def test_custom_gamma_and_beta_applied():
    layer = LayerNormalization(3)
    with torch.no_grad():
        layer.gamma.copy_(torch.tensor([1.0, 2.0, 3.0]))
        layer.beta.copy_(torch.tensor([0.5, -0.5, 1.0]))
    x = torch.randn(5, 3)
    base = LayerNormalization(3, gamma=False, beta=False)(x)
    expected = base * torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([0.5, -0.5, 1.0])
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_constant_input_gives_beta():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.beta.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.full((2, 4), 7.0)
    y = layer(x)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(2, 4)
    assert torch.all(torch.isfinite(y))
    assert torch.allclose(y, expected, atol=1e-4)


def test_normalizes_each_sample_independently():
    layer = LayerNormalization(5)
    x = torch.randn(4, 5)
    full = layer(x)
    for i in range(4):
        single = layer(x[i:i + 1])
        assert torch.allclose(full[i:i + 1], single, atol=1e-5)


def test_invariant_to_shift_and_scale_of_input():
    torch.manual_seed(2)
    layer = LayerNormalization(8, epsilon=1e-10)
    x = torch.randn(3, 8)
    y1 = layer(x)
    y2 = layer(x * 10 + 100)
    assert torch.allclose(y1, y2, atol=1e-3)


def test_gradients_flow_to_input_and_parameters():
    layer = LayerNormalization(6)
    x = torch.randn(4, 6, requires_grad=True)
    layer(x).pow(2).sum().backward()
    assert x.grad is not None
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None
    assert layer.gamma.grad.shape == (6,)
    assert layer.beta.grad.shape == (6,)


def test_reset_parameters_restores_defaults():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.gamma.fill_(3.0)
        layer.beta.fill_(2.0)
    layer.reset_parameters()
    assert torch.all(layer.gamma == 1)
    assert torch.all(layer.beta == 0)


def test_state_dict_contains_gamma_and_beta():
    layer = LayerNormalization(4)
    keys = set(layer.state_dict().keys())
    assert keys == {'gamma', 'beta'}


def test_single_element_tuple_normal_shape():
    layer = LayerNormalization((6,))
    assert tuple(layer.gamma.shape) == (6,)
    x = torch.randn(2, 6)
    assert layer(x).shape == (2, 6)
