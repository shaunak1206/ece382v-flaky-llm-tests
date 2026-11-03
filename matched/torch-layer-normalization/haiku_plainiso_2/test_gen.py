"""Unit tests for LayerNormalization in torch_layer_normalization."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_layer_normalization import LayerNormalization


def _seed():
    torch.manual_seed(1234)


def _manual_normalize(x, eps):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def _manual_forward(x, gamma, beta, eps):
    y = _manual_normalize(x, eps)
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y = y + beta
    return y


def test_int_normal_shape_sets_last_dim():
    layer = LayerNormalization(4)
    assert layer.normal_shape == torch.Size([4])
    assert layer.gamma.shape == (4,)
    assert layer.beta.shape == (4,)


def test_tuple_normal_shape_uses_last_dim():
    layer = LayerNormalization((2, 3, 5))
    assert layer.normal_shape == torch.Size([5])
    assert layer.gamma.shape == (5,)
    assert layer.beta.shape == (5,)


def test_list_normal_shape_uses_last_dim():
    layer = LayerNormalization([7, 8])
    assert layer.normal_shape == torch.Size([8])
    assert layer.gamma.shape == (8,)


def test_default_parameters_are_ones_and_zeros():
    layer = LayerNormalization(6)
    assert isinstance(layer.gamma, nn.Parameter)
    assert isinstance(layer.beta, nn.Parameter)
    assert layer.gamma.requires_grad
    assert layer.beta.requires_grad
    assert torch.all(layer.gamma == 1).item()
    assert torch.all(layer.beta == 0).item()


def test_gamma_false_registers_none_and_keeps_beta():
    layer = LayerNormalization(4, gamma=False)
    assert layer.gamma is None
    names = dict(layer.named_parameters())
    assert 'gamma' not in names
    assert 'beta' in names


def test_beta_false_registers_none_and_keeps_gamma():
    layer = LayerNormalization(4, beta=False)
    assert layer.beta is None
    names = dict(layer.named_parameters())
    assert 'beta' not in names
    assert 'gamma' in names


def test_no_affine_parameters_when_both_disabled():
    _seed()
    layer = LayerNormalization(4, gamma=False, beta=False)
    assert list(layer.parameters()) == []
    x = torch.randn(3, 4)
    assert torch.allclose(layer(x), _manual_normalize(x, layer.epsilon), atol=1e-6)


def test_reset_parameters_restores_defaults():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.gamma.fill_(-3.0)
        layer.beta.fill_(2.5)
    layer.reset_parameters()
    assert torch.all(layer.gamma == 1).item()
    assert torch.all(layer.beta == 0).item()


def test_forward_output_has_zero_mean_and_unit_variance():
    _seed()
    layer = LayerNormalization(16)
    x = torch.randn(8, 16) * 5.0 + 3.0
    y = layer(x)
    mean = y.mean(dim=-1)
    var = ((y - y.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_forward_matches_manual_formula_with_random_affine():
    _seed()
    layer = LayerNormalization(10)
    with torch.no_grad():
        layer.gamma.copy_(torch.randn(10))
        layer.beta.copy_(torch.randn(10))
    x = torch.randn(4, 3, 10)
    expected = _manual_forward(x, layer.gamma, layer.beta, layer.epsilon)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_forward_matches_functional_layer_norm():
    _seed()
    layer = LayerNormalization(12, epsilon=1e-5)
    with torch.no_grad():
        layer.gamma.copy_(torch.rand(12) + 0.5)
        layer.beta.copy_(torch.randn(12))
    x = torch.randn(5, 12)
    expected = F.layer_norm(x, (12,), layer.gamma, layer.beta, 1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_affine_parameters_scale_and_shift_output():
    _seed()
    x = torch.randn(3, 6)
    base = LayerNormalization(6)
    scaled = LayerNormalization(6)
    with torch.no_grad():
        scaled.gamma.fill_(2.0)
        scaled.beta.fill_(3.0)
    assert torch.allclose(scaled(x), 2.0 * base(x) + 3.0, atol=1e-5)


def test_forward_preserves_input_shape():
    _seed()
    layer = LayerNormalization(4)
    for shape in [(5, 4), (2, 3, 4), (2, 1, 3, 4)]:
        x = torch.randn(*shape)
        assert layer(x).shape == x.shape


def test_rows_are_normalized_independently():
    _seed()
    layer = LayerNormalization(8)
    x = torch.randn(4, 8)
    y_before = layer(x)
    x2 = x.clone()
    x2[1] = x2[1] * 100.0 + 7.0
    y_after = layer(x2)
    assert torch.allclose(y_before[0], y_after[0], atol=1e-6)
    assert torch.allclose(y_before[2:], y_after[2:], atol=1e-6)


def test_output_invariant_to_row_scale_and_shift():
    _seed()
    layer = LayerNormalization(8)
    x = torch.randn(3, 8)
    y = layer(x)
    assert torch.allclose(layer(x * 10.0), y, atol=1e-4)
    assert torch.allclose(layer(x + 5.0), y, atol=1e-4)


def test_constant_row_maps_to_beta():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.beta.fill_(0.5)
    x = torch.full((2, 4), 3.0)
    assert torch.allclose(layer(x), torch.full((2, 4), 0.5), atol=1e-4)


def test_custom_epsilon_is_stored_and_damps_output():
    x = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    small = LayerNormalization(4, epsilon=1e-10)
    large = LayerNormalization(4, epsilon=1.0)
    assert large.epsilon == 1.0
    y_small = small(x)
    y_large = large(x)
    assert torch.allclose(y_large, _manual_normalize(x, 1.0), atol=1e-6)
    assert y_large.abs().max() < y_small.abs().max()


def test_repr_reports_configuration():
    layer = LayerNormalization(4, gamma=False, epsilon=1e-5)
    text = repr(layer)
    assert 'LayerNormalization' in text
    assert 'normal_shape=torch.Size([4])' in text
    assert 'gamma=False' in text
    assert 'beta=True' in text
    assert 'epsilon=1e-05' in text


def test_gradients_reach_gamma_and_beta():
    _seed()
    layer = LayerNormalization(5)
    x = torch.randn(4, 5)
    with torch.no_grad():
        y_hat = layer(x)  # gamma=1 and beta=0 at construction, so this is the normalized input
    out = layer(x)
    out.sum().backward()
    assert layer.beta.grad is not None
    assert layer.gamma.grad is not None
    assert torch.allclose(layer.beta.grad, torch.full((5,), 4.0), atol=1e-5)
    assert torch.allclose(layer.gamma.grad, y_hat.sum(dim=0), atol=1e-4)


def test_gradcheck_passes_in_double_precision():
    torch.manual_seed(0)
    layer = LayerNormalization(4).double()
    with torch.no_grad():
        layer.gamma.copy_(torch.rand(4, dtype=torch.float64) + 0.5)
        layer.beta.copy_(torch.randn(4, dtype=torch.float64))
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: layer(inp), (x,))


def test_forward_does_not_modify_input():
    _seed()
    layer = LayerNormalization(6)
    x = torch.randn(3, 6)
    original = x.clone()
    layer(x)
    assert torch.equal(x, original)


def test_mismatched_last_dimension_raises():
    layer = LayerNormalization(4)
    x = torch.randn(2, 5)
    with pytest.raises(RuntimeError):
        layer(x)
