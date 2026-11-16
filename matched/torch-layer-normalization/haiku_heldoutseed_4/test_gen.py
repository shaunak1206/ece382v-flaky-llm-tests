import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_layer_normalization import LayerNormalization


def _set_identity_affine(layer):
    """Set every 'weight' parameter to ones and every 'bias' parameter to zeros."""
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.fill_(1.0)
            else:
                param.zero_()


def _set_affine(layer, weight_value, bias_value):
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.fill_(weight_value)
            else:
                param.fill_(bias_value)


def test_layer_is_nn_module():
    layer = LayerNormalization(normalized_shape=4)
    assert isinstance(layer, nn.Module)


def test_default_construction_with_int_shape():
    layer = LayerNormalization(normalized_shape=8)
    out = layer(torch.randn(2, 8))
    assert out.shape == (2, 8)


def test_default_construction_with_tuple_shape():
    layer = LayerNormalization(normalized_shape=(3, 4))
    out = layer(torch.randn(2, 3, 4))
    assert out.shape == (2, 3, 4)


def test_forward_preserves_shape_for_3d_input():
    layer = LayerNormalization(normalized_shape=16)
    x = torch.randn(5, 7, 16)
    assert layer(x).shape == x.shape


def test_output_has_zero_mean_and_unit_variance_without_affine():
    torch.manual_seed(0)
    layer = LayerNormalization(normalized_shape=32, elementwise_affine=False)
    x = torch.randn(4, 10, 32) * 3.0 + 5.0
    out = layer(x)
    means = out.mean(dim=-1)
    variances = out.var(dim=-1, unbiased=False)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    assert torch.allclose(variances, torch.ones_like(variances), atol=1e-3)


def test_multi_dimensional_shape_normalizes_jointly():
    torch.manual_seed(1)
    layer = LayerNormalization(normalized_shape=(3, 4), elementwise_affine=False)
    x = torch.randn(2, 3, 4) * 2.0 + 1.0
    out = layer(x)
    flat = out.reshape(2, -1)
    assert torch.allclose(flat.mean(dim=-1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(flat.var(dim=-1, unbiased=False), torch.ones(2), atol=1e-3)


def test_constant_input_yields_zeros_due_to_eps():
    layer = LayerNormalization(normalized_shape=6, elementwise_affine=False)
    x = torch.full((3, 6), 7.5)
    out = layer(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_no_parameters_when_affine_disabled():
    layer = LayerNormalization(normalized_shape=10, elementwise_affine=False)
    assert len(list(layer.parameters())) == 0


def test_parameters_exist_when_affine_enabled():
    layer = LayerNormalization(normalized_shape=10, elementwise_affine=True)
    params = list(layer.parameters())
    assert len(params) == 2
    total = sum(p.numel() for p in params)
    assert total == 2 * 10


def test_identity_affine_matches_non_affine():
    torch.manual_seed(2)
    x = torch.randn(4, 5, 12)
    plain = LayerNormalization(normalized_shape=12, elementwise_affine=False)
    affine = LayerNormalization(normalized_shape=12, elementwise_affine=True)
    _set_identity_affine(affine)
    assert torch.allclose(plain(x), affine(x), atol=1e-6)


def test_affine_scale_and_shift_applied():
    torch.manual_seed(3)
    x = torch.randn(3, 9)
    plain = LayerNormalization(normalized_shape=9, elementwise_affine=False)
    affine = LayerNormalization(normalized_shape=9, elementwise_affine=True)
    _set_affine(affine, weight_value=2.0, bias_value=3.0)
    expected = 2.0 * plain(x) + 3.0
    assert torch.allclose(affine(x), expected, atol=1e-6)


def test_matches_functional_layer_norm_with_same_weights():
    torch.manual_seed(4)
    x = torch.randn(6, 5, 10)
    layer = LayerNormalization(normalized_shape=10, elementwise_affine=True)
    with torch.no_grad():
        for name, param in layer.named_parameters():
            param.copy_(torch.randn_like(param))
    weight = dict(layer.named_parameters())["weight"]
    bias = dict(layer.named_parameters())["bias"]
    expected = F.layer_norm(x, (10,), weight=weight, bias=bias, eps=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_rows_are_normalized_independently():
    torch.manual_seed(5)
    layer = LayerNormalization(normalized_shape=8, elementwise_affine=False)
    x = torch.randn(4, 8)
    out_before = layer(x)
    x_modified = x.clone()
    x_modified[0] = x_modified[0] * 100.0 + 50.0
    out_after = layer(x_modified)
    assert torch.allclose(out_before[1:], out_after[1:], atol=1e-6)


def test_scale_invariance_without_affine():
    torch.manual_seed(6)
    layer = LayerNormalization(normalized_shape=16, elementwise_affine=False)
    x = torch.randn(3, 16)
    assert torch.allclose(layer(x), layer(x * 10.0), atol=1e-4)


def test_shift_invariance_without_affine():
    torch.manual_seed(7)
    layer = LayerNormalization(normalized_shape=16, elementwise_affine=False)
    x = torch.randn(3, 16)
    assert torch.allclose(layer(x), layer(x + 42.0), atol=1e-4)


def test_gradients_flow_to_input_and_parameters():
    torch.manual_seed(8)
    layer = LayerNormalization(normalized_shape=6, elementwise_affine=True)
    x = torch.randn(2, 6, requires_grad=True)
    loss = (layer(x) ** 2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for param in layer.parameters():
        assert param.grad is not None


def test_double_precision_supported():
    torch.manual_seed(9)
    layer = LayerNormalization(normalized_shape=7, elementwise_affine=False).double()
    x = torch.randn(4, 7, dtype=torch.float64)
    out = layer(x)
    assert out.dtype == torch.float64
    assert torch.allclose(out.mean(dim=-1), torch.zeros(4, dtype=torch.float64), atol=1e-10)


def test_custom_eps_reduces_output_for_small_variance():
    x = torch.full((1, 4), 1.0) + torch.tensor([[0.0, 1e-3, -1e-3, 0.0]])
    small_eps = LayerNormalization(normalized_shape=4, eps=1e-12, elementwise_affine=False)
    large_eps = LayerNormalization(normalized_shape=4, eps=1.0, elementwise_affine=False)
    assert small_eps(x).abs().max() > large_eps(x).abs().max()


def test_mismatched_last_dimension_raises():
    layer = LayerNormalization(normalized_shape=8)
    x = torch.randn(2, 5)
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        layer(x)


def test_forward_is_deterministic():
    torch.manual_seed(10)
    layer = LayerNormalization(normalized_shape=12, elementwise_affine=True)
    x = torch.randn(3, 12)
    first = layer(x)
    second = layer(x)
    assert torch.equal(first, second)
