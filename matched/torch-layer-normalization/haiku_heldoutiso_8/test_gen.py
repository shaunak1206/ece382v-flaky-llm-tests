import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_layer_normalization import LayerNormalization


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


def _reference_layer_norm(x, shape, weight=None, bias=None, eps=1e-5):
    dims = tuple(range(x.dim() - len(shape), x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
    out = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def test_import_exposes_class():
    assert isinstance(LayerNormalization, type)
    assert issubclass(LayerNormalization, nn.Module)


def test_is_nn_module_instance():
    layer = LayerNormalization(8)
    assert isinstance(layer, nn.Module)


def test_default_eps_is_small_positive():
    layer = LayerNormalization(8)
    assert layer.eps == pytest.approx(1e-5)


def test_custom_eps_is_stored():
    layer = LayerNormalization(8, eps=1e-3)
    assert layer.eps == pytest.approx(1e-3)


def test_int_normalized_shape_becomes_tuple():
    layer = LayerNormalization(8)
    assert tuple(layer.normalized_shape) == (8,)


def test_tuple_normalized_shape_is_kept():
    layer = LayerNormalization((4, 6))
    assert tuple(layer.normalized_shape) == (4, 6)


def test_forward_preserves_shape():
    layer = LayerNormalization(8)
    x = torch.randn(3, 5, 8)
    assert layer(x).shape == x.shape


def test_forward_zero_mean_over_last_dim():
    layer = LayerNormalization(16)
    x = torch.randn(4, 16) * 3.0 + 7.0
    out = layer(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(4), atol=1e-5)


def test_forward_unit_variance_over_last_dim():
    layer = LayerNormalization(16)
    x = torch.randn(4, 16) * 5.0 - 2.0
    out = layer(x)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(4), atol=1e-3)


def test_forward_matches_reference_computation():
    layer = LayerNormalization(10, eps=1e-5)
    x = torch.randn(2, 3, 10)
    expected = _reference_layer_norm(x, (10,), layer.weight, layer.bias, eps=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_forward_matches_functional_layer_norm():
    layer = LayerNormalization(10)
    x = torch.randn(5, 10)
    expected = F.layer_norm(x, (10,), layer.weight, layer.bias, layer.eps)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_multi_dimensional_normalized_shape():
    layer = LayerNormalization((4, 6))
    x = torch.randn(2, 4, 6) * 2.0 + 1.0
    out = layer(x)
    flat = out.reshape(2, -1)
    assert torch.allclose(flat.mean(dim=-1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(flat.var(dim=-1, unbiased=False), torch.ones(2), atol=1e-3)


def test_default_affine_parameters_are_identity_init():
    layer = LayerNormalization(8)
    assert torch.allclose(layer.weight, torch.ones(8))
    assert torch.allclose(layer.bias, torch.zeros(8))


def test_affine_parameters_are_applied():
    layer = LayerNormalization(8)
    with torch.no_grad():
        layer.weight.fill_(2.0)
        layer.bias.fill_(3.0)
    x = torch.randn(4, 8)
    base = LayerNormalization(8)
    expected = base(x) * 2.0 + 3.0
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_parameters_are_registered():
    layer = LayerNormalization(8)
    names = dict(layer.named_parameters())
    assert "weight" in names
    assert "bias" in names
    assert sum(p.numel() for p in layer.parameters()) == 16


def test_no_affine_has_no_parameters():
    layer = LayerNormalization(8, elementwise_affine=False)
    assert len(list(layer.parameters())) == 0


def test_no_affine_output_matches_reference():
    layer = LayerNormalization(8, elementwise_affine=False)
    x = torch.randn(3, 8)
    expected = _reference_layer_norm(x, (8,), eps=layer.eps)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_each_sample_normalized_independently():
    layer = LayerNormalization(8)
    x = torch.randn(6, 8)
    batched = layer(x)
    for i in range(x.shape[0]):
        assert torch.allclose(layer(x[i:i + 1])[0], batched[i], atol=1e-6)


def test_constant_input_is_finite():
    layer = LayerNormalization(8)
    x = torch.full((2, 8), 4.2)
    out = layer(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)


def test_backward_produces_gradients():
    layer = LayerNormalization(8)
    x = torch.randn(4, 8, requires_grad=True)
    loss = (layer(x) ** 2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_float64_input_preserves_dtype():
    layer = LayerNormalization(8).double()
    x = torch.randn(3, 8, dtype=torch.float64)
    out = layer(x)
    assert out.dtype == torch.float64
    expected = _reference_layer_norm(x, (8,), layer.weight, layer.bias, eps=layer.eps)
    assert torch.allclose(out, expected, atol=1e-10)


def test_mismatched_last_dim_raises():
    layer = LayerNormalization(8)
    x = torch.randn(3, 5)
    with pytest.raises(RuntimeError):
        layer(x)
