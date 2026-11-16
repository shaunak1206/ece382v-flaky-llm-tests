import math

import pytest
import torch
import torch.nn as nn

import torch_layer_normalization
from torch_layer_normalization import LayerNormalization


# Helpers (plain functions, not tests). They do not touch any global state.

def _reference_layer_norm(x, n_dims, eps=1e-5):
    """Reference normalization over the last n_dims dimensions, without affine parameters."""
    dims = tuple(range(x.dim() - n_dims, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)  # biased variance
    return (x - mean) / torch.sqrt(var + eps)


def _make_layer(normalized_shape, dtype=torch.float64):
    layer = LayerNormalization(normalized_shape)
    return layer.to(dtype)


def test_package_exports_layer_and_version():
    assert torch_layer_normalization.__version__ == "0.11.0"
    assert issubclass(LayerNormalization, nn.Module)


def test_output_shape_matches_input_shape():
    torch.manual_seed(0)
    layer = _make_layer(8)
    x = torch.randn(4, 3, 8, dtype=torch.float64)
    out = layer(x)
    assert out.shape == x.shape


def test_matches_reference_computation_on_last_dim():
    torch.manual_seed(1)
    layer = _make_layer(6)
    x = torch.randn(5, 6, dtype=torch.float64) * 3.0 + 2.0
    out = layer(x)
    expected = _reference_layer_norm(x, n_dims=1)
    # Default affine parameters are gamma=1 and beta=0, so output equals the normalized input.
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_output_has_zero_mean_and_unit_variance_per_sample():
    torch.manual_seed(2)
    layer = _make_layer(16)
    x = torch.randn(10, 16, dtype=torch.float64) * 5.0 - 3.0
    out = layer(x)
    means = out.mean(dim=-1)
    variances = out.var(dim=-1, unbiased=False)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-6)
    assert torch.allclose(variances, torch.ones_like(variances), atol=1e-4)


def test_normalizes_over_multiple_trailing_dims():
    torch.manual_seed(3)
    layer = _make_layer((3, 4))
    x = torch.randn(2, 5, 3, 4, dtype=torch.float64) * 2.0 + 1.0
    out = layer(x)
    expected = _reference_layer_norm(x, n_dims=2)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_constant_input_gives_zero_output():
    layer = _make_layer(5)
    x = torch.full((3, 5), 7.5, dtype=torch.float64)
    out = layer(x)
    # Variance is zero, so the normalized value is (x - mean) / sqrt(eps) = 0.
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_each_sample_is_normalized_independently():
    torch.manual_seed(4)
    layer = _make_layer(7)
    x = torch.randn(6, 7, dtype=torch.float64)
    batch_out = layer(x)
    single_outs = torch.stack([layer(x[i]) for i in range(x.shape[0])])
    assert torch.allclose(batch_out, single_outs, atol=1e-10)


def test_invariant_to_positive_scaling_of_input():
    torch.manual_seed(5)
    layer = _make_layer(10)
    x = torch.randn(4, 10, dtype=torch.float64)
    out_a = layer(x)
    out_b = layer(3.0 * x)
    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_invariant_to_constant_shift_of_input():
    torch.manual_seed(6)
    layer = _make_layer(10)
    x = torch.randn(4, 10, dtype=torch.float64)
    out_a = layer(x)
    out_b = layer(x + 100.0)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_forward_is_deterministic_for_same_input():
    torch.manual_seed(7)
    layer = _make_layer(9)
    x = torch.randn(3, 9, dtype=torch.float64)
    first = layer(x)
    second = layer(x)
    assert torch.equal(first, second)


def test_affine_parameters_have_normalized_shape_and_defaults():
    torch.manual_seed(8)
    layer = _make_layer(12)
    params = list(layer.parameters())
    assert len(params) == 2
    for p in params:
        assert p.shape == (12,)
    gamma, beta = params
    assert torch.allclose(gamma, torch.ones_like(gamma))
    assert torch.allclose(beta, torch.zeros_like(beta))


def test_affine_parameters_scale_and_shift_output():
    torch.manual_seed(9)
    layer = _make_layer(6)
    gamma, beta = list(layer.parameters())
    with torch.no_grad():
        gamma.fill_(2.0)
        beta.fill_(1.0)
    x = torch.randn(4, 6, dtype=torch.float64)
    out = layer(x)
    expected = 2.0 * _reference_layer_norm(x, n_dims=1) + 1.0
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_gradients_flow_to_input_and_parameters():
    torch.manual_seed(10)
    layer = _make_layer(8)
    x = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    out = layer(x)
    loss = (out ** 2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for p in layer.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_preserves_float64_dtype():
    torch.manual_seed(11)
    layer = _make_layer(4, dtype=torch.float64)
    x = torch.randn(2, 4, dtype=torch.float64)
    out = layer(x)
    assert out.dtype == torch.float64


def test_mismatched_last_dimension_raises():
    layer = _make_layer(8)
    x = torch.randn(3, 5, dtype=torch.float64)
    with pytest.raises((RuntimeError, ValueError)):
        layer(x)


def test_output_values_are_finite_for_large_inputs():
    torch.manual_seed(12)
    layer = _make_layer(10)
    x = torch.randn(4, 10, dtype=torch.float64) * 1e6
    out = layer(x)
    assert torch.isfinite(out).all()
    assert math.isclose(float(out.mean(dim=-1).abs().max()), 0.0, abs_tol=1e-6)
