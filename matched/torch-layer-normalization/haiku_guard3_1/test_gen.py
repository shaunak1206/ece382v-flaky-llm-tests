import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_layer_normalization import LayerNormalization
from torch_layer_normalization import __version__


ATOL = 1e-4


def test_version_string_is_defined():
    torch.manual_seed(0)
    assert isinstance(__version__, str)
    assert __version__ == '0.11.0'


def test_is_nn_module_subclass():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    assert isinstance(layer, nn.Module)


def test_forward_preserves_input_shape_3d():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    x = torch.randn(2, 3, 4)
    y = layer(x)
    assert y.shape == x.shape


def test_forward_preserves_input_shape_2d():
    torch.manual_seed(0)
    layer = LayerNormalization(6)
    x = torch.randn(5, 6)
    assert layer(x).shape == (5, 6)


def test_output_has_zero_mean_over_last_dim():
    torch.manual_seed(0)
    layer = LayerNormalization(8)
    x = torch.randn(4, 3, 8) * 5.0 + 2.0
    y = layer(x)
    means = y.mean(dim=-1)
    assert torch.allclose(means, torch.zeros_like(means), atol=ATOL)


def test_output_has_unit_variance_over_last_dim():
    torch.manual_seed(0)
    layer = LayerNormalization(8)
    x = torch.randn(4, 3, 8) * 5.0 + 2.0
    y = layer(x)
    variances = y.var(dim=-1, unbiased=False)
    assert torch.allclose(variances, torch.ones_like(variances), atol=ATOL)


def test_matches_functional_layer_norm_without_affine():
    torch.manual_seed(0)
    layer = LayerNormalization(5)
    x = torch.randn(3, 4, 5)
    expected = F.layer_norm(x, (5,))
    assert torch.allclose(layer(x), expected, atol=ATOL)


def test_multi_dimensional_normalized_shape_normalizes_last_two_dims():
    torch.manual_seed(0)
    layer = LayerNormalization((3, 4))
    x = torch.randn(2, 3, 4) * 3.0 - 1.0
    y = layer(x)
    flat = y.reshape(2, -1)
    assert torch.allclose(flat.mean(dim=-1), torch.zeros(2), atol=ATOL)
    assert torch.allclose(flat.var(dim=-1, unbiased=False), torch.ones(2), atol=ATOL)


def test_constant_input_gives_zero_output():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    x = torch.full((2, 4), 7.5)
    y = layer(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=ATOL)


def test_scale_invariance_of_input():
    torch.manual_seed(0)
    layer = LayerNormalization(6)
    x = torch.randn(3, 6)
    assert torch.allclose(layer(x), layer(10.0 * x), atol=ATOL)


def test_shift_invariance_of_input():
    torch.manual_seed(0)
    layer = LayerNormalization(6)
    x = torch.randn(3, 6)
    assert torch.allclose(layer(x), layer(x + 42.0), atol=ATOL)


def test_samples_are_normalized_independently():
    torch.manual_seed(0)
    layer = LayerNormalization(5)
    x = torch.randn(4, 5)
    y_full = layer(x)
    y_single = layer(x[2:3])
    assert torch.allclose(y_full[2:3], y_single, atol=ATOL)


def test_batch_size_one_works():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    x = torch.randn(1, 4)
    y = layer(x)
    assert y.shape == (1, 4)
    assert abs(float(y.mean())) < ATOL


def test_repeated_forward_is_deterministic():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    x = torch.randn(2, 4)
    first = layer(x)
    second = layer(x)
    assert torch.equal(first, second)


def test_non_contiguous_input_matches_contiguous_input():
    torch.manual_seed(0)
    layer = LayerNormalization(5)
    base = torch.randn(5, 4)
    x = base.t()  # shape (4, 5), non-contiguous
    assert not x.is_contiguous()
    assert torch.allclose(layer(x), layer(x.contiguous()), atol=ATOL)


def test_large_magnitude_inputs_stay_finite():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    x = torch.randn(3, 4) * 1e4
    y = layer(x)
    assert torch.isfinite(y).all()


def test_parameters_have_expected_total_size():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    total = sum(p.numel() for p in layer.parameters())
    # scale and bias vectors of length 4
    assert total == 8


def test_backward_produces_finite_input_gradient():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    x = torch.randn(2, 4, requires_grad=True)
    loss = layer(x).pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_gradcheck_in_double_precision():
    torch.manual_seed(0)
    layer = LayerNormalization(3).double()
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,))


def test_mismatched_last_dimension_raises():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    x = torch.randn(2, 5)
    with pytest.raises((RuntimeError, ValueError)):
        layer(x)
