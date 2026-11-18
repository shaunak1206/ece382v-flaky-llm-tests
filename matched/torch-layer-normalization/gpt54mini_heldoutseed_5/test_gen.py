import inspect

import pytest
import torch
import torch.nn as nn

import torch_layer_normalization
from torch_layer_normalization import LayerNormalization


def _make_layer_norm(normalized_shape, eps=1e-5, elementwise_affine=True):
    """
    Create LayerNormalization while being tolerant to small signature differences.
    """
    sig = inspect.signature(LayerNormalization)
    params = sig.parameters

    candidates = []
    # Most likely PyTorch-style signatures
    if "normalized_shape" in params:
        candidates.append(
            dict(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
        )
    if "shape" in params:
        candidates.append(dict(shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine))
    if "normalized_ndim" in params:
        candidates.append(
            dict(
                normalized_ndim=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
        )

    # Positional fallback
    candidates.append((normalized_shape, eps, elementwise_affine))
    candidates.append((normalized_shape,))

    last_err = None
    for args in candidates:
        try:
            if isinstance(args, dict):
                return LayerNormalization(**args)
            return LayerNormalization(*args)
        except TypeError as e:
            last_err = e
    raise last_err


def test_module_exports_layer_normalization():
    assert hasattr(torch_layer_normalization, "LayerNormalization")
    assert LayerNormalization is torch_layer_normalization.LayerNormalization


def test_version_string():
    assert isinstance(torch_layer_normalization.__version__, str)
    assert torch_layer_normalization.__version__ == "0.11.0"


def test_layer_normalization_is_nn_module():
    layer = _make_layer_norm(4)
    assert isinstance(layer, nn.Module)


def test_forward_preserves_input_shape_2d():
    layer = _make_layer_norm(4)
    x = torch.randn(2, 4)
    y = layer(x)
    assert y.shape == x.shape


def test_forward_preserves_input_shape_3d():
    layer = _make_layer_norm(6)
    x = torch.randn(3, 5, 6)
    y = layer(x)
    assert y.shape == x.shape


def test_forward_normalizes_last_dimension_mean_near_zero():
    layer = _make_layer_norm(8, elementwise_affine=False)
    x = torch.randn(16, 8) * 10 + 5
    y = layer(x)
    means = y.mean(dim=-1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5, rtol=1e-4)


def test_forward_normalizes_last_dimension_variance_near_one():
    layer = _make_layer_norm(8, elementwise_affine=False)
    x = torch.randn(32, 8) * 7 - 2
    y = layer(x)
    vars_ = y.var(dim=-1, unbiased=False)
    assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-4, rtol=1e-3)


def test_forward_is_deterministic_in_eval_mode():
    layer = _make_layer_norm(5)
    layer.eval()
    x = torch.randn(4, 5)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2)


def test_forward_accepts_non_contiguous_input():
    layer = _make_layer_norm(4)
    x = torch.randn(2, 4, 3).transpose(1, 2)  # shape (2, 3, 4), non-contiguous
    assert not x.is_contiguous()
    y = layer(x)
    assert y.shape == x.shape


def test_state_dict_has_affine_parameters_by_default():
    layer = _make_layer_norm(4)
    state = layer.state_dict()
    keys = set(state.keys())
    # Support common parameter naming conventions
    assert any(k in keys for k in ("weight", "gamma"))
    assert any(k in keys for k in ("bias", "beta"))


def test_state_dict_empty_when_affine_disabled():
    layer = _make_layer_norm(4, elementwise_affine=False)
    state = layer.state_dict()
    assert len(state) == 0


def test_parameters_require_grad_by_default():
    layer = _make_layer_norm(4)
    params = list(layer.parameters())
    assert params, "Expected learnable parameters by default"
    assert all(p.requires_grad for p in params)


def test_backward_populates_input_gradient():
    layer = _make_layer_norm(4)
    x = torch.randn(3, 4, requires_grad=True)
    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_affine_parameters_affect_output():
    layer = _make_layer_norm(4)
    x = torch.randn(2, 4)

    y1 = layer(x)

    # If the module uses common naming conventions, changing weight/bias should change output.
    with torch.no_grad():
        if hasattr(layer, "weight") and layer.weight is not None:
            layer.weight.fill_(2.0)
        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.fill_(1.0)

    y2 = layer(x)
    assert not torch.allclose(y1, y2)


def test_repr_contains_class_name():
    layer = _make_layer_norm(4)
    assert "LayerNormalization" in repr(layer)


def test_custom_eps_does_not_change_output_shape():
    layer = _make_layer_norm(4, eps=1e-3)
    x = torch.randn(2, 4)
    y = layer(x)
    assert y.shape == x.shape


def test_multiple_forward_calls_with_different_inputs_work():
    layer = _make_layer_norm(4)
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 4) * 3 + 1
    y1 = layer(x1)
    y2 = layer(x2)
    assert y1.shape == x1.shape
    assert y2.shape == x2.shape
    assert not torch.allclose(y1, y2)
