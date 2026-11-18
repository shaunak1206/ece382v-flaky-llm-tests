import importlib
import math

import pytest
import torch

from torch_layer_normalization import LayerNormalization, __version__


def test_version_string_matches_package():
    assert __version__ == "0.11.0"


def test_layer_normalization_is_exported_from_package_root():
    module = importlib.import_module("torch_layer_normalization")
    assert hasattr(module, "LayerNormalization")
    assert module.LayerNormalization is LayerNormalization


def test_layer_normalization_can_be_imported_from_submodule():
    submodule = importlib.import_module("torch_layer_normalization.layer_normalization")
    assert hasattr(submodule, "LayerNormalization")
    assert submodule.LayerNormalization is LayerNormalization


def test_layer_normalization_instantiation_with_int_shape():
    layer = LayerNormalization(4)
    assert isinstance(layer, torch.nn.Module)
    assert getattr(layer, "normalized_shape", None) == 4 or tuple(layer.normalized_shape) == (4,)


def test_layer_normalization_instantiation_with_tuple_shape():
    layer = LayerNormalization((2, 3))
    normalized_shape = layer.normalized_shape
    assert tuple(normalized_shape) == (2, 3)


def test_forward_preserves_input_shape_2d():
    torch.manual_seed(0)
    layer = LayerNormalization(5)
    x = torch.randn(7, 5)
    y = layer(x)
    assert y.shape == x.shape


def test_forward_preserves_input_shape_3d():
    torch.manual_seed(0)
    layer = LayerNormalization((3, 4))
    x = torch.randn(2, 3, 4)
    y = layer(x)
    assert y.shape == x.shape


def test_forward_without_affine_has_zero_mean_along_last_dim():
    torch.manual_seed(0)
    layer = LayerNormalization(6, elementwise_affine=False)
    x = torch.randn(4, 6)
    y = layer(x)
    mean = y.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6, rtol=1e-6)


def test_forward_without_affine_has_unit_variance_along_last_dim():
    torch.manual_seed(0)
    layer = LayerNormalization(8, elementwise_affine=False)
    x = torch.randn(3, 8)
    y = layer(x)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5, rtol=1e-5)


def test_affine_parameters_exist_by_default():
    layer = LayerNormalization(5)
    assert hasattr(layer, "weight")
    assert hasattr(layer, "bias")
    assert isinstance(layer.weight, torch.nn.Parameter)
    assert isinstance(layer.bias, torch.nn.Parameter)
    assert tuple(layer.weight.shape) == (5,)
    assert tuple(layer.bias.shape) == (5,)


def test_affine_parameters_absent_when_disabled():
    layer = LayerNormalization(5, elementwise_affine=False)
    assert getattr(layer, "weight", None) is None
    assert getattr(layer, "bias", None) is None


def test_state_dict_contains_affine_parameters_when_enabled():
    layer = LayerNormalization(4)
    state = layer.state_dict()
    assert "weight" in state
    assert "bias" in state
    assert tuple(state["weight"].shape) == (4,)
    assert tuple(state["bias"].shape) == (4,)


def test_state_dict_is_empty_when_affine_disabled():
    layer = LayerNormalization(4, elementwise_affine=False)
    state = layer.state_dict()
    assert state == {}


def test_manual_weight_and_bias_shift_output_as_expected():
    torch.manual_seed(0)
    layer = LayerNormalization(3, elementwise_affine=True)
    with torch.no_grad():
        layer.weight.fill_(2.0)
        layer.bias.fill_(0.5)
    x = torch.randn(2, 3)
    y = layer(x)
    plain = LayerNormalization(3, elementwise_affine=False)(x)
    expected = plain * 2.0 + 0.5
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_matches_torch_layer_norm_without_affine():
    torch.manual_seed(0)
    normalized_shape = (4,)
    x = torch.randn(2, 4)
    custom = LayerNormalization(normalized_shape, elementwise_affine=False, eps=1e-5)
    expected = torch.nn.functional.layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5)
    y = custom(x)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_matches_torch_layer_norm_with_affine():
    torch.manual_seed(0)
    normalized_shape = (4,)
    x = torch.randn(3, 4)

    custom = LayerNormalization(normalized_shape, elementwise_affine=True, eps=1e-5)
    reference = torch.nn.LayerNorm(normalized_shape, elementwise_affine=True, eps=1e-5)

    with torch.no_grad():
        custom.weight.copy_(reference.weight)
        custom.bias.copy_(reference.bias)

    y_custom = custom(x)
    y_reference = reference(x)
    assert torch.allclose(y_custom, y_reference, atol=1e-6, rtol=1e-6)
