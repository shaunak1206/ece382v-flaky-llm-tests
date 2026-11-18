import inspect

import pytest
import torch

import torch_layer_normalization as tln


def test_package_exposes_layer_normalization():
    assert hasattr(tln, "LayerNormalization")
    assert tln.LayerNormalization is not None


def test_package_version_matches_expected():
    assert tln.__version__ == "0.11.0"


def test_layer_normalization_is_callable_and_instantiable():
    layer = tln.LayerNormalization(4)
    assert callable(layer)
    assert isinstance(layer, torch.nn.Module)


def test_layer_normalization_has_weight_and_bias_when_affine():
    layer = tln.LayerNormalization(4)
    assert hasattr(layer, "weight")
    assert hasattr(layer, "bias")
    assert layer.weight.shape == (4,)
    assert layer.bias.shape == (4,)


def test_layer_normalization_default_parameters_are_initialized_reasonably():
    layer = tln.LayerNormalization(4)
    torch.testing.assert_close(layer.weight, torch.ones(4))
    torch.testing.assert_close(layer.bias, torch.zeros(4))


def test_forward_matches_torch_layernorm_for_1d_normalized_shape():
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    custom = tln.LayerNormalization(4)
    reference = torch.nn.LayerNorm(4)

    # Ensure identical parameters for a fair comparison
    with torch.no_grad():
        reference.weight.copy_(custom.weight)
        reference.bias.copy_(custom.bias)

    y_custom = custom(x)
    y_ref = reference(x)
    torch.testing.assert_close(y_custom, y_ref, rtol=1e-5, atol=1e-6)


def test_forward_matches_torch_layernorm_for_tuple_normalized_shape():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4)
    custom = tln.LayerNormalization((3, 4))
    reference = torch.nn.LayerNorm((3, 4))

    with torch.no_grad():
        reference.weight.copy_(custom.weight)
        reference.bias.copy_(custom.bias)

    y_custom = custom(x)
    y_ref = reference(x)
    torch.testing.assert_close(y_custom, y_ref, rtol=1e-5, atol=1e-6)


def test_forward_preserves_input_shape():
    x = torch.randn(5, 7, 9)
    layer = tln.LayerNormalization((7, 9))
    y = layer(x)
    assert y.shape == x.shape


def test_forward_works_for_batch_size_one():
    x = torch.randn(1, 6)
    layer = tln.LayerNormalization(6)
    y = layer(x)
    assert y.shape == (1, 6)
    assert torch.isfinite(y).all()


def test_output_has_zero_mean_and_unit_variance_over_normalized_dims():
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    layer = tln.LayerNormalization(8)
    y = layer(x)

    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)

    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(var, torch.ones_like(var), atol=1e-4, rtol=1e-4)


def test_backward_computes_gradients_for_input_and_parameters():
    torch.manual_seed(123)
    x = torch.randn(3, 4, requires_grad=True)
    layer = tln.LayerNormalization(4)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert x.grad.shape == x.shape
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad.shape == layer.bias.shape


def test_layer_normalization_repr_contains_class_name_and_shape():
    layer = tln.LayerNormalization((2, 3))
    rep = repr(layer)
    assert "LayerNormalization" in rep
    assert "2" in rep and "3" in rep


def test_module_has_expected_init_signature_keywords():
    sig = inspect.signature(tln.LayerNormalization)
    params = sig.parameters
    assert len(params) >= 1
    # normalized shape should be the first argument
    first_param = next(iter(params.values()))
    assert first_param.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
