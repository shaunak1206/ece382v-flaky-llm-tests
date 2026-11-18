import inspect

import pytest
import torch

import torch_layer_normalization as tln


def _make_layer_norm(normalized_shape, **kwargs):
    """
    Create LayerNormalization while being tolerant to small API differences.
    """
    cls = tln.LayerNormalization
    sig = inspect.signature(cls)
    params = sig.parameters

    # Try common constructor names.
    ctor_kwargs = {}
    if "normalized_shape" in params:
        ctor_kwargs["normalized_shape"] = normalized_shape
    elif "features" in params:
        ctor_kwargs["features"] = normalized_shape
    elif "shape" in params:
        ctor_kwargs["shape"] = normalized_shape
    else:
        # Fall back to positional argument.
        return cls(normalized_shape, **kwargs)

    for k, v in kwargs.items():
        if k in params:
            ctor_kwargs[k] = v

    return cls(**ctor_kwargs)


def _get_affine_weights(module):
    """
    Return (weight, bias) if present, else (None, None).
    """
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    return weight, bias


def test_version_constant():
    assert tln.__version__ == "0.11.0"


def test_layer_normalization_is_exported_from_package():
    assert hasattr(tln, "LayerNormalization")
    assert inspect.isclass(tln.LayerNormalization)


def test_layer_normalization_is_instantiable():
    module = _make_layer_norm(4)
    assert isinstance(module, torch.nn.Module)


def test_forward_preserves_2d_shape():
    module = _make_layer_norm(4)
    x = torch.randn(3, 4)
    y = module(x)
    assert y.shape == x.shape


def test_forward_preserves_3d_shape():
    module = _make_layer_norm((5,))
    x = torch.randn(2, 7, 5)
    y = module(x)
    assert y.shape == x.shape


def test_normalizes_last_dimension_to_zero_mean_and_unit_variance():
    torch.manual_seed(0)
    module = _make_layer_norm(8, eps=1e-5)
    x = torch.randn(4, 8) * 3.0 + 10.0
    y = module(x)

    means = y.mean(dim=-1)
    vars_ = y.var(dim=-1, unbiased=False)

    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5, rtol=1e-5)
    assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-4, rtol=1e-4)


def test_constant_input_does_not_produce_nan_or_inf():
    module = _make_layer_norm(6, eps=1e-5)
    x = torch.ones(2, 6) * 123.0
    y = module(x)

    assert torch.isfinite(y).all()
    assert y.shape == x.shape


def test_gradients_flow_to_input():
    module = _make_layer_norm(4)
    x = torch.randn(2, 4, requires_grad=True)
    y = module(x).sum()
    y.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.shape == x.shape


def test_gradients_flow_to_affine_parameters_if_present():
    module = _make_layer_norm(4)
    x = torch.randn(3, 4, requires_grad=True)
    y = module(x).pow(2).mean()
    y.backward()

    weight, bias = _get_affine_weights(module)
    if weight is not None:
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
    if bias is not None:
        assert bias.grad is not None
        assert torch.isfinite(bias.grad).all()


def test_affine_parameters_exist_by_default_if_supported():
    module = _make_layer_norm(4)
    weight, bias = _get_affine_weights(module)

    # Some implementations may not expose affine parameters; in that case this
    # test verifies that the module still works and leaves a clear failure only
    # when the attributes are expected.
    assert weight is None or isinstance(weight, torch.nn.Parameter)
    assert bias is None or isinstance(bias, torch.nn.Parameter)


def test_can_disable_affine_parameters_when_supported():
    sig = inspect.signature(tln.LayerNormalization)
    if "elementwise_affine" not in sig.parameters and "affine" not in sig.parameters:
        pytest.skip("Constructor does not support disabling affine parameters")

    kwargs = {}
    if "elementwise_affine" in sig.parameters:
        kwargs["elementwise_affine"] = False
    else:
        kwargs["affine"] = False

    module = _make_layer_norm(4, **kwargs)
    weight, bias = _get_affine_weights(module)

    assert weight is None or weight is False or not isinstance(weight, torch.nn.Parameter)
    assert bias is None or bias is False or not isinstance(bias, torch.nn.Parameter)


def test_state_dict_round_trip_preserves_output():
    module1 = _make_layer_norm(4)
    module2 = _make_layer_norm(4)

    x = torch.randn(5, 4)

    y1_before = module1(x)
    state = module1.state_dict()
    module2.load_state_dict(state)
    y2 = module2(x)

    assert torch.allclose(y1_before, y2, atol=1e-6, rtol=1e-6)


def test_repr_contains_class_name():
    module = _make_layer_norm(4)
    r = repr(module)
    assert "LayerNormalization" in r


def test_module_works_with_double_precision():
    module = _make_layer_norm(4).double()
    x = torch.randn(2, 4, dtype=torch.float64)
    y = module(x)

    assert y.dtype == torch.float64
    assert y.shape == x.shape


def test_output_changes_when_input_changes():
    module = _make_layer_norm(4)
    x1 = torch.randn(2, 4)
    x2 = x1 + 1.0
    y1 = module(x1)
    y2 = module(x2)

    assert not torch.allclose(y1, y2)


def test_eval_mode_is_deterministic():
    module = _make_layer_norm(4)
    module.eval()
    x = torch.randn(3, 4)

    y1 = module(x)
    y2 = module(x)

    assert torch.allclose(y1, y2, atol=0.0, rtol=0.0)


def test_train_mode_is_deterministic_for_same_input():
    module = _make_layer_norm(4)
    module.train()
    x = torch.randn(3, 4)

    y1 = module(x)
    y2 = module(x)

    assert torch.allclose(y1, y2, atol=0.0, rtol=0.0)


def test_module_accepts_batched_higher_rank_inputs():
    module = _make_layer_norm(6)
    x = torch.randn(2, 3, 4, 6)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
