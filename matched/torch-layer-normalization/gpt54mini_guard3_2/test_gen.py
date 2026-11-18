import inspect

import pytest
import torch
import torch.nn as nn

import torch_layer_normalization as tln


def make_layer_norm(feature_dim=4, **kwargs):
    """
    Try a few common constructor conventions for LayerNormalization.
    This keeps the tests resilient to small API differences while still
    exercising the exported class.
    """
    cls = tln.LayerNormalization
    sig = inspect.signature(cls)
    params = sig.parameters

    # Prefer explicit keyword if supported.
    if "normalized_shape" in params:
        try:
            return cls(normalized_shape=feature_dim, **kwargs)
        except TypeError:
            try:
                return cls(normalized_shape=(feature_dim,), **kwargs)
            except TypeError:
                pass

    if "input_size" in params:
        try:
            return cls(input_size=feature_dim, **kwargs)
        except TypeError:
            pass

    if "shape" in params:
        try:
            return cls(shape=feature_dim, **kwargs)
        except TypeError:
            try:
                return cls(shape=(feature_dim,), **kwargs)
            except TypeError:
                pass

    # Positional fallbacks.
    for args in [
        (feature_dim,),
        ((feature_dim,),),
    ]:
        try:
            return cls(*args, **kwargs)
        except TypeError:
            continue

    raise AssertionError(
        f"Could not instantiate {cls.__name__}; signature is {sig!s}"
    )


def test_package_exports_layer_normalization():
    assert hasattr(tln, "LayerNormalization")
    assert tln.LayerNormalization is not None


def test_version_string_matches_expected():
    assert tln.__version__ == "0.11.0"


def test_layer_normalization_is_nn_module_subclass():
    assert issubclass(tln.LayerNormalization, nn.Module)


def test_layer_normalization_can_be_instantiated():
    module = make_layer_norm(4)
    assert isinstance(module, nn.Module)


def test_layer_normalization_forward_preserves_shape_2d():
    torch.manual_seed(0)
    module = make_layer_norm(4)
    x = torch.randn(3, 4)
    y = module(x)
    assert y.shape == x.shape


def test_layer_normalization_forward_preserves_shape_3d():
    torch.manual_seed(0)
    module = make_layer_norm(5)
    x = torch.randn(2, 3, 5)
    y = module(x)
    assert y.shape == x.shape


def test_layer_normalization_output_has_zero_mean_per_sample():
    torch.manual_seed(0)
    module = make_layer_norm(4)
    x = torch.randn(6, 4)
    y = module(x)
    means = y.mean(dim=-1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5, rtol=1e-5)


def test_layer_normalization_output_has_unit_variance_per_sample():
    torch.manual_seed(0)
    module = make_layer_norm(4)
    x = torch.randn(6, 4)
    y = module(x)
    variances = y.var(dim=-1, unbiased=False)
    assert torch.allclose(variances, torch.ones_like(variances), atol=1e-4, rtol=1e-4)


def test_layer_normalization_supports_backward():
    torch.manual_seed(0)
    module = make_layer_norm(4)
    x = torch.randn(3, 4, requires_grad=True)
    y = module(x)
    loss = y.pow(2).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_layer_normalization_parameters_require_grad_when_present():
    module = make_layer_norm(4)
    params = list(module.named_parameters())
    assert len(params) >= 0

    for _, p in params:
        assert p.requires_grad


def test_layer_normalization_is_deterministic_for_same_input():
    torch.manual_seed(0)
    module = make_layer_norm(4)
    x = torch.randn(2, 4)
    y1 = module(x)
    y2 = module(x.clone())
    assert torch.allclose(y1, y2, atol=0.0, rtol=0.0)


def test_layer_normalization_state_dict_roundtrip():
    torch.manual_seed(0)
    module1 = make_layer_norm(4)
    module2 = make_layer_norm(4)

    x = torch.randn(2, 4)
    out1 = module1(x)

    state = module1.state_dict()
    module2.load_state_dict(state)
    out2 = module2(x)

    assert torch.allclose(out1, out2, atol=1e-6, rtol=1e-6)


def test_layer_normalization_module_repr_contains_class_name():
    module = make_layer_norm(4)
    assert "LayerNormalization" in repr(module)
