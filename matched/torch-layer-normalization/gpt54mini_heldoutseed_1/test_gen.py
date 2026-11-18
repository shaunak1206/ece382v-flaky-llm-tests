import inspect

import pytest
import torch
import torch.nn as nn

import torch_layer_normalization as tln


def _layer_norm_cls():
    assert hasattr(tln, "LayerNormalization"), "LayerNormalization not re-exported from package"
    return tln.LayerNormalization


def _make_layer(norm_shape):
    cls = _layer_norm_cls()
    sig = inspect.signature(cls)
    kwargs = {}

    if "normalized_shape" in sig.parameters:
        kwargs["normalized_shape"] = norm_shape
    elif "shape" in sig.parameters:
        kwargs["shape"] = norm_shape
    elif "num_features" in sig.parameters:
        kwargs["num_features"] = norm_shape[-1] if isinstance(norm_shape, (tuple, list)) else norm_shape
    elif "channels" in sig.parameters:
        kwargs["channels"] = norm_shape[-1] if isinstance(norm_shape, (tuple, list)) else norm_shape

    try:
        return cls(**kwargs)
    except TypeError:
        # Fall back to positional construction for implementations that do not use keyword args.
        try:
            return cls(norm_shape)
        except TypeError:
            if isinstance(norm_shape, (tuple, list)) and len(norm_shape) == 1:
                return cls(norm_shape[0])
            raise


def _forward(layer, x):
    layer.eval()
    with torch.no_grad():
        return layer(x)


def _assert_normalized_last_dim(y, atol_mean=1e-5, atol_var=1e-4):
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=atol_mean, rtol=0)
    assert torch.allclose(var, torch.ones_like(var), atol=atol_var, rtol=0)


def test_package_version_string():
    assert tln.__version__ == "0.11.0"


def test_layer_normalization_is_reexported_from_package():
    from torch_layer_normalization.layer_normalization import LayerNormalization as Imported

    assert tln.LayerNormalization is Imported


def test_layer_normalization_class_is_subclass_of_torch_nn_module():
    cls = _layer_norm_cls()
    assert inspect.isclass(cls)
    assert issubclass(cls, nn.Module)


def test_layer_normalization_has_forward_method():
    cls = _layer_norm_cls()
    assert hasattr(cls, "forward")
    assert callable(getattr(cls, "forward"))


def test_layer_normalization_can_be_instantiated_with_1d_shape():
    layer = _make_layer((4,))
    assert isinstance(layer, nn.Module)


def test_layer_normalization_can_be_instantiated_with_scalar_shape():
    layer = _make_layer(4)
    assert isinstance(layer, nn.Module)


def test_layer_normalization_forward_preserves_2d_shape():
    layer = _make_layer((6,))
    x = torch.randn(3, 6)
    y = _forward(layer, x)
    assert y.shape == x.shape


def test_layer_normalization_forward_preserves_3d_shape():
    layer = _make_layer((8,))
    x = torch.randn(2, 5, 8)
    y = _forward(layer, x)
    assert y.shape == x.shape


def test_layer_normalization_normalizes_last_dimension_on_2d_input():
    layer = _make_layer((10,))
    x = torch.randn(4, 10)
    y = _forward(layer, x)
    _assert_normalized_last_dim(y)


def test_layer_normalization_normalizes_last_dimension_on_3d_input():
    layer = _make_layer((7,))
    x = torch.randn(2, 3, 7)
    y = _forward(layer, x)
    _assert_normalized_last_dim(y)


def test_layer_normalization_is_deterministic_in_eval_mode():
    layer = _make_layer((5,))
    x = torch.randn(3, 5)
    y1 = _forward(layer, x)
    y2 = _forward(layer, x)
    assert torch.allclose(y1, y2)


def test_layer_normalization_supports_backward():
    layer = _make_layer((6,))
    layer.train()
    x = torch.randn(4, 6, requires_grad=True)
    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_layer_normalization_parameters_exist_when_affine_is_enabled():
    layer = _make_layer((4,))
    params = dict(layer.named_parameters())
    # Many layer norm implementations expose weight/bias when affine=True.
    assert params, "Expected learnable parameters on LayerNormalization"
