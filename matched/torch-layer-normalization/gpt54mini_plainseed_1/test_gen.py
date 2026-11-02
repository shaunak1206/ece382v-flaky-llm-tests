import inspect

import pytest
import torch
import torch.nn.functional as F

import torch_layer_normalization as tln
from torch_layer_normalization.layer_normalization import LayerNormalization as LayerNormalizationImpl


def _make_layer_norm(normalized_shape, **kwargs):
    """
    Instantiate LayerNormalization with a few common constructor signatures.
    """
    cls = tln.LayerNormalization
    attempts = [
        lambda: cls(normalized_shape, **kwargs),
        lambda: cls(shape=normalized_shape, **kwargs),
        lambda: cls(normalized_dimensions=normalized_shape, **kwargs),
    ]
    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as e:
            last_error = e
    raise last_error


def _forward(module, x):
    """
    Call module with a few common forward signatures.
    """
    try:
        return module(x)
    except TypeError:
        pass

    # fallback for uncommon API naming
    for name in ("forward", "__call__"):
        if hasattr(module, name):
            fn = getattr(module, name)
            try:
                return fn(x)
            except TypeError:
                continue
    raise TypeError("Could not invoke LayerNormalization forward")


def test_version_string():
    assert isinstance(tln.__version__, str)
    assert tln.__version__ == "0.11.0"


def test_layer_normalization_is_exported_from_package():
    assert hasattr(tln, "LayerNormalization")
    assert tln.LayerNormalization is LayerNormalizationImpl
    assert inspect.isclass(tln.LayerNormalization)


def test_all_star_import_exports_layer_normalization():
    namespace = {}
    exec("from torch_layer_normalization import *", namespace)
    assert "LayerNormalization" in namespace
    assert namespace["LayerNormalization"] is tln.LayerNormalization


def test_layer_normalization_class_name():
    assert tln.LayerNormalization.__name__ == "LayerNormalization"


def test_constructor_accepts_one_dimensional_shape():
    mod = _make_layer_norm(4)
    assert isinstance(mod, torch.nn.Module)


def test_forward_preserves_shape_2d_input():
    mod = _make_layer_norm(4)
    x = torch.randn(2, 4)
    y = _forward(mod, x)
    assert y.shape == x.shape


def test_forward_preserves_shape_3d_input():
    mod = _make_layer_norm(5)
    x = torch.randn(3, 2, 5)
    y = _forward(mod, x)
    assert y.shape == x.shape


def test_forward_matches_torch_layer_norm_default_settings():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    mod = _make_layer_norm(4)
    y = _forward(mod, x)
    expected = F.layer_norm(x, (4,))
    assert torch.allclose(y, expected, atol=1e-5, rtol=1e-5)


def test_forward_matches_torch_layer_norm_with_affine_parameters():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    mod = _make_layer_norm(4)

    # If the implementation exposes affine parameters, set them to deterministic values.
    if hasattr(mod, "weight") and mod.weight is not None:
        with torch.no_grad():
            mod.weight.fill_(1.5)
    if hasattr(mod, "bias") and mod.bias is not None:
        with torch.no_grad():
            mod.bias.fill_(-0.25)

    y = _forward(mod, x)

    weight = getattr(mod, "weight", None)
    bias = getattr(mod, "bias", None)
    if weight is not None and bias is not None:
        expected = F.layer_norm(x, (4,), weight=weight, bias=bias)
        assert torch.allclose(y, expected, atol=1e-5, rtol=1e-5)
    else:
        # If the implementation is non-affine, at least ensure output is finite.
        assert torch.isfinite(y).all()


def test_backward_computes_gradients_for_input():
    mod = _make_layer_norm(4)
    x = torch.randn(2, 4, requires_grad=True)
    y = _forward(mod, x)
    loss = y.pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_backward_computes_gradients_for_parameters_if_present():
    mod = _make_layer_norm(4)
    x = torch.randn(2, 4, requires_grad=True)
    y = _forward(mod, x)
    y.sum().backward()

    params = list(mod.named_parameters())
    if params:
        for name, p in params:
            assert p.grad is not None, name
            assert p.grad.shape == p.shape, name
            assert torch.isfinite(p.grad).all(), name


def test_repr_includes_class_name():
    mod = _make_layer_norm(4)
    r = repr(mod)
    assert "LayerNormalization" in r


def test_multiple_calls_are_deterministic_in_eval_mode():
    mod = _make_layer_norm(4)
    mod.eval()
    x = torch.randn(2, 4)
    y1 = _forward(mod, x)
    y2 = _forward(mod, x)
    assert torch.allclose(y1, y2, atol=0, rtol=0)


def test_module_can_be_moved_to_cpu_and_still_run():
    mod = _make_layer_norm(4).to("cpu")
    x = torch.randn(2, 4, device="cpu")
    y = _forward(mod, x)
    assert y.device.type == "cpu"
    assert y.shape == x.shape
