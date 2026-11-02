import inspect

import pytest
import torch
import torch.nn.functional as F

import torch_layer_normalization as tln


def _make_module(*, normalized_shape=None, shape=None, eps=None, elementwise_affine=True):
    """
    Create LayerNormalization with a best-effort constructor adaptation.

    The project may expose slightly different constructor parameter names.
    This helper inspects the signature and supplies the most likely arguments.
    """
    cls = tln.LayerNormalization
    sig = inspect.signature(cls)
    params = sig.parameters

    kwargs = {}
    if normalized_shape is not None:
        if "normalized_shape" in params:
            kwargs["normalized_shape"] = normalized_shape
        elif "shape" in params:
            kwargs["shape"] = shape if shape is not None else normalized_shape
        elif "normalized_shape_or_shape" in params:
            kwargs["normalized_shape_or_shape"] = normalized_shape
        else:
            # Fall back to positional construction.
            args = [normalized_shape]
            if eps is not None or "eps" in params or "epsilon" in params:
                if eps is None:
                    eps = 1e-5
                if "eps" in params:
                    kwargs["eps"] = eps
                elif "epsilon" in params:
                    kwargs["epsilon"] = eps
            if "elementwise_affine" in params:
                kwargs["elementwise_affine"] = elementwise_affine
            return cls(*args, **kwargs)

    if eps is not None:
        if "eps" in params:
            kwargs["eps"] = eps
        elif "epsilon" in params:
            kwargs["epsilon"] = eps

    if "elementwise_affine" in params:
        kwargs["elementwise_affine"] = elementwise_affine

    if shape is not None and "shape" in params and "shape" not in kwargs:
        kwargs["shape"] = shape

    return cls(**kwargs)


def _layer_norm_reference(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    return F.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)


def test_version_string():
    assert isinstance(tln.__version__, str)
    assert tln.__version__ == "0.11.0"


def test_layer_normalization_is_exported():
    assert hasattr(tln, "LayerNormalization")
    assert inspect.isclass(tln.LayerNormalization)


def test_layer_normalization_instantiates():
    module = _make_module(normalized_shape=4)
    assert isinstance(module, torch.nn.Module)


def test_layer_normalization_has_trainable_parameters_by_default():
    module = _make_module(normalized_shape=4)
    params = dict(module.named_parameters())
    assert params, "Expected at least one parameter when elementwise_affine=True"


def test_layer_normalization_can_disable_affine_parameters():
    module = _make_module(normalized_shape=4, elementwise_affine=False)
    params = list(module.parameters())
    assert len(params) == 0


def test_layer_normalization_forward_shape_2d():
    torch.manual_seed(0)
    module = _make_module(normalized_shape=4)
    x = torch.randn(3, 4)
    y = module(x)
    assert y.shape == x.shape


def test_layer_normalization_forward_shape_3d_last_dim():
    torch.manual_seed(0)
    module = _make_module(normalized_shape=5)
    x = torch.randn(2, 3, 5)
    y = module(x)
    assert y.shape == x.shape


def test_layer_normalization_matches_torch_layer_norm_without_affine():
    torch.manual_seed(0)
    eps = 1e-5
    module = _make_module(normalized_shape=4, eps=eps, elementwise_affine=False)
    x = torch.randn(2, 4)
    y = module(x)
    expected = _layer_norm_reference(x, (4,), eps=eps)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-5)


def test_layer_normalization_matches_torch_layer_norm_with_identity_affine():
    torch.manual_seed(0)
    eps = 1e-5
    module = _make_module(normalized_shape=4, eps=eps, elementwise_affine=True)
    x = torch.randn(2, 4)

    with torch.no_grad():
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.fill_(1.0)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.zero_()

    y = module(x)
    expected = _layer_norm_reference(
        x,
        (4,),
        weight=torch.ones(4),
        bias=torch.zeros(4),
        eps=eps,
    )
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-5)


def test_layer_normalization_backward_produces_input_grad():
    torch.manual_seed(0)
    module = _make_module(normalized_shape=4)
    x = torch.randn(2, 4, requires_grad=True)
    y = module(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_layer_normalization_backward_produces_parameter_grads():
    torch.manual_seed(0)
    module = _make_module(normalized_shape=4)
    x = torch.randn(2, 4, requires_grad=True)
    y = module(x)
    y.sum().backward()

    params = list(module.parameters())
    assert params, "Expected affine parameters to exist"
    for p in params:
        assert p.grad is not None
        assert p.grad.shape == p.shape
        assert torch.isfinite(p.grad).all()


def test_layer_normalization_is_deterministic_in_eval_mode():
    torch.manual_seed(0)
    module = _make_module(normalized_shape=4)
    module.eval()
    x = torch.randn(2, 4)
    y1 = module(x)
    y2 = module(x)
    assert torch.allclose(y1, y2, atol=0, rtol=0)


def test_layer_normalization_repr_contains_class_name():
    module = _make_module(normalized_shape=4)
    rep = repr(module)
    assert "LayerNormalization" in rep or module.__class__.__name__ in rep


def test_layer_normalization_forward_preserves_dtype():
    module = _make_module(normalized_shape=4)
    x = torch.randn(2, 4, dtype=torch.float32)
    y = module(x)
    assert y.dtype == x.dtype
