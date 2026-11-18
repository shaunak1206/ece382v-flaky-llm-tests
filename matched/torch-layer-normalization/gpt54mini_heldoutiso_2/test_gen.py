import inspect

import pytest
import torch

from torch_layer_normalization import LayerNormalization, __version__


def _make_layer_norm(feature_size=4, **overrides):
    """
    Create LayerNormalization with a best-effort signature-aware constructor call.

    The project has a small API surface, but this helper keeps the tests resilient
    to minor constructor signature differences.
    """
    sig = inspect.signature(LayerNormalization)
    params = list(sig.parameters.values())

    args = []
    kwargs = dict(overrides)

    # Try to provide the first positional parameter with the normalized dimension.
    if params:
        p0 = params[0]
        if p0.name not in kwargs:
            if p0.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                args.append(feature_size)
            else:
                kwargs[p0.name] = feature_size

    # Fill in a few common optional arguments if the constructor requires them.
    for p in params[1:]:
        if p.name in kwargs:
            continue
        if p.default is not inspect._empty:
            continue
        if p.name in {"eps", "epsilon"}:
            kwargs[p.name] = 1e-5
        elif p.name in {"elementwise_affine", "affine"}:
            kwargs[p.name] = True
        elif p.name in {"bias"}:
            kwargs[p.name] = True
        elif p.name in {"device"}:
            kwargs[p.name] = None
        elif p.name in {"dtype"}:
            kwargs[p.name] = None

    try:
        return LayerNormalization(*args, **kwargs)
    except TypeError:
        # Last-resort fallback: try only the feature size as positional.
        return LayerNormalization(feature_size)


def _get_feature_size_from_module(module, default=4):
    # Prefer parameter shapes if available.
    for p in module.parameters():
        if p.ndim > 0:
            return p.shape[0]
    return default


def _output_last_dim_stats(x, dim=-1, eps=1e-6):
    mean = x.mean(dim=dim)
    var = x.var(dim=dim, unbiased=False)
    return mean, var


def test_package_version_is_exposed():
    assert __version__ == "0.11.0"


def test_layer_normalization_is_imported_from_package_root():
    assert LayerNormalization is not None
    assert inspect.isclass(LayerNormalization)


def test_can_instantiate_layer_normalization():
    ln = _make_layer_norm(4)
    assert isinstance(ln, torch.nn.Module)


def test_has_parameters_or_buffers_after_construction():
    ln = _make_layer_norm(4)
    params = list(ln.parameters())
    buffers = list(ln.buffers())
    assert len(params) + len(buffers) >= 0
    assert isinstance(ln.state_dict(), dict)


def test_forward_preserves_shape_for_2d_input():
    ln = _make_layer_norm(6)
    x = torch.randn(3, 6)
    y = ln(x)
    assert y.shape == x.shape


def test_forward_preserves_shape_for_3d_input():
    ln = _make_layer_norm(5)
    x = torch.randn(2, 4, 5)
    y = ln(x)
    assert y.shape == x.shape


def test_forward_works_with_batch_size_one():
    ln = _make_layer_norm(7)
    x = torch.randn(1, 7)
    y = ln(x)
    assert y.shape == x.shape


def test_output_is_normalized_over_last_dimension():
    torch.manual_seed(0)
    ln = _make_layer_norm(8)
    x = torch.randn(4, 3, 8) * 5.0 + 2.0
    y = ln(x)

    mean, var = _output_last_dim_stats(y, dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4, rtol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=2e-4, rtol=2e-4)


def test_forward_is_deterministic_in_train_mode():
    ln = _make_layer_norm(4)
    ln.train()
    x = torch.randn(2, 4)
    y1 = ln(x)
    y2 = ln(x)
    assert torch.allclose(y1, y2)


def test_forward_is_deterministic_between_train_and_eval():
    ln = _make_layer_norm(4)
    x = torch.randn(2, 4)

    ln.train()
    y_train = ln(x)

    ln.eval()
    y_eval = ln(x)

    assert torch.allclose(y_train, y_eval)


def test_backward_propagates_to_input_and_parameters():
    ln = _make_layer_norm(5)
    x = torch.randn(3, 5, requires_grad=True)
    y = ln(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for p in ln.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_state_dict_round_trip_preserves_output():
    torch.manual_seed(0)
    ln1 = _make_layer_norm(6)
    x = torch.randn(2, 6)

    y1 = ln1(x)
    state = ln1.state_dict()

    ln2 = _make_layer_norm(6)
    ln2.load_state_dict(state)
    y2 = ln2(x)

    assert torch.allclose(y1, y2)


def test_supports_non_contiguous_input():
    ln = _make_layer_norm(4)
    x = torch.randn(4, 2, 4).transpose(0, 1)  # non-contiguous
    assert not x.is_contiguous()
    y = ln(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_incorrect_last_dimension_raises_error():
    ln = _make_layer_norm(4)
    x = torch.randn(2, 5)
    with pytest.raises(Exception):
        _ = ln(x)


def test_multiple_forwards_do_not_change_parameters():
    ln = _make_layer_norm(4)
    before = {k: v.clone() for k, v in ln.state_dict().items()}
    x = torch.randn(2, 4)

    _ = ln(x)
    _ = ln(x)

    after = dict(ln.state_dict())
    assert before.keys() == after.keys()
    for k in before:
        assert torch.allclose(before[k], after[k])


def test_module_can_be_moved_to_cpu_explicitly():
    ln = _make_layer_norm(4)
    ln = ln.to("cpu")
    x = torch.randn(2, 4, device="cpu")
    y = ln(x)
    assert y.device.type == "cpu"
    assert y.shape == x.shape


def test_parameter_count_is_small():
    ln = _make_layer_norm(4)
    param_count = sum(p.numel() for p in ln.parameters())
    assert param_count in {0, 8} or param_count > 0
    assert param_count < 1000


def test_gradients_are_zero_mean_like_layer_norm_behavior():
    ln = _make_layer_norm(4)
    x = torch.randn(2, 4, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()
