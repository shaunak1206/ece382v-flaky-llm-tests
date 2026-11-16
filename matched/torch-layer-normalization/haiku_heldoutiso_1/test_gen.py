import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_layer_normalization import LayerNormalization
import torch_layer_normalization


ATOL = 1e-4


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


def test_version_string():
    assert torch_layer_normalization.__version__ == '0.11.0'


def test_layer_is_nn_module():
    layer = LayerNormalization(8)
    assert isinstance(layer, nn.Module)


def test_forward_preserves_shape_2d():
    layer = LayerNormalization(8)
    x = torch.randn(4, 8)
    out = layer(x)
    assert out.shape == x.shape


def test_forward_preserves_shape_3d():
    layer = LayerNormalization(16)
    x = torch.randn(2, 5, 16)
    out = layer(x)
    assert out.shape == x.shape


def test_output_is_zero_mean_unit_variance_over_last_dim():
    layer = LayerNormalization(32)
    x = torch.randn(6, 32) * 5.0 + 3.0
    out = layer(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(6), atol=1e-4)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(6), atol=1e-3)


def test_multi_dimensional_normalized_shape_normalizes_over_all_trailing_dims():
    layer = LayerNormalization((3, 4))
    x = torch.randn(2, 3, 4) * 2.0 + 1.0
    out = layer(x)
    flat = out.reshape(2, -1)
    assert torch.allclose(flat.mean(dim=-1), torch.zeros(2), atol=1e-4)
    assert torch.allclose(flat.var(dim=-1, unbiased=False), torch.ones(2), atol=1e-3)


def test_matches_functional_layer_norm_with_default_init():
    dim = 10
    layer = LayerNormalization(dim)
    x = torch.randn(3, 7, dim)
    expected = F.layer_norm(x, (dim,))
    out = layer(x)
    assert torch.allclose(out, expected, atol=ATOL)


def test_invariant_to_positive_affine_transform_of_input():
    layer = LayerNormalization(12)
    x = torch.randn(5, 12)
    out_original = layer(x)
    out_transformed = layer(x * 7.0 + 3.0)
    assert torch.allclose(out_original, out_transformed, atol=1e-3)


def test_constant_input_gives_finite_zeros():
    layer = LayerNormalization(6)
    x = torch.full((2, 6), 4.2)
    out = layer(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)


def test_samples_are_normalized_independently():
    layer = LayerNormalization(8)
    x = torch.randn(4, 8)
    batched = layer(x)
    for i in range(x.shape[0]):
        single = layer(x[i:i + 1])
        assert torch.allclose(batched[i:i + 1], single, atol=ATOL)


def test_repeated_forward_is_deterministic():
    layer = LayerNormalization(8)
    layer.eval()
    x = torch.randn(3, 8)
    first = layer(x)
    second = layer(x)
    assert torch.equal(first, second)


def test_train_and_eval_modes_produce_same_output():
    layer = LayerNormalization(8)
    x = torch.randn(3, 8)
    layer.train()
    out_train = layer(x)
    layer.eval()
    out_eval = layer(x)
    assert torch.allclose(out_train, out_eval, atol=1e-6)


def test_gradients_flow_to_input():
    layer = LayerNormalization(8)
    x = torch.randn(4, 8, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_learnable_parameters_receive_gradients():
    layer = LayerNormalization(8)
    params = list(layer.parameters())
    assert len(params) > 0
    x = torch.randn(4, 8)
    out = layer(x)
    loss = (out * torch.arange(8, dtype=out.dtype)).sum()
    loss.backward()
    for p in params:
        if p.requires_grad:
            assert p.grad is not None


def test_forward_under_no_grad_does_not_track_graph():
    layer = LayerNormalization(8)
    x = torch.randn(2, 8, requires_grad=True)
    with torch.no_grad():
        out = layer(x)
    assert not out.requires_grad


def test_double_precision_dtype_is_preserved():
    layer = LayerNormalization(8).double()
    x = torch.randn(3, 8, dtype=torch.float64)
    out = layer(x)
    assert out.dtype == torch.float64
    assert torch.allclose(out.mean(dim=-1), torch.zeros(3, dtype=torch.float64), atol=1e-8)


def test_mismatched_last_dimension_raises():
    layer = LayerNormalization(8)
    x = torch.randn(2, 5)
    with pytest.raises((RuntimeError, ValueError)):
        layer(x)


def test_state_dict_roundtrip_reproduces_output():
    src = LayerNormalization(8)
    dst = LayerNormalization(8)
    dst.load_state_dict(src.state_dict())
    x = torch.randn(4, 8)
    assert torch.allclose(src(x), dst(x), atol=1e-6)
