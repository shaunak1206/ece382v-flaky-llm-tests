import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_layer_normalization import LayerNormalization, __version__


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_version_string():
    assert __version__ == '0.11.0'


def test_is_nn_module():
    layer = LayerNormalization(4)
    assert isinstance(layer, nn.Module)


def test_forward_preserves_shape_3d():
    layer = LayerNormalization(4)
    x = torch.randn(2, 3, 4)
    assert layer(x).shape == x.shape


def test_forward_preserves_shape_2d():
    layer = LayerNormalization(5)
    x = torch.randn(7, 5)
    assert layer(x).shape == (7, 5)


def test_output_has_zero_mean_and_unit_std_over_last_dim():
    layer = LayerNormalization(8)
    x = torch.randn(4, 3, 8) * 5.0 + 2.0
    y = layer(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(4, 3), atol=1e-5)
    assert torch.allclose(y.var(dim=-1, unbiased=False), torch.ones(4, 3), atol=1e-3)


def test_matches_functional_layer_norm_with_default_affine():
    layer = LayerNormalization(6)
    x = torch.randn(2, 5, 6)
    expected = F.layer_norm(x, normalized_shape=(6,), eps=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_normalizes_over_multiple_trailing_dims():
    layer = LayerNormalization((3, 4))
    x = torch.randn(2, 3, 4) * 3.0 + 1.0
    expected = F.layer_norm(x, normalized_shape=(3, 4), eps=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_each_sample_normalized_independently():
    layer = LayerNormalization(4)
    x = torch.randn(3, 4)
    batch_out = layer(x)
    for i in range(x.shape[0]):
        single_out = layer(x[i:i + 1])
        assert torch.allclose(batch_out[i:i + 1], single_out, atol=1e-6)


def test_scale_invariance():
    layer = LayerNormalization(8)
    x = torch.randn(4, 8)
    assert torch.allclose(layer(x), layer(x * 10.0), atol=1e-4)


def test_shift_invariance():
    layer = LayerNormalization(8)
    x = torch.randn(4, 8)
    assert torch.allclose(layer(x), layer(x + 7.5), atol=1e-4)


def test_constant_input_gives_zeros_without_nan():
    layer = LayerNormalization(5)
    x = torch.full((2, 5), 3.0)
    y = layer(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-4)


def test_parameters_have_expected_count():
    layer = LayerNormalization(10)
    total = sum(p.numel() for p in layer.parameters())
    assert total == 2 * 10


def test_parameters_are_trainable():
    layer = LayerNormalization(4)
    params = list(layer.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_gradient_flows_to_input():
    layer = LayerNormalization(4)
    x = torch.randn(2, 4, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_gradient_flows_to_parameters():
    layer = LayerNormalization(4)
    x = torch.randn(3, 4)
    (layer(x) * torch.arange(4, dtype=torch.float32)).sum().backward()
    for p in layer.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_repeated_forward_is_deterministic():
    layer = LayerNormalization(6)
    x = torch.randn(2, 6)
    assert torch.equal(layer(x), layer(x))


def test_train_and_eval_modes_give_same_output():
    layer = LayerNormalization(6)
    x = torch.randn(2, 6)
    layer.train()
    out_train = layer(x)
    layer.eval()
    out_eval = layer(x)
    assert torch.allclose(out_train, out_eval, atol=1e-6)


def test_works_with_float64():
    layer = LayerNormalization(4).double()
    x = torch.randn(3, 4, dtype=torch.float64)
    y = layer(x)
    assert y.dtype == torch.float64
    expected = F.layer_norm(x, normalized_shape=(4,), eps=1e-5)
    assert torch.allclose(y, expected, atol=1e-10)


def test_mismatched_last_dim_raises():
    layer = LayerNormalization(4)
    x = torch.randn(2, 5)
    with pytest.raises(RuntimeError):
        layer(x)


def test_state_dict_round_trip():
    src = LayerNormalization(4)
    dst = LayerNormalization(4)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(torch.randn_like(p))
    dst.load_state_dict(src.state_dict())
    x = torch.randn(2, 4)
    assert torch.allclose(src(x), dst(x), atol=1e-6)
