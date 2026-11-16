import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_layer_normalization import LayerNormalization


FEATURES = 8


@pytest.fixture
def layer():
    torch.manual_seed(0)
    return LayerNormalization(FEATURES)


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return torch.randn(4, FEATURES) * 3.0 + 2.0


def test_is_nn_module():
    assert issubclass(LayerNormalization, nn.Module)


def test_output_shape_2d(layer, batch):
    assert layer(batch).shape == batch.shape


def test_output_shape_3d(layer):
    x = torch.randn(2, 5, FEATURES)
    assert layer(x).shape == (2, 5, FEATURES)


def test_output_is_finite(layer, batch):
    assert torch.isfinite(layer(batch)).all()


def test_last_dim_has_zero_mean(layer, batch):
    out = layer(batch)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(batch.shape[0]), atol=1e-5)


def test_last_dim_has_unit_std(layer, batch):
    out = layer(batch)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(batch.shape[0]), atol=1e-3)


def test_matches_functional_layer_norm(layer, batch):
    weight = torch.ones(FEATURES)
    bias = torch.zeros(FEATURES)
    expected = F.layer_norm(batch, (FEATURES,), weight, bias, eps=1e-5)
    assert torch.allclose(layer(batch), expected, atol=1e-5)


def test_shift_invariance(layer, batch):
    out = layer(batch)
    shifted = layer(batch + 7.5)
    assert torch.allclose(out, shifted, atol=1e-4)


def test_scale_invariance(layer, batch):
    out = layer(batch)
    scaled = layer(batch * 4.0)
    assert torch.allclose(out, scaled, atol=1e-3)


def test_constant_input_gives_zero_output(layer):
    x = torch.full((3, FEATURES), 5.0)
    out = layer(x)
    assert torch.allclose(out, torch.zeros_like(x), atol=1e-4)


def test_rows_are_independent(layer, batch):
    full = layer(batch)
    single = layer(batch[1:2])
    assert torch.allclose(full[1:2], single, atol=1e-6)


def test_deterministic_repeated_calls(layer, batch):
    assert torch.equal(layer(batch), layer(batch))


def test_train_and_eval_modes_match(layer, batch):
    layer.train()
    out_train = layer(batch)
    layer.eval()
    out_eval = layer(batch)
    assert torch.allclose(out_train, out_eval, atol=1e-6)


def test_gradient_flows_to_input(layer, batch):
    x = batch.clone().requires_grad_(True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_parameters_are_trainable(layer):
    params = list(layer.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_parameter_count_matches_features(layer):
    total = sum(p.numel() for p in layer.parameters())
    assert total == 2 * FEATURES


def test_state_dict_roundtrip(layer, batch):
    torch.manual_seed(42)
    other = LayerNormalization(FEATURES)
    for p in other.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p))
    other.load_state_dict(layer.state_dict())
    assert torch.allclose(layer(batch), other(batch), atol=1e-6)


def test_mismatched_feature_size_raises(layer):
    wrong = torch.randn(4, FEATURES + 3)
    with pytest.raises(RuntimeError):
        layer(wrong)
