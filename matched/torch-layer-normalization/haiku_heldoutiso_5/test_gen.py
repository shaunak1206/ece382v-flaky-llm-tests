import pytest
import torch
import torch.nn as nn

from torch_layer_normalization import LayerNormalization
import torch_layer_normalization


FEATURES = 4


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture
def layer():
    return LayerNormalization(FEATURES)


@pytest.fixture
def batch():
    return torch.randn(3, FEATURES) * 2.0 + 5.0


def reference_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def test_version_string():
    assert torch_layer_normalization.__version__ == '0.11.0'


def test_exported_class_is_nn_module():
    assert issubclass(LayerNormalization, nn.Module)


def test_construct_with_int_and_tuple_shapes():
    a = LayerNormalization(FEATURES)
    b = LayerNormalization((FEATURES,))
    x = torch.randn(2, FEATURES)
    assert a(x).shape == b(x).shape == x.shape


def test_forward_preserves_2d_shape(layer):
    x = torch.randn(5, FEATURES)
    assert layer(x).shape == x.shape


def test_forward_preserves_3d_shape():
    layer = LayerNormalization(FEATURES)
    x = torch.randn(2, 7, FEATURES)
    assert layer(x).shape == (2, 7, FEATURES)


def test_output_has_zero_mean_over_last_dim(layer, batch):
    y = layer(batch)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(batch.shape[0]), atol=1e-4)


def test_output_has_unit_variance_over_last_dim(layer, batch):
    y = layer(batch)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(var, torch.ones(batch.shape[0]), atol=1e-3)


def test_matches_reference_computation(layer, batch):
    y = layer(batch)
    expected = reference_norm(batch)
    assert torch.allclose(y, expected, atol=1e-4)


def test_invariant_to_positive_scaling_of_input(layer, batch):
    y1 = layer(batch)
    y2 = layer(batch * 7.5)
    assert torch.allclose(y1, y2, atol=1e-3)


def test_invariant_to_constant_shift_of_input(layer, batch):
    y1 = layer(batch)
    y2 = layer(batch + 3.0)
    assert torch.allclose(y1, y2, atol=1e-4)


def test_normalizes_each_sample_independently(layer, batch):
    full = layer(batch)
    single = layer(batch[1:2])
    assert torch.allclose(full[1:2], single, atol=1e-6)


def test_constant_input_gives_finite_output(layer):
    x = torch.ones(2, FEATURES)
    y = layer(x)
    assert torch.isfinite(y).all()
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-4)


def test_parameters_are_trainable_and_sized_to_features(layer):
    params = list(layer.parameters())
    assert len(params) > 0
    for p in params:
        assert p.requires_grad
        assert p.shape == (FEATURES,)


def test_default_parameter_count_is_two_feature_vectors(layer):
    total = sum(p.numel() for p in layer.parameters())
    assert total == 2 * FEATURES


def test_zero_parameters_give_zero_output(layer, batch):
    with torch.no_grad():
        for p in layer.parameters():
            p.zero_()
    y = layer(batch)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_unit_parameters_add_one_to_normalized_output(layer, batch):
    # With gamma=1 and beta=1 the output is normalized * 1 + 1.
    with torch.no_grad():
        for p in layer.parameters():
            p.fill_(1.0)
    y = layer(batch)
    expected = reference_norm(batch) + 1.0
    assert torch.allclose(y, expected, atol=1e-4)


def test_backward_produces_gradients(layer, batch):
    x = batch.clone().requires_grad_(True)
    loss = layer(x).pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for p in layer.parameters():
        assert p.grad is not None


def test_gradcheck_in_double_precision():
    layer = LayerNormalization(FEATURES).double()
    x = torch.randn(2, FEATURES, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)


def test_train_and_eval_modes_match(layer, batch):
    layer.train()
    y_train = layer(batch)
    layer.eval()
    y_eval = layer(batch)
    assert torch.allclose(y_train, y_eval, atol=1e-6)


def test_mismatched_feature_dimension_raises(layer):
    x = torch.randn(2, FEATURES + 1)
    with pytest.raises((RuntimeError, ValueError)):
        layer(x)
