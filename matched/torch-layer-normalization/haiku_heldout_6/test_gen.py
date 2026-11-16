import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_layer_normalization import LayerNormalization, __version__


FEATURES = 4


@pytest.fixture
def layer():
    torch.manual_seed(0)
    return LayerNormalization((FEATURES,))


@pytest.fixture
def sample_input():
    torch.manual_seed(1)
    return torch.randn(8, 3, FEATURES)


def manual_layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__ == '0.11.0'


def test_is_nn_module(layer):
    assert isinstance(layer, nn.Module)


def test_output_shape_matches_input_shape(layer, sample_input):
    out = layer(sample_input)
    assert out.shape == sample_input.shape


def test_output_has_zero_mean_over_last_dim(layer, sample_input):
    out = layer(sample_input)
    means = out.mean(dim=-1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)


def test_output_has_unit_variance_over_last_dim(layer, sample_input):
    out = layer(sample_input)
    variances = out.var(dim=-1, unbiased=False)
    assert torch.allclose(variances, torch.ones_like(variances), atol=1e-3)


def test_matches_manual_reference(layer, sample_input):
    out = layer(sample_input)
    expected = manual_layer_norm(sample_input)
    assert torch.allclose(out, expected, atol=1e-5)


def test_matches_functional_layer_norm(layer, sample_input):
    out = layer(sample_input)
    expected = F.layer_norm(sample_input, (FEATURES,), eps=1e-5)
    assert torch.allclose(out, expected, atol=1e-5)


def test_samples_are_normalized_independently(layer, sample_input):
    batch_out = layer(sample_input)
    for i in range(sample_input.shape[0]):
        single_out = layer(sample_input[i:i + 1])
        assert torch.allclose(batch_out[i:i + 1], single_out, atol=1e-6)


def test_scale_invariance(layer, sample_input):
    out = layer(sample_input)
    scaled_out = layer(sample_input * 7.5)
    assert torch.allclose(out, scaled_out, atol=1e-4)


def test_shift_invariance(layer, sample_input):
    out = layer(sample_input)
    shifted_out = layer(sample_input + 3.25)
    assert torch.allclose(out, shifted_out, atol=1e-4)


def test_constant_input_gives_finite_output(layer):
    x = torch.full((2, 5, FEATURES), 2.0)
    out = layer(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)


def test_backward_produces_finite_input_gradients(layer, sample_input):
    x = sample_input.clone().requires_grad_(True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_input_gradients_match_functional_layer_norm(layer, sample_input):
    x1 = sample_input.clone().requires_grad_(True)
    weights = torch.linspace(-1.0, 1.0, steps=sample_input.numel()).reshape(sample_input.shape)
    (layer(x1) * weights).sum().backward()

    x2 = sample_input.clone().requires_grad_(True)
    (F.layer_norm(x2, (FEATURES,), eps=1e-5) * weights).sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-4)


def test_parameter_count_with_default_affine(layer):
    total = sum(p.numel() for p in layer.parameters())
    assert total == 2 * FEATURES


def test_three_dimensional_input_normalizes_last_dim():
    torch.manual_seed(2)
    layer = LayerNormalization((5,))
    x = torch.randn(2, 7, 5) * 10 + 4
    out = layer(x)
    assert out.shape == (2, 7, 5)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 7), atol=1e-5)


def test_output_is_deterministic(layer, sample_input):
    out1 = layer(sample_input)
    out2 = layer(sample_input)
    assert torch.equal(out1, out2)


def test_train_and_eval_modes_match(layer, sample_input):
    layer.train()
    out_train = layer(sample_input)
    layer.eval()
    out_eval = layer(sample_input)
    assert torch.allclose(out_train, out_eval, atol=1e-6)


def test_float64_dtype_is_preserved():
    layer = LayerNormalization((FEATURES,)).double()
    x = torch.randn(3, FEATURES, dtype=torch.float64)
    out = layer(x)
    assert out.dtype == torch.float64
    assert torch.allclose(out, manual_layer_norm(x), atol=1e-10)


def test_state_dict_roundtrip_preserves_output(layer, sample_input):
    other = LayerNormalization((FEATURES,))
    other.load_state_dict(layer.state_dict())
    assert torch.allclose(layer(sample_input), other(sample_input), atol=1e-6)


def test_mismatched_feature_size_raises(layer):
    bad_input = torch.randn(2, FEATURES + 1)
    with pytest.raises((RuntimeError, ValueError)):
        layer(bad_input)
