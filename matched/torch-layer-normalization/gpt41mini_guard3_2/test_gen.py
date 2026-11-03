import pytest
import torch
import torch_layer_normalization
import random
import numpy as np


@pytest.fixture(autouse=True)
def _reset_seed():
    # This runs before every test for global RNG reset if needed.
    # Tests will set within each test anyway.
    yield
    # no reset needed after test


def _seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_layernorm_basic_forward_shape_and_dtype():
    _seed_all()
    # input: (batch, d)
    x = torch.randn(10, 5, dtype=torch.float32)
    ln = torch_layer_normalization.LayerNormalization(5)

    y = ln(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_layernorm_forward_two_different_inputs_produce_different_returns():
    _seed_all()
    x1 = torch.randn(6, 7)
    x2 = torch.randn(6, 7)
    ln = torch_layer_normalization.LayerNormalization(7)

    y1 = ln(x1)
    y2 = ln(x2)

    # Outputs should not be identical for different inputs
    assert not torch.allclose(y1, y2)


def test_layernorm_output_mean_zero_and_variance_one_per_sample():
    _seed_all()
    # Input shape arbitrary
    x = torch.randn(8, 10, dtype=torch.float64)
    ln = torch_layer_normalization.LayerNormalization(10)

    y = ln(x)

    # Check mean close to zero and variance close to one along last dim, per sample
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-6)


def test_layernorm_with_different_normalized_shape():
    _seed_all()
    # normalized_shape can be tuple
    x = torch.randn(4, 3, 5)
    normalized_shape = (3, 5)
    ln = torch_layer_normalization.LayerNormalization(normalized_shape)

    y = ln(x)
    assert y.shape == x.shape


def test_layernorm_with_affine_off_means_weight_bias_not_applied():
    _seed_all()
    x = torch.randn(2, 4)
    ln = torch_layer_normalization.LayerNormalization(4, elementwise_affine=False)

    y = ln(x)
    # Since affine off, weight and bias attributes should not be present as parameters
    assert not hasattr(ln, 'weight') or ln.weight is None
    assert not hasattr(ln, 'bias') or ln.bias is None

    # Output should still have zero mean and unit var per sample
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-6)


def test_layernorm_weight_and_bias_parameters_present_when_affine_true():
    _seed_all()
    ln = torch_layer_normalization.LayerNormalization(3, elementwise_affine=True)

    assert isinstance(ln.weight, torch.nn.Parameter)
    assert isinstance(ln.bias, torch.nn.Parameter)
    # Defaults weight close to ones, bias close to zeros
    assert torch.allclose(ln.weight.data, torch.ones(3), atol=1e-6)
    assert torch.allclose(ln.bias.data, torch.zeros(3), atol=1e-6)


def test_layernorm_forward_is_deterministic_with_seed():
    _seed_all(123)
    x = torch.randn(5, 5)
    ln = torch_layer_normalization.LayerNormalization(5)

    y1 = ln(x)

    _seed_all(123)
    x2 = torch.randn(5, 5)
    ln2 = torch_layer_normalization.LayerNormalization(5)
    y2 = ln2(x2)

    # Since seed is same and ln initialized freshly, outputs for same input should match
    # But x and x2 are different draws, so y1 and y2 different inputs for different tensors
    # So let's compare y1 and y2 only if x and x2 same:
    assert not torch.allclose(x, x2) or torch.allclose(y1, y2)


def test_layernorm_forward_preserves_dtype_on_double_and_float():
    _seed_all()
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(3, 7, dtype=dtype)
        ln = torch_layer_normalization.LayerNormalization(7)
        y = ln(x)
        assert y.dtype == dtype
        # mean ~0 variance ~1 on double and float
        mean = y.mean(dim=-1)
        var = y.var(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-6)


def test_layernorm_forward_with_large_input_values():
    _seed_all()
    # Large magnitude inputs
    x = torch.randn(5, 3) * 1e6
    ln = torch_layer_normalization.LayerNormalization(3)
    y = ln(x)

    # Check mean ~0 and var ~1 even at large scale
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_layernorm_consistent_with_manual_layernorm():
    _seed_all()
    x = torch.randn(10, 4)
    normalized_shape = 4
    ln = torch_layer_normalization.LayerNormalization(normalized_shape)

    # manual layer norm calculation:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    eps = ln.eps
    expected = (x - mean) / torch.sqrt(var + eps)
    if ln.elementwise_affine:
        expected = expected * ln.weight + ln.bias

    actual = ln(x)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_layernorm_invalid_normalized_shape_raises():
    # normalized_shape must be int or tuple of ints
    with pytest.raises(TypeError):
        torch_layer_normalization.LayerNormalization('invalid')

    with pytest.raises(TypeError):
        torch_layer_normalization.LayerNormalization([1, 2, 3])  # list not accepted


def test_layernorm_eps_property():
    _seed_all()
    ln = torch_layer_normalization.LayerNormalization(5)
    # eps should be a small float number > 0
    assert isinstance(ln.eps, float)
    assert ln.eps > 0
    assert abs(ln.eps - 1e-5) < 1e-6


def test_layernorm_weight_and_bias_are_trainable_parameters():
    _seed_all()
    ln = torch_layer_normalization.LayerNormalization(3, elementwise_affine=True)
    pnames = {p[0] for p in ln.named_parameters()}
    assert 'weight' in pnames and 'bias' in pnames


def test_layernorm_no_parameters_when_affine_false():
    _seed_all()
    ln = torch_layer_normalization.LayerNormalization(3, elementwise_affine=False)
    pnames = list(ln.named_parameters())
    assert len(pnames) == 0


def test_layernorm_state_dict_contains_correct_keys():
    _seed_all()
    ln = torch_layer_normalization.LayerNormalization(4, elementwise_affine=True)
    sd = ln.state_dict()
    # keys: weight, bias
    assert 'weight' in sd
    assert 'bias' in sd

    ln2 = torch_layer_normalization.LayerNormalization(4, elementwise_affine=False)
    sd2 = ln2.state_dict()
    # no weight or bias keys
    assert 'weight' not in sd2
    assert 'bias' not in sd2


def test_layernorm_repr_contains_classname_and_normalized_shape():
    _seed_all()
    normalized_shape = (2, 3)
    ln = torch_layer_normalization.LayerNormalization(normalized_shape)
    r = repr(ln)
    assert 'LayerNormalization' in r
    for n in (str(normalized_shape), 'eps', 'elementwise_affine'):
        assert n in r


def test_layernorm_forward_single_element_tensor():
    _seed_all()
    x = torch.tensor([[3.14]])
    ln = torch_layer_normalization.LayerNormalization(1)
    y = ln(x)
    # With single element variance zero, should not produce NaNs
    assert torch.isfinite(y).all()


def test_layernorm_affine_parameters_can_be_updated():
    _seed_all()
    ln = torch_layer_normalization.LayerNormalization(3, elementwise_affine=True)
    optimizer = torch.optim.SGD(ln.parameters(), lr=0.1)

    x = torch.randn(10, 3)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # Check parameters changed from initial values (weight ~1, bias ~0)
    weight_diff = (ln.weight.data - torch.ones(3)).abs().sum()
    bias_diff = (ln.bias.data - torch.zeros(3)).abs().sum()
    assert weight_diff > 0 or bias_diff > 0


def test_layernorm_forward_with_multidimensional_normalized_shape_and_input():
    _seed_all()
    x = torch.randn(2, 3, 4, 5)
    normalized_shape = (4, 5)
    ln = torch_layer_normalization.LayerNormalization(normalized_shape)
    y = ln(x)

    assert y.shape == x.shape
    # mean and var computed along last 2 dims
    mean = y.mean(dim=(-2, -1))
    var = y.var(dim=(-2, -1), unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-6)
