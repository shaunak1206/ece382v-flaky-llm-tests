# tests/test_layer_normalization.py
import math
import random
import pytest
import torch
import numpy as np

from torch_layer_normalization import LayerNormalization


def _seed_all():
    # Seed all RNGs used by the code under test to ensure reproducibility
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)


def test_default_construction_and_forward_shapes():
    _seed_all()
    # Test default initialization for a 2D input (batch, features)
    ln = LayerNormalization(normalized_shape=5)
    x = torch.randn(3, 5)
    y = ln(x)
    assert y.shape == x.shape
    # mean and std per sample (last dimension)
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    # output should have approx zero mean and unit variance (per sample)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_normalized_shape_tuple_and_tensor_input():
    _seed_all()
    ln = LayerNormalization(normalized_shape=(3, 4))
    x = torch.randn(2, 3, 4)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean(dim=(-2, -1))
    std = y.std(dim=(-2, -1), unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_gamma_beta_parameters_affine():
    _seed_all()
    normalized_shape = 7
    ln = LayerNormalization(normalized_shape, elementwise_affine=True)
    # gamma and beta should be parameters of correct shape
    assert isinstance(ln.weight, torch.nn.Parameter)
    assert ln.weight.shape == (normalized_shape,)
    assert isinstance(ln.bias, torch.nn.Parameter)
    assert ln.bias.shape == (normalized_shape,)

    x = torch.randn(4, normalized_shape)
    y = ln(x)
    # output shape same as input
    assert y.shape == x.shape


def test_no_affine_parameters_when_elementwise_affine_false():
    _seed_all()
    ln = LayerNormalization(4, elementwise_affine=False)
    # no weight or bias parameters when affine is False
    assert not hasattr(ln, "weight")
    assert not hasattr(ln, "bias")

    x = torch.randn(2, 4)
    y = ln(x)
    assert y.shape == x.shape


def test_forward_output_properties_with_affine(monkeypatch):
    _seed_all()
    # monkeypatch to fixed weight and bias to test their effect
    ln = LayerNormalization(3, elementwise_affine=True)

    with torch.no_grad():
        ln.weight.fill_(2.0)
        ln.bias.fill_(0.5)

    x = torch.randn(5, 3)
    y = ln(x)
    # mean and std along last dims before scaling
    mean_unscaled = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + ln.eps)
    # after scaling by weight=2 and shifting by bias=0.5
    expected = 2 * mean_unscaled + 0.5
    assert torch.allclose(y, expected, atol=1e-5)


def test_eps_effect_on_variance():
    _seed_all()
    eps_small = 1e-8
    eps_large = 1e-3
    ln_small_eps = LayerNormalization(5, eps=eps_small)
    ln_large_eps = LayerNormalization(5, eps=eps_large)
    x = torch.randn(4, 5)

    y_small = ln_small_eps(x)
    y_large = ln_large_eps(x)

    # With larger epsilon, variance is biased slightly towards 1/(1+eps),
    # so y_large std should be a bit smaller than 1 but close.
    std_small = y_small.std(dim=-1, unbiased=False)
    std_large = y_large.std(dim=-1, unbiased=False)

    assert torch.allclose(std_small, torch.ones_like(std_small), atol=1e-5)
    assert torch.all(std_large < 1.1)
    assert torch.all(std_large > 0.8)


def test_invalid_normalized_shape_type_raises():
    _seed_all()
    with pytest.raises(TypeError):
        LayerNormalization(normalized_shape="invalid")

    with pytest.raises(TypeError):
        LayerNormalization(normalized_shape=[1, "b"])


def test_zero_dimensional_input_raises():
    _seed_all()
    ln = LayerNormalization(1)
    x = torch.tensor(1.0)
    with pytest.raises(IndexError):
        ln(x)


def test_state_dict_and_load_state_dict():
    _seed_all()
    ln1 = LayerNormalization(3, elementwise_affine=True)
    ln2 = LayerNormalization(3, elementwise_affine=True)
    x = torch.randn(2, 3)
    _ = ln1(x)
    # Modify ln1 params to non-default values
    with torch.no_grad():
        ln1.weight.fill_(42)
        ln1.bias.fill_(-1)

    sd = ln1.state_dict()
    ln2.load_state_dict(sd)
    # ln2 params should now equal ln1's
    assert torch.allclose(ln2.weight, ln1.weight)
    assert torch.allclose(ln2.bias, ln1.bias)


def test_repr_contains_normalized_shape_and_affine_flag():
    _seed_all()
    ln = LayerNormalization((2, 3), elementwise_affine=True)
    r = repr(ln)
    assert "normalized_shape=(2, 3)" in r
    assert "elementwise_affine=True" in r


def test_forward_on_gpu_if_available():
    _seed_all()
    if torch.cuda.is_available():
        ln = LayerNormalization(4).cuda()
        x = torch.randn(3, 4).cuda()
        y = ln(x)
        assert y.device.type == "cuda"
        assert y.shape == x.shape


def test_backward_pass_and_gradients():
    _seed_all()
    ln = LayerNormalization(3, elementwise_affine=True)
    x = torch.randn(5, 3, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    # grad for input should exist
    assert x.grad is not None
    # grad for weight and bias should exist
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_layernorm_with_batch_and_sequence_dimensions():
    _seed_all()
    ln = LayerNormalization(6)
    x = torch.randn(10, 8, 6)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_layernorm_float64_precision():
    _seed_all()
    ln = LayerNormalization(5)
    x = torch.randn(3, 5, dtype=torch.float64)
    y = ln(x)
    assert y.dtype == torch.float64
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-10)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-10)


def test_layernorm_different_input_dtypes():
    _seed_all()
    ln = LayerNormalization(4)
    for dtype in [torch.float16, torch.float32]:
        x = torch.randn(7, 4, dtype=dtype)
        y = ln(x)
        assert y.dtype == dtype
        mean = y.mean(dim=-1)
        std = y.std(dim=-1, unbiased=False)
        # fp16 is less precise, allow looser tolerance
        atol = 1e-2 if dtype == torch.float16 else 1e-5
        assert torch.allclose(mean, torch.zeros_like(mean), atol=atol)
        assert torch.allclose(std, torch.ones_like(std), atol=atol)


def test_can_handle_singleton_batch_and_feature():
    _seed_all()
    ln = LayerNormalization(1)
    x = torch.randn(1, 1)
    y = ln(x)
    # Should not error and output shape matches input
    assert y.shape == x.shape


def test_inplace_modification_of_input_is_prevented():
    _seed_all()
    ln = LayerNormalization(3)
    x = torch.randn(2, 3)
    x_orig = x.clone()
    y = ln(x)
    # Input tensor should remain unchanged
    assert torch.allclose(x, x_orig)


def test_forward_with_non_contiguous_input():
    _seed_all()
    ln = LayerNormalization(2)
    x = torch.randn(4, 2, 3)
    x_t = x.transpose(1, 2)  # non-contiguous view
    y = ln(x_t)
    assert y.shape == x_t.shape
    # output mean and std along last dimension approximately 0 and 1
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_forward_with_large_values_stable():
    _seed_all()
    ln = LayerNormalization(3)
    x = torch.randn(5, 3) * 1e6
    y = ln(x)
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4)
