import pytest
import torch
import numpy as np

from torch_layer_normalization import LayerNormalization


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_output_shape(dtype):
    torch.manual_seed(42)
    np.random.seed(42)
    # Create input tensor of batch size 4 with 10 features
    x = torch.randn(4, 10, dtype=dtype)
    layer_norm = LayerNormalization(10)
    out = layer_norm(x)
    assert out.shape == x.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_output_dtype(dtype):
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(2, 5, dtype=dtype)
    layer_norm = LayerNormalization(5)
    out = layer_norm(x)
    assert out.dtype == dtype


@pytest.mark.parametrize("eps", [1e-5, 1e-3, 1e-9])
def test_forward_eps_effect(eps):
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(3, 8)
    ln1 = LayerNormalization(8, eps=eps)
    ln2 = LayerNormalization(8, eps=1e-12)
    out1 = ln1(x)
    out2 = ln2(x)
    # When eps is larger, output can slightly differ
    diff = (out1 - out2).abs().max()
    assert diff >= 0  # diff is non-negative
    # The outputs should be finite
    assert torch.isfinite(out1).all()
    assert torch.isfinite(out2).all()


def test_forward_mean_zero_variance_one():
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(10, 7)
    ln = LayerNormalization(7)
    out = ln(x)
    # Verify mean and variance along last dimension almost 0 and 1 respectively
    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-6)


def test_with_learnable_parameters():
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(5, 4)
    ln = LayerNormalization(4)
    # By default weight and bias are learnable and initialized to ones and zeros
    assert isinstance(ln.weight, torch.nn.Parameter)
    assert isinstance(ln.bias, torch.nn.Parameter)
    out = ln(x)
    # weight and bias have correct shape
    assert ln.weight.shape == (4,)
    assert ln.bias.shape == (4,)
    # Output after applying scale (weight) and shift (bias)
    # Change weight and bias and check output changes accordingly
    with torch.no_grad():
        ln.weight.fill_(2.0)
        ln.bias.fill_(3.0)
    out2 = ln(x)
    # out2 should be roughly out * 2 + 3
    expected = out * 2 + 3
    assert torch.allclose(out2, expected, atol=1e-6)


def test_invalid_normalized_shape_raises():
    # normalized_shape must be int or tuple of int > 0
    with pytest.raises(TypeError):
        LayerNormalization('a')
    with pytest.raises(ValueError):
        LayerNormalization(0)
    with pytest.raises(TypeError):
        LayerNormalization([2, 'b'])


def test_normalized_shape_tuple_and_int_equivalence():
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(3, 5)
    ln1 = LayerNormalization(5)
    ln2 = LayerNormalization((5,))
    out1 = ln1(x)
    out2 = ln2(x)
    assert torch.allclose(out1, out2, atol=1e-7)


def test_layer_norm_on_multi_dimensional_input():
    torch.manual_seed(42)
    np.random.seed(42)
    # 3D input (batch_size, seq_len, features)
    x = torch.randn(2, 4, 6)
    ln = LayerNormalization(6)
    out = ln(x)
    # Confirm shape unchanged
    assert out.shape == x.shape
    # Check mean and variance along last dim for each element in (batch_size, seq_len)
    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-6)


def test_weight_and_bias_requires_grad_true():
    ln = LayerNormalization(3)
    assert ln.weight.requires_grad
    assert ln.bias.requires_grad


def test_functional_equivalence_to_manual_layer_norm():
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(10, 7)
    ln = LayerNormalization(7)
    # baseline output
    out = ln(x)
    # manual computation: (x - mean) / sqrt(var + eps) * weight + bias
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    eps = ln.eps
    manual_norm = (x - mean) / torch.sqrt(var + eps)
    manual_out = manual_norm * ln.weight + ln.bias
    assert torch.allclose(out, manual_out, atol=1e-6)


def test_reset_parameters_sets_weight_and_bias_to_default():
    ln = LayerNormalization(4)
    with torch.no_grad():
        ln.weight.fill_(5.0)
        ln.bias.fill_(-3.0)
    ln.reset_parameters()
    # after reset, weight should be ones and bias zeros
    assert torch.allclose(ln.weight, torch.ones_like(ln.weight))
    assert torch.allclose(ln.bias, torch.zeros_like(ln.bias))


def test_no_grad_forward_does_not_track_grad():
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(2, 3, requires_grad=True)
    ln = LayerNormalization(3)
    with torch.no_grad():
        out = ln(x)
        assert out.requires_grad is False


def test_layer_norm_output_is_deterministic_for_same_seed():
    torch.manual_seed(1234)
    np.random.seed(1234)
    x = torch.randn(5, 5)
    ln = LayerNormalization(5)
    out1 = ln(x)
    torch.manual_seed(1234)
    np.random.seed(1234)
    x = torch.randn(5, 5)
    ln2 = LayerNormalization(5)
    out2 = ln2(x)
    assert torch.allclose(out1, out2, atol=1e-7)


def test_layer_norm_can_handle_batch_size_1():
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(1, 10)
    ln = LayerNormalization(10)
    out = ln(x)
    assert out.shape == (1, 10)


def test_gradients_exist_after_backward():
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(3, 4, requires_grad=True)
    ln = LayerNormalization(4)
    out = ln(x)
    out.sum().backward()
    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_repr_contains_class_name_and_shape():
    ln = LayerNormalization(7)
    r = repr(ln)
    assert "LayerNormalization" in r
    # Should mention normalized shape
    assert str(7) in r or str((7,)) in r


def test_double_backward_works_correctly():
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(2, 6, requires_grad=True)
    ln = LayerNormalization(6)
    out = ln(x)
    grad_outputs = torch.ones_like(out)
    grad = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=grad_outputs, create_graph=True)[0]
    grad_grad = torch.autograd.grad(outputs=grad.sum(), inputs=x)[0]
    assert grad_grad.shape == x.shape


def test_layer_norm_with_non_contiguous_input():
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(4, 5)
    x_t = x.t()  # transpose to make non-contiguous
    assert not x_t.is_contiguous()
    ln = LayerNormalization(4)
    out = ln(x_t)
    # output shape same as input
    assert out.shape == x_t.shape
    # Still normalizes last dimension
    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-6)
