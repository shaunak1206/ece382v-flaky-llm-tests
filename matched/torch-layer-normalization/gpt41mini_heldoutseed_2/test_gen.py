import math
import torch
import pytest
from torch_layer_normalization import LayerNormalization

@pytest.mark.parametrize("normalized_shape", [1, 3, 7])
def test_forward_shape_and_type(normalized_shape):
    ln = LayerNormalization(normalized_shape)
    x = torch.randn(2, normalized_shape, 5)
    y = ln(x)
    assert y.shape == x.shape
    assert isinstance(y, torch.Tensor)


def test_forward_on_2d_tensor():
    normalized_shape = 4
    ln = LayerNormalization(normalized_shape)
    x = torch.randn(10, normalized_shape)
    y = ln(x)
    assert y.shape == x.shape


def test_forward_on_1d_tensor():
    normalized_shape = 3
    ln = LayerNormalization(normalized_shape)
    x = torch.randn(normalized_shape)
    y = ln(x)
    assert y.shape == x.shape


def test_default_parameters():
    normalized_shape = 5
    ln = LayerNormalization(normalized_shape)
    # weight and bias should be parameter tensors of correct size
    assert ln.weight.shape == (normalized_shape,)
    assert ln.bias.shape == (normalized_shape,)
    # weight initialized to ones, bias to zeros
    assert torch.allclose(ln.weight, torch.ones(normalized_shape))
    assert torch.allclose(ln.bias, torch.zeros(normalized_shape))


def test_forward_normalizes_mean_zero_and_var_one():
    normalized_shape = 6
    ln = LayerNormalization(normalized_shape, eps=1e-5)
    x = torch.randn(4, normalized_shape)
    y = ln(x)
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    # mean close to zero and var close to one
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_forward_with_different_eps():
    normalized_shape = 3
    eps = 1e-3
    ln = LayerNormalization(normalized_shape, eps=eps)
    x = torch.randn(7, normalized_shape)
    y = ln(x)
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_forward_with_elementwise_affine_false():
    normalized_shape = 4
    ln = LayerNormalization(normalized_shape, elementwise_affine=False)
    x = torch.randn(10, normalized_shape)
    y = ln(x)
    # weight and bias should be None
    assert ln.weight is None
    assert ln.bias is None
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    # mean close to zero and var close to one
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_parameters_requires_grad():
    normalized_shape = 5
    ln = LayerNormalization(normalized_shape)
    # weight and bias require grad by default
    assert ln.weight.requires_grad
    assert ln.bias.requires_grad


def test_parameters_dont_exist_when_elementwise_affine_false():
    normalized_shape = 3
    ln = LayerNormalization(normalized_shape, elementwise_affine=False)
    assert not hasattr(ln, "weight") or ln.weight is None
    assert not hasattr(ln, "bias") or ln.bias is None


def test_forward_with_high_dim_input():
    normalized_shape = 4
    ln = LayerNormalization(normalized_shape)
    x = torch.randn(2, 3, 5, normalized_shape)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_forward_preserves_dtype():
    normalized_shape = 4
    ln = LayerNormalization(normalized_shape)
    x = torch.randn(2, normalized_shape).double()
    y = ln(x)
    assert y.dtype == torch.float64


def test_forward_with_requires_grad_input():
    normalized_shape = 4
    ln = LayerNormalization(normalized_shape)
    x = torch.randn(2, normalized_shape, requires_grad=True)
    y = ln(x)
    y.mean().backward()
    assert x.grad is not None


def test_invalid_normalized_shape_type():
    with pytest.raises(TypeError):
        LayerNormalization('invalid')  # string instead of int or tuple


def test_invalid_normalized_shape_value():
    # Negative or zero normalized shape
    with pytest.raises(ValueError):
        LayerNormalization(0)
    with pytest.raises(ValueError):
        LayerNormalization(-3)


def test_forward_with_integer_tensor():
    normalized_shape = 3
    ln = LayerNormalization(normalized_shape)
    # integers are generally not supported by LayerNorm, but let's test that it errors
    x = torch.randint(0, 10, (2, normalized_shape))
    with pytest.raises(RuntimeError):
        ln(x)


def test_repr_returns_expected_string():
    normalized_shape = 3
    ln = LayerNormalization(normalized_shape)
    r = repr(ln)
    assert "LayerNormalization" in r
    assert str(normalized_shape) in r


def test_forward_consistency_between_calls():
    normalized_shape = 5
    ln = LayerNormalization(normalized_shape)
    x = torch.randn(2, normalized_shape)
    y1 = ln(x)
    y2 = ln(x)
    # For the same input and parameters y should be the same
    assert torch.allclose(y1, y2)


def test_forward_with_cuda_if_available():
    if torch.cuda.is_available():
        normalized_shape = 4
        ln = LayerNormalization(normalized_shape).cuda()
        x = torch.randn(2, normalized_shape).cuda()
        y = ln(x)
        assert y.device.type == 'cuda'
        mean = y.mean(dim=-1)
        var = y.var(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)
