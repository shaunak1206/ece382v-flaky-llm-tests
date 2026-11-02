import pytest
import torch
import math
from torch_layer_normalization.layer_normalization import LayerNormalization


def test_layernorm_default_behavior():
    # default LayerNormalization should have normalized_shape=None, elementwise_affine=True
    # but normalized_shape is required positional argument, so must provide.
    # Test default settings with normalized_shape=3
    ln = LayerNormalization(3)
    assert ln.normalized_shape == (3,)
    assert ln.elementwise_affine is True
    # weight and bias initialized to ones and zeros respectively, shape matches normalized_shape
    assert ln.weight.shape == (3,)
    assert torch.allclose(ln.weight, torch.ones(3))
    assert ln.bias.shape == (3,)
    assert torch.allclose(ln.bias, torch.zeros(3))


def test_layernorm_forward_zero():
    # input tensor of zeros, output should be zeros because (x-mean)/std == 0, bias == 0
    ln = LayerNormalization(2)
    x = torch.zeros(4, 2)
    y = ln(x)
    assert torch.allclose(y, torch.zeros_like(x), atol=1e-7)


def test_layernorm_forward_constant_input():
    # input tensor all same value, output should be zeros (since variance=0, use epsilon)
    ln = LayerNormalization(2)
    x = torch.full((5, 2), 3.14)
    y = ln(x)
    assert y.shape == x.shape
    assert torch.allclose(y, torch.zeros_like(x), atol=1e-6)


def test_layernorm_forward_known_values():
    # Check output against manual calculation
    ln = LayerNormalization(3, eps=1e-5)
    ln.weight.data.fill_(1.0)
    ln.bias.data.fill_(0.0)
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 3.0, 4.0]])
    # mean per sample: [2.0, 3.0], var per sample: [0.6666667, 0.6666667]
    expected = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, unbiased=False, keepdim=True) + ln.eps)
    y = ln(x)
    assert torch.allclose(y, expected, atol=1e-6)


def test_layernorm_forward_with_affine_parameters():
    # test with weight and bias different from default
    ln = LayerNormalization(2)
    ln.weight.data.fill_(2.0)
    ln.bias.data.fill_(1.0)
    x = torch.tensor([[1.0, 3.0], [4.0, 2.0]])
    y = ln(x)
    # Expected: normalized output * 2 + 1
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    expected = ((x - mean) / torch.sqrt(var + ln.eps)) * 2.0 + 1.0
    assert torch.allclose(y, expected, atol=1e-6)


def test_layernorm_no_affine():
    # elementwise_affine=False disables learnable params, output normalized only
    ln = LayerNormalization(3, elementwise_affine=False)
    assert not hasattr(ln, 'weight')
    assert not hasattr(ln, 'bias')
    x = torch.randn(4, 3)
    y = ln(x)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    expected = (x - mean) / torch.sqrt(var + ln.eps)
    assert torch.allclose(y, expected, atol=1e-6)


def test_layernorm_different_normalized_shape():
    # normalized_shape can be tuple or int
    for shape in [(3,), 3, (2, 3)]:
        ln = LayerNormalization(shape)
        if isinstance(shape, int):
            assert ln.normalized_shape == (shape,)
        else:
            assert ln.normalized_shape == shape
        # Check weight shape matches normalized_shape
        expected_num = 1
        for d in ln.normalized_shape:
            expected_num *= d
        assert ln.weight.numel() == expected_num
        # Forward pass shape test
        input_shape = (4,) + ln.normalized_shape
        x = torch.randn(input_shape)
        y = ln(x)
        assert y.shape == x.shape


def test_layernorm_eps_small_and_large():
    # eps affects numerical stability but output shape and stats remain consistent
    ln_small_eps = LayerNormalization(4, eps=1e-12)
    ln_large_eps = LayerNormalization(4, eps=1e-1)
    x = torch.randn(5, 4)
    y_small = ln_small_eps(x)
    y_large = ln_large_eps(x)
    # Both outputs have same shape
    assert y_small.shape == x.shape
    assert y_large.shape == x.shape
    # Outputs differ numerically but not by huge margin
    diff = torch.abs(y_small - y_large).mean().item()
    assert diff < 10  # very loose bound


def test_layernorm_backward_computation():
    # test backward pass gradients produce expected gradients flow
    ln = LayerNormalization(3)
    x = torch.randn(2, 3, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    # Gradients should be finite
    assert torch.isfinite(x.grad).all()


def test_layernorm_weight_bias_gradients():
    ln = LayerNormalization(3)
    x = torch.randn(5, 3, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    # weight and bias gradients exist and are finite
    assert ln.weight.grad is not None
    assert torch.isfinite(ln.weight.grad).all()
    assert ln.bias.grad is not None
    assert torch.isfinite(ln.bias.grad).all()


def test_layernorm_reset_parameters_idempotent():
    ln = LayerNormalization(4)
    # Modify parameters
    ln.weight.data.fill_(10.0)
    ln.bias.data.fill_(5.0)
    ln.reset_parameters()
    assert torch.allclose(ln.weight, torch.ones_like(ln.weight))
    assert torch.allclose(ln.bias, torch.zeros_like(ln.bias))
    # Calling again shouldn't break anything
    ln.reset_parameters()
    assert torch.allclose(ln.weight, torch.ones_like(ln.weight))
    assert torch.allclose(ln.bias, torch.zeros_like(ln.bias))


def test_layernorm_repr_contains_expected_info():
    ln = LayerNormalization(2, eps=1e-4, elementwise_affine=False)
    r = repr(ln)
    assert "LayerNormalization" in r
    assert "normalized_shape=(2,)" in r
    assert "eps=0.0001" in r or "eps=1e-4" in r  # string representation can vary but should include value
    assert "elementwise_affine=False" in r


def test_layernorm_forward_high_dim_input():
    # normalized_shape corresponds to last N dims, so test with 3D input and normalized_shape 2 dims
    ln = LayerNormalization((2, 3))
    x = torch.randn(4, 5, 2, 3)
    y = ln(x)
    assert y.shape == x.shape
    # output means over last dims per sample should be close to zero
    mean = y.mean(dim=(-2, -1))
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    # output variance over last dims should be close to one
    var = y.var(dim=(-2, -1), unbiased=False)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


def test_layernorm_forward_with_requires_grad_false_input():
    # input tensor without requires_grad should still produce output without requires_grad
    ln = LayerNormalization(3)
    x = torch.randn(4, 3)
    y = ln(x)
    # output requires_grad equals input requires_grad (both False)
    assert y.requires_grad is False


def test_layernorm_forward_preserves_dtype_and_device():
    ln = LayerNormalization(3)
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(2, 3, dtype=dtype)
        y = ln(x)
        assert y.dtype == dtype
    # test on cpu at least
    x_cpu = torch.randn(2, 3)
    y_cpu = ln(x_cpu)
    assert y_cpu.device == x_cpu.device


def test_layernorm_gradients_do_not_accumulate_across_forward_calls():
    ln = LayerNormalization(3)
    x = torch.randn(2, 3, requires_grad=True)
    y1 = ln(x)
    y1.sum().backward()
    grad1 = x.grad.clone()
    x.grad.zero_()
    y2 = ln(x)
    y2.sum().backward()
    grad2 = x.grad.clone()
    assert torch.allclose(grad1, grad2, atol=1e-7)


def test_layernorm_does_not_change_parameters_after_forward():
    ln = LayerNormalization(3)
    old_weight = ln.weight.clone()
    old_bias = ln.bias.clone()
    x = torch.randn(4, 3)
    _ = ln(x)
    assert torch.allclose(ln.weight, old_weight)
    assert torch.allclose(ln.bias, old_bias)


def test_layernorm_invalid_normalized_shape_raises():
    # Passing 0 or negative number or invalid type should raise TypeError or ValueError
    with pytest.raises(TypeError):
        LayerNormalization('invalid')
    with pytest.raises(ValueError):
        LayerNormalization(0)
    with pytest.raises(ValueError):
        LayerNormalization(-1)


def test_layernorm_gradient_check():
    # test numeric gradient approx via torch.autograd.gradcheck with double precision
    ln = LayerNormalization(3).double()
    x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(ln, (x,), raise_exception=True)


def test_layernorm_inplace_operations_not_supported():
    # ensure forward output tensor does not share storage with input (no inplace modification)
    ln = LayerNormalization(3)
    x = torch.randn(4, 3)
    y = ln(x)
    # Changing output does not affect input
    y.add_(1.0)
    assert not torch.allclose(x, y)


def test_layernorm_forward_multiple_batches_same_output_properties():
    # For multiple batches, output mean close to 0 and var close to 1 per sample
    ln = LayerNormalization(4)
    x = torch.randn(10, 4)
    y = ln(x)
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)
