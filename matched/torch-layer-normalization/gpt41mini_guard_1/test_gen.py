import pytest
import torch
import math
from torch_layer_normalization import LayerNormalization


def test_default_initialization():
    ln = LayerNormalization(10)
    assert ln.normalized_shape == (10,)
    assert ln.eps == 1e-5
    assert ln.elementwise_affine is True
    # weight and bias should be initialized properly
    assert ln.weight.shape == (10,)
    assert ln.bias.shape == (10,)
    # weight should be ones and bias zeros initially
    assert torch.allclose(ln.weight, torch.ones(10))
    assert torch.allclose(ln.bias, torch.zeros(10))


def test_initialization_without_affine():
    ln = LayerNormalization(7, elementwise_affine=False)
    assert ln.weight is None
    assert ln.bias is None


def test_forward_shape_and_dtype():
    ln = LayerNormalization(5)
    x = torch.randn(3, 5)
    y = ln(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_forward_correctness_simple_input():
    # LayerNorm should normalize last dimension to mean=0 and variance=1
    torch.manual_seed(0)
    x = torch.randn(4, 6, 10)
    ln = LayerNormalization(10, elementwise_affine=False, eps=1e-6)
    y = ln(x)
    # Compute mean and var along last dim
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    # mean and var should be approx 0 and 1, shape (4,6)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6, rtol=0)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-6, rtol=0)


def test_forward_with_affine_scaling_and_bias():
    ln = LayerNormalization(3)
    # Set weight and bias to specific values
    with torch.no_grad():
        ln.weight.copy_(torch.tensor([2.0, 3.0, 4.0]))
        ln.bias.copy_(torch.tensor([1.0, -1.0, 0.5]))
    x = torch.randn(2, 3)
    # Manually compute normalized output to check affine transform
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    std = torch.sqrt(var + ln.eps)
    expected = (x - mean) / std
    expected = expected * ln.weight + ln.bias
    out = ln(x)
    assert torch.allclose(out, expected, atol=1e-7, rtol=0)


def test_forward_eps_effect():
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    ln1 = LayerNormalization(4, eps=1e-5, elementwise_affine=False)
    ln2 = LayerNormalization(4, eps=1e-1, elementwise_affine=False)
    y1 = ln1(x)
    y2 = ln2(x)
    # Since eps affects denominator, outputs should differ
    assert not torch.allclose(y1, y2, atol=1e-3)
    # Check y1 closer to normalized with smaller epsilon than y2
    mean1 = y1.mean(dim=-1)
    mean2 = y2.mean(dim=-1)
    var1 = y1.var(dim=-1, unbiased=False)
    var2 = y2.var(dim=-1, unbiased=False)
    assert torch.allclose(mean1, torch.zeros_like(mean1), atol=1e-5, rtol=0)
    assert torch.allclose(mean2, torch.zeros_like(mean2), atol=1e-5, rtol=0)
    assert torch.abs(var1 - 1) < torch.abs(var2 - 1)


def test_forward_input_different_shapes():
    for shape in [(1,), (2, 5), (3, 7, 11), (4, 6, 8, 3)]:
        normalized_shape = shape[-1]
        ln = LayerNormalization(normalized_shape)
        x = torch.randn(*shape)
        y = ln(x)
        assert y.shape == x.shape
        # mean and var along last dimension approx 0 and 1
        mean = y.mean(dim=-1)
        var = y.var(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5, rtol=0)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5, rtol=0)


def test_forward_with_different_dtypes():
    for dtype in [torch.float32, torch.float64, torch.float16]:
        # LayerNorm for float16 may lose precision but should not error
        x = torch.randn(3, 8, dtype=dtype)
        ln = LayerNormalization(8)
        ln = ln.to(dtype)
        y = ln(x)
        assert y.shape == x.shape
        # Check dtype is preserved
        assert y.dtype == dtype


def test_layernorm_parameters_autograd():
    ln = LayerNormalization(4)
    x = torch.randn(2, 4, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    if ln.elementwise_affine:
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None


def test_layernorm_no_affine_no_grad():
    ln = LayerNormalization(3, elementwise_affine=False)
    x = torch.randn(2, 3, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert ln.weight is None
    assert ln.bias is None


def test_repr_contains_expected_strings():
    ln = LayerNormalization(5, eps=1e-4)
    rep = repr(ln)
    assert "LayerNormalization" in rep
    assert "normalized_shape=(5,)" in rep
    assert "eps=0.0001" in rep


def test_state_dict_and_load_state_dict():
    ln1 = LayerNormalization(6)
    with torch.no_grad():
        ln1.weight.mul_(2.0)
        ln1.bias.add_(3.0)
    sd = ln1.state_dict()
    ln2 = LayerNormalization(6)
    ln2.load_state_dict(sd)
    for k in sd:
        assert torch.allclose(sd[k], ln2.state_dict()[k], atol=1e-7, rtol=0)


def test_layernorm_with_single_integer_normalized_shape():
    # normalized_shape can be int per PyTorch API interface
    ln = LayerNormalization(4)
    x = torch.randn(2, 4)
    y = ln(x)
    assert y.shape == x.shape


def test_layernorm_with_tuple_normalized_shape():
    ln = LayerNormalization((3, 2))
    x = torch.randn(1, 3, 2)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean(dim=(-2, -1))
    var = y.var(dim=(-2, -1), unbiased=False)
    # mean approx 0 and var approx 1
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


def test_missing_cuda_device_returns_same_device_cpu():
    # If CUDA unavailable or input on CPU, output should be on CPU
    ln = LayerNormalization(5)
    x = torch.randn(2, 5)
    y = ln(x)
    assert y.device == x.device == torch.device("cpu")


def test_layernorm_handles_zero_dimensional_input():
    ln = LayerNormalization(1)
    x = torch.tensor(0.5)
    with pytest.raises(RuntimeError):
        ln(x)  # LayerNorm expects input of shape at least (normalized_shape)


def test_inplace_modification_of_input_not_allowed():
    ln = LayerNormalization(3)
    x = torch.randn(2, 3)
    x_clone = x.clone()
    y = ln(x)
    # input should not be modified inplace
    assert torch.allclose(x, x_clone)


@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_layernorm_no_change_to_weight_and_bias_after_forward(elementwise_affine):
    ln = LayerNormalization(4, elementwise_affine=elementwise_affine)
    x = torch.randn(2, 4)
    before_weight = ln.weight.clone() if ln.weight is not None else None
    before_bias = ln.bias.clone() if ln.bias is not None else None
    _ = ln(x)
    if elementwise_affine:
        assert torch.allclose(ln.weight, before_weight)
        assert torch.allclose(ln.bias, before_bias)
    else:
        assert ln.weight is None
        assert ln.bias is None


def test_layernorm_with_batch_size_1():
    ln = LayerNormalization(8)
    x = torch.randn(1, 8)
    y = ln(x)
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)
