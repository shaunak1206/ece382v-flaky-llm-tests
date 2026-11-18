import pytest
import torch
from torch_layer_normalization import LayerNormalization


def test_default_init():
    ln = LayerNormalization(normalized_shape=10)
    assert isinstance(ln, LayerNormalization)
    assert ln.normalized_shape == (10,)
    # default eps should be a float
    assert isinstance(ln.eps, float)
    # default elementwise_affine should be True
    assert ln.elementwise_affine is True
    # check weight and bias shapes
    assert ln.weight.shape == (10,)
    assert ln.bias.shape == (10,)
    # initial weight is all ones, bias is all zeros
    assert torch.allclose(ln.weight.data, torch.ones(10))
    assert torch.allclose(ln.bias.data, torch.zeros(10))


def test_init_tuple_shape():
    ln = LayerNormalization(normalized_shape=(2, 3))
    assert ln.normalized_shape == (2, 3)
    assert ln.weight.shape == (2, 3)
    assert ln.bias.shape == (2, 3)


def test_init_elementwise_affine_false():
    ln = LayerNormalization(normalized_shape=4, elementwise_affine=False)
    assert ln.elementwise_affine is False
    assert ln.weight is None
    assert ln.bias is None


def test_forward_output_shape():
    ln = LayerNormalization(normalized_shape=3)
    # input shape ends with normalized_shape
    x = torch.randn(5, 3)
    y = ln(x)
    assert y.shape == x.shape


def test_forward_normalization_values():
    torch.manual_seed(42)
    ln = LayerNormalization(normalized_shape=2, eps=1e-5, elementwise_affine=False)
    x = torch.randn(10, 2)
    y = ln(x)
    # For each sample, mean over last dims should be nearly 0, var nearly 1
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_forward_elementwise_affine():
    torch.manual_seed(777)
    ln = LayerNormalization(normalized_shape=3, eps=1e-5, elementwise_affine=True)
    # Force weight and bias to known values
    ln.weight.data = torch.tensor([1.0, 2.0, 3.0])
    ln.bias.data = torch.tensor([0.5, -0.5, 0.0])
    x = torch.randn(4, 3)
    y = ln(x)
    # Normalize manually
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    expected = (x - mean) / torch.sqrt(var + ln.eps)
    expected = expected * ln.weight + ln.bias
    assert torch.allclose(y, expected)


def test_forward_with_high_dim_input():
    ln = LayerNormalization(normalized_shape=(2, 3))
    x = torch.randn(5, 7, 2, 3)
    y = ln(x)
    assert y.shape == x.shape


def test_backward_gradients():
    ln = LayerNormalization(normalized_shape=5)
    x = torch.randn(4, 5, requires_grad=True)
    y = ln(x)
    y.sum().backward()
    # gradients should be set for input and parameters (if affine)
    assert x.grad is not None
    if ln.elementwise_affine:
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None


def test_invalid_normalized_shape_raises():
    with pytest.raises(TypeError):
        # normalized_shape must be int or tuple/list of ints
        LayerNormalization(normalized_shape='invalid')


def test_weight_and_bias_are_parameters():
    ln = LayerNormalization(normalized_shape=3)
    parameters = dict(ln.named_parameters())
    assert 'weight' in parameters
    assert 'bias' in parameters


def test_weight_and_bias_none_if_affine_false():
    ln = LayerNormalization(normalized_shape=3, elementwise_affine=False)
    assert ln.weight is None
    assert ln.bias is None


def test_can_use_cuda_if_available():
    if torch.cuda.is_available():
        ln = LayerNormalization(normalized_shape=4).cuda()
        x = torch.randn(2, 4).cuda()
        y = ln(x)
        assert y.device.type == 'cuda'


def test_can_handle_batch_size_0():
    ln = LayerNormalization(normalized_shape=3)
    x = torch.randn(0, 3)
    y = ln(x)
    assert y.shape == x.shape


def test_repr_contains_correct_info():
    ln = LayerNormalization(normalized_shape=7, eps=1e-4, elementwise_affine=False)
    r = repr(ln)
    assert "normalized_shape=(7,)" in r
    assert "eps=0.0001" in r
    assert "elementwise_affine=False" in r


def test_eps_effect_on_output():
    ln_eps_small = LayerNormalization(normalized_shape=3, eps=1e-12, elementwise_affine=False)
    ln_eps_large = LayerNormalization(normalized_shape=3, eps=1.0, elementwise_affine=False)
    x = torch.randn(8, 3)
    y_small = ln_eps_small(x)
    y_large = ln_eps_large(x)
    # Larger eps smooths variance estimate thus it will slightly change outputs
    assert not torch.allclose(y_small, y_large)
    # Both outputs should be normalized (mean zero)
    mean_small = y_small.mean(dim=-1)
    mean_large = y_large.mean(dim=-1)
    assert torch.allclose(mean_small, torch.zeros_like(mean_small), atol=1e-4)
    assert torch.allclose(mean_large, torch.zeros_like(mean_large), atol=1e-4)


def test_normalized_shape_list_input():
    ln = LayerNormalization(normalized_shape=[4])
    assert ln.normalized_shape == (4,)
    x = torch.randn(2, 4)
    y = ln(x)
    assert y.shape == x.shape


def test_forward_preserves_input_dtype():
    ln = LayerNormalization(normalized_shape=3)
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(6, 3).to(dtype)
        y = ln(x)
        assert y.dtype == dtype


def test_forward_preserves_input_device():
    ln = LayerNormalization(normalized_shape=3)
    x_cpu = torch.randn(2,3)
    y_cpu = ln(x_cpu)
    assert y_cpu.device == x_cpu.device


def test_forward_with_non_contiguous_input():
    ln = LayerNormalization(normalized_shape=3)
    x = torch.randn(5, 3)
    x_t = x.transpose(0, 1)  # non-contiguous in memory
    # We input non-contiguous, but shape last dims must match normalized_shape
    with pytest.raises(RuntimeError):
        # as last dim is not normalized_shape, should raise
        ln(x_t)


def test_inplace_operations_do_not_change_weight_bias():
    ln = LayerNormalization(normalized_shape=3)
    before_w = ln.weight.clone()
    before_b = ln.bias.clone()
    x = torch.randn(2, 3)
    _ = ln(x)
    # parameters should stay the same
    assert torch.allclose(ln.weight, before_w)
    assert torch.allclose(ln.bias, before_b)
