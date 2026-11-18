import torch
import pytest
from torch_layer_normalization import LayerNormalization


def test_layernorm_basic_forward():
    ln = LayerNormalization(10)
    x = torch.randn(4, 10)
    y = ln(x)
    assert y.shape == x.shape
    # Output should have mean close to zero and std close to one per sample
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_layernorm_forward_different_dim():
    ln = LayerNormalization(5)
    x = torch.randn(3, 7, 5)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_layernorm_forward_1d_input():
    ln = LayerNormalization(8)
    x = torch.randn(8)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean()
    std = y.std(unbiased=False)
    assert torch.isclose(mean, torch.tensor(0.), atol=1e-5)
    assert torch.isclose(std, torch.tensor(1.), atol=1e-5)


def test_layernorm_with_bias_and_scale():
    ln = LayerNormalization(6, eps=1e-6, elementwise_affine=True)
    x = torch.randn(2, 6)
    y = ln(x)
    # parameters gamma and beta shape
    assert ln.weight.shape == (6,)
    assert ln.bias.shape == (6,)
    # output shape matches input shape
    assert y.shape == x.shape


def test_layernorm_without_affine():
    ln = LayerNormalization(4, elementwise_affine=False)
    x = torch.randn(3, 4)
    y = ln(x)
    # no weight or bias attributes
    assert not hasattr(ln, 'weight')
    assert not hasattr(ln, 'bias')
    # output normalized
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_layernorm_eps_effect():
    x = torch.randn(2, 3)
    ln1 = LayerNormalization(3, eps=1e-5)
    ln2 = LayerNormalization(3, eps=1e-1)
    y1 = ln1(x)
    y2 = ln2(x)
    # Outputs are different due to different eps
    assert not torch.allclose(y1, y2)


def test_layernorm_train_and_eval_modes():
    ln = LayerNormalization(5)
    ln.train()
    x = torch.randn(3, 5)
    y_train = ln(x)
    ln.eval()
    y_eval = ln(x)
    # In both modes output should be normalized
    mean_train = y_train.mean(dim=-1)
    std_train = y_train.std(dim=-1, unbiased=False)
    mean_eval = y_eval.mean(dim=-1)
    std_eval = y_eval.std(dim=-1, unbiased=False)
    assert torch.allclose(mean_train, torch.zeros_like(mean_train), atol=1e-5)
    assert torch.allclose(std_train, torch.ones_like(std_train), atol=1e-5)
    assert torch.allclose(mean_eval, torch.zeros_like(mean_eval), atol=1e-5)
    assert torch.allclose(std_eval, torch.ones_like(std_eval), atol=1e-5)


def test_layernorm_parameter_gradients_enabled():
    ln = LayerNormalization(4)
    x = torch.randn(2, 4, requires_grad=True)
    y = ln(x)
    y.mean().backward()
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None
    assert x.grad is not None


def test_layernorm_parameter_gradients_disabled():
    ln = LayerNormalization(4, elementwise_affine=False)
    x = torch.randn(2, 4, requires_grad=True)
    y = ln(x)
    y.mean().backward()
    # weight and bias do not exist
    with pytest.raises(AttributeError):
        _ = ln.weight.grad
    with pytest.raises(AttributeError):
        _ = ln.bias.grad
    assert x.grad is not None


def test_layernorm_different_input_dtypes():
    ln = LayerNormalization(3)
    for dtype in [torch.float32, torch.float64, torch.float16]:
        x = torch.randn(2, 3, dtype=dtype)
        y = ln(x)
        assert y.dtype == dtype


def test_layernorm_invalid_normalized_shape_argument():
    with pytest.raises(TypeError):
        LayerNormalization('not an int or list')


def test_layernorm_normalized_shape_list():
    ln = LayerNormalization([3])
    x = torch.randn(2, 3)
    y = ln(x)
    assert y.shape == x.shape


def test_layernorm_normalized_shape_tuple():
    ln = LayerNormalization((3,))
    x = torch.randn(2, 3)
    y = ln(x)
    assert y.shape == x.shape


def test_layernorm_normalized_shape_multiple_dims():
    # Normalizing last 2 dims (say 3x4)
    ln = LayerNormalization((3, 4))
    x = torch.randn(2, 3, 4)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean(dim=(-2, -1))
    std = y.std(dim=(-2, -1), unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_layernorm_zero_input():
    ln = LayerNormalization(5)
    x = torch.zeros(2, 5)
    y = ln(x)
    # Since input is zero, output should equal bias if affine, else zero
    if ln.elementwise_affine:
        expected = ln.bias.unsqueeze(0).expand_as(x)
        assert torch.allclose(y, expected, atol=1e-5)
    else:
        assert torch.allclose(y, torch.zeros_like(x))


def test_layernorm_repr_contains_classname():
    ln = LayerNormalization(3)
    r = repr(ln)
    assert "LayerNormalization" in r


def test_layernorm_forward_with_requires_grad():
    ln = LayerNormalization(6)
    x = torch.randn(2, 6, requires_grad=True)
    y = ln(x)
    y.sum().backward()
    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_layernorm_forward_device_cpu_and_cuda():
    ln = LayerNormalization(7)
    x_cpu = torch.randn(2, 7)
    y_cpu = ln(x_cpu)
    assert y_cpu.device.type == 'cpu'
    if torch.cuda.is_available():
        ln_cuda = LayerNormalization(7).cuda()
        x_cuda = torch.randn(2, 7).cuda()
        y_cuda = ln_cuda(x_cuda)
        assert y_cuda.device.type == 'cuda'
        mean = y_cuda.mean(dim=-1)
        std = y_cuda.std(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-5)
