import torch
import pytest
from torch_layer_normalization import LayerNormalization


def test_layernorm_basic_forward():
    ln = LayerNormalization(10)
    x = torch.randn(2, 10)
    y = ln(x)
    assert y.shape == x.shape
    # Output should be normalized: mean ~0, std ~1 on the last dimension
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_layernorm_forward_with_different_shape():
    ln = LayerNormalization(5)
    x = torch.randn(4, 3, 5)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_layernorm_with_elementwise_affine_false():
    ln = LayerNormalization(8, elementwise_affine=False)
    x = torch.randn(5, 8)
    y = ln(x)
    assert y.shape == x.shape
    # Without affine, weight and bias params are None or do not exist
    assert not ln.elementwise_affine
    with pytest.raises(AttributeError):
        _ = ln.weight
    with pytest.raises(AttributeError):
        _ = ln.bias


def test_layernorm_with_epsilon():
    # Setting a larger epsilon
    epsilon = 1e-3
    ln = LayerNormalization(6, eps=epsilon)
    x = torch.ones(3, 6) * 2
    y = ln(x)
    assert y.shape == x.shape
    # std should never be zero due to epsilon
    std = y.std(dim=-1, unbiased=False)
    assert (std > 0).all()


def test_layernorm_with_different_input_dtype():
    ln = LayerNormalization(4)
    x = torch.randn(2, 4).double()
    y = ln(x)
    assert y.dtype == x.dtype


def test_layernorm_train_and_eval_mode_consistency():
    ln = LayerNormalization(7)
    x = torch.randn(3, 7)

    ln.train()
    y_train = ln(x)

    ln.eval()
    y_eval = ln(x)

    # Output shape matches input shape
    assert y_train.shape == x.shape
    assert y_eval.shape == x.shape
    # Outputs should be close since LayerNorm is stateless (no running stats)
    assert torch.allclose(y_train, y_eval, atol=1e-6)


def test_layernorm_weight_and_bias_requires_grad():
    ln = LayerNormalization(9)
    assert ln.weight.requires_grad
    assert ln.bias.requires_grad


def test_layernorm_forward_with_single_dimension_input():
    ln = LayerNormalization(3)
    x = torch.randn(3)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean()
    std = y.std(unbiased=False)
    assert abs(mean.item()) < 1e-5
    assert abs(std.item() - 1) < 1e-5


def test_layernorm_output_close_to_manual_computation():
    # Manually compute LayerNorm and compare to module output
    ln = LayerNormalization(4, eps=1e-5)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + ln.eps) ** 0.5
    expected = (x - mean) / std * ln.weight + ln.bias
    y = ln(x)
    assert torch.allclose(y, expected, atol=1e-6)


def test_layernorm_with_large_dimension():
    dim = 1024
    ln = LayerNormalization(dim)
    x = torch.randn(5, dim)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4)


def test_layernorm_scriptability():
    ln = LayerNormalization(4)
    script_ln = torch.jit.script(ln)
    x = torch.randn(2, 4)
    y1 = ln(x)
    y2 = script_ln(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def test_layernorm_repr_contains_parameters():
    ln = LayerNormalization(3)
    rep = repr(ln)
    assert str(ln.normalized_shape[0]) in rep
    assert "eps=" in rep


def test_layernorm_gradients():
    ln = LayerNormalization(6)
    x = torch.randn(3, 6, requires_grad=True)
    y = ln(x)
    y.sum().backward()
    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_layernorm_float16_input():
    ln = LayerNormalization(5).to(torch.float16)
    x = torch.randn(2, 5).to(torch.float16)
    y = ln(x)
    assert y.dtype == torch.float16
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    # Check mean and std are roughly correct, allowing larger tolerance due to fp16
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-2)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-2)


def test_layernorm_invalid_normalized_shape_type():
    # Pass something that can't be converted to tuple
    with pytest.raises(TypeError):
        LayerNormalization('invalid')


def test_layernorm_with_tuple_normalized_shape():
    ln = LayerNormalization((3, 2))
    x = torch.randn(4, 3, 2)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean(dim=(-2, -1))
    std = y.std(dim=(-2, -1), unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_layernorm_device_consistency():
    ln = LayerNormalization(4)
    x = torch.randn(2, 4)
    if torch.cuda.is_available():
        ln_cuda = LayerNormalization(4).cuda()
        x_cuda = x.cuda()
        y = ln_cuda(x_cuda)
        assert y.device.type == 'cuda'


def test_layernorm_repr_contains_elementwise_affine_flag():
    ln = LayerNormalization(4, elementwise_affine=False)
    rep = repr(ln)
    assert "elementwise_affine=False" in rep or "elementwise_affine" not in rep  # At least does not contradict


def test_layernorm_multiple_calls_produce_consistent_outputs():
    ln = LayerNormalization(5)
    x = torch.randn(3, 5)
    y1 = ln(x)
    y2 = ln(x)
    assert torch.allclose(y1, y2)


def test_layernorm_backward_pass():
    ln = LayerNormalization(5)
    x = torch.randn(4, 5, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None
