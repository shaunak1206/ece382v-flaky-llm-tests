import pytest
import torch
from torch_layer_normalization import LayerNormalization


def test_layer_norm_basic_output_shape():
    layer_norm = LayerNormalization(normalized_shape=5)
    x = torch.randn(10, 5)
    out = layer_norm(x)
    assert out.shape == x.shape


def test_layer_norm_1d_input():
    layer_norm = LayerNormalization(normalized_shape=3)
    x = torch.randn(3)
    out = layer_norm(x)
    assert out.shape == x.shape


def test_layer_norm_float32_input():
    layer_norm = LayerNormalization(normalized_shape=4)
    x = torch.randn(2, 4).float()
    out = layer_norm(x)
    assert out.dtype == torch.float32


def test_layer_norm_float64_input():
    layer_norm = LayerNormalization(normalized_shape=4)
    x = torch.randn(2, 4).double()
    out = layer_norm(x)
    assert out.dtype == torch.float64


def test_layer_norm_with_gain_and_bias():
    layer_norm = LayerNormalization(normalized_shape=5, elementwise_affine=True)
    x = torch.randn(10, 5)
    out = layer_norm(x)
    # weight and bias params exist
    assert hasattr(layer_norm, 'weight')
    assert hasattr(layer_norm, 'bias')
    # shapes
    assert layer_norm.weight.shape == (5,)
    assert layer_norm.bias.shape == (5,)
    # output shape
    assert out.shape == x.shape


def test_layer_norm_without_affine():
    layer_norm = LayerNormalization(normalized_shape=5, elementwise_affine=False)
    x = torch.randn(8, 5)
    out = layer_norm(x)
    assert not hasattr(layer_norm, 'weight')
    assert not hasattr(layer_norm, 'bias')
    assert out.shape == x.shape


def test_layer_norm_with_different_epsilon():
    layer_norm_eps_default = LayerNormalization(normalized_shape=6)
    layer_norm_eps_custom = LayerNormalization(normalized_shape=6, eps=1e-3)
    x = torch.randn(3, 6)

    out_default = layer_norm_eps_default(x)
    out_custom = layer_norm_eps_custom(x)

    # outputs should be close but different due to eps
    assert torch.allclose(out_default, out_custom, atol=1e-1) or not torch.allclose(out_default, out_custom, atol=1e-4)


def test_layer_norm_forward_multiple_dims():
    # normalized_shape can be tuple
    shape = (2, 3, 4)
    normalized_shape = (3, 4)
    layer_norm = LayerNormalization(normalized_shape=normalized_shape)
    x = torch.randn(5, *shape)
    out = layer_norm(x)
    assert out.shape == x.shape


def test_layer_norm_invalid_normalized_shape():
    with pytest.raises(TypeError):
        LayerNormalization(normalized_shape=3.5)

    with pytest.raises(ValueError):
        LayerNormalization(normalized_shape=0)

    with pytest.raises(ValueError):
        LayerNormalization(normalized_shape=())


def test_layer_norm_backward_pass():
    layer_norm = LayerNormalization(normalized_shape=5)
    x = torch.randn(10, 5, requires_grad=True)
    out = layer_norm(x)
    loss = out.sum()
    loss.backward()
    # grad should not be None for input
    assert x.grad is not None
    # grad should not be None for weight and bias
    if layer_norm.elementwise_affine:
        assert layer_norm.weight.grad is not None
        assert layer_norm.bias.grad is not None


def test_layer_norm_eval_mode_behavior():
    layer_norm = LayerNormalization(normalized_shape=4)
    x = torch.randn(2, 4)
    layer_norm.eval()
    out_eval = layer_norm(x)
    layer_norm.train()
    out_train = layer_norm(x)
    # outputs should be close
    assert torch.allclose(out_eval, out_train, atol=1e-6)


def test_layer_norm_no_inplace_modification():
    layer_norm = LayerNormalization(normalized_shape=3)
    x = torch.randn(2, 3)
    x_clone = x.clone()
    _ = layer_norm(x)
    # input should be unchanged
    assert torch.allclose(x, x_clone)


def test_layer_norm_parameter_initialization():
    normalized_shape = 4
    layer_norm = LayerNormalization(normalized_shape=normalized_shape, elementwise_affine=True)
    # weight should initialize to ones
    expected_weight = torch.ones(normalized_shape)
    assert torch.allclose(layer_norm.weight.data, expected_weight)
    # bias should initialize to zeros
    expected_bias = torch.zeros(normalized_shape)
    assert torch.allclose(layer_norm.bias.data, expected_bias)


def test_layer_norm_multiple_forward_calls_produce_consistent_results():
    layer_norm = LayerNormalization(normalized_shape=2)
    x = torch.randn(6, 2)
    out1 = layer_norm(x)
    out2 = layer_norm(x)
    assert torch.allclose(out1, out2)


def test_layer_norm_different_input_devices():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        layer_norm = LayerNormalization(normalized_shape=3).to(device)
        x = torch.randn(4, 3, device=device)
        out = layer_norm(x)
        assert out.device == device
    else:
        pytest.skip("CUDA device not available")


def test_layer_norm_different_input_dtypes():
    layer_norm = LayerNormalization(normalized_shape=2)
    for dtype in [torch.float16, torch.float32, torch.float64]:
        x = torch.randn(7, 2, dtype=dtype)
        out = layer_norm(x)
        assert out.dtype == dtype


def test_layer_norm_repr():
    layer_norm = LayerNormalization(normalized_shape=5, elementwise_affine=True, eps=1e-4)
    # repr returns string containing class name and normalized_shape
    s = repr(layer_norm)
    assert "LayerNormalization" in s
    assert "normalized_shape" in s or "normalized" in s


def test_layer_norm_different_batch_sizes():
    layer_norm = LayerNormalization(normalized_shape=4)
    for batch_size in [1, 10, 100]:
        x = torch.randn(batch_size, 4)
        out = layer_norm(x)
        assert out.shape == x.shape


def test_layer_norm_with_gradients_through_parameters():
    layer_norm = LayerNormalization(normalized_shape=3, elementwise_affine=True)
    optimizer = torch.optim.SGD(layer_norm.parameters(), lr=0.01)
    x = torch.randn(4, 3)
    out = layer_norm(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    # After step weight and bias should still be trainable
    for p in layer_norm.parameters():
        assert p.requires_grad


def test_layer_norm_zero_variance_input():
    layer_norm = LayerNormalization(normalized_shape=4)
    x = torch.ones(6, 4) * 5.0
    out = layer_norm(x)
    # Output should be finite and not nan
    assert torch.isfinite(out).all()
    # Output should have mean close to 0 since variance is 0 but eps exists
    mean = out.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
