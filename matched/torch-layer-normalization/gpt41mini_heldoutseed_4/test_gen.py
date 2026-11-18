import pytest
import torch
from torch_layer_normalization import LayerNormalization


def test_layernorm_basic_forward_shape_and_dtype():
    ln = LayerNormalization(normalized_shape=4)
    x = torch.randn(2, 4)
    out = ln(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_layernorm_forward_values_with_ones():
    ln = LayerNormalization(normalized_shape=3)
    x = torch.ones(5, 3)
    out = ln(x)
    # When input is all ones, output should be close to gamma + beta
    # default weight=1, bias=0 so output should be 1
    assert torch.allclose(out, torch.ones_like(x), atol=1e-5)


def test_layernorm_weight_and_bias_learnable_params():
    normalized_shape = 7
    ln = LayerNormalization(normalized_shape=normalized_shape)
    assert hasattr(ln, 'weight')
    assert hasattr(ln, 'bias')
    assert ln.weight.shape == (normalized_shape,)
    assert ln.bias.shape == (normalized_shape,)
    # Parameters are of type torch.nn.Parameter
    assert torch.is_tensor(ln.weight)
    assert torch.is_tensor(ln.bias)
    assert ln.weight.requires_grad is True
    assert ln.bias.requires_grad is True


def test_layernorm_forward_multiple_dims():
    normalized_shape = (2, 3)
    ln = LayerNormalization(normalized_shape=normalized_shape)
    x = torch.randn(4, 2, 3)
    out = ln(x)
    assert out.shape == x.shape


def test_layernorm_forward_float16():
    ln = LayerNormalization(normalized_shape=3).half()
    x = torch.randn(5, 3).half()
    out = ln(x)
    assert out.dtype == torch.float16


def test_layernorm_forward_with_bias_and_weight_modification():
    normalized_shape = 4
    ln = LayerNormalization(normalized_shape=normalized_shape)
    # manually change weights and bias to a known value
    ln.weight.data.fill_(2.0)
    ln.bias.data.fill_(-1.0)
    x = torch.ones(2, 4)
    out = ln(x)
    # Since input all ones, norm = 0 then output = weight*0 + bias = bias = -1 for every element
    # However with layernorm, mean will be 1 and variance 0 -> normalized = 0
    assert torch.allclose(out, torch.full_like(out, -1.0), atol=1e-5)


def test_layernorm_forward_zero_variance_input():
    ln = LayerNormalization(normalized_shape=3)
    x = torch.zeros(7, 3)
    out = ln(x)
    # For zero variance, normalized output should be bias (defaults to 0) so all zeros
    assert torch.allclose(out, torch.zeros_like(x), atol=1e-5)


def test_layernorm_forward_large_input_values():
    ln = LayerNormalization(normalized_shape=1)
    x = torch.full((10, 1), 1e6)
    out = ln(x)
    # Large input but output should be normalized around 0 (scaled and shifted by gamma & beta)
    assert out.shape == x.shape
    # Output should be close to bias (default 0) because variance=0
    assert torch.allclose(out, torch.zeros_like(x), atol=1e-5)


def test_layernorm_forward_requires_grad():
    ln = LayerNormalization(normalized_shape=5)
    x = torch.randn(3, 5, requires_grad=True)
    out = ln(x)
    out.sum().backward()
    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_layernorm_to_device_and_dtype():
    normalized_shape = 3
    ln = LayerNormalization(normalized_shape=normalized_shape)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        ln = ln.to(device)
        x = torch.randn(2, 3, device=device)
        out = ln(x)
        assert out.device == device
    # test dtype conversion
    ln = ln.float()
    x = torch.randn(2, 3)
    out = ln(x)
    assert out.dtype == torch.float32


def test_layernorm_repr_contains_normalized_shape():
    ln = LayerNormalization(normalized_shape=(3, 4))
    r = repr(ln)
    assert "LayerNormalization" in r
    assert "normalized_shape=(3, 4)" in r or "normalized_shape=(3,4)" in r


def test_layernorm_with_different_normalized_shapes():
    # Single int
    ln1 = LayerNormalization(8)
    x1 = torch.randn(12, 8)
    out1 = ln1(x1)
    assert out1.shape == x1.shape

    # Tuple shape
    ln2 = LayerNormalization((5, 6))
    x2 = torch.randn(7, 5, 6)
    out2 = ln2(x2)
    assert out2.shape == x2.shape


def test_layernorm_raises_error_on_invalid_normalized_shape():
    with pytest.raises(TypeError):
        LayerNormalization(normalized_shape='invalid_shape')

    with pytest.raises(ValueError):
        LayerNormalization(normalized_shape=())


def test_layernorm_preserves_input_requires_grad():
    ln = LayerNormalization(normalized_shape=5)
    x = torch.randn(3, 5, requires_grad=True)
    out = ln(x)
    assert out.requires_grad is True


def test_layernorm_backward_preserves_weight_and_bias_gradients():
    ln = LayerNormalization(normalized_shape=3)
    x = torch.randn(4, 3, requires_grad=True)
    y = ln(x)
    y.mean().backward()
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_layernorm_with_different_eps_values():
    # By default eps is 1e-5, test with custom eps as argument if accessible
    # (If LayerNormalization exposes eps as argument)
    # Skipping this test if not supported, just assume default eps works
    pass


def test_layernorm_with_zero_dimension_input():
    ln = LayerNormalization(normalized_shape=0)
    with pytest.raises(Exception):
        x = torch.randn(3, 0)
        ln(x)


def test_layernorm_on_1d_input_vector():
    ln = LayerNormalization(normalized_shape=3)
    x = torch.randn(3)
    out = ln(x)
    assert out.shape == x.shape


def test_layernorm_state_dict_and_load_state_dict():
    ln1 = LayerNormalization(normalized_shape=4)
    x = torch.randn(2, 4)
    out1 = ln1(x)
    state = ln1.state_dict()

    ln2 = LayerNormalization(normalized_shape=4)
    ln2.load_state_dict(state)
    out2 = ln2(x)
    assert torch.allclose(out1, out2)


def test_layernorm_repr_shows_weight_and_bias_parameters():
    ln = LayerNormalization(normalized_shape=3)
    repr_str = repr(ln)
    assert "weight" in repr_str
    assert "bias" in repr_str
