import pytest
import torch
from torch_layer_normalization import LayerNormalization


@pytest.mark.parametrize("normalized_shape", [
    1,
    4,
    (4,),
    (4, 5),
    torch.Size([4, 5])
])
def test_layernorm_shapes_and_forward(normalized_shape):
    ln = LayerNormalization(normalized_shape)
    batch_size = 2
    # Make input shape: (batch_size, *normalized_shape)
    if isinstance(normalized_shape, int):
        input_shape = (batch_size, normalized_shape)
    elif isinstance(normalized_shape, torch.Size):
        input_shape = (batch_size, *normalized_shape)
    else:
        input_shape = (batch_size, *normalized_shape)
    x = torch.randn(input_shape)
    y = ln(x)
    assert y.shape == x.shape
    # Check output dtype and device equal input
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_layernorm_default_eps_and_elementwise_affine():
    ln = LayerNormalization(3)
    assert isinstance(ln.eps, float)
    assert ln.eps > 0
    assert ln.elementwise_affine is True
    x = torch.randn(2, 3)
    y = ln(x)
    assert y.shape == x.shape


def test_layernorm_runs_with_elementwise_affine_false():
    ln = LayerNormalization(3, elementwise_affine=False)
    x = torch.randn(2, 3)
    y = ln(x)
    assert y.shape == x.shape
    # Should have no weight and bias parameters in this mode
    assert not hasattr(ln, 'weight')
    assert not hasattr(ln, 'bias')
    # Output should still be normalized approximately zero mean and unit var per last dims
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


def test_layernorm_forward_value_close_to_torch_layernorm():
    normalized_shape = 5
    ln = LayerNormalization(normalized_shape)
    # Reduce randomness variance by fixing weight and bias to ones and zeros
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.fill_(0)
    x = torch.randn(3, normalized_shape)
    y1 = ln(x)
    y2 = torch.nn.functional.layer_norm(x, (normalized_shape,), ln.weight, ln.bias, ln.eps)
    assert torch.allclose(y1, y2, atol=1e-6)


def test_layernorm_backward_and_gradients():
    ln = LayerNormalization(3)
    x = torch.randn(2, 3, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None
    # Gradients shapes
    assert ln.weight.grad.shape == ln.weight.shape
    assert ln.bias.grad.shape == ln.bias.shape


def test_layernorm_forward_with_different_dtypes():
    normalized_shape = 3
    for dtype in [torch.float32, torch.float64]:
        ln = LayerNormalization(normalized_shape).to(dtype=dtype)
        x = torch.randn(2, normalized_shape, dtype=dtype)
        y = ln(x)
        assert y.dtype == dtype


def test_layernorm_input_ndim_error():
    ln = LayerNormalization(3)
    # Input last dimensions must correspond to normalized_shape
    x = torch.randn(2, 4)  # Mismatch, normalized_shape=3
    with pytest.raises(RuntimeError):
        ln(x)


def test_layernorm_no_inplace_modification():
    ln = LayerNormalization(3)
    x = torch.randn(2, 3)
    x_clone = x.clone()
    y = ln(x)
    # Input should not be modified inplace
    assert torch.allclose(x, x_clone)


def test_layernorm_weight_and_bias_parameters_initialization():
    normalized_shape = (3, 4)
    ln = LayerNormalization(normalized_shape)
    assert ln.weight.shape == torch.Size(normalized_shape)
    assert ln.bias.shape == torch.Size(normalized_shape)
    # Default weight initialized to ones, bias to zeros
    assert torch.all(ln.weight == 1)
    assert torch.all(ln.bias == 0)


def test_layernorm_no_weight_bias_params_when_elementwise_affine_false():
    ln = LayerNormalization(3, elementwise_affine=False)
    assert not any(hasattr(ln, attr) for attr in ["weight", "bias"])
    # Confirm parameters count is 0
    params = list(ln.parameters())
    assert len(params) == 0


def test_layernorm_str_and_repr_contains_version_and_shape():
    normalized_shape = 3
    ln = LayerNormalization(normalized_shape)
    s = str(ln)
    r = repr(ln)
    # Should mention normalized_shape in str or repr
    assert str(normalized_shape) in s or str(normalized_shape) in r


@pytest.mark.parametrize("eps", [1e-5, 1e-8, 1e-2])
def test_layernorm_custom_eps(eps):
    ln = LayerNormalization(3, eps=eps)
    assert ln.eps == eps
    x = torch.randn(2, 3)
    y = ln(x)
    assert y.shape == x.shape


def test_layernorm_multiple_batch_dimensions():
    normalized_shape = (4,)
    ln = LayerNormalization(normalized_shape)
    x = torch.randn(2, 3, 4)
    y = ln(x)
    # Output shape matches input
    assert y.shape == x.shape
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    # Check normalization per last dimension
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


def test_layernorm_with_gpu_if_available():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        ln = LayerNormalization(3).to(device)
        x = torch.randn(2, 3, device=device)
        y = ln(x)
        assert y.device == device
        # Check output shape
        assert y.shape == x.shape


def test_layernorm_return_type_is_tensor():
    ln = LayerNormalization(3)
    x = torch.randn(2, 3)
    y = ln(x)
    assert isinstance(y, torch.Tensor)


def test_layernorm_serialization_and_deserialization(tmp_path):
    ln = LayerNormalization(3)
    x = torch.randn(2, 3)
    y1 = ln(x)

    path = tmp_path / "ln.pt"
    torch.save(ln.state_dict(), path)

    ln2 = LayerNormalization(3)
    ln2.load_state_dict(torch.load(path))
    y2 = ln2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def test_layernorm_with_noncontiguous_input():
    ln = LayerNormalization(3)
    x = torch.randn(2, 6)[:, ::2]  # non-contiguous
    assert not x.is_contiguous()
    y = ln(x)
    assert y.shape == x.shape


def test_layernorm_repr_contains_class_name():
    ln = LayerNormalization(3)
    r = repr(ln)
    # It should contain class name
    assert "LayerNormalization" in r


def test_layernorm_with_normalized_shape_tensor_size():
    # Check that normalized_shape can be torch.Size and works
    normalized_shape = torch.Size([3, 4])
    ln = LayerNormalization(normalized_shape)
    x = torch.randn(2, 3, 4)
    y = ln(x)
    # Output shape equal input shape
    assert y.shape == x.shape


def test_layernorm_invalid_normalized_shape_type():
    # normalized_shape must be int or tuple or torch.Size but test invalid type
    with pytest.raises(TypeError):
        LayerNormalization("invalid_shape")


def test_layernorm_forward_with_half_precision_inputs():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        ln = LayerNormalization(3).to(device).to(torch.float16)
        x = torch.randn(2, 3, device=device).to(torch.float16)
        y = ln(x)
        assert y.dtype == torch.float16
        assert y.shape == x.shape
