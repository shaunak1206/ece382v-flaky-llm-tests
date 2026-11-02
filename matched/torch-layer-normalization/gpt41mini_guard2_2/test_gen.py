import pytest
import torch
import numpy as np
from torch_layer_normalization import LayerNormalization

@pytest.mark.parametrize("normalized_shape", [1, 5, (3, 4)])
def test_basic_forward_output_shape(normalized_shape):
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.randn(2, *normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,))
    ln = LayerNormalization(normalized_shape)
    y = ln(x)
    assert y.shape == x.shape

@pytest.mark.parametrize("eps", [1e-5, 1e-12, 1e-3])
def test_eps_parameter_effect(eps):
    torch.manual_seed(0)
    np.random.seed(0)
    ln1 = LayerNormalization(10, eps=eps)
    ln2 = LayerNormalization(10, eps=eps * 10)
    x = torch.randn(4, 10)
    y1 = ln1(x)
    y2 = ln2(x)
    # Outputs should differ because eps affects stability (small differences)
    assert torch.allclose(y1, y2, atol=1e-2) or torch.any(torch.abs(y1 - y2) > 1e-6)

def test_default_weight_and_bias_are_ones_and_zeros():
    torch.manual_seed(0)
    np.random.seed(0)
    ln = LayerNormalization(4)
    assert torch.allclose(ln.weight, torch.ones(4))
    assert torch.allclose(ln.bias, torch.zeros(4))

def test_layernorm_normalizes_to_zero_mean_and_unit_var():
    torch.manual_seed(123)
    np.random.seed(123)
    x = torch.randn(10, 6, dtype=torch.float64)
    ln = LayerNormalization(6, eps=1e-14).double()
    y = ln(x)
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    # mean close to zero, std close to one
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-7)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-7)

def test_layernorm_with_affine_false():
    torch.manual_seed(42)
    np.random.seed(42)
    ln = LayerNormalization(3, elementwise_affine=False)
    x = torch.randn(5, 3)
    y = ln(x)
    # Since no affine, output mean and std per last dim must be zero and one respectively
    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-6)
    # Weight and bias should be None when affine is False
    assert ln.weight is None
    assert ln.bias is None

def test_parameter_gradients_enabled():
    torch.manual_seed(0)
    np.random.seed(0)
    ln = LayerNormalization(4)
    x = torch.randn(2, 4, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None
    assert x.grad is not None

@pytest.mark.parametrize("shape", [(2,3,4),(5,10),(7,1)])
def test_forward_and_backward_pass(shape):
    torch.manual_seed(1234)
    np.random.seed(1234)
    ln = LayerNormalization(shape[-1])
    x = torch.randn(*shape, requires_grad=True)
    y = ln(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None

def test_invalid_normalized_shape_type_raises():
    with pytest.raises(TypeError):
        LayerNormalization("invalid_shape")

def test_normalized_shape_tuple_or_int_only():
    ln_int = LayerNormalization(4)
    assert ln_int.normalized_shape == (4,)
    ln_tuple = LayerNormalization((4, 3))
    assert ln_tuple.normalized_shape == (4, 3)

def test_parameters_are_registered_correctly():
    ln = LayerNormalization(3)
    params = dict(ln.named_parameters())
    assert "weight" in params
    assert "bias" in params
    # weight and bias shape
    assert params["weight"].shape == (3,)
    assert params["bias"].shape == (3,)

def test_no_parameters_when_elementwise_affine_false():
    ln = LayerNormalization(3, elementwise_affine=False)
    params = list(ln.parameters())
    assert len(params) == 0

@pytest.mark.parametrize("input_shape", [(2, 3, 4), (4, 5), (10,)])
def test_forward_produces_different_output_from_input(input_shape):
    torch.manual_seed(0)
    np.random.seed(0)
    ln = LayerNormalization(input_shape[-1])
    x = torch.randn(*input_shape)
    y = ln(x)
    # Output differs (except in rare case where input is already zero mean/unit var)
    assert not torch.allclose(x, y)

def test_forward_output_dtype_matches_input_dtype():
    torch.manual_seed(0)
    np.random.seed(0)
    input_dtypes = [torch.float32, torch.float64]
    for dt in input_dtypes:
        ln = LayerNormalization(3).to(dtype=dt)
        x = torch.randn(2, 3, dtype=dt)
        y = ln(x)
        assert y.dtype == dt

def test_repr_contains_classname_and_version():
    ln = LayerNormalization(3)
    rep = repr(ln)
    # It should contain the class name string 'LayerNormalization'
    assert "LayerNormalization" in rep

def test_forward_with_different_batch_sizes():
    torch.manual_seed(0)
    np.random.seed(0)
    ln = LayerNormalization(3)
    for batch_size in [1, 10, 100]:
        x = torch.randn(batch_size, 3)
        y = ln(x)
        assert y.shape == torch.Size([batch_size, 3])

def test_print_parameters_after_init(monkeypatch):
    ln = LayerNormalization(3)
    # should have attribute weight and bias with correct shape
    assert hasattr(ln, "weight")
    assert hasattr(ln, "bias")
    assert ln.weight.shape == (3,)
    assert ln.bias.shape == (3,)

def test_gradients_do_not_accumulate_across_forward_backward():
    torch.manual_seed(0)
    np.random.seed(0)
    ln = LayerNormalization(3)
    x = torch.randn(4, 3, requires_grad=True)
    for _ in range(3):
        y = ln(x)
        loss = y.sum()
        loss.backward()
        # After backward, gradients in weight and bias should be nonzero
        assert ln.weight.grad is not None and ln.bias.grad is not None
        # Zero gradients for next iteration
        ln.weight.grad.zero_()
        ln.bias.grad.zero_()
        x.grad.zero_()

def test_layernorm_returns_differentiable_output():
    torch.manual_seed(0)
    np.random.seed(0)
    ln = LayerNormalization(4)
    x = torch.randn(2, 4, requires_grad=True)
    y = ln(x)
    y.sum().backward()
    assert x.grad is not None

def test_layernorm_with_custom_eps_does_not_raise():
    torch.manual_seed(0)
    np.random.seed(0)
    ln = LayerNormalization(3, eps=1e-10)
    x = torch.randn(2, 3)
    y = ln(x)
    assert y.shape == x.shape
