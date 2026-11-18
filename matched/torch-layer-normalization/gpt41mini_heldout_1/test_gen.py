import torch
import pytest
from torch_layer_normalization import LayerNormalization


def test_default_initialization():
    ln = LayerNormalization(10)
    assert ln.normalized_shape == (10,)
    assert ln.elementwise_affine is True
    assert ln.eps == 1e-5
    assert ln.weight.shape == (10,)
    assert ln.bias.shape == (10,)
    assert torch.allclose(ln.weight, torch.ones(10))
    assert torch.allclose(ln.bias, torch.zeros(10))


def test_initialization_elementwise_affine_false():
    ln = LayerNormalization(5, elementwise_affine=False)
    assert ln.elementwise_affine is False
    assert not hasattr(ln, "weight")
    assert not hasattr(ln, "bias")


def test_forward_output_shape():
    ln = LayerNormalization(4)
    x = torch.randn(2, 3, 4)
    y = ln(x)
    assert y.shape == x.shape


def test_forward_mean_zero_and_variance_one():
    torch.manual_seed(42)
    ln = LayerNormalization(7)
    x = torch.randn(3, 7)
    y = ln(x)
    mean = y.mean(-1)
    var = y.var(-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


def test_forward_with_non_flat_normalized_shape():
    # normalized shape can be tuple of shapes (multi-dim)
    ln = LayerNormalization((3, 2))
    x = torch.randn(4, 3, 2)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean(dim=(-2, -1))
    var = y.var(dim=(-2, -1), unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


def test_forward_with_elementwise_affine_false_produces_correct_output():
    ln = LayerNormalization(3, elementwise_affine=False)
    x = torch.randn(5, 3)
    y = ln(x)
    mean = y.mean(-1)
    var = y.var(-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


def test_eps_effect_on_output():
    x = torch.randn(4, 5)
    ln1 = LayerNormalization(5, eps=1e-3)
    ln2 = LayerNormalization(5, eps=1e-6)
    y1 = ln1(x)
    y2 = ln2(x)
    # Larger eps reduces normalization effect slightly, so outputs won't be exactly equal
    assert not torch.allclose(y1, y2, atol=1e-4)
    # Both outputs still have near zero mean and near one variance (within tolerance due to eps)
    mean1 = y1.mean(-1)
    var1 = y1.var(-1, unbiased=False)
    mean2 = y2.mean(-1)
    var2 = y2.var(-1, unbiased=False)
    assert torch.allclose(mean1, torch.zeros_like(mean1), atol=1e-3)
    assert torch.allclose(var1, torch.ones_like(var1), atol=1e-3)
    assert torch.allclose(mean2, torch.zeros_like(mean2), atol=1e-4)
    assert torch.allclose(var2, torch.ones_like(var2), atol=1e-4)


def test_layernorm_preserves_dtype_and_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(6, 8, device=device, dtype=torch.float64)
    ln = LayerNormalization(8).to(device).to(torch.float64)
    y = ln(x)
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_weight_and_bias_are_parameters_when_elementwise_affine_true():
    ln = LayerNormalization(3)
    params = dict(ln.named_parameters())
    assert "weight" in params
    assert "bias" in params
    assert isinstance(params["weight"], torch.nn.Parameter)
    assert isinstance(params["bias"], torch.nn.Parameter)


def test_forward_backward_pass():
    ln = LayerNormalization(10)
    x = torch.randn(4, 10, requires_grad=True)
    y = ln(x)
    y.sum().backward()
    assert x.grad is not None
    # weight and bias should also have gradients
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_invalid_normalized_shape_type():
    with pytest.raises(TypeError):
        LayerNormalization("invalid_shape")  # Must be int or tuple


def test_zero_dimension_normalized_shape():
    with pytest.raises(ValueError):
        LayerNormalization(0)  # Normalized shape cannot be zero


def test_forward_with_higher_dimensional_input():
    ln = LayerNormalization(2)
    x = torch.randn(3, 4, 5, 2)
    y = ln(x)
    assert y.shape == x.shape
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


def test_repr_contains_correct_info():
    ln = LayerNormalization(3)
    r = repr(ln)
    assert "LayerNormalization" in r
    assert "normalized_shape=(3,)" in r
    assert "eps=1e-05" in r.lower()
    assert "elementwise_affine=True" in r


def test_state_dict_contains_weight_and_bias():
    ln = LayerNormalization(4)
    sd = ln.state_dict()
    assert "weight" in sd
    assert "bias" in sd


def test_state_dict_does_not_contain_weight_and_bias_if_no_affine():
    ln = LayerNormalization(4, elementwise_affine=False)
    sd = ln.state_dict()
    assert "weight" not in sd
    assert "bias" not in sd


def test_loading_state_dict_restores_parameters():
    ln1 = LayerNormalization(3)
    ln2 = LayerNormalization(3)
    # modify ln1 parameters
    with torch.no_grad():
        ln1.weight.add_(1.23)
        ln1.bias.sub_(0.45)
    sd = ln1.state_dict()
    ln2.load_state_dict(sd)
    assert torch.allclose(ln1.weight, ln2.weight)
    assert torch.allclose(ln1.bias, ln2.bias)
