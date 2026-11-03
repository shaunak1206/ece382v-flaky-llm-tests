import math
import torch
import pytest
from torch_layer_normalization import LayerNormalization

@pytest.mark.parametrize("normalized_shape", [
    10,
    (10,),
    (2, 5),
])
def test_forward_output_shape(normalized_shape):
    ln = LayerNormalization(normalized_shape)
    x = torch.randn(2, *normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,))
    y = ln(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_elementwise_affine_parameter_presence(elementwise_affine):
    ln = LayerNormalization(5, elementwise_affine=elementwise_affine)
    if elementwise_affine:
        assert hasattr(ln, "weight") and ln.weight is not None
        assert hasattr(ln, "bias") and ln.bias is not None
        assert ln.weight.shape == (5,)
        assert ln.bias.shape == (5,)
    else:
        assert not hasattr(ln, "weight") or ln.weight is None
        assert not hasattr(ln, "bias") or ln.bias is None


def test_forward_computes_mean_and_var_correctly():
    ln = LayerNormalization(3, elementwise_affine=False)
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    # Manually compute LN output
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    expected = (x - mean) / torch.sqrt(var + ln.eps)
    out = ln(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_forward_with_affine_parameters():
    ln = LayerNormalization(3, elementwise_affine=True)
    with torch.no_grad():
        ln.weight.fill_(2.0)
        ln.bias.fill_(-1.0)
    x = torch.tensor([[1., 2., 3.]])
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + ln.eps)
    expected = normalized * 2.0 + (-1.0)
    out = ln(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_backward_pass():
    ln = LayerNormalization(4, elementwise_affine=True)
    x = torch.randn(2, 4, requires_grad=True)
    y = ln(x)
    y.mean().backward()
    assert x.grad is not None
    if ln.elementwise_affine:
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None


def test_eps_prevents_divide_by_zero():
    ln = LayerNormalization(3, elementwise_affine=False, eps=1e-12)
    x = torch.ones(5, 3) * 3.0  # zero variance
    y = ln(x)
    assert torch.all(torch.isfinite(y))


def test_invalid_normalized_shape_raises():
    with pytest.raises(TypeError):
        LayerNormalization('invalid')

    with pytest.raises(ValueError):
        LayerNormalization(0)

    with pytest.raises(ValueError):
        LayerNormalization(())


def test_normalized_shape_accepts_tuple_and_int():
    # int normalized_shape becomes (normalized_shape,)
    ln1 = LayerNormalization(3)
    ln2 = LayerNormalization((3,))
    x1 = torch.randn(2, 3)
    x2 = torch.randn(2, 3)
    y1 = ln1(x1)
    y2 = ln2(x2)
    assert y1.shape == x1.shape
    assert y2.shape == x2.shape


def test_forward_with_multidimensional_normalized_shape():
    shape = (2, 3)
    ln = LayerNormalization(shape)
    x = torch.randn(4, *shape)
    y = ln(x)
    assert y.shape == x.shape


def test_repr_contains_expected_info():
    ln = LayerNormalization(5, elementwise_affine=True, eps=1e-3)
    r = repr(ln)
    assert "LayerNormalization" in r
    assert "elementwise_affine=True" in r
    assert "eps=0.001" in r


def test_weight_and_bias_initialize_to_ones_and_zeros():
    ln = LayerNormalization(4)
    if ln.elementwise_affine:
        assert torch.allclose(ln.weight, torch.ones_like(ln.weight))
        assert torch.allclose(ln.bias, torch.zeros_like(ln.bias))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_layernorm_on_device(device):
    shape = (3,)
    ln = LayerNormalization(shape).to(device)
    x = torch.randn(5, *shape).to(device)
    y = ln(x)
    assert y.device.type == device
    assert y.shape == x.shape


def test_state_dict_and_load_state_dict():
    ln1 = LayerNormalization(3, elementwise_affine=True)
    ln2 = LayerNormalization(3, elementwise_affine=True)
    with torch.no_grad():
        ln1.weight.fill_(7.0)
        ln1.bias.fill_(-3.0)
    sd = ln1.state_dict()
    ln2.load_state_dict(sd)
    for p1, p2 in zip([ln1.weight, ln1.bias], [ln2.weight, ln2.bias]):
        assert torch.allclose(p1, p2)


def test_output_mean_zero_and_var_one_when_affine_false():
    ln = LayerNormalization(4, elementwise_affine=False)
    x = torch.randn(10, 4) * 10 + 5
    y = ln(x)
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


def test_layernorm_does_not_mutate_input():
    ln = LayerNormalization(3)
    x = torch.randn(2, 3)
    x_clone = x.clone()
    _ = ln(x)
    assert torch.allclose(x, x_clone)


def test_forward_with_single_element_normalized_shape():
    ln = LayerNormalization(1)
    x = torch.randn(7, 1)
    y = ln(x)
    assert y.shape == x.shape


def test_layernorm_with_requires_grad_behavior():
    ln = LayerNormalization(6)
    x = torch.randn(3, 6)
    y = ln(x)
    y.sum().backward()
    assert x.grad is None  # x requires_grad=False by default

    x2 = x.clone().detach().requires_grad_(True)
    y2 = ln(x2)
    y2.sum().backward()
    assert x2.grad is not None


def test_forward_matches_pytorch_native_layernorm():
    import torch.nn as nn
    normalized_shape = (4,)
    ln_custom = LayerNormalization(normalized_shape)
    ln_native = nn.LayerNorm(normalized_shape, elementwise_affine=False)
    x = torch.randn(5, 4)
    out_custom = ln_custom(x)
    out_native = ln_native(x)
    assert torch.allclose(out_custom, out_native, atol=1e-5)
