import math

import pytest
import torch
import torch.nn.functional as F

import torch_layer_normalization
from torch_layer_normalization import LayerNormalization, __version__


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


def test_package_exposes_layer_normalization():
    assert hasattr(torch_layer_normalization, "LayerNormalization")
    assert LayerNormalization is torch_layer_normalization.LayerNormalization


def test_version_is_non_empty_string():
    assert isinstance(__version__, str)
    assert __version__.strip() != ""


def test_layer_is_torch_module():
    assert isinstance(LayerNormalization(4), torch.nn.Module)


def test_output_shape_matches_input_2d():
    layer = LayerNormalization(8)
    x = torch.randn(5, 8)
    assert layer(x).shape == x.shape


def test_output_shape_matches_input_3d():
    layer = LayerNormalization(6)
    x = torch.randn(2, 3, 6)
    assert layer(x).shape == x.shape


def test_matches_functional_layer_norm_2d():
    layer = LayerNormalization(8)
    x = torch.randn(4, 8)
    expected = F.layer_norm(x, (8,), layer.weight, layer.bias, 1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_matches_functional_layer_norm_multi_dim_shape():
    layer = LayerNormalization((3, 5))
    x = torch.randn(2, 4, 3, 5)
    expected = F.layer_norm(x, (3, 5), layer.weight, layer.bias, 1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_output_has_zero_mean_and_unit_variance_per_row():
    layer = LayerNormalization(16, elementwise_affine=False)
    x = torch.randn(10, 16) * 5.0 + 3.0
    y = layer(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(10), atol=1e-5)
    assert torch.allclose(y.var(dim=-1, unbiased=False), torch.ones(10), atol=1e-4)


def test_int_and_tuple_normalized_shape_are_equivalent():
    x = torch.randn(3, 7)
    out_int = LayerNormalization(7)(x)
    out_tuple = LayerNormalization((7,))(x)
    assert torch.allclose(out_int, out_tuple, atol=1e-6)


def test_affine_parameters_are_registered_with_expected_shapes():
    layer = LayerNormalization((2, 3))
    params = dict(layer.named_parameters())
    assert set(params.keys()) == {"weight", "bias"}
    assert tuple(params["weight"].shape) == (2, 3)
    assert tuple(params["bias"].shape) == (2, 3)
    assert len(list(layer.parameters())) == 2


def test_default_affine_initialization_is_identity():
    layer = LayerNormalization(5)
    assert torch.allclose(layer.weight.detach(), torch.ones(5))
    assert torch.allclose(layer.bias.detach(), torch.zeros(5))


def test_elementwise_affine_false_has_no_parameters():
    layer = LayerNormalization(5, elementwise_affine=False)
    assert len(list(layer.parameters())) == 0


def test_elementwise_affine_false_matches_functional_without_affine():
    layer = LayerNormalization(9, elementwise_affine=False)
    x = torch.randn(4, 9)
    expected = F.layer_norm(x, (9,), None, None, 1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_custom_affine_scale_and_shift_are_applied():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        layer.bias.copy_(torch.tensor([0.5, -0.5, 1.0, -1.0]))
    x = torch.randn(6, 4)
    expected = F.layer_norm(x, (4,), layer.weight, layer.bias, 1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_constant_input_outputs_bias():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.full((3, 4), 7.5)
    out = layer(x)
    expected = layer.bias.detach().expand(3, 4)
    assert torch.allclose(out, expected, atol=1e-6)


def test_output_is_invariant_to_positive_scale_and_shift_of_input():
    layer = LayerNormalization(10, elementwise_affine=False)
    x = torch.randn(4, 10)
    assert torch.allclose(layer(x), layer(3.0 * x + 2.0), atol=1e-4)


def test_rows_are_normalized_independently_of_batch():
    layer = LayerNormalization(6)
    x = torch.randn(5, 6)
    batched = layer(x)
    for i in range(x.shape[0]):
        single = layer(x[i : i + 1])
        assert torch.allclose(batched[i : i + 1], single, atol=1e-6)


def test_custom_eps_matches_functional_with_same_eps():
    eps = 0.5
    layer = LayerNormalization(8, eps=eps)
    x = torch.randn(3, 8) * 0.1
    expected = F.layer_norm(x, (8,), layer.weight, layer.bias, eps)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_gradients_flow_to_input_and_parameters():
    layer = LayerNormalization(5)
    x = torch.randn(4, 5, requires_grad=True)
    target = torch.randn(4, 5)
    loss = (layer(x) * target).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad.shape == layer.bias.shape


def test_float64_inputs_are_supported():
    layer = LayerNormalization(4).double()
    x = torch.randn(3, 4, dtype=torch.float64)
    out = layer(x)
    assert out.dtype == torch.float64
    expected = F.layer_norm(x, (4,), layer.weight, layer.bias, 1e-5)
    assert torch.allclose(out, expected, atol=1e-10)


def test_state_dict_roundtrip_reproduces_output():
    src = LayerNormalization(6)
    with torch.no_grad():
        src.weight.uniform_(0.5, 1.5)
        src.bias.uniform_(-1.0, 1.0)
    dst = LayerNormalization(6)
    dst.load_state_dict(src.state_dict())
    x = torch.randn(2, 6)
    assert torch.allclose(src(x), dst(x), atol=1e-7)


def test_train_and_eval_modes_give_same_output():
    layer = LayerNormalization(7)
    x = torch.randn(3, 7)
    layer.train()
    out_train = layer(x)
    layer.eval()
    with torch.no_grad():
        out_eval = layer(x)
    assert torch.allclose(out_train, out_eval, atol=1e-7)


def test_mismatched_feature_size_raises_runtime_error():
    layer = LayerNormalization(4)
    x = torch.randn(2, 5)
    with pytest.raises(RuntimeError):
        layer(x)
