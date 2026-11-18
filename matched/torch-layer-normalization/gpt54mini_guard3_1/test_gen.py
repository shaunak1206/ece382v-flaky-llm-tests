import inspect

import pytest
import torch

import torch_layer_normalization as tln
from torch_layer_normalization import LayerNormalization


def test_version_string():
    assert tln.__version__ == "0.11.0"


def test_layer_normalization_is_reexported():
    assert LayerNormalization is tln.LayerNormalization
    assert inspect.isclass(LayerNormalization)


def test_layer_normalization_is_instantiable_with_common_signature():
    module = LayerNormalization(4)
    assert isinstance(module, torch.nn.Module)


def test_forward_preserves_shape_for_2d_input():
    torch.manual_seed(0)
    module = LayerNormalization(4)
    x = torch.randn(3, 4)
    y = module(x)
    assert y.shape == x.shape


def test_forward_preserves_shape_for_3d_input():
    torch.manual_seed(1)
    module = LayerNormalization((2, 4))
    x = torch.randn(5, 2, 4)
    y = module(x)
    assert y.shape == x.shape


def test_output_has_zero_mean_and_unit_variance_when_affine_disabled_2d():
    torch.manual_seed(2)
    module = LayerNormalization(4, elementwise_affine=False)
    x = torch.randn(7, 4)
    y = module(x)

    means = y.mean(dim=-1)
    vars_ = y.var(dim=-1, unbiased=False)

    assert torch.allclose(means, torch.zeros_like(means), atol=1e-6, rtol=1e-6)
    assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-5, rtol=1e-5)


def test_output_has_zero_mean_and_unit_variance_when_affine_disabled_3d():
    torch.manual_seed(3)
    module = LayerNormalization((2, 4), elementwise_affine=False)
    x = torch.randn(6, 2, 4)
    y = module(x)

    means = y.mean(dim=(-2, -1))
    vars_ = y.var(dim=(-2, -1), unbiased=False)

    assert torch.allclose(means, torch.zeros_like(means), atol=1e-6, rtol=1e-6)
    assert torch.allclose(vars_, torch.ones_like(vars_), atol=1e-5, rtol=1e-5)


def test_output_is_unchanged_for_constant_input_when_affine_disabled():
    module = LayerNormalization(5, elementwise_affine=False)
    x = torch.full((4, 5), 3.25)
    y = module(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6, rtol=1e-6)


def test_affine_parameters_exist_by_default():
    module = LayerNormalization(4)
    assert hasattr(module, "weight")
    assert hasattr(module, "bias")
    assert isinstance(module.weight, torch.nn.Parameter)
    assert isinstance(module.bias, torch.nn.Parameter)
    assert module.weight.shape == torch.Size([4])
    assert module.bias.shape == torch.Size([4])


def test_affine_parameters_absent_when_disabled():
    module = LayerNormalization(4, elementwise_affine=False)
    assert module.weight is None
    assert module.bias is None
    assert list(module.parameters()) == []


def test_parameters_receive_gradients():
    torch.manual_seed(4)
    module = LayerNormalization(4)
    x = torch.randn(2, 4, requires_grad=True)
    y = module(x).sum()
    y.backward()

    assert x.grad is not None
    assert module.weight.grad is not None
    assert module.bias.grad is not None
    assert x.grad.shape == x.shape
    assert module.weight.grad.shape == module.weight.shape
    assert module.bias.grad.shape == module.bias.shape


def test_state_dict_roundtrip_preserves_output():
    torch.manual_seed(5)
    module = LayerNormalization(4)
    x = torch.randn(3, 4)

    y1 = module(x)

    state = module.state_dict()
    clone = LayerNormalization(4)
    clone.load_state_dict(state)

    y2 = clone(x)
    assert torch.allclose(y1, y2, atol=1e-7, rtol=1e-7)


def test_repeated_forward_calls_are_deterministic_in_eval_mode():
    torch.manual_seed(6)
    module = LayerNormalization(4)
    module.eval()
    x = torch.randn(8, 4)

    y1 = module(x)
    y2 = module(x)

    assert torch.allclose(y1, y2, atol=0.0, rtol=0.0)


def test_batch_items_are_normalized_independently():
    module = LayerNormalization(4, elementwise_affine=False)

    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x2 = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    x = torch.stack([x1, x2], dim=0)

    y = module(x)

    expected_row_means = torch.zeros(2)
    expected_row_vars = torch.ones(2)

    assert torch.allclose(y.mean(dim=-1), expected_row_means, atol=1e-6, rtol=1e-6)
    assert torch.allclose(y.var(dim=-1, unbiased=False), expected_row_vars, atol=1e-5, rtol=1e-5)


def test_module_repr_contains_class_name():
    module = LayerNormalization(4)
    assert "LayerNormalization" in repr(module)
