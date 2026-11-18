import inspect

import pytest
import torch

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


def test_package_exports_expected_names():
    assert torch_gpt_2.BytePairEncoding is BytePairEncoding
    assert torch_gpt_2.GPT_2 is GPT_2
    assert torch_gpt_2.gelu is gelu
    assert torch_gpt_2.generate is generate


def test_generate_reexport_matches_module_function():
    import torch_gpt_2.gen as gen_mod

    assert torch_gpt_2.generate is gen_mod.generate


def test_gpt2_is_a_torch_module_subclass():
    assert issubclass(GPT_2, torch.nn.Module)


def test_gpt2_class_name_and_module():
    assert GPT_2.__name__ == "GPT_2"
    assert GPT_2.__module__.startswith("torch_gpt_2")


def test_bytepairencoding_is_a_class():
    assert inspect.isclass(BytePairEncoding)
    assert BytePairEncoding.__name__ == "BytePairEncoding"


def test_bytepairencoding_module_origin():
    assert BytePairEncoding.__module__.startswith("keras_gpt_2")


def test_bytepairencoding_has_expected_common_methods():
    # The exact implementation may vary, but these names are expected for a BPE helper.
    for method_name in ("encode", "decode"):
        assert hasattr(BytePairEncoding, method_name), method_name
        assert callable(getattr(BytePairEncoding, method_name)), method_name


def test_gelu_scalar_zero_is_zero():
    out = gelu(torch.tensor(0.0))
    assert out.shape == torch.Size([])
    assert out.item() == pytest.approx(0.0, abs=1e-12)


def test_gelu_preserves_shape_and_dtype_for_tensor_input():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    y = gelu(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_gelu_is_order_preserving_on_sorted_inputs():
    x = torch.tensor([-4.0, -1.5, -0.25, 0.0, 0.5, 2.0, 5.0], dtype=torch.float32)
    y = gelu(x)
    assert torch.all(y[1:] >= y[:-1])


def test_gelu_does_not_mutate_input_tensor():
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    x_before = x.clone()
    _ = gelu(x)
    assert torch.equal(x, x_before)


def test_gelu_matches_known_reference_values():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    y = gelu(x)
    expected = torch.tensor(
        [-0.04540231, -0.15880801, 0.0, 0.84119199, 1.95459769],
        dtype=torch.float32,
    )
    assert torch.allclose(y, expected, rtol=1e-4, atol=1e-4)
