import math
import types
import inspect

import pytest
import torch

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate, get_bpe_from_files


def test_package_exports_expected_symbols():
    assert hasattr(torch_gpt_2, "BytePairEncoding")
    assert hasattr(torch_gpt_2, "get_bpe_from_files")
    assert hasattr(torch_gpt_2, "GPT_2")
    assert hasattr(torch_gpt_2, "gelu")
    assert hasattr(torch_gpt_2, "generate")


def test_byte_pair_encoding_is_reexported_same_object():
    import keras_gpt_2

    assert BytePairEncoding is keras_gpt_2.BytePairEncoding


def test_get_bpe_from_files_is_reexported_same_object():
    import keras_gpt_2

    assert get_bpe_from_files is keras_gpt_2.get_bpe_from_files


def test_generate_is_reexported_same_object():
    from torch_gpt_2.gen import generate as gen_generate

    assert generate is gen_generate


def test_gpt2_symbol_is_class():
    assert inspect.isclass(GPT_2)


def test_gelu_is_callable():
    assert callable(gelu)


def test_gelu_approximation_matches_reference_formula_on_tensor():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)
    y = gelu(x)
    expected = 0.5 * x * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))
    )
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_gelu_returns_zero_for_zero_input():
    x = torch.zeros(4, dtype=torch.float32)
    y = gelu(x)
    assert torch.equal(y, torch.zeros_like(x))


def test_gelu_preserves_shape():
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    y = gelu(x)
    assert y.shape == x.shape


def test_gelu_preserves_dtype():
    x = torch.randn(5, dtype=torch.float64)
    y = gelu(x)
    assert y.dtype == x.dtype


def test_gelu_is_elementwise_for_batched_inputs():
    x = torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.float32)
    y = gelu(x)
    y_flat = gelu(x.reshape(-1)).reshape_as(x)
    assert torch.allclose(y, y_flat, atol=1e-7, rtol=1e-7)


def test_gelu_behaves_reasonably_for_positive_values():
    x = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32)
    y = gelu(x)
    assert torch.all(y > 0)
    assert torch.all(y < x)


def test_gelu_behaves_reasonably_for_negative_values():
    x = torch.tensor([-0.5, -1.0, -2.0, -4.0], dtype=torch.float32)
    y = gelu(x)
    assert torch.all(y < 0)
    assert torch.all(y > x)


def test_package_allows_star_import_of_key_symbols():
    namespace = {}
    exec("from torch_gpt_2 import *", namespace)
    assert namespace["BytePairEncoding"] is BytePairEncoding
    assert namespace["GPT_2"] is GPT_2
    assert namespace["gelu"] is gelu
    assert namespace["generate"] is generate
    assert namespace["get_bpe_from_files"] is get_bpe_from_files
