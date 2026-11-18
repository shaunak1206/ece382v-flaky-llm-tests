import importlib
import inspect
import math

import torch
import torch.nn as nn
from keras_gpt_2 import BytePairEncoding as KerasBytePairEncoding
from keras_gpt_2 import get_bpe_from_files as keras_get_bpe_from_files


def test_package_exports_generate_and_bpe_helpers():
    mod = importlib.import_module("torch_gpt_2")
    assert hasattr(mod, "generate")
    assert hasattr(mod, "BytePairEncoding")
    assert hasattr(mod, "get_bpe_from_files")


def test_byte_pair_encoding_is_reexported_identity():
    mod = importlib.import_module("torch_gpt_2")
    assert mod.BytePairEncoding is KerasBytePairEncoding


def test_get_bpe_from_files_is_reexported_identity():
    mod = importlib.import_module("torch_gpt_2")
    assert mod.get_bpe_from_files is keras_get_bpe_from_files


def test_generate_is_reexported_from_gen_module():
    mod = importlib.import_module("torch_gpt_2")
    gen_mod = importlib.import_module("torch_gpt_2.gen")
    assert mod.generate is gen_mod.generate


def test_gelu_is_available_and_callable():
    mod = importlib.import_module("torch_gpt_2")
    assert hasattr(mod, "gelu")
    assert callable(mod.gelu)


def test_gelu_matches_zero_at_origin():
    mod = importlib.import_module("torch_gpt_2")
    x = torch.tensor(0.0)
    y = mod.gelu(x)
    assert torch.is_tensor(y)
    assert y.item() == 0.0


def test_gelu_preserves_input_shape():
    mod = importlib.import_module("torch_gpt_2")
    x = torch.randn(2, 3, 4)
    y = mod.gelu(x)
    assert y.shape == x.shape


def test_gelu_positive_input_is_not_amplified():
    mod = importlib.import_module("torch_gpt_2")
    x = torch.tensor([0.1, 1.0, 3.0])
    y = mod.gelu(x)
    assert torch.all(y <= x + 1e-6)


def test_gelu_negative_input_remains_non_positive():
    mod = importlib.import_module("torch_gpt_2")
    x = torch.tensor([-0.1, -1.0, -3.0])
    y = mod.gelu(x)
    assert torch.all(y <= 1e-6)


def test_gelu_outputs_are_finite_for_typical_inputs():
    mod = importlib.import_module("torch_gpt_2")
    x = torch.linspace(-10, 10, steps=100)
    y = mod.gelu(x)
    assert torch.isfinite(y).all()


def test_gpt2_is_available_and_is_a_class():
    mod = importlib.import_module("torch_gpt_2")
    assert hasattr(mod, "GPT_2")
    assert inspect.isclass(mod.GPT_2)


def test_gpt2_is_a_torch_module_subclass():
    mod = importlib.import_module("torch_gpt_2")
    assert issubclass(mod.GPT_2, nn.Module)


def test_gpt2_has_forward_method():
    mod = importlib.import_module("torch_gpt_2")
    assert hasattr(mod.GPT_2, "forward")
    assert callable(getattr(mod.GPT_2, "forward"))


def test_gpt2_construction_signature_is_introspectable():
    mod = importlib.import_module("torch_gpt_2")
    sig = inspect.signature(mod.GPT_2)
    assert len(sig.parameters) >= 1


def test_package_star_import_exposes_expected_symbols():
    namespace = {}
    exec("from torch_gpt_2 import *", namespace)
    assert "generate" in namespace
    assert "BytePairEncoding" in namespace
    assert "get_bpe_from_files" in namespace
    assert "GPT_2" in namespace
    assert "gelu" in namespace


def test_generate_has_callable_signature():
    mod = importlib.import_module("torch_gpt_2")
    sig = inspect.signature(mod.generate)
    assert callable(mod.generate)
    assert len(sig.parameters) >= 1
