import importlib
import inspect

import pytest

import torch_gpt_2
from keras_gpt_2 import BytePairEncoding, get_bpe_from_files


def test_package_exports_expected_names():
    assert hasattr(torch_gpt_2, "BytePairEncoding")
    assert hasattr(torch_gpt_2, "get_bpe_from_files")
    assert hasattr(torch_gpt_2, "GPT_2")
    assert hasattr(torch_gpt_2, "gelu")
    assert hasattr(torch_gpt_2, "generate")


def test_bytepairencoding_reexport_is_same_object():
    assert torch_gpt_2.BytePairEncoding is BytePairEncoding


def test_get_bpe_from_files_reexport_is_same_object():
    assert torch_gpt_2.get_bpe_from_files is get_bpe_from_files


def test_gpt2_reexport_is_same_object_as_net_module():
    net = importlib.import_module("torch_gpt_2.net")
    assert torch_gpt_2.GPT_2 is net.GPT_2


def test_gelu_reexport_is_same_object_as_net_module():
    net = importlib.import_module("torch_gpt_2.net")
    assert torch_gpt_2.gelu is net.gelu


def test_generate_reexport_is_same_object_as_gen_module():
    gen = importlib.import_module("torch_gpt_2.gen")
    assert torch_gpt_2.generate is gen.generate


def test_bytepairencoding_looks_like_a_class():
    assert inspect.isclass(torch_gpt_2.BytePairEncoding)


def test_gpt2_looks_like_a_class():
    assert inspect.isclass(torch_gpt_2.GPT_2)


def test_gelu_is_callable():
    assert callable(torch_gpt_2.gelu)


def test_generate_is_callable():
    assert callable(torch_gpt_2.generate)


def test_module_dir_contains_reexported_names():
    names = dir(torch_gpt_2)
    for name in ["BytePairEncoding", "get_bpe_from_files", "GPT_2", "gelu", "generate"]:
        assert name in names


def test_reload_picks_up_patched_generate(monkeypatch):
    gen = importlib.import_module("torch_gpt_2.gen")

    sentinel = object()
    monkeypatch.setattr(gen, "generate", sentinel, raising=True)

    reloaded = importlib.reload(torch_gpt_2)
    assert reloaded.generate is sentinel


def test_reload_picks_up_patched_gpt2(monkeypatch):
    net = importlib.import_module("torch_gpt_2.net")

    sentinel = object()
    monkeypatch.setattr(net, "GPT_2", sentinel, raising=True)

    reloaded = importlib.reload(torch_gpt_2)
    assert reloaded.GPT_2 is sentinel


def test_reload_picks_up_patched_gelu(monkeypatch):
    net = importlib.import_module("torch_gpt_2.net")

    sentinel = lambda *args, **kwargs: "patched"  # noqa: E731
    monkeypatch.setattr(net, "gelu", sentinel, raising=True)

    reloaded = importlib.reload(torch_gpt_2)
    assert reloaded.gelu is sentinel


def test_reload_picks_up_patched_bytepairencoding(monkeypatch):
    keras_mod = importlib.import_module("keras_gpt_2")

    sentinel = object()
    monkeypatch.setattr(keras_mod, "BytePairEncoding", sentinel, raising=True)

    reloaded = importlib.reload(torch_gpt_2)
    assert reloaded.BytePairEncoding is sentinel
