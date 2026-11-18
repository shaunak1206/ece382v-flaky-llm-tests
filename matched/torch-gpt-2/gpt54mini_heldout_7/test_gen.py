import importlib

import keras_gpt_2
import torch_gpt_2
import torch_gpt_2.gen
import torch_gpt_2.net


def test_bytepairencoding_is_reexported_from_keras_gpt_2():
    assert torch_gpt_2.BytePairEncoding is keras_gpt_2.BytePairEncoding


def test_get_bpe_from_files_is_reexported_from_keras_gpt_2():
    assert torch_gpt_2.get_bpe_from_files is keras_gpt_2.get_bpe_from_files


def test_gpt_2_is_reexported_from_net_module():
    assert torch_gpt_2.GPT_2 is torch_gpt_2.net.GPT_2


def test_gelu_is_reexported_from_net_module():
    assert torch_gpt_2.gelu is torch_gpt_2.net.gelu


def test_generate_is_reexported_from_gen_module():
    assert torch_gpt_2.generate is torch_gpt_2.gen.generate


def test_package_exports_expected_symbols():
    for name in [
        "BytePairEncoding",
        "get_bpe_from_files",
        "GPT_2",
        "gelu",
        "generate",
    ]:
        assert hasattr(torch_gpt_2, name), name


def test_bytepairencoding_name_and_module_are_preserved():
    assert torch_gpt_2.BytePairEncoding.__name__ == keras_gpt_2.BytePairEncoding.__name__
    assert torch_gpt_2.BytePairEncoding.__module__ == keras_gpt_2.BytePairEncoding.__module__


def test_gpt_2_name_and_module_are_preserved():
    assert torch_gpt_2.GPT_2.__name__ == torch_gpt_2.net.GPT_2.__name__
    assert torch_gpt_2.GPT_2.__module__ == torch_gpt_2.net.GPT_2.__module__


def test_gelu_name_and_module_are_preserved():
    assert torch_gpt_2.gelu.__name__ == torch_gpt_2.net.gelu.__name__
    assert torch_gpt_2.gelu.__module__ == torch_gpt_2.net.gelu.__module__


def test_generate_name_and_module_are_preserved():
    assert torch_gpt_2.generate.__name__ == torch_gpt_2.gen.generate.__name__
    assert torch_gpt_2.generate.__module__ == torch_gpt_2.gen.generate.__module__


def test_reloading_package_picks_up_monkeypatched_generate(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(torch_gpt_2.gen, "generate", sentinel, raising=True)

    reloaded = importlib.reload(torch_gpt_2)
    assert reloaded.generate is sentinel


def test_reloading_package_picks_up_monkeypatched_gelu(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(torch_gpt_2.net, "gelu", sentinel, raising=True)

    reloaded = importlib.reload(torch_gpt_2)
    assert reloaded.gelu is sentinel


def test_reloading_package_picks_up_monkeypatched_gpt_2(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(torch_gpt_2.net, "GPT_2", sentinel, raising=True)

    reloaded = importlib.reload(torch_gpt_2)
    assert reloaded.GPT_2 is sentinel


def test_reloading_package_picks_up_monkeypatched_bytepairencoding(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(keras_gpt_2, "BytePairEncoding", sentinel, raising=True)

    reloaded = importlib.reload(torch_gpt_2)
    assert reloaded.BytePairEncoding is sentinel


def test_reloading_package_picks_up_monkeypatched_get_bpe_from_files(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(keras_gpt_2, "get_bpe_from_files", sentinel, raising=True)

    reloaded = importlib.reload(torch_gpt_2)
    assert reloaded.get_bpe_from_files is sentinel
