import importlib
import sys
import types

import pytest


def _build_fake_modules():
    fake_net = types.ModuleType("torch_gpt_2.net")

    class GPT_2:
        def __init__(self, value=None):
            self.value = value

    def gelu(x):
        return ("gelu", x)

    def net_only_symbol():
        return "net-only"

    fake_net.GPT_2 = GPT_2
    fake_net.gelu = gelu
    fake_net.net_only_symbol = net_only_symbol
    fake_net.__all__ = ["GPT_2", "gelu", "net_only_symbol"]

    fake_loader = types.ModuleType("torch_gpt_2.loader")

    def loader_only_symbol():
        return "loader-only"

    fake_loader.loader_only_symbol = loader_only_symbol
    fake_loader.__all__ = ["loader_only_symbol"]

    fake_gen = types.ModuleType("torch_gpt_2.gen")

    def generate(*args, **kwargs):
        return ("generate", args, kwargs)

    fake_gen.generate = generate
    fake_gen.__all__ = ["generate"]

    fake_keras = types.ModuleType("keras_gpt_2")

    class BytePairEncoding:
        def __init__(self, vocab=None):
            self.vocab = vocab

    def get_bpe_from_files(*args, **kwargs):
        return ("bpe", args, kwargs)

    fake_keras.BytePairEncoding = BytePairEncoding
    fake_keras.get_bpe_from_files = get_bpe_from_files
    fake_keras.__all__ = ["BytePairEncoding", "get_bpe_from_files"]

    return fake_net, fake_loader, fake_gen, fake_keras


def _import_package_with_fakes(monkeypatch):
    fake_net, fake_loader, fake_gen, fake_keras = _build_fake_modules()

    # Replace any previously imported modules so the package __init__ sees our fakes.
    for name in [
        "torch_gpt_2",
        "torch_gpt_2.net",
        "torch_gpt_2.loader",
        "torch_gpt_2.gen",
        "keras_gpt_2",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    monkeypatch.setitem(sys.modules, "torch_gpt_2.net", fake_net)
    monkeypatch.setitem(sys.modules, "torch_gpt_2.loader", fake_loader)
    monkeypatch.setitem(sys.modules, "torch_gpt_2.gen", fake_gen)
    monkeypatch.setitem(sys.modules, "keras_gpt_2", fake_keras)

    pkg = importlib.import_module("torch_gpt_2")
    return pkg, fake_net, fake_loader, fake_gen, fake_keras


def test_import_exposes_generate_from_gen(monkeypatch):
    pkg, _, _, fake_gen, _ = _import_package_with_fakes(monkeypatch)
    assert pkg.generate is fake_gen.generate


def test_import_exposes_bytepairencoding_from_keras_gpt_2(monkeypatch):
    pkg, _, _, _, fake_keras = _import_package_with_fakes(monkeypatch)
    assert pkg.BytePairEncoding is fake_keras.BytePairEncoding


def test_import_exposes_get_bpe_from_files_from_keras_gpt_2(monkeypatch):
    pkg, _, _, _, fake_keras = _import_package_with_fakes(monkeypatch)
    assert pkg.get_bpe_from_files is fake_keras.get_bpe_from_files


def test_import_exposes_gpt_2_from_net(monkeypatch):
    pkg, fake_net, _, _, _ = _import_package_with_fakes(monkeypatch)
    assert pkg.GPT_2 is fake_net.GPT_2


def test_import_exposes_gelu_from_net(monkeypatch):
    pkg, fake_net, _, _, _ = _import_package_with_fakes(monkeypatch)
    assert pkg.gelu is fake_net.gelu


def test_star_import_from_net_exports_non_private_symbols(monkeypatch):
    pkg, _, _, _, _ = _import_package_with_fakes(monkeypatch)
    assert pkg.net_only_symbol() == "net-only"


def test_star_import_from_loader_exports_non_private_symbols(monkeypatch):
    pkg, _, _, _, _ = _import_package_with_fakes(monkeypatch)
    assert pkg.loader_only_symbol() == "loader-only"


def test_generate_callable_behavior_is_preserved(monkeypatch):
    pkg, _, _, _, _ = _import_package_with_fakes(monkeypatch)
    result = pkg.generate(1, 2, mode="sample")
    assert result == ("generate", (1, 2), {"mode": "sample"})


def test_bytepairencoding_is_a_class(monkeypatch):
    pkg, _, _, _, _ = _import_package_with_fakes(monkeypatch)
    assert isinstance(pkg.BytePairEncoding, type)


def test_bytepairencoding_can_be_instantiated(monkeypatch):
    pkg, _, _, _, _ = _import_package_with_fakes(monkeypatch)
    bpe = pkg.BytePairEncoding(vocab={"a": 1})
    assert bpe.vocab == {"a": 1}


def test_get_bpe_from_files_callable_returns_expected_value(monkeypatch):
    pkg, _, _, _, _ = _import_package_with_fakes(monkeypatch)
    result = pkg.get_bpe_from_files("encoder.json", "vocab.bpe")
    assert result == ("bpe", ("encoder.json", "vocab.bpe"), {})


def test_gelu_callable_behavior_is_preserved(monkeypatch):
    pkg, _, _, _, _ = _import_package_with_fakes(monkeypatch)
    assert pkg.gelu(3.14) == ("gelu", 3.14)


def test_gpt2_class_can_be_instantiated(monkeypatch):
    pkg, _, _, _, _ = _import_package_with_fakes(monkeypatch)
    model = pkg.GPT_2(value=123)
    assert model.value == 123
