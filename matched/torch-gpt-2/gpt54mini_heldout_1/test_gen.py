import importlib
import sys
import types

import pytest


def _make_module(name, attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _reload_torch_gpt_2(monkeypatch, net_attrs=None, loader_attrs=None, gen_attrs=None, keras_attrs=None):
    """
    Reload torch_gpt_2 with fake dependency modules so we can test __init__.py
    re-export behavior deterministically.
    """
    net_attrs = net_attrs or {}
    loader_attrs = loader_attrs or {}
    gen_attrs = gen_attrs or {}
    keras_attrs = keras_attrs or {}

    fake_net = _make_module("torch_gpt_2.net", net_attrs)
    fake_loader = _make_module("torch_gpt_2.loader", loader_attrs)
    fake_gen = _make_module("torch_gpt_2.gen", gen_attrs)
    fake_keras = _make_module("keras_gpt_2", keras_attrs)

    monkeypatch.setitem(sys.modules, "torch_gpt_2.net", fake_net)
    monkeypatch.setitem(sys.modules, "torch_gpt_2.loader", fake_loader)
    monkeypatch.setitem(sys.modules, "torch_gpt_2.gen", fake_gen)
    monkeypatch.setitem(sys.modules, "keras_gpt_2", fake_keras)

    import torch_gpt_2

    return importlib.reload(torch_gpt_2), fake_net, fake_loader, fake_gen, fake_keras


def test_reexports_symbol_from_net(monkeypatch):
    pkg, _, _, _, _ = _reload_torch_gpt_2(monkeypatch, net_attrs={"net_symbol": 123})
    assert pkg.net_symbol == 123


def test_reexports_symbol_from_loader(monkeypatch):
    pkg, _, _, _, _ = _reload_torch_gpt_2(monkeypatch, loader_attrs={"loader_symbol": "loaded"})
    assert pkg.loader_symbol == "loaded"


def test_reexports_generate_from_gen(monkeypatch):
    sentinel = object()

    def fake_generate():
        return sentinel

    pkg, _, _, fake_gen, _ = _reload_torch_gpt_2(monkeypatch, gen_attrs={"generate": fake_generate})
    assert pkg.generate is fake_gen.generate
    assert pkg.generate() is sentinel


def test_reexports_byte_pair_encoding_from_keras_gpt_2(monkeypatch):
    class FakeBPE:
        pass

    pkg, _, _, _, fake_keras = _reload_torch_gpt_2(monkeypatch, keras_attrs={"BytePairEncoding": FakeBPE})
    assert pkg.BytePairEncoding is fake_keras.BytePairEncoding


def test_reexports_get_bpe_from_files_from_keras_gpt_2(monkeypatch):
    def fake_get_bpe_from_files(*args, **kwargs):
        return args, kwargs

    pkg, _, _, _, fake_keras = _reload_torch_gpt_2(
        monkeypatch, keras_attrs={"get_bpe_from_files": fake_get_bpe_from_files}
    )
    assert pkg.get_bpe_from_files is fake_keras.get_bpe_from_files
    assert pkg.get_bpe_from_files(1, a=2) == ((1,), {"a": 2})


def test_star_import_includes_net_symbols(monkeypatch):
    pkg, _, _, _, _ = _reload_torch_gpt_2(monkeypatch, net_attrs={"alpha": "a", "beta": "b"})

    ns = {}
    exec("from torch_gpt_2 import *", ns, ns)

    assert ns["alpha"] == "a"
    assert ns["beta"] == "b"
    assert ns["generate"] is pkg.generate
    assert ns["BytePairEncoding"] is pkg.BytePairEncoding


def test_star_import_includes_loader_symbols(monkeypatch):
    pkg, _, _, _, _ = _reload_torch_gpt_2(monkeypatch, loader_attrs={"gamma": 3.14})

    ns = {}
    exec("from torch_gpt_2 import *", ns, ns)

    assert ns["gamma"] == 3.14
    assert ns["generate"] is pkg.generate


def test_star_import_includes_keras_symbols(monkeypatch):
    class FakeBPE:
        pass

    pkg, _, _, _, _ = _reload_torch_gpt_2(monkeypatch, keras_attrs={"BytePairEncoding": FakeBPE})

    ns = {}
    exec("from torch_gpt_2 import *", ns, ns)

    assert ns["BytePairEncoding"] is pkg.BytePairEncoding
    assert ns["get_bpe_from_files"] is pkg.get_bpe_from_files


def test_star_import_does_not_export_underscore_prefixed_net_symbols(monkeypatch):
    pkg, _, _, _, _ = _reload_torch_gpt_2(monkeypatch, net_attrs={"_hidden": "secret", "visible": "ok"})

    ns = {}
    exec("from torch_gpt_2 import *", ns, ns)

    assert "visible" in ns
    assert "_hidden" not in ns


def test_star_import_does_not_export_underscore_prefixed_loader_symbols(monkeypatch):
    pkg, _, _, _, _ = _reload_torch_gpt_2(monkeypatch, loader_attrs={"_internal": 7, "public_name": 8})

    ns = {}
    exec("from torch_gpt_2 import *", ns, ns)

    assert "public_name" in ns
    assert "_internal" not in ns


def test_loader_symbol_can_override_net_symbol_on_same_name(monkeypatch):
    pkg, _, _, _, _ = _reload_torch_gpt_2(
        monkeypatch,
        net_attrs={"shared_name": "net"},
        loader_attrs={"shared_name": "loader"},
    )

    # Because __init__.py does `from .net import *` followed by `from .loader import *`,
    # later imports should win on name collisions.
    assert pkg.shared_name == "loader"


def test_reload_updates_reexported_generate(monkeypatch):
    def gen_v1():
        return "v1"

    def gen_v2():
        return "v2"

    pkg, _, _, _, _ = _reload_torch_gpt_2(monkeypatch, gen_attrs={"generate": gen_v1})
    assert pkg.generate() == "v1"

    # Reload with a different fake gen module; exported name should update too.
    pkg2, _, _, _, _ = _reload_torch_gpt_2(monkeypatch, gen_attrs={"generate": gen_v2})
    assert pkg2.generate() == "v2"


def test_reload_updates_reexported_bytepairencoding(monkeypatch):
    class BPE1:
        pass

    class BPE2:
        pass

    pkg, _, _, _, _ = _reload_torch_gpt_2(monkeypatch, keras_attrs={"BytePairEncoding": BPE1})
    assert pkg.BytePairEncoding is BPE1

    pkg2, _, _, _, _ = _reload_torch_gpt_2(monkeypatch, keras_attrs={"BytePairEncoding": BPE2})
    assert pkg2.BytePairEncoding is BPE2


def test_reexports_are_accessible_via_from_import(monkeypatch):
    class FakeBPE:
        pass

    _reload_torch_gpt_2(
        monkeypatch,
        net_attrs={"n1": 1},
        loader_attrs={"l1": 2},
        gen_attrs={"generate": lambda: "gen"},
        keras_attrs={"BytePairEncoding": FakeBPE, "get_bpe_from_files": lambda: "bpe"},
    )

    from torch_gpt_2 import BytePairEncoding, generate, get_bpe_from_files, l1, n1

    assert n1 == 1
    assert l1 == 2
    assert generate() == "gen"
    assert BytePairEncoding is FakeBPE
    assert get_bpe_from_files() == "bpe"
