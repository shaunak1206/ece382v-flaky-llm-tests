import importlib
import sys
import types

import pytest


def _make_fake_embedding_sim_module(**attrs):
    mod = types.ModuleType("torch_embed_sim.embedding_sim")
    for name, value in attrs.items():
        setattr(mod, name, value)
    return mod


def _reload_torch_embed_sim(monkeypatch, fake_submodule):
    # Ensure the relative import in torch_embed_sim.__init__ resolves to our fake module.
    monkeypatch.setitem(sys.modules, "torch_embed_sim.embedding_sim", fake_submodule)

    import torch_embed_sim

    return importlib.reload(torch_embed_sim)


def test_package_exports_embedding_sim_from_real_module():
    import torch_embed_sim
    from torch_embed_sim.embedding_sim import EmbeddingSim as direct

    assert torch_embed_sim.EmbeddingSim is direct


def test_from_package_import_embedding_sim_matches_direct_import():
    from torch_embed_sim import EmbeddingSim
    from torch_embed_sim.embedding_sim import EmbeddingSim as direct

    assert EmbeddingSim is direct


def test_package_module_has_embedding_sim_attribute():
    import torch_embed_sim

    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert torch_embed_sim.EmbeddingSim is not None


def test_package_dir_contains_embedding_sim():
    import torch_embed_sim

    assert "EmbeddingSim" in dir(torch_embed_sim)


def test_reload_reexports_symbol_from_fake_submodule(monkeypatch):
    sentinel = object()
    fake = _make_fake_embedding_sim_module(EmbeddingSim=sentinel)
    pkg = _reload_torch_embed_sim(monkeypatch, fake)

    assert pkg.EmbeddingSim is sentinel


def test_reload_allows_star_imported_symbol_to_be_used_from_package_namespace(monkeypatch):
    sentinel = object()
    fake = _make_fake_embedding_sim_module(EmbeddingSim=sentinel)
    pkg = _reload_torch_embed_sim(monkeypatch, fake)

    ns = {}
    exec("from torch_embed_sim import EmbeddingSim", ns, ns)

    assert ns["EmbeddingSim"] is sentinel
    assert pkg.EmbeddingSim is sentinel


def test_reload_reexports_multiple_names_when_submodule_defines_all(monkeypatch):
    sentinel = object()
    another = object()
    fake = _make_fake_embedding_sim_module(
        EmbeddingSim=sentinel,
        OTHER_NAME=another,
        __all__=["EmbeddingSim", "OTHER_NAME"],
    )
    pkg = _reload_torch_embed_sim(monkeypatch, fake)

    assert pkg.EmbeddingSim is sentinel
    assert pkg.OTHER_NAME is another


def test_reload_respects_submodule_all_on_star_import(monkeypatch):
    sentinel = object()
    hidden = object()
    fake = _make_fake_embedding_sim_module(
        EmbeddingSim=sentinel,
        HIDDEN=hidden,
        __all__=["EmbeddingSim"],
    )
    pkg = _reload_torch_embed_sim(monkeypatch, fake)

    assert pkg.EmbeddingSim is sentinel
    assert not hasattr(pkg, "HIDDEN")


def test_reload_keeps_export_name_stable_across_multiple_reloads(monkeypatch):
    first = object()
    second = object()

    fake1 = _make_fake_embedding_sim_module(EmbeddingSim=first)
    pkg = _reload_torch_embed_sim(monkeypatch, fake1)
    assert pkg.EmbeddingSim is first

    fake2 = _make_fake_embedding_sim_module(EmbeddingSim=second)
    pkg = _reload_torch_embed_sim(monkeypatch, fake2)
    assert pkg.EmbeddingSim is second


def test_reload_uses_module_object_from_sys_modules(monkeypatch):
    sentinel = object()
    fake = _make_fake_embedding_sim_module(EmbeddingSim=sentinel)
    pkg = _reload_torch_embed_sim(monkeypatch, fake)

    assert sys.modules["torch_embed_sim.embedding_sim"] is fake
    assert pkg.EmbeddingSim is sentinel


def test_reload_does_not_require_real_embedding_sim_implementation(monkeypatch):
    sentinel = object()
    fake = _make_fake_embedding_sim_module(EmbeddingSim=sentinel)
    pkg = _reload_torch_embed_sim(monkeypatch, fake)

    assert getattr(pkg, "EmbeddingSim") is sentinel


def test_reload_allows_import_module_to_return_same_package_object(monkeypatch):
    sentinel = object()
    fake = _make_fake_embedding_sim_module(EmbeddingSim=sentinel)
    pkg1 = _reload_torch_embed_sim(monkeypatch, fake)
    pkg2 = importlib.import_module("torch_embed_sim")

    assert pkg1 is pkg2
    assert pkg2.EmbeddingSim is sentinel


def test_package_import_does_not_break_when_reloaded_with_fake_submodule(monkeypatch):
    sentinel = object()
    fake = _make_fake_embedding_sim_module(EmbeddingSim=sentinel)
    pkg = _reload_torch_embed_sim(monkeypatch, fake)

    assert isinstance(pkg, types.ModuleType)
    assert pkg.__name__ == "torch_embed_sim"
    assert pkg.EmbeddingSim is sentinel
