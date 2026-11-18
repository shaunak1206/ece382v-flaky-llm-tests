import importlib
import sys
import types

import pytest


def _import_package():
    import torch_embed_sim

    return torch_embed_sim


def _real_submodule():
    return importlib.import_module("torch_embed_sim.embedding_sim")


def _reload_package_with_submodule(monkeypatch, submodule):
    """
    Force torch_embed_sim to re-execute __init__.py while using a provided
    torch_embed_sim.embedding_sim module object.
    """
    monkeypatch.setitem(sys.modules, "torch_embed_sim.embedding_sim", submodule)
    pkg = _import_package()
    return importlib.reload(pkg)


def test_package_exports_embedding_sim_from_real_submodule():
    pkg = _import_package()
    sub = _real_submodule()

    assert hasattr(pkg, "EmbeddingSim")
    assert pkg.EmbeddingSim is sub.EmbeddingSim


def test_embedding_sim_is_accessible_via_from_import():
    from torch_embed_sim import EmbeddingSim

    sub = _real_submodule()
    assert EmbeddingSim is sub.EmbeddingSim


def test_embedding_sim_is_present_in_dir():
    pkg = _import_package()

    assert "EmbeddingSim" in dir(pkg)


def test_embedding_sim_is_present_in_module_dict():
    pkg = _import_package()

    assert "EmbeddingSim" in pkg.__dict__
    assert pkg.__dict__["EmbeddingSim"] is pkg.EmbeddingSim


def test_package_embedding_sim_is_a_class():
    pkg = _import_package()

    assert isinstance(pkg.EmbeddingSim, type)


def test_package_embedding_sim_matches_submodule_object_identity():
    pkg = _import_package()
    sub = _real_submodule()

    assert pkg.EmbeddingSim is sub.EmbeddingSim
    assert pkg.EmbeddingSim.__module__ == sub.EmbeddingSim.__module__


def test_reloading_package_preserves_embedding_sim_identity():
    pkg = _import_package()
    before = pkg.EmbeddingSim

    reloaded = importlib.reload(pkg)

    assert reloaded.EmbeddingSim is before


def test_reload_uses_fake_submodule_public_names(monkeypatch):
    fake = types.ModuleType("torch_embed_sim.embedding_sim")

    class EmbeddingSim:
        sentinel = "fake"

    fake.EmbeddingSim = EmbeddingSim
    fake.PublicValue = 123

    pkg = _reload_package_with_submodule(monkeypatch, fake)

    assert pkg.EmbeddingSim is EmbeddingSim
    assert pkg.PublicValue == 123


def test_reload_excludes_private_names_from_fake_submodule(monkeypatch):
    fake = types.ModuleType("torch_embed_sim.embedding_sim")

    class EmbeddingSim:
        pass

    fake.EmbeddingSim = EmbeddingSim
    fake._hidden_value = "secret"
    fake.visible_value = "shown"

    pkg = _reload_package_with_submodule(monkeypatch, fake)

    assert hasattr(pkg, "EmbeddingSim")
    assert hasattr(pkg, "visible_value")
    assert not hasattr(pkg, "_hidden_value")


def test_reload_overwrites_previous_public_names_from_fake_submodule(monkeypatch):
    fake1 = types.ModuleType("torch_embed_sim.embedding_sim")

    class EmbeddingSim1:
        pass

    fake1.EmbeddingSim = EmbeddingSim1
    fake1.marker = "first"

    pkg = _reload_package_with_submodule(monkeypatch, fake1)
    assert pkg.marker == "first"

    fake2 = types.ModuleType("torch_embed_sim.embedding_sim")

    class EmbeddingSim2:
        pass

    fake2.EmbeddingSim = EmbeddingSim2
    fake2.marker = "second"

    pkg = _reload_package_with_submodule(monkeypatch, fake2)

    assert pkg.EmbeddingSim is EmbeddingSim2
    assert pkg.marker == "second"


def test_reload_with_fake_submodule_keeps_only_public_exports(monkeypatch):
    fake = types.ModuleType("torch_embed_sim.embedding_sim")

    class EmbeddingSim:
        pass

    def helper():
        return "ok"

    fake.EmbeddingSim = EmbeddingSim
    fake.helper = helper
    fake._internal = object()

    pkg = _reload_package_with_submodule(monkeypatch, fake)

    assert "EmbeddingSim" in pkg.__dict__
    assert "helper" in pkg.__dict__
    assert "_internal" not in pkg.__dict__
    assert pkg.helper is helper


def test_reload_with_fake_submodule_import_star_behavior(monkeypatch):
    fake = types.ModuleType("torch_embed_sim.embedding_sim")

    class EmbeddingSim:
        pass

    fake.EmbeddingSim = EmbeddingSim
    fake.alpha = 1
    fake.beta = 2

    pkg = _reload_package_with_submodule(monkeypatch, fake)

    namespace = {}
    exec("from torch_embed_sim import *", namespace)

    assert namespace["EmbeddingSim"] is EmbeddingSim
    assert namespace["alpha"] == 1
    assert namespace["beta"] == 2
    assert "_internal" not in namespace
