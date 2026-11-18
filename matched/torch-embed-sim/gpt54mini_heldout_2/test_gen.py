import importlib
import sys
import types

import pytest

import torch_embed_sim


def _make_fake_embedding_sim_module(name="torch_embed_sim.embedding_sim", *, all_names=None, extra_attrs=None):
    mod = types.ModuleType(name)
    if all_names is not None:
        mod.__all__ = list(all_names)
    if extra_attrs:
        for k, v in extra_attrs.items():
            setattr(mod, k, v)
    return mod


def _reload_package_with_fake_submodule(monkeypatch, fake_submodule):
    monkeypatch.setitem(sys.modules, "torch_embed_sim.embedding_sim", fake_submodule)
    return importlib.reload(torch_embed_sim)


def test_package_reexports_symbol_listed_in_submodule___all__(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["EmbeddingSim"],
        extra_attrs={"EmbeddingSim": object()},
    )
    pkg = _reload_package_with_fake_submodule(monkeypatch, fake)
    assert hasattr(pkg, "EmbeddingSim")
    assert pkg.EmbeddingSim is fake.EmbeddingSim


def test_package_does_not_export_private_names_from_submodule(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["public_name"],
        extra_attrs={"public_name": 1, "_private_name": 2},
    )
    pkg = _reload_package_with_fake_submodule(monkeypatch, fake)
    assert hasattr(pkg, "public_name")
    assert not hasattr(pkg, "_private_name")


def test_package_does_not_export_names_not_in___all__(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["kept_name"],
        extra_attrs={"kept_name": 1, "dropped_name": 2},
    )
    pkg = _reload_package_with_fake_submodule(monkeypatch, fake)
    assert hasattr(pkg, "kept_name")
    assert not hasattr(pkg, "dropped_name")


def test_star_import_from_package_imports_reexported_name(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["EmbeddingSim"],
        extra_attrs={"EmbeddingSim": "sentinel"},
    )
    _reload_package_with_fake_submodule(monkeypatch, fake)

    ns = {}
    exec("from torch_embed_sim import *", ns, ns)
    assert ns["EmbeddingSim"] == "sentinel"


def test_star_import_from_package_does_not_import_private_name(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["public_name"],
        extra_attrs={"public_name": 1, "_private_name": 2},
    )
    _reload_package_with_fake_submodule(monkeypatch, fake)

    ns = {}
    exec("from torch_embed_sim import *", ns, ns)
    assert "public_name" in ns
    assert "_private_name" not in ns


def test_package_attribute_is_same_object_as_submodule_attribute(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["EmbeddingSim"],
        extra_attrs={"EmbeddingSim": object()},
    )
    pkg = _reload_package_with_fake_submodule(monkeypatch, fake)
    submod = sys.modules["torch_embed_sim.embedding_sim"]
    assert pkg.EmbeddingSim is submod.EmbeddingSim


def test_reload_reflects_updated_submodule_exports(monkeypatch):
    fake1 = _make_fake_embedding_sim_module(
        all_names=["first_name"],
        extra_attrs={"first_name": "a"},
    )
    pkg = _reload_package_with_fake_submodule(monkeypatch, fake1)
    assert hasattr(pkg, "first_name")
    assert not hasattr(pkg, "second_name")

    fake2 = _make_fake_embedding_sim_module(
        all_names=["second_name"],
        extra_attrs={"second_name": "b"},
    )
    monkeypatch.setitem(sys.modules, "torch_embed_sim.embedding_sim", fake2)
    pkg2 = importlib.reload(torch_embed_sim)

    assert not hasattr(pkg2, "first_name")
    assert hasattr(pkg2, "second_name")
    assert pkg2.second_name == "b"


def test_package_does_not_expose_underlying_submodule_name_by_default(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["visible"],
        extra_attrs={"visible": 123},
    )
    pkg = _reload_package_with_fake_submodule(monkeypatch, fake)
    assert not hasattr(pkg, "embedding_sim")
    assert hasattr(pkg, "visible")


def test_importing_package_multiple_times_is_stable(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["EmbeddingSim"],
        extra_attrs={"EmbeddingSim": object()},
    )
    pkg1 = _reload_package_with_fake_submodule(monkeypatch, fake)
    pkg2 = importlib.reload(torch_embed_sim)
    assert pkg1.EmbeddingSim is pkg2.EmbeddingSim


def test_package_dir_contains_reexported_name(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["EmbeddingSim"],
        extra_attrs={"EmbeddingSim": object()},
    )
    pkg = _reload_package_with_fake_submodule(monkeypatch, fake)
    assert "EmbeddingSim" in dir(pkg)


def test_package_getattr_via_direct_access_works_after_reload(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["EmbeddingSim"],
        extra_attrs={"EmbeddingSim": 42},
    )
    pkg = _reload_package_with_fake_submodule(monkeypatch, fake)
    assert getattr(pkg, "EmbeddingSim") == 42


def test_package_exports_match_fake_submodule___all__(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["alpha", "beta"],
        extra_attrs={"alpha": 1, "beta": 2, "gamma": 3},
    )
    pkg = _reload_package_with_fake_submodule(monkeypatch, fake)
    exported = {name for name in dir(pkg) if not name.startswith("_")}
    assert {"alpha", "beta"}.issubset(exported)
    assert "gamma" not in exported


def test_package_allows_multiple_public_symbols_from_submodule(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["EmbeddingSim", "OtherThing"],
        extra_attrs={"EmbeddingSim": "x", "OtherThing": "y"},
    )
    pkg = _reload_package_with_fake_submodule(monkeypatch, fake)
    assert pkg.EmbeddingSim == "x"
    assert pkg.OtherThing == "y"


def test_reload_with_fake_submodule_preserves_module_identity(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["EmbeddingSim"],
        extra_attrs={"EmbeddingSim": "value"},
    )
    monkeypatch.setitem(sys.modules, "torch_embed_sim.embedding_sim", fake)
    pkg_before = torch_embed_sim
    pkg_after = importlib.reload(torch_embed_sim)
    assert pkg_before is pkg_after
    assert pkg_after.EmbeddingSim == "value"


def test_package_does_not_require_submodule_private_names_to_be_present(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["public"],
        extra_attrs={"public": "ok"},
    )
    pkg = _reload_package_with_fake_submodule(monkeypatch, fake)
    assert hasattr(pkg, "public")
    assert not hasattr(pkg, "_anything")


def test_from_import_star_honors_updated___all__(monkeypatch):
    fake = _make_fake_embedding_sim_module(
        all_names=["first", "second"],
        extra_attrs={"first": 1, "second": 2},
    )
    _reload_package_with_fake_submodule(monkeypatch, fake)

    ns = {}
    exec("from torch_embed_sim import *", ns, ns)
    assert ns["first"] == 1
    assert ns["second"] == 2
    assert "_anything" not in ns
