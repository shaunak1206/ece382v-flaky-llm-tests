import importlib
import inspect
import sys

import pytest


def _get_pkg():
    return importlib.import_module("torch_embed_sim")


def _get_embedding_sim():
    pkg = _get_pkg()
    return pkg.EmbeddingSim


def _get_submodule():
    return importlib.import_module("torch_embed_sim.embedding_sim")


def test_package_import_exposes_embedding_sim():
    pkg = _get_pkg()
    assert hasattr(pkg, "EmbeddingSim"), "torch_embed_sim should expose EmbeddingSim at package level"


def test_embedding_sim_is_accessible_via_from_import():
    namespace = {}
    exec("from torch_embed_sim import EmbeddingSim", namespace)
    assert "EmbeddingSim" in namespace
    assert namespace["EmbeddingSim"] is _get_embedding_sim()


def test_embedding_sim_matches_submodule_attribute():
    pkg = _get_pkg()
    submod = _get_submodule()
    assert pkg.EmbeddingSim is submod.EmbeddingSim


def test_embedding_sim_is_a_class():
    EmbeddingSim = _get_embedding_sim()
    assert inspect.isclass(EmbeddingSim)


def test_embedding_sim_class_name_is_correct():
    EmbeddingSim = _get_embedding_sim()
    assert EmbeddingSim.__name__ == "EmbeddingSim"


def test_embedding_sim_qualname_contains_class_name():
    EmbeddingSim = _get_embedding_sim()
    assert "EmbeddingSim" in EmbeddingSim.__qualname__


def test_embedding_sim_defined_in_embedding_sim_module():
    EmbeddingSim = _get_embedding_sim()
    assert EmbeddingSim.__module__ == "torch_embed_sim.embedding_sim"


def test_embedding_sim_is_same_object_after_reload():
    pkg1 = _get_pkg()
    EmbeddingSim1 = pkg1.EmbeddingSim
    pkg2 = importlib.reload(pkg1)
    EmbeddingSim2 = pkg2.EmbeddingSim
    assert EmbeddingSim1 is EmbeddingSim2


def test_package_star_import_includes_embedding_sim():
    namespace = {}
    exec("from torch_embed_sim import *", namespace)
    assert "EmbeddingSim" in namespace
    assert namespace["EmbeddingSim"] is _get_embedding_sim()


def test_module_attribute_lookup_returns_embedding_sim():
    pkg = _get_pkg()
    assert getattr(pkg, "EmbeddingSim") is _get_embedding_sim()


def test_embedding_sim_has_a_callable_constructor():
    EmbeddingSim = _get_embedding_sim()
    assert callable(EmbeddingSim)


def test_embedding_sim_signature_can_be_introspected():
    EmbeddingSim = _get_embedding_sim()
    sig = inspect.signature(EmbeddingSim)
    assert isinstance(sig, inspect.Signature)


def test_importing_package_does_not_replace_sys_modules_entry():
    pkg1 = _get_pkg()
    before = sys.modules["torch_embed_sim"]
    pkg2 = importlib.import_module("torch_embed_sim")
    after = sys.modules["torch_embed_sim"]
    assert before is after
    assert pkg1 is pkg2


def test_embedding_sim_object_identity_is_stable_across_import_paths():
    direct = _get_embedding_sim()
    from_import_ns = {}
    exec("from torch_embed_sim import EmbeddingSim as ES", from_import_ns)
    assert direct is from_import_ns["ES"]


def test_embedding_sim_submodule_and_package_are_consistent():
    pkg = _get_pkg()
    submod = _get_submodule()
    assert hasattr(submod, "EmbeddingSim")
    assert submod.EmbeddingSim is pkg.EmbeddingSim
