import importlib
import inspect
import sys

import pytest

import torch_embed_sim
import torch_embed_sim.embedding_sim as embedding_sim_mod


def test_package_import_exposes_embedding_sim():
    assert hasattr(torch_embed_sim, "EmbeddingSim"), "torch_embed_sim should re-export EmbeddingSim"


def test_embedding_sim_is_same_object_as_submodule_definition():
    assert torch_embed_sim.EmbeddingSim is embedding_sim_mod.EmbeddingSim


def test_from_import_returns_embedding_sim_class():
    from torch_embed_sim import EmbeddingSim

    assert EmbeddingSim is torch_embed_sim.EmbeddingSim


def test_star_import_brings_embedding_sim_into_namespace():
    namespace = {}
    exec("from torch_embed_sim import *", namespace)

    assert "EmbeddingSim" in namespace
    assert namespace["EmbeddingSim"] is torch_embed_sim.EmbeddingSim


def test_dir_contains_embedding_sim():
    assert "EmbeddingSim" in dir(torch_embed_sim)


def test_module_dict_contains_embedding_sim():
    assert torch_embed_sim.__dict__["EmbeddingSim"] is torch_embed_sim.EmbeddingSim


def test_submodule_is_importable():
    mod = importlib.import_module("torch_embed_sim.embedding_sim")
    assert mod is embedding_sim_mod
    assert hasattr(mod, "EmbeddingSim")


def test_package_name_is_torch_embed_sim():
    assert torch_embed_sim.__name__ == "torch_embed_sim"


def test_embedding_sim_is_a_class():
    assert inspect.isclass(torch_embed_sim.EmbeddingSim)


def test_embedding_sim_class_can_be_reflected_from_submodule():
    cls = getattr(importlib.import_module("torch_embed_sim.embedding_sim"), "EmbeddingSim")
    assert cls.__module__ == "torch_embed_sim.embedding_sim"


def test_reimporting_package_preserves_embedding_sim_identity():
    mod1 = torch_embed_sim
    mod2 = importlib.import_module("torch_embed_sim")
    assert mod1 is mod2
    assert mod2.EmbeddingSim is embedding_sim_mod.EmbeddingSim


def test_reloading_package_keeps_embedding_sim_accessible():
    reloaded = importlib.reload(torch_embed_sim)
    assert hasattr(reloaded, "EmbeddingSim")
    assert reloaded.EmbeddingSim is embedding_sim_mod.EmbeddingSim


def test_reloading_submodule_keeps_embedding_sim_accessible():
    reloaded_submodule = importlib.reload(embedding_sim_mod)
    assert hasattr(reloaded_submodule, "EmbeddingSim")
    assert reloaded_submodule.EmbeddingSim is torch_embed_sim.EmbeddingSim


def test_from_import_after_reload_still_works():
    importlib.reload(torch_embed_sim)
    from torch_embed_sim import EmbeddingSim

    assert EmbeddingSim is torch_embed_sim.EmbeddingSim


def test_star_import_after_reload_still_works():
    importlib.reload(torch_embed_sim)
    namespace = {}
    exec("from torch_embed_sim import *", namespace)

    assert namespace["EmbeddingSim"] is torch_embed_sim.EmbeddingSim


def test_package_and_submodule_share_same_embedding_sim_name():
    assert torch_embed_sim.EmbeddingSim.__name__ == embedding_sim_mod.EmbeddingSim.__name__


def test_embedding_sim_is_accessible_through_sys_modules():
    assert sys.modules["torch_embed_sim"] is torch_embed_sim
    assert sys.modules["torch_embed_sim.embedding_sim"] is embedding_sim_mod
    assert sys.modules["torch_embed_sim"].EmbeddingSim is sys.modules["torch_embed_sim.embedding_sim"].EmbeddingSim
