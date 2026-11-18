import importlib
import inspect

import pytest

import torch_embed_sim
import torch_embed_sim.embedding_sim as embedding_sim


def test_package_import_succeeds():
    assert torch_embed_sim is not None


def test_embedding_sim_is_exposed_on_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")


def test_embedding_sim_is_same_object_as_submodule_attribute():
    assert torch_embed_sim.EmbeddingSim is embedding_sim.EmbeddingSim


def test_direct_from_import_matches_package_attribute():
    from torch_embed_sim import EmbeddingSim

    assert EmbeddingSim is torch_embed_sim.EmbeddingSim


def test_submodule_contains_embedding_sim_name_in_dict():
    assert "EmbeddingSim" in embedding_sim.__dict__
    assert embedding_sim.__dict__["EmbeddingSim"] is torch_embed_sim.EmbeddingSim


def test_embedding_sim_is_a_class():
    assert inspect.isclass(torch_embed_sim.EmbeddingSim)


def test_embedding_sim_module_attribute_points_to_submodule():
    assert torch_embed_sim.EmbeddingSim.__module__ == embedding_sim.EmbeddingSim.__module__


def test_embedding_sim_name_matches_class_name():
    assert torch_embed_sim.EmbeddingSim.__name__ == "EmbeddingSim"


def test_dir_of_package_includes_embedding_sim():
    assert "EmbeddingSim" in dir(torch_embed_sim)


def test_star_import_from_package_exports_embedding_sim():
    namespace = {}
    exec("from torch_embed_sim import *", namespace)
    assert "EmbeddingSim" in namespace
    assert namespace["EmbeddingSim"] is torch_embed_sim.EmbeddingSim


def test_reload_package_keeps_embedding_sim_accessible():
    reloaded = importlib.reload(torch_embed_sim)
    assert hasattr(reloaded, "EmbeddingSim")
    assert reloaded.EmbeddingSim is embedding_sim.EmbeddingSim


def test_reload_submodule_keeps_embedding_sim_accessible():
    reloaded = importlib.reload(embedding_sim)
    assert hasattr(reloaded, "EmbeddingSim")
    assert torch_embed_sim.EmbeddingSim is reloaded.EmbeddingSim


def test_package_dict_contains_embedding_sim_key():
    assert "EmbeddingSim" in torch_embed_sim.__dict__


def test_package_attribute_is_same_as_getattr():
    assert getattr(torch_embed_sim, "EmbeddingSim") is torch_embed_sim.EmbeddingSim


def test_embedding_sim_identity_is_stable_with_multiple_import_styles():
    pkg = importlib.import_module("torch_embed_sim")
    subpkg = importlib.import_module("torch_embed_sim.embedding_sim")
    from torch_embed_sim import EmbeddingSim

    assert pkg.EmbeddingSim is subpkg.EmbeddingSim is EmbeddingSim
