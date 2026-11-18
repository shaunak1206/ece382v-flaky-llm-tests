import importlib
import inspect

import torch_embed_sim


def test_package_exports_embedding_sim():
    assert hasattr(torch_embed_sim, "EmbeddingSim")


def test_embedding_sim_is_class():
    assert inspect.isclass(torch_embed_sim.EmbeddingSim)


def test_from_import_matches_package_attribute():
    from torch_embed_sim import EmbeddingSim

    assert EmbeddingSim is torch_embed_sim.EmbeddingSim


def test_submodule_class_matches_package_attribute():
    submodule = importlib.import_module("torch_embed_sim.embedding_sim")
    assert torch_embed_sim.EmbeddingSim is submodule.EmbeddingSim


def test_embedding_sim_name_is_stable():
    assert torch_embed_sim.EmbeddingSim.__name__ == "EmbeddingSim"


def test_embedding_sim_qualname_contains_class_name():
    assert "EmbeddingSim" in torch_embed_sim.EmbeddingSim.__qualname__


def test_embedding_sim_module_attribute_points_to_submodule():
    assert torch_embed_sim.EmbeddingSim.__module__ == "torch_embed_sim.embedding_sim"


def test_dir_includes_embedding_sim():
    assert "EmbeddingSim" in dir(torch_embed_sim)


def test_package_dict_contains_embedding_sim():
    assert "EmbeddingSim" in torch_embed_sim.__dict__


def test_import_module_returns_same_object_on_reimport():
    mod1 = importlib.import_module("torch_embed_sim")
    mod2 = importlib.import_module("torch_embed_sim")
    assert mod1.EmbeddingSim is mod2.EmbeddingSim


def test_star_import_exposes_embedding_sim():
    namespace = {}
    exec("from torch_embed_sim import *", namespace)
    assert "EmbeddingSim" in namespace
    assert namespace["EmbeddingSim"] is torch_embed_sim.EmbeddingSim


def test_embedding_sim_is_callable():
    assert callable(torch_embed_sim.EmbeddingSim)
