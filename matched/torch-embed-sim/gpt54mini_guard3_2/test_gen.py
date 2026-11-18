import importlib
import inspect

import pytest

import torch_embed_sim
from torch_embed_sim import EmbeddingSim
from torch_embed_sim import embedding_sim as embedding_sim_module


def test_embedding_sim_is_exported_from_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert torch_embed_sim.EmbeddingSim is EmbeddingSim


def test_embedding_sim_matches_submodule_definition():
    assert EmbeddingSim is embedding_sim_module.EmbeddingSim


def test_embedding_sim_is_a_class():
    assert inspect.isclass(EmbeddingSim)
    assert isinstance(EmbeddingSim, type)


def test_embedding_sim_has_expected_name():
    assert EmbeddingSim.__name__ == "EmbeddingSim"


def test_embedding_sim_module_name_points_inside_package():
    assert EmbeddingSim.__module__.startswith("torch_embed_sim")


def test_embedding_sim_is_in_module_dict():
    assert "EmbeddingSim" in torch_embed_sim.__dict__
    assert torch_embed_sim.__dict__["EmbeddingSim"] is EmbeddingSim


def test_star_import_exposes_embedding_sim():
    namespace = {}
    exec("from torch_embed_sim import *", namespace)
    assert "EmbeddingSim" in namespace
    assert namespace["EmbeddingSim"] is EmbeddingSim


def test_multiple_imports_return_same_object():
    mod1 = importlib.import_module("torch_embed_sim")
    mod2 = importlib.import_module("torch_embed_sim")
    assert mod1.EmbeddingSim is mod2.EmbeddingSim is EmbeddingSim


def test_reload_preserves_embedding_sim_export():
    reloaded = importlib.reload(torch_embed_sim)
    assert hasattr(reloaded, "EmbeddingSim")
    assert reloaded.EmbeddingSim is EmbeddingSim


def test_embedding_sim_mro_starts_with_itself_and_object():
    mro = inspect.getmro(EmbeddingSim)
    assert mro[0] is EmbeddingSim
    assert mro[-1] is object


def test_embedding_sim_repr_contains_class_name():
    # repr(type) is stable and should identify the class by name.
    assert "EmbeddingSim" in repr(EmbeddingSim)


def test_embedding_sim_is_callable_like_a_class():
    assert callable(EmbeddingSim)
    assert hasattr(EmbeddingSim, "__call__")
