import importlib
import sys
import types

import pytest

import torch_embed_sim


def test_package_exports_embeddingsim():
    assert hasattr(torch_embed_sim, "EmbeddingSim")


def test_embeddingsim_is_a_class():
    assert isinstance(torch_embed_sim.EmbeddingSim, type)


def test_embeddingsim_reexport_matches_submodule():
    from torch_embed_sim.embedding_sim import EmbeddingSim as SubmoduleEmbeddingSim

    assert torch_embed_sim.EmbeddingSim is SubmoduleEmbeddingSim


def test_embeddingsim_available_in_module_dict():
    assert torch_embed_sim.__dict__["EmbeddingSim"] is torch_embed_sim.EmbeddingSim


def test_import_star_exposes_embeddingsim():
    ns = {}
    exec("from torch_embed_sim import *", ns)
    assert "EmbeddingSim" in ns
    assert ns["EmbeddingSim"] is torch_embed_sim.EmbeddingSim


def test_reload_preserves_embeddingsim_identity():
    before = torch_embed_sim.EmbeddingSim
    reloaded = importlib.reload(torch_embed_sim)
    assert reloaded.EmbeddingSim is before


def test_importing_package_does_not_replace_sys_modules_entry():
    assert sys.modules["torch_embed_sim"] is torch_embed_sim


def test_embeddingsim_qualified_name_looks_like_class():
    qualname = torch_embed_sim.EmbeddingSim.__qualname__
    assert isinstance(qualname, str)
    assert qualname


def test_embeddingsim_module_attribute_points_to_embedding_sim_module():
    assert torch_embed_sim.EmbeddingSim.__module__ == "torch_embed_sim.embedding_sim"


def test_import_module_object_and_symbol_consistency():
    module = importlib.import_module("torch_embed_sim")
    assert module is torch_embed_sim
    assert module.EmbeddingSim is torch_embed_sim.EmbeddingSim
