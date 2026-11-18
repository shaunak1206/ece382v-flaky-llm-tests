import importlib
import inspect
import sys
from types import ModuleType

import torch_embed_sim


def test_package_imports_as_module():
    assert isinstance(torch_embed_sim, ModuleType)


def test_embedding_sim_is_exported_from_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")


def test_embedding_sim_import_from_package_equals_attribute():
    from torch_embed_sim import EmbeddingSim as imported_embedding_sim

    assert imported_embedding_sim is torch_embed_sim.EmbeddingSim


def test_embedding_sim_is_same_as_submodule_symbol():
    submodule = importlib.import_module("torch_embed_sim.embedding_sim")
    assert torch_embed_sim.EmbeddingSim is submodule.EmbeddingSim


def test_embedding_sim_object_is_not_none():
    assert torch_embed_sim.EmbeddingSim is not None


def test_embedding_sim_has_expected_name():
    assert torch_embed_sim.EmbeddingSim.__name__ == "EmbeddingSim"


def test_embedding_sim_module_points_to_embedding_sim_submodule():
    assert torch_embed_sim.EmbeddingSim.__module__ == "torch_embed_sim.embedding_sim"


def test_star_import_brings_embedding_sim_into_local_namespace():
    namespace = {}
    exec("from torch_embed_sim import *", namespace)
    assert "EmbeddingSim" in namespace
    assert namespace["EmbeddingSim"] is torch_embed_sim.EmbeddingSim


def test_package_reimport_preserves_embedding_sim_identity():
    module_before = torch_embed_sim.EmbeddingSim
    reloaded = importlib.reload(torch_embed_sim)
    assert reloaded.EmbeddingSim is module_before


def test_submodule_reimport_preserves_package_reference_identity():
    package_before = torch_embed_sim.EmbeddingSim
    submodule = importlib.import_module("torch_embed_sim.embedding_sim")
    reloaded_submodule = importlib.reload(submodule)
    assert torch_embed_sim.EmbeddingSim is package_before
    assert reloaded_submodule.EmbeddingSim is package_before


def test_embedding_sim_is_defined_in_submodule_source_file():
    submodule = importlib.import_module("torch_embed_sim.embedding_sim")
    source_file = inspect.getsourcefile(submodule)
    assert source_file is not None
    assert source_file.endswith("embedding_sim.py")


def test_package_does_not_shadow_embedding_sim_with_different_object():
    submodule = importlib.import_module("torch_embed_sim.embedding_sim")
    assert id(torch_embed_sim.EmbeddingSim) == id(submodule.EmbeddingSim)


def test_embedding_sim_is_accessible_via_sys_modules_import_path():
    assert "torch_embed_sim" in sys.modules
    assert "torch_embed_sim.embedding_sim" in sys.modules
    assert getattr(sys.modules["torch_embed_sim"], "EmbeddingSim") is torch_embed_sim.EmbeddingSim


def test_from_package_import_does_not_create_new_class_object():
    from torch_embed_sim import EmbeddingSim as first
    from torch_embed_sim import EmbeddingSim as second

    assert first is second is torch_embed_sim.EmbeddingSim
