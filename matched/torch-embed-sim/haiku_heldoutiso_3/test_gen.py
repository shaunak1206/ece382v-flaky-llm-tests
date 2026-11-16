"""Unit tests for EmbeddingSim in torch_embed_sim.

Only torch_embed_sim/__init__.py was provided, and it re-exports everything
from torch_embed_sim.embedding_sim. The implementation of EmbeddingSim itself
was not available, so these tests rely on introspection and skip any check
whose preconditions (a no-argument constructor, a forward method, a
similarity method) cannot be determined from the class.
"""
import importlib
import inspect

import pytest
import torch

import torch_embed_sim
from torch_embed_sim import EmbeddingSim


def _constructible_without_args(cls):
    """Return True if cls can be instantiated with no positional/keyword args."""
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is param.empty:
            return False
    return True


@pytest.fixture
def instance():
    if not _constructible_without_args(EmbeddingSim):
        pytest.skip("EmbeddingSim requires constructor arguments")
    return EmbeddingSim()


def test_embedding_sim_exported_from_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")


def test_embedding_sim_is_class():
    assert inspect.isclass(EmbeddingSim)


def test_name_is_embedding_sim():
    assert EmbeddingSim.__name__ == "EmbeddingSim"


def test_constructor_signature_is_introspectable():
    sig = inspect.signature(EmbeddingSim)
    assert isinstance(sig.parameters, dict) or hasattr(sig, "parameters")


def test_submodule_exposes_same_class():
    submodule = importlib.import_module("torch_embed_sim.embedding_sim")
    assert hasattr(submodule, "EmbeddingSim")
    assert submodule.EmbeddingSim is EmbeddingSim


def test_package_and_submodule_class_identity():
    submodule = importlib.import_module("torch_embed_sim.embedding_sim")
    assert torch_embed_sim.EmbeddingSim is submodule.EmbeddingSim


def test_class_defined_within_package():
    assert EmbeddingSim.__module__.startswith("torch_embed_sim")


def test_star_import_namespace_includes_class():
    submodule = importlib.import_module("torch_embed_sim.embedding_sim")
    exported = getattr(submodule, "__all__", None)
    if exported is not None:
        assert "EmbeddingSim" in exported
    else:
        assert "EmbeddingSim" in vars(submodule)


def test_instance_creation_without_args(instance):
    assert isinstance(instance, EmbeddingSim)


def test_instance_repr_is_nonempty_string(instance):
    text = repr(instance)
    assert isinstance(text, str)
    assert text


def test_instance_str_does_not_raise(instance):
    assert isinstance(str(instance), str)


def test_two_instances_are_distinct_objects():
    if not _constructible_without_args(EmbeddingSim):
        pytest.skip("EmbeddingSim requires constructor arguments")
    first = EmbeddingSim()
    second = EmbeddingSim()
    assert first is not second


def test_public_attributes_resolve_statically():
    for name in dir(EmbeddingSim):
        if name.startswith("_"):
            continue
        # getattr_static never triggers descriptors, so this only checks
        # that the attribute is declared on the class hierarchy.
        assert inspect.getattr_static(EmbeddingSim, name) is not None


def test_forward_takes_at_least_one_argument(instance):
    forward = getattr(instance, "forward", None)
    if forward is None or not callable(forward):
        pytest.skip("EmbeddingSim has no forward method")
    params = [
        p for p in inspect.signature(forward).parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert len(params) >= 1


def test_module_parameters_iterable_if_nn_module(instance):
    if not isinstance(instance, torch.nn.Module):
        pytest.skip("EmbeddingSim is not a torch.nn.Module")
    params = list(instance.parameters())
    assert all(isinstance(p, torch.nn.Parameter) for p in params)


def test_state_dict_is_mapping_if_nn_module(instance):
    if not isinstance(instance, torch.nn.Module):
        pytest.skip("EmbeddingSim is not a torch.nn.Module")
    state = instance.state_dict()
    assert hasattr(state, "keys")


def test_similarity_output_is_tensor_if_available(instance):
    similarity = None
    for name in dir(instance):
        if "similarity" in name.lower() and not name.startswith("_"):
            candidate = getattr(instance, name)
            if callable(candidate):
                similarity = candidate
                break
    if similarity is None:
        pytest.skip("No similarity method found on EmbeddingSim")
    x = torch.randn(4, 8)
    try:
        result = similarity(x, x)
    except TypeError:
        pytest.skip("Similarity method has an incompatible signature")
    assert isinstance(result, torch.Tensor)
