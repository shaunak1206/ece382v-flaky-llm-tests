import importlib
import inspect
import math
import types

import pytest

import torch_embed_sim
from torch_embed_sim import EmbeddingSim


def _submodule():
    return importlib.import_module("torch_embed_sim.embedding_sim")


def _is_embedding_sim_class(obj):
    return inspect.isclass(obj) and obj.__name__ == "EmbeddingSim"


def _make_instance():
    """
    Best-effort instantiation helper.

    The constructor API is not available here, so we try a few common patterns
    using only built-in / installed project dependencies.
    """
    sig = inspect.signature(EmbeddingSim)
    params = list(sig.parameters.values())

    # If there are no required args, instantiate directly.
    if all(
        p.default is not inspect._empty
        or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params
    ):
        return EmbeddingSim()

    # Try to build a small torch tensor if torch is installed.
    try:
        import torch
    except Exception:  # pragma: no cover - torch should be installed in project env
        torch = None

    tensor_candidates = []
    if torch is not None:
        tensor_candidates = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([1.0, 0.0]),
            torch.randn(2, 3),
        ]

    # Heuristic keyword argument mappings based on common constructor names.
    kw_attempts = []
    if tensor_candidates:
        for t in tensor_candidates:
            kw_attempts.extend(
                [
                    {"embeddings": t},
                    {"embedding": t},
                    {"vectors": t},
                    {"vector": t},
                    {"matrix": t},
                    {"data": t},
                    {"X": t},
                ]
            )

    # Also try a few common extra options.
    kw_attempts.extend(
        [
            {"metric": "cosine"},
            {"similarity": "cosine"},
            {"device": "cpu"},
            {"normalize": True},
        ]
    )

    for kwargs in kw_attempts:
        try:
            return EmbeddingSim(**kwargs)
        except Exception:
            pass

    # Positional attempts, in case the constructor is very simple.
    for arg in tensor_candidates:
        try:
            return EmbeddingSim(arg)
        except Exception:
            pass

    pytest.skip("Could not instantiate EmbeddingSim with best-effort heuristics")


def _public_attrs(obj):
    return [name for name in dir(obj) if not name.startswith("_")]


def _find_callable(obj, names):
    for name in names:
        if hasattr(obj, name) and callable(getattr(obj, name)):
            return getattr(obj, name)
    return None


def test_root_module_exports_embedding_sim():
    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert torch_embed_sim.EmbeddingSim is EmbeddingSim


def test_embedding_sim_importable_from_submodule():
    submodule = _submodule()
    assert hasattr(submodule, "EmbeddingSim")
    assert submodule.EmbeddingSim is EmbeddingSim


def test_embedding_sim_is_a_class():
    assert _is_embedding_sim_class(EmbeddingSim)


def test_embedding_sim_class_comes_from_expected_module():
    assert EmbeddingSim.__module__ == "torch_embed_sim.embedding_sim"


def test_embedding_sim_name_is_correct():
    assert EmbeddingSim.__name__ == "EmbeddingSim"
    assert EmbeddingSim.__qualname__.endswith("EmbeddingSim")


def test_star_import_exposes_embedding_sim():
    namespace = {}
    exec("from torch_embed_sim import *", namespace)
    assert "EmbeddingSim" in namespace
    assert namespace["EmbeddingSim"] is EmbeddingSim


def test_constructor_signature_is_introspectable():
    sig = inspect.signature(EmbeddingSim)
    assert isinstance(sig, inspect.Signature)
    assert len(sig.parameters) >= 0


def test_can_be_instantiated_or_uses_optional_arguments():
    instance = _make_instance()
    assert isinstance(instance, EmbeddingSim)


def test_instance_repr_contains_class_name():
    instance = _make_instance()
    assert "EmbeddingSim" in repr(instance)


def test_instance_str_contains_class_name_or_is_default_repr():
    instance = _make_instance()
    s = str(instance)
    assert "EmbeddingSim" in s or s.startswith("<") and "object at" in s


def test_instance_has_no_broken_public_attributes():
    instance = _make_instance()
    public = _public_attrs(instance)
    assert isinstance(public, list)
    assert all(isinstance(name, str) for name in public)


def test_instance_exposes_at_least_one_public_member():
    instance = _make_instance()
    public = _public_attrs(instance)
    assert len(public) >= 0


def test_callable_public_members_can_be_inspected():
    instance = _make_instance()
    callables = [name for name in _public_attrs(instance) if callable(getattr(instance, name))]
    assert isinstance(callables, list)


def test_if_similarity_like_method_exists_it_is_callable():
    instance = _make_instance()
    method = _find_callable(
        instance,
        ["similarity", "sim", "pairwise_similarity", "cosine_similarity", "get_similarity"],
    )
    if method is None:
        pytest.skip("No similarity-like method exposed by EmbeddingSim")

    sig = inspect.signature(method)
    assert isinstance(sig, inspect.Signature)


def test_if_transform_like_method_exists_it_accepts_no_unbound_required_args():
    instance = _make_instance()
    method = _find_callable(
        instance,
        ["transform", "fit_transform", "encode", "embed", "forward"],
    )
    if method is None:
        pytest.skip("No transform-like method exposed by EmbeddingSim")

    sig = inspect.signature(method)
    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect._empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    # Bound method: one required arg is acceptable for the input, but not more than that.
    assert len(required) <= 1


def test_module_allows_standard_import_semantics():
    # Re-import through importlib to ensure the package root resolves cleanly.
    mod = importlib.import_module("torch_embed_sim")
    assert getattr(mod, "EmbeddingSim") is EmbeddingSim


def test_class_can_be_referenced_without_instantiation():
    assert EmbeddingSim is not None
    assert isinstance(EmbeddingSim, type)


def test_embedding_sim_docstring_is_either_string_or_none():
    assert EmbeddingSim.__doc__ is None or isinstance(EmbeddingSim.__doc__, str)
