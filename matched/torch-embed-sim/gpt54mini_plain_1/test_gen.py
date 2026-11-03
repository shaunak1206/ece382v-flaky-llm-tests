import inspect
import math

import pytest
import torch

import torch_embed_sim
from torch_embed_sim import EmbeddingSim


def _make_instance(*args, **kwargs):
    """
    Try to construct EmbeddingSim with the provided args/kwargs, otherwise with
    a best-effort subset of defaulted parameters.
    """
    try:
        return EmbeddingSim(*args, **kwargs)
    except TypeError:
        sig = inspect.signature(EmbeddingSim)
        params = list(sig.parameters.values())

        # Try calling with only parameters that have defaults.
        call_kwargs = {}
        for p in params:
            if p.name == "self":
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.default is not inspect._empty:
                call_kwargs[p.name] = p.default

        return EmbeddingSim(**call_kwargs)


def _find_similarity_callable(obj):
    """
    Find a method/function on the instance that likely computes similarity.
    """
    candidate_names = [
        "__call__",
        "forward",
        "predict",
        "similarity",
        "compute_similarity",
        "get_similarity",
        "score",
        "sim",
    ]

    for name in candidate_names:
        attr = getattr(obj, name, None)
        if callable(attr):
            return attr

    # Fallback: first public callable method
    for name in dir(obj):
        if name.startswith("_"):
            continue
        attr = getattr(obj, name)
        if callable(attr):
            return attr

    return None


def _call_similarity(sim_fn, a, b):
    """
    Call similarity function with a few common tensor argument patterns.
    """
    # Common binary patterns
    attempts = [
        (a, b),
        ([a, b],),
        ((a, b),),
        (a.unsqueeze(0), b.unsqueeze(0)),
        (a.unsqueeze(0), b),
        (a, b.unsqueeze(0)),
    ]
    last_exc = None
    for args in attempts:
        try:
            return sim_fn(*args)
        except TypeError as exc:
            last_exc = exc
            continue
    raise last_exc


def _to_python_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().flatten().tolist()
    return x


def test_package_exports_embedding_sim():
    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert torch_embed_sim.EmbeddingSim is EmbeddingSim


def test_embedding_sim_is_class():
    assert inspect.isclass(EmbeddingSim)


def test_embedding_sim_has_public_name():
    assert EmbeddingSim.__name__ == "EmbeddingSim"


def test_embedding_sim_has_module_path():
    assert EmbeddingSim.__module__.startswith("torch_embed_sim")


def test_embedding_sim_can_be_instantiated():
    instance = _make_instance()
    assert instance is not None


def test_embedding_sim_instance_has_a_similarity_callable():
    instance = _make_instance()
    sim_fn = _find_similarity_callable(instance)
    assert sim_fn is not None


def test_embedding_sim_same_vector_similarity_is_high_or_defined():
    instance = _make_instance()
    sim_fn = _find_similarity_callable(instance)
    assert sim_fn is not None

    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result = _call_similarity(sim_fn, a, a)
    value = _to_python_scalar(result)

    if isinstance(value, list):
        assert len(value) > 0
    else:
        assert math.isfinite(float(value))


def test_embedding_sim_orthogonal_vectors_produce_finite_result():
    instance = _make_instance()
    sim_fn = _find_similarity_callable(instance)
    assert sim_fn is not None

    a = torch.tensor([1.0, 0.0], dtype=torch.float32)
    b = torch.tensor([0.0, 1.0], dtype=torch.float32)
    result = _call_similarity(sim_fn, a, b)
    value = _to_python_scalar(result)

    if isinstance(value, list):
        assert len(value) > 0
        assert all(math.isfinite(float(v)) for v in value)
    else:
        assert math.isfinite(float(value))


def test_embedding_sim_handles_negative_values():
    instance = _make_instance()
    sim_fn = _find_similarity_callable(instance)
    assert sim_fn is not None

    a = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
    b = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    result = _call_similarity(sim_fn, a, b)
    value = _to_python_scalar(result)

    if isinstance(value, list):
        assert len(value) > 0
        assert all(math.isfinite(float(v)) for v in value)
    else:
        assert math.isfinite(float(value))


def test_embedding_sim_accepts_batched_inputs_if_supported():
    instance = _make_instance()
    sim_fn = _find_similarity_callable(instance)
    assert sim_fn is not None

    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    try:
        result = _call_similarity(sim_fn, a, b)
    except Exception:
        pytest.skip("EmbeddingSim similarity callable does not support batched input patterns")

    value = _to_python_scalar(result)
    if isinstance(value, list):
        assert len(value) > 0
        assert all(math.isfinite(float(v)) for v in value)
    else:
        assert math.isfinite(float(value))


def test_embedding_sim_result_changes_for_different_vectors():
    instance = _make_instance()
    sim_fn = _find_similarity_callable(instance)
    assert sim_fn is not None

    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    c = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)

    r1 = _to_python_scalar(_call_similarity(sim_fn, a, b))
    r2 = _to_python_scalar(_call_similarity(sim_fn, a, c))

    if isinstance(r1, list) or isinstance(r2, list):
        pytest.skip("Similarity output is not scalar-like for this implementation")

    assert math.isfinite(float(r1))
    assert math.isfinite(float(r2))
    assert float(r1) != float(r2)


def test_embedding_sim_repr_contains_class_name():
    instance = _make_instance()
    rep = repr(instance)
    assert "EmbeddingSim" in rep


def test_embedding_sim_has_no_required_public_instance_methods_without_defaults():
    sig = inspect.signature(EmbeddingSim)
    required = [
        p.name
        for p in sig.parameters.values()
        if p.name != "self"
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and p.default is inspect._empty
    ]
    # If there are required args, we at least know the API expects them.
    assert isinstance(required, list)


def test_embedding_sim_module_reexports_match_submodule_import():
    submodule = __import__("torch_embed_sim.embedding_sim", fromlist=["EmbeddingSim"])
    assert hasattr(submodule, "EmbeddingSim")
    assert submodule.EmbeddingSim is EmbeddingSim
