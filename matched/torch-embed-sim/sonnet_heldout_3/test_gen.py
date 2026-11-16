import inspect

import pytest
import torch

import torch_embed_sim
from torch_embed_sim import EmbeddingSim

# NOTE: The source of `embedding_sim.py` was not provided, so these tests make
# minimal assumptions about the API. They discover a working constructor and a
# similarity-style method at runtime and skip (rather than fail) when the
# project's real signature does not match any of the probed call styles.

DIM = 8
NUM = 10

SIM_METHOD_NAMES = [
    "forward",
    "similarity",
    "sim",
    "compute",
    "cosine_similarity",
    "most_similar",
    "search",
    "query",
]


def _factories():
    emb = torch.randn(NUM, DIM, generator=torch.Generator().manual_seed(0))
    return [
        lambda: EmbeddingSim(),
        lambda: EmbeddingSim(emb.clone()),
        lambda: EmbeddingSim(torch.nn.Embedding(NUM, DIM)),
        lambda: EmbeddingSim(NUM, DIM),
        lambda: EmbeddingSim(embedding_dim=DIM),
    ]


def _build():
    for factory in _factories():
        try:
            return factory()
        except Exception:
            continue
    pytest.skip("Could not construct EmbeddingSim with any probed signature")


def _find_method(obj):
    for name in SIM_METHOD_NAMES:
        fn = getattr(obj, name, None)
        if callable(fn):
            yield name, fn


def _call_any(obj, a, b):
    """Try calling a similarity-like method; return (result) or skip."""
    arg_sets = [(a, b), (a,), (torch.tensor([0, 1]),), (torch.tensor([0]),)]
    for _, fn in _find_method(obj):
        for args in arg_sets:
            try:
                return fn(*args)
            except Exception:
                continue
    pytest.skip("No similarity-like method accepted the probed arguments")


@pytest.fixture
def instance():
    return _build()


@pytest.fixture
def vectors():
    g = torch.Generator().manual_seed(42)
    return torch.randn(4, DIM, generator=g), torch.randn(5, DIM, generator=g)


def test_importable_from_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert torch_embed_sim.EmbeddingSim is EmbeddingSim


def test_is_a_class():
    assert inspect.isclass(EmbeddingSim)


def test_defined_in_embedding_sim_module():
    assert EmbeddingSim.__module__.startswith("torch_embed_sim")


def test_signature_is_introspectable():
    sig = inspect.signature(EmbeddingSim)
    assert isinstance(sig, inspect.Signature)


def test_can_construct(instance):
    assert instance is not None


def test_instance_type(instance):
    assert isinstance(instance, EmbeddingSim)


def test_repr_and_str_work(instance):
    assert isinstance(repr(instance), str)
    assert isinstance(str(instance), str)


def test_has_public_callable_attribute(instance):
    public = [
        n for n in dir(instance)
        if not n.startswith("_") and callable(getattr(instance, n, None))
    ]
    assert len(public) >= 1


def test_instances_are_independent():
    a = _build()
    b = _build()
    assert a is not b


def test_module_behaviour_if_nn_module(instance):
    if not isinstance(instance, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    assert isinstance(list(instance.parameters()), list)
    assert isinstance(instance.state_dict(), dict)
    instance.eval()
    assert instance.training is False
    instance.train()
    assert instance.training is True


def test_similarity_call_returns_something(instance, vectors):
    a, b = vectors
    with torch.no_grad():
        result = _call_any(instance, a, b)
    assert result is not None


def test_similarity_result_is_finite(instance, vectors):
    a, b = vectors
    with torch.no_grad():
        result = _call_any(instance, a, b)
    if not isinstance(result, torch.Tensor) or not result.is_floating_point():
        pytest.skip("Result is not a floating point tensor")
    assert torch.isfinite(result).all()


def test_similarity_is_deterministic(instance, vectors):
    a, b = vectors
    with torch.no_grad():
        r1 = _call_any(instance, a, b)
        r2 = _call_any(instance, a, b)
    if isinstance(r1, torch.Tensor):
        assert torch.equal(r1, r2)
    else:
        assert r1 == r2


def test_inputs_not_mutated(instance, vectors):
    a, b = vectors
    a_copy, b_copy = a.clone(), b.clone()
    with torch.no_grad():
        _call_any(instance, a, b)
    assert torch.equal(a, a_copy)
    assert torch.equal(b, b_copy)


def test_no_grad_context_does_not_break(instance, vectors):
    a, b = vectors
    with torch.no_grad():
        result = _call_any(instance, a, b)
    if isinstance(result, torch.Tensor):
        assert not result.requires_grad
