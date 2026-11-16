"""
Pytest unit tests for EmbeddingSim in torch_embed_sim.

The source of torch_embed_sim/embedding_sim.py was not provided, so these tests
rely on conventional names and behaviour. Tests that need a specific method or
constructor signature locate it at runtime and skip when it is absent, so the
suite stays runnable without guessing the exact API.
"""
import inspect

import pytest
import torch

import torch_embed_sim


EPS = 1e-5
SIM_METHOD_NAMES = ("similarity", "sim", "cosine_similarity", "compute_similarity", "score")
FORWARD_METHOD_NAMES = ("forward", "__call__", "embed", "encode")


def _get_cls():
    cls = getattr(torch_embed_sim, "EmbeddingSim", None)
    assert cls is not None, "EmbeddingSim must be exported from torch_embed_sim"
    return cls


def _make_instance(cls):
    """Instantiate cls with no arguments, or skip if a constructor argument is required."""
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        for p in sig.parameters.values():
            if p.default is inspect.Parameter.empty and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                pytest.skip("EmbeddingSim requires constructor arguments not known to these tests")
    return cls()


def _find_method(obj, names):
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return method
    return None


@pytest.fixture
def instance():
    cls = _get_cls()
    return _make_instance(cls)


@pytest.fixture
def sim_fn(instance):
    method = _find_method(instance, SIM_METHOD_NAMES)
    if method is None:
        pytest.skip("No similarity method found on EmbeddingSim")
    return method


def _vectors(seed=0, batch=4, dim=8):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, dim, generator=gen)
    b = torch.randn(batch, dim, generator=gen)
    return a, b


def test_embedding_sim_is_exported_and_is_a_class():
    cls = _get_cls()
    assert inspect.isclass(cls)


def test_embedding_sim_module_attribute_matches_submodule():
    from torch_embed_sim import embedding_sim

    assert torch_embed_sim.EmbeddingSim is embedding_sim.EmbeddingSim


def test_embedding_sim_signature_is_introspectable():
    cls = _get_cls()
    sig = inspect.signature(cls)
    assert isinstance(sig, inspect.Signature)


def test_instance_construction_without_arguments(instance):
    assert instance is not None
    assert isinstance(instance, _get_cls())


def test_instance_str_and_repr_return_strings(instance):
    assert isinstance(repr(instance), str)
    assert isinstance(str(instance), str)


def test_instances_are_independent():
    cls = _get_cls()
    first = _make_instance(cls)
    second = _make_instance(cls)
    assert first is not second


def test_forward_like_method_returns_tensor_when_present(instance):
    method = _find_method(instance, FORWARD_METHOD_NAMES)
    if method is None:
        pytest.skip("No forward-like method found on EmbeddingSim")
    torch.manual_seed(0)
    x = torch.randn(3, 8)
    try:
        out = method(x)
    except TypeError:
        pytest.skip("Forward-like method does not accept a single tensor argument")
    assert isinstance(out, torch.Tensor)


def test_self_similarity_is_one(sim_fn):
    a, _ = _vectors()
    out = sim_fn(a, a)
    out = torch.as_tensor(out, dtype=torch.float32).reshape(-1)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-4)


def test_similarity_is_symmetric(sim_fn):
    a, b = _vectors(seed=1)
    ab = torch.as_tensor(sim_fn(a, b), dtype=torch.float32)
    ba = torch.as_tensor(sim_fn(b, a), dtype=torch.float32)
    assert torch.allclose(ab, ba, atol=1e-5)


def test_similarity_is_bounded_in_unit_interval(sim_fn):
    a, b = _vectors(seed=2, batch=16, dim=32)
    out = torch.as_tensor(sim_fn(a, b), dtype=torch.float32)
    assert torch.all(out <= 1.0 + EPS)
    assert torch.all(out >= -1.0 - EPS)


def test_similarity_is_scale_invariant(sim_fn):
    a, b = _vectors(seed=3)
    base = torch.as_tensor(sim_fn(a, b), dtype=torch.float32)
    scaled = torch.as_tensor(sim_fn(a * 7.5, b * 0.25), dtype=torch.float32)
    assert torch.allclose(base, scaled, atol=1e-4)


def test_orthogonal_vectors_have_zero_similarity(sim_fn):
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    out = torch.as_tensor(sim_fn(a, b), dtype=torch.float32).reshape(-1)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_opposite_vectors_have_negative_one_similarity(sim_fn):
    a = torch.tensor([[1.0, 2.0, 3.0]])
    out = torch.as_tensor(sim_fn(a, -a), dtype=torch.float32).reshape(-1)
    assert torch.allclose(out, -torch.ones_like(out), atol=1e-4)


def test_similarity_is_deterministic(sim_fn):
    a, b = _vectors(seed=4)
    first = torch.as_tensor(sim_fn(a, b), dtype=torch.float32)
    second = torch.as_tensor(sim_fn(a, b), dtype=torch.float32)
    assert torch.equal(first, second)


def test_similarity_output_shape_matches_batch(sim_fn):
    a, b = _vectors(seed=5, batch=6, dim=10)
    out = torch.as_tensor(sim_fn(a, b))
    assert out.numel() == 6


def test_similarity_rejects_mismatched_dimensions(sim_fn):
    a = torch.randn(2, 8)
    b = torch.randn(2, 5)
    with pytest.raises((RuntimeError, ValueError, AssertionError, IndexError, TypeError)):
        sim_fn(a, b)


def test_similarity_does_not_modify_inputs(sim_fn):
    a, b = _vectors(seed=6)
    a_before = a.clone()
    b_before = b.clone()
    sim_fn(a, b)
    assert torch.equal(a, a_before)
    assert torch.equal(b, b_before)


def test_similarity_supports_autograd(sim_fn):
    a, b = _vectors(seed=7)
    a = a.clone().requires_grad_(True)
    out = torch.as_tensor(sim_fn(a, b))
    if not out.requires_grad:
        pytest.skip("Similarity method does not produce a differentiable output")
    out.sum().backward()
    assert a.grad is not None
    assert torch.all(torch.isfinite(a.grad))
