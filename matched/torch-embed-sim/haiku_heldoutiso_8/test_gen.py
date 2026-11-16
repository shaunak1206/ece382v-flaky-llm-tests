"""Unit tests for EmbeddingSim in torch_embed_sim.

Note: only torch_embed_sim/__init__.py was provided, so the EmbeddingSim
constructor and similarity method names are not known for certain. The
helpers below probe a few likely names and skip tests when none exist,
so adjust `SIM_METHOD_NAMES` and `_make_sim` if the real API differs.
"""
import math

import pytest

torch = pytest.importorskip("torch")

import torch_embed_sim
from torch_embed_sim import EmbeddingSim


SIM_METHOD_NAMES = (
    "similarity",
    "cosine_similarity",
    "compute_similarity",
    "sim",
)


def _make_sim():
    """Create an EmbeddingSim with default arguments, skipping if that is not supported."""
    try:
        return EmbeddingSim()
    except TypeError:
        pytest.skip("EmbeddingSim requires constructor arguments not known to these tests")


def _similarity_fn(sim):
    """Return the similarity callable from the instance, skipping if none is found."""
    for name in SIM_METHOD_NAMES:
        method = getattr(sim, name, None)
        if callable(method):
            return method
    pytest.skip("No known similarity method found on EmbeddingSim")


def _vec(values):
    return torch.tensor(values, dtype=torch.float32)


def _as_float(result):
    if isinstance(result, torch.Tensor):
        return float(result.reshape(-1)[0].item()) if result.numel() > 1 else float(result.item())
    return float(result)


def test_package_exports_embedding_sim():
    assert hasattr(torch_embed_sim, "EmbeddingSim")


def test_embedding_sim_is_a_class():
    assert isinstance(EmbeddingSim, type)


def test_star_import_matches_submodule_definition():
    from torch_embed_sim.embedding_sim import EmbeddingSim as submodule_cls

    assert torch_embed_sim.EmbeddingSim is submodule_cls


def test_instantiation_returns_embedding_sim_instance():
    sim = _make_sim()
    assert isinstance(sim, EmbeddingSim)


def test_similarity_identical_vectors_is_one():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    v = _vec([1.0, 2.0, 3.0])
    assert _as_float(fn(v, v)) == pytest.approx(1.0, abs=1e-6)


def test_similarity_orthogonal_vectors_is_zero():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    a = _vec([1.0, 0.0, 0.0])
    b = _vec([0.0, 1.0, 0.0])
    assert _as_float(fn(a, b)) == pytest.approx(0.0, abs=1e-6)


def test_similarity_opposite_vectors_is_minus_one():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    a = _vec([1.0, -2.0, 3.0])
    b = _vec([-1.0, 2.0, -3.0])
    assert _as_float(fn(a, b)) == pytest.approx(-1.0, abs=1e-6)


def test_similarity_is_symmetric():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    a = _vec([0.3, -1.2, 2.5, 0.7])
    b = _vec([1.1, 0.4, -0.9, 2.0])
    assert _as_float(fn(a, b)) == pytest.approx(_as_float(fn(b, a)), abs=1e-6)


def test_similarity_is_scale_invariant():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    a = _vec([1.0, 2.0, 3.0])
    b = _vec([4.0, 5.0, 6.0])
    base = _as_float(fn(a, b))
    scaled = _as_float(fn(a * 10.0, b * 0.5))
    assert scaled == pytest.approx(base, abs=1e-5)


def test_similarity_known_value():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    a = _vec([1.0, 0.0])
    b = _vec([1.0, 1.0])
    expected = 1.0 / math.sqrt(2.0)
    assert _as_float(fn(a, b)) == pytest.approx(expected, abs=1e-6)


def test_similarity_result_is_within_bounds():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        a = torch.randn(8, generator=gen)
        b = torch.randn(8, generator=gen)
        value = _as_float(fn(a, b))
        assert -1.0 - 1e-6 <= value <= 1.0 + 1e-6


def test_similarity_is_deterministic_across_calls():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    a = _vec([0.5, -0.25, 1.5])
    b = _vec([2.0, 0.1, -0.7])
    first = _as_float(fn(a, b))
    second = _as_float(fn(a, b))
    assert first == second


def test_similarity_zero_vector_is_finite():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    zero = _vec([0.0, 0.0, 0.0])
    other = _vec([1.0, 2.0, 3.0])
    value = _as_float(fn(zero, other))
    assert math.isfinite(value)


def test_similarity_mismatched_dimensions_raises():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    a = _vec([1.0, 2.0, 3.0])
    b = _vec([1.0, 2.0])
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        fn(a, b)


def test_similarity_accepts_float_tensors_of_single_element_shape():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    a = _vec([2.0])
    b = _vec([3.0])
    assert _as_float(fn(a, b)) == pytest.approx(1.0, abs=1e-6)


def test_similarity_does_not_modify_inputs():
    sim = _make_sim()
    fn = _similarity_fn(sim)
    a = _vec([1.0, 2.0, 3.0])
    b = _vec([3.0, 2.0, 1.0])
    a_before = a.clone()
    b_before = b.clone()
    fn(a, b)
    assert torch.equal(a, a_before)
    assert torch.equal(b, b_before)
