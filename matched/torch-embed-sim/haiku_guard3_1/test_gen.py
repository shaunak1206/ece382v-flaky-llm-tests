"""
Unit tests for EmbeddingSim in torch_embed_sim/embedding_sim.py.

NOTE: Only torch_embed_sim/__init__.py was provided, and it does `from .embedding_sim import *`.
The tests below assume the following interface. Check them against embedding_sim.py and
adjust the constructor, attribute, and method names if they differ:

    EmbeddingSim(num_embeddings: int, embedding_dim: int)  -> torch.nn.Module
    sim.weight                 : parameter of shape (num_embeddings, embedding_dim)
    sim(indices)               : embeddings for integer indices, shape (..., embedding_dim)
    sim.similarity(i, j)       : cosine similarity between rows i and j, a float in [-1, 1]

Every test seeds torch at the top of its own body, so results do not depend on test order.
"""

import pytest
import torch

from torch_embed_sim import EmbeddingSim


def _make_sim(num_embeddings=4, embedding_dim=3):
    sim = EmbeddingSim(num_embeddings, embedding_dim)
    return sim


def _set_weight(sim, rows):
    """Overwrite the weight matrix with explicit values (no randomness)."""
    with torch.no_grad():
        sim.weight.copy_(torch.tensor(rows, dtype=sim.weight.dtype))


def test_package_exports_embedding_sim():
    torch.manual_seed(0)
    import torch_embed_sim

    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert isinstance(EmbeddingSim, type)


def test_constructor_sets_weight_shape():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=5, embedding_dim=7)

    assert tuple(sim.weight.shape) == (5, 7)


def test_constructor_weight_is_float_tensor():
    torch.manual_seed(0)
    sim = _make_sim()

    assert sim.weight.dtype == torch.float32


def test_constructor_weight_is_finite():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=10, embedding_dim=16)

    assert torch.isfinite(sim.weight).all()


def test_forward_output_shape_for_batch_of_indices():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=4, embedding_dim=3)

    out = sim(torch.tensor([0, 2, 3]))

    assert tuple(out.shape) == (3, 3)


def test_forward_single_index_returns_vector():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=4, embedding_dim=3)

    out = sim(torch.tensor(1))

    assert tuple(out.shape) == (3,)


def test_forward_returns_matching_weight_rows():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=4, embedding_dim=2)
    _set_weight(sim, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]])

    out = sim(torch.tensor([3, 0]))

    expected = torch.tensor([[-1.0, 2.0], [1.0, 0.0]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_forward_is_deterministic_under_same_seed():
    torch.manual_seed(1234)
    sim_a = _make_sim(num_embeddings=6, embedding_dim=4)
    torch.manual_seed(1234)
    sim_b = _make_sim(num_embeddings=6, embedding_dim=4)

    assert torch.equal(sim_a.weight, sim_b.weight)


def test_forward_rejects_out_of_range_index():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=4, embedding_dim=3)

    with pytest.raises(IndexError):
        sim(torch.tensor([4]))


def test_self_similarity_is_one():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=4, embedding_dim=3)
    _set_weight(sim, [[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 0.0, 4.0]])

    for i in range(4):
        assert sim.similarity(i, i) == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_rows_have_zero_similarity():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=2, embedding_dim=2)
    _set_weight(sim, [[1.0, 0.0], [0.0, 1.0]])

    assert sim.similarity(0, 1) == pytest.approx(0.0, abs=1e-6)


def test_opposite_rows_have_minus_one_similarity():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=2, embedding_dim=3)
    _set_weight(sim, [[1.0, 2.0, -3.0], [-2.0, -4.0, 6.0]])

    assert sim.similarity(0, 1) == pytest.approx(-1.0, abs=1e-6)


def test_similarity_is_symmetric():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=5, embedding_dim=4)

    for i in range(5):
        for j in range(5):
            assert sim.similarity(i, j) == pytest.approx(sim.similarity(j, i), abs=1e-6)


def test_similarity_is_bounded_for_random_weights():
    torch.manual_seed(42)
    sim = _make_sim(num_embeddings=8, embedding_dim=5)

    for i in range(8):
        for j in range(8):
            value = sim.similarity(i, j)
            assert -1.0 - 1e-6 <= value <= 1.0 + 1e-6


def test_similarity_is_scale_invariant():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=2, embedding_dim=3)
    _set_weight(sim, [[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]])
    before = sim.similarity(0, 1)

    _set_weight(sim, [[10.0, 20.0, 30.0], [4.0, -1.0, 0.5]])
    after = sim.similarity(0, 1)

    assert after == pytest.approx(before, abs=1e-6)


def test_forward_gradients_flow_to_weight():
    torch.manual_seed(0)
    sim = _make_sim(num_embeddings=4, embedding_dim=3)

    out = sim(torch.tensor([1, 2]))
    loss = out.sum()
    loss.backward()

    assert sim.weight.grad is not None
    assert torch.any(sim.weight.grad != 0)
    # Rows that were never looked up receive no gradient.
    assert torch.all(sim.weight.grad[0] == 0)
    assert torch.all(sim.weight.grad[3] == 0)
