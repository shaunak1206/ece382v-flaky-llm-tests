"""Unit tests for EmbeddingSim exported from torch_embed_sim.

The implementation source (torch_embed_sim/embedding_sim.py) was not provided, so
this file assumes the following interface. Adjust the helpers if the real API differs:

    EmbeddingSim(num_embeddings: int, embedding_dim: int)
    .embed(ids: LongTensor of shape (n,)) -> FloatTensor of shape (n, embedding_dim)
    .similarity(a: FloatTensor (n, d), b: FloatTensor (n, d)) -> FloatTensor of shape (n,)
        cosine similarity per row, in [-1, 1]
"""

import pytest
import torch

from torch_embed_sim import EmbeddingSim


ATOL = 1e-5


def test_embedding_sim_is_exported_as_class():
    torch.manual_seed(0)
    assert isinstance(EmbeddingSim, type)


def test_constructor_builds_instance():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=5, embedding_dim=3)
    assert isinstance(sim, EmbeddingSim)


def test_embed_output_shape_matches_ids_and_dim():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=10, embedding_dim=4)
    ids = torch.tensor([0, 3, 7])
    out = sim.embed(ids)
    assert tuple(out.shape) == (3, 4)


def test_embed_returns_float32():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=10, embedding_dim=4)
    out = sim.embed(torch.tensor([1, 2]))
    assert out.dtype == torch.float32


def test_embed_is_deterministic_for_same_ids():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=10, embedding_dim=4)
    ids = torch.tensor([2, 5, 2])
    first = sim.embed(ids)
    second = sim.embed(ids)
    assert torch.allclose(first, second, atol=ATOL)


def test_embed_repeated_id_gives_identical_rows():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=10, embedding_dim=4)
    out = sim.embed(torch.tensor([6, 6]))
    assert torch.allclose(out[0], out[1], atol=ATOL)


def test_embed_empty_ids_gives_zero_rows():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=10, embedding_dim=4)
    out = sim.embed(torch.tensor([], dtype=torch.long))
    assert tuple(out.shape) == (0, 4)


def test_embed_out_of_range_id_raises():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=5, embedding_dim=3)
    with pytest.raises((IndexError, RuntimeError)):
        sim.embed(torch.tensor([5]))


def test_same_seed_gives_same_weights():
    torch.manual_seed(123)
    sim_a = EmbeddingSim(num_embeddings=8, embedding_dim=4)
    ids = torch.arange(8)
    out_a = sim_a.embed(ids)

    torch.manual_seed(123)
    sim_b = EmbeddingSim(num_embeddings=8, embedding_dim=4)
    out_b = sim_b.embed(ids)

    assert torch.allclose(out_a, out_b, atol=ATOL)


def test_similarity_of_vector_with_itself_is_one():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=4, embedding_dim=3)
    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    out = sim.similarity(x, x)
    assert torch.allclose(out, torch.ones(2), atol=ATOL)


def test_similarity_is_symmetric():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=4, embedding_dim=3)
    a = torch.tensor([[1.0, 0.0, 2.0], [0.3, -1.0, 0.7]])
    b = torch.tensor([[0.5, 1.5, -1.0], [2.0, 0.1, 0.4]])
    assert torch.allclose(sim.similarity(a, b), sim.similarity(b, a), atol=ATOL)


def test_similarity_orthogonal_vectors_is_zero():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=4, embedding_dim=3)
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(sim.similarity(a, b), torch.zeros(1), atol=ATOL)


def test_similarity_is_scale_invariant():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=4, embedding_dim=3)
    a = torch.tensor([[1.0, 2.0, -0.5]])
    b = torch.tensor([[0.3, -1.0, 2.0]])
    base = sim.similarity(a, b)
    scaled = sim.similarity(10.0 * a, b)
    assert torch.allclose(base, scaled, atol=ATOL)


def test_similarity_stays_within_unit_range():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=4, embedding_dim=5)
    a = torch.randn(50, 5)
    b = torch.randn(50, 5)
    out = sim.similarity(a, b)
    assert torch.all(out <= 1.0 + ATOL)
    assert torch.all(out >= -1.0 - ATOL)


def test_similarity_zero_vector_is_finite():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=4, embedding_dim=3)
    a = torch.zeros(1, 3)
    b = torch.tensor([[1.0, 2.0, 3.0]])
    out = sim.similarity(a, b)
    assert torch.isfinite(out).all()


def test_similarity_mismatched_dimensions_raises():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=4, embedding_dim=3)
    a = torch.ones(2, 3)
    b = torch.ones(2, 4)
    with pytest.raises((ValueError, RuntimeError)):
        sim.similarity(a, b)
