"""Unit tests for EmbeddingSim in torch_embed_sim/embedding_sim.py.

Assumed API (adjust if the implementation differs):
    EmbeddingSim(num_embeddings: int, embedding_dim: int)
    - .num_embeddings and .embedding_dim attributes
    - .weight: torch.Tensor of shape (num_embeddings, embedding_dim)
    - .similarity(i, j) -> float, cosine similarity between rows i and j
    - .most_similar(i, k) -> list of k row indices (excluding i), best first
"""

import inspect
import math

import pytest
import torch

import torch_embed_sim
from torch_embed_sim import EmbeddingSim


@pytest.fixture
def seeded():
    torch.manual_seed(0)
    yield


@pytest.fixture
def sim(seeded):
    return EmbeddingSim(num_embeddings=5, embedding_dim=4)


def _cosine(a, b):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


# --- Import / export surface -------------------------------------------------

def test_embedding_sim_is_exported_from_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")


def test_embedding_sim_is_a_class():
    assert inspect.isclass(EmbeddingSim)


def test_star_import_exposes_embedding_sim():
    namespace = {}
    exec("from torch_embed_sim import *", namespace)
    assert "EmbeddingSim" in namespace


# --- Construction ------------------------------------------------------------

def test_constructor_stores_num_embeddings(sim):
    assert sim.num_embeddings == 5


def test_constructor_stores_embedding_dim(sim):
    assert sim.embedding_dim == 4


def test_weight_shape_matches_configuration(sim):
    assert tuple(sim.weight.shape) == (5, 4)


def test_weight_is_float_tensor(sim):
    assert sim.weight.dtype == torch.float32


def test_construction_is_deterministic_under_seed():
    torch.manual_seed(123)
    a = EmbeddingSim(num_embeddings=3, embedding_dim=2)
    torch.manual_seed(123)
    b = EmbeddingSim(num_embeddings=3, embedding_dim=2)
    assert torch.equal(a.weight, b.weight)


def test_different_seeds_give_different_weights():
    torch.manual_seed(1)
    a = EmbeddingSim(num_embeddings=3, embedding_dim=2)
    torch.manual_seed(2)
    b = EmbeddingSim(num_embeddings=3, embedding_dim=2)
    assert not torch.equal(a.weight, b.weight)


# --- Similarity --------------------------------------------------------------

def test_self_similarity_is_one(sim):
    assert math.isclose(sim.similarity(0, 0), 1.0, rel_tol=1e-5)


def test_similarity_is_symmetric(sim):
    assert math.isclose(sim.similarity(1, 3), sim.similarity(3, 1), rel_tol=1e-6)


def test_similarity_matches_independent_cosine(sim):
    expected = _cosine(sim.weight[2], sim.weight[4])
    assert math.isclose(sim.similarity(2, 4), expected, rel_tol=1e-5, abs_tol=1e-6)


def test_similarity_bounded_between_minus_one_and_one(sim):
    for i in range(sim.num_embeddings):
        for j in range(sim.num_embeddings):
            value = sim.similarity(i, j)
            assert -1.0 - 1e-6 <= value <= 1.0 + 1e-6


def test_orthogonal_vectors_have_zero_similarity():
    s = EmbeddingSim(num_embeddings=2, embedding_dim=2)
    with torch.no_grad():
        s.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert math.isclose(s.similarity(0, 1), 0.0, abs_tol=1e-6)


def test_parallel_vectors_have_unit_similarity():
    s = EmbeddingSim(num_embeddings=2, embedding_dim=3)
    with torch.no_grad():
        s.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]))
    assert math.isclose(s.similarity(0, 1), 1.0, rel_tol=1e-5)


def test_opposite_vectors_have_minus_one_similarity():
    s = EmbeddingSim(num_embeddings=2, embedding_dim=2)
    with torch.no_grad():
        s.weight.copy_(torch.tensor([[1.0, 1.0], [-1.0, -1.0]]))
    assert math.isclose(s.similarity(0, 1), -1.0, rel_tol=1e-5)


# --- Ranking -----------------------------------------------------------------

def test_most_similar_excludes_query_index(sim):
    result = sim.most_similar(0, 3)
    assert 0 not in result


def test_most_similar_returns_requested_count(sim):
    assert len(sim.most_similar(0, 3)) == 3


def test_most_similar_is_sorted_by_descending_similarity(sim):
    result = sim.most_similar(1, 4)
    scores = [sim.similarity(1, idx) for idx in result]
    assert scores == sorted(scores, reverse=True)


def test_most_similar_top_result_is_true_argmax():
    s = EmbeddingSim(num_embeddings=4, embedding_dim=2)
    with torch.no_grad():
        s.weight.copy_(torch.tensor([
            [1.0, 0.0],
            [0.9, 0.1],   # closest to row 0
            [0.0, 1.0],
            [-1.0, 0.0],
        ]))
    assert s.most_similar(0, 1) == [1]
