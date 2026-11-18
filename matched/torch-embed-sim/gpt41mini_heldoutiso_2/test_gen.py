import torch
import pytest
from torch_embed_sim import EmbeddingSim


def test_similarity_same_vector():
    # similarity of a vector with itself should be close to 1
    emb_sim = EmbeddingSim()
    vec = torch.randn(10)
    sim = emb_sim.similarity(vec, vec)
    assert torch.isclose(sim, torch.tensor(1.0), atol=1e-6)


def test_similarity_different_vectors_same_dim():
    emb_sim = EmbeddingSim()
    vec1 = torch.tensor([1., 0., 0.])
    vec2 = torch.tensor([0., 1., 0.])
    sim = emb_sim.similarity(vec1, vec2)
    # orthogonal vectors cosine similarity is 0
    assert torch.isclose(sim, torch.tensor(0.), atol=1e-6)


def test_similarity_input_requires_grad():
    emb_sim = EmbeddingSim()
    vec1 = torch.randn(5, requires_grad=True)
    vec2 = torch.randn(5, requires_grad=True)
    sim = emb_sim.similarity(vec1, vec2)
    sim.backward()
    # gradients should be populated
    assert vec1.grad is not None
    assert vec2.grad is not None


def test_similarity_handles_batch_vectors():
    emb_sim = EmbeddingSim()
    vec1 = torch.randn(3, 10)
    vec2 = torch.randn(3, 10)
    sim = emb_sim.similarity(vec1, vec2)
    # should return a tensor of shape (3,)
    assert sim.shape == (3,)
    # values should be in [-1,1]
    assert torch.all(sim <= 1)
    assert torch.all(sim >= -1)


def test_similarity_tensor_input_type():
    emb_sim = EmbeddingSim()
    vec1 = torch.randn(4)
    vec2 = torch.randn(4)
    out = emb_sim.similarity(vec1, vec2)
    assert isinstance(out, torch.Tensor)


def test_similarity_raises_on_different_dims():
    emb_sim = EmbeddingSim()
    vec1 = torch.randn(5)
    vec2 = torch.randn(4)
    with pytest.raises(RuntimeError):
        emb_sim.similarity(vec1, vec2)


def test_similarity_batch_and_single_vector_inputs():
    emb_sim = EmbeddingSim()
    vec1 = torch.randn(7)
    vec2 = torch.randn(3, 7)
    sim = emb_sim.similarity(vec1, vec2)
    # should return a tensor of shape (3,)
    assert sim.shape == (3,)


def test_similarity_with_zero_vector():
    emb_sim = EmbeddingSim()
    vec1 = torch.zeros(10)
    vec2 = torch.randn(10)
    sim = emb_sim.similarity(vec1, vec2)
    # cosine similarity with zero vector may result in nan or 0, test it does not throw
    assert sim.shape == ()
    # check for nan or finite number
    assert not torch.isnan(sim)


def test_similarity_all_zero_batch_vectors():
    emb_sim = EmbeddingSim()
    vec1 = torch.zeros(2, 5)
    vec2 = torch.zeros(2, 5)
    sim = emb_sim.similarity(vec1, vec2)
    # cosine similarity between zero vectors, result could be nan or 0
    assert sim.shape == (2,)
    assert not torch.isnan(sim).any()


def test_similarity_normalized_vectors_is_dot_product():
    emb_sim = EmbeddingSim()
    v1 = torch.tensor([1.0, 0.0])
    v2 = torch.tensor([0.0, 1.0])
    # normalize vectors manually
    v1_norm = v1 / v1.norm()
    v2_norm = v2 / v2.norm()
    sim1 = emb_sim.similarity(v1_norm, v2_norm)
    # Should equal dot product of normalized vectors (which is 0)
    dot = torch.dot(v1_norm, v2_norm)
    assert torch.isclose(sim1, dot, atol=1e-6)


def test_similarity_with_large_vectors():
    emb_sim = EmbeddingSim()
    vec1 = torch.randn(1000)
    vec2 = torch.randn(1000)
    sim = emb_sim.similarity(vec1, vec2)
    assert sim.shape == ()
    assert sim <= 1
    assert sim >= -1


def test_similarity_multiple_calls_consistent():
    emb_sim = EmbeddingSim()
    vec1 = torch.randn(8)
    vec2 = torch.randn(8)
    sim1 = emb_sim.similarity(vec1, vec2)
    sim2 = emb_sim.similarity(vec1, vec2)
    assert torch.isclose(sim1, sim2, atol=1e-8)


def test_similarity_input_dtypes_float32_float64():
    emb_sim = EmbeddingSim()
    vec1 = torch.randn(10, dtype=torch.float32)
    vec2 = torch.randn(10, dtype=torch.float64)
    sim = emb_sim.similarity(vec1, vec2)
    assert isinstance(sim, torch.Tensor)
    # result will be float64 after promotion, check dtype
    assert sim.dtype == torch.float64 or sim.dtype == torch.float32


def test_similarity_with_nan_values():
    emb_sim = EmbeddingSim()
    vec1 = torch.tensor([float('nan'), 0., 1.])
    vec2 = torch.tensor([1., 0., 0.])
    sim = emb_sim.similarity(vec1, vec2)
    # similarity with nan input results in nan output
    assert torch.isnan(sim)


def test_similarity_with_inf_values():
    emb_sim = EmbeddingSim()
    vec1 = torch.tensor([float('inf'), 0., 1.])
    vec2 = torch.tensor([1., 0., 0.])
    sim = emb_sim.similarity(vec1, vec2)
    # cosine similarity with inf input usually results in nan or inf
    assert sim.isnan() or sim.isinf()


def test_similarity_with_different_device_inputs():
    emb_sim = EmbeddingSim()
    if torch.cuda.is_available():
        vec1 = torch.randn(5, device='cpu')
        vec2 = torch.randn(5, device='cuda')
        with pytest.raises(RuntimeError):
            emb_sim.similarity(vec1, vec2)
    else:
        pytest.skip("CUDA not available - skipping device mismatch test.")


def test_similarity_requires_grad_output():
    emb_sim = EmbeddingSim()
    vec1 = torch.randn(5, requires_grad=True)
    vec2 = torch.randn(5, requires_grad=True)
    sim = emb_sim.similarity(vec1, vec2)
    assert sim.requires_grad


def test_similarity_multiple_vector_pairs():
    emb_sim = EmbeddingSim()
    vecs1 = torch.randn(4, 6)
    vecs2 = torch.randn(4, 6)
    sims = emb_sim.similarity(vecs1, vecs2)
    assert sims.shape == (4,)
    assert torch.all(sims <= 1)
    assert torch.all(sims >= -1)


def test_similarity_empty_vectors_raises():
    emb_sim = EmbeddingSim()
    vec1 = torch.tensor([])
    vec2 = torch.tensor([])
    with pytest.raises(RuntimeError):
        emb_sim.similarity(vec1, vec2)
