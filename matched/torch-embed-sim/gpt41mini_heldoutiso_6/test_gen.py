import pytest
import torch
from torch_embed_sim import EmbeddingSim


def test_init_with_empty_embedding_tensor():
    emb = EmbeddingSim(torch.tensor([]))
    assert isinstance(emb, EmbeddingSim)
    assert emb.embeddings.numel() == 0


def test_init_with_2d_embedding_tensor():
    embeddings = torch.randn(10, 5)
    emb = EmbeddingSim(embeddings)
    assert torch.allclose(emb.embeddings, embeddings)
    assert emb.embeddings.shape == (10, 5)


def test_init_raises_with_non_tensor_input():
    with pytest.raises(TypeError):
        EmbeddingSim([[1, 2], [3, 4]])


def test_init_raises_if_embedding_tensor_not_2d():
    with pytest.raises(ValueError):
        EmbeddingSim(torch.randn(3))


def test_clone_creates_independent_copy():
    embeddings = torch.randn(7, 3)
    emb = EmbeddingSim(embeddings)
    clone = emb.clone()
    assert isinstance(clone, EmbeddingSim)
    assert torch.allclose(clone.embeddings, emb.embeddings)
    # Modify clone embeddings and check original unchanged
    clone.embeddings[0, 0] += 1
    assert not torch.allclose(clone.embeddings, emb.embeddings)


def test_embed_idx_returns_correct_embeddings():
    embeddings = torch.randn(5, 4)
    emb = EmbeddingSim(embeddings)
    idxs = torch.tensor([0, 2, 4])
    selected = emb.embed_idx(idxs)
    assert selected.shape == (3, 4)
    assert torch.allclose(selected, embeddings[idxs])


def test_embed_idx_raises_with_invalid_index():
    embeddings = torch.randn(5, 4)
    emb = EmbeddingSim(embeddings)
    idxs = torch.tensor([10])  # Out of range
    with pytest.raises(IndexError):
        emb.embed_idx(idxs)


def test_embed_idx_works_with_single_scalar_index():
    embeddings = torch.randn(5, 4)
    emb = EmbeddingSim(embeddings)
    idx = 3
    selected = emb.embed_idx(idx)
    assert torch.allclose(selected, embeddings[idx])


def test_similarity_returns_tensor_of_expected_shape():
    embeddings = torch.randn(6, 8)
    emb = EmbeddingSim(embeddings)
    vecs = torch.randn(4, 8)
    sim = emb.similarity(vecs)
    assert isinstance(sim, torch.Tensor)
    assert sim.shape == (6, 4)


def test_similarity_with_normalized_embeddings_produces_cosine_sim():
    embeddings = torch.randn(3, 5)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    emb = EmbeddingSim(embeddings)
    vecs = torch.randn(2, 5)
    vecs = vecs / vecs.norm(dim=1, keepdim=True)
    sim = emb.similarity(vecs)
    # Cosine similarity values must be in [-1, 1]
    assert torch.all(sim <= 1)
    assert torch.all(sim >= -1)


def test_similarity_raises_if_dimensionality_mismatch():
    embeddings = torch.randn(4, 7)
    emb = EmbeddingSim(embeddings)
    vecs = torch.randn(2, 8)
    with pytest.raises(RuntimeError):
        emb.similarity(vecs)


def test_repr_contains_shape_information():
    embeddings = torch.randn(9, 2)
    emb = EmbeddingSim(embeddings)
    r = repr(emb)
    assert str(emb.embeddings.shape) in r


def test_eq_and_ne_match_based_on_embeddings():
    embeddings1 = torch.randn(5, 3)
    embeddings2 = embeddings1.clone()
    embeddings3 = torch.randn(5, 3)
    emb1 = EmbeddingSim(embeddings1)
    emb2 = EmbeddingSim(embeddings2)
    emb3 = EmbeddingSim(embeddings3)
    assert emb1 == emb2
    assert emb1 != emb3
    assert (emb1 == "not an emb") is False


def test_embedding_sim_is_device_independent():
    embeddings = torch.randn(4, 4)
    emb = EmbeddingSim(embeddings)
    # Should hold on CPU and if CUDA available, on CUDA too
    cpu_clone = emb.clone()
    assert cpu_clone.embeddings.device.type == 'cpu'
    if torch.cuda.is_available():
        emb_cuda = EmbeddingSim(embeddings.to('cuda'))
        assert emb_cuda.embeddings.device.type == 'cuda'


def test_similarity_with_empty_embeddings_and_vectors():
    emb = EmbeddingSim(torch.empty((0, 10)))
    vecs = torch.empty((0, 10))
    sim = emb.similarity(vecs)
    assert sim.shape == (0, 0)


def test_embed_idx_with_empty_index_returns_empty_tensor():
    embeddings = torch.randn(3, 3)
    emb = EmbeddingSim(embeddings)
    idxs = torch.tensor([], dtype=torch.long)
    result = emb.embed_idx(idxs)
    assert result.shape == (0, 3)


def test_clone_and_modify_do_not_affect_original():
    embeddings = torch.randn(2, 3)
    emb = EmbeddingSim(embeddings)
    clone = emb.clone()
    clone.embeddings += 1
    assert not torch.allclose(emb.embeddings, clone.embeddings)


def test_embed_idx_accepts_python_list():
    embeddings = torch.randn(4, 6)
    emb = EmbeddingSim(embeddings)
    idxs = [1, 3]
    selected = emb.embed_idx(idxs)
    assert selected.shape == (2, 6)
    assert torch.allclose(selected, embeddings[idxs])


def test_similarity_accepts_single_vector():
    embeddings = torch.randn(5, 7)
    emb = EmbeddingSim(embeddings)
    vec = torch.randn(7)
    sim = emb.similarity(vec)
    assert sim.shape == (5,)


def test_similarity_with_batched_vectors_and_single_vector_are_consistent():
    embeddings = torch.randn(3, 4)
    emb = EmbeddingSim(embeddings)
    vec_single = torch.randn(4)
    vec_batch = vec_single.unsqueeze(0)
    sim_single = emb.similarity(vec_single)
    sim_batch = emb.similarity(vec_batch)
    assert torch.allclose(sim_single.unsqueeze(1), sim_batch.transpose(0, 1))


def test_repr_returns_string_without_error():
    embeddings = torch.randn(1, 1)
    emb = EmbeddingSim(embeddings)
    r = repr(emb)
    assert isinstance(r, str)
