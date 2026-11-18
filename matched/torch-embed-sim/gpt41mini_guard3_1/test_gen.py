import math
import torch
import random
import numpy as np
import pytest

from torch_embed_sim import EmbeddingSim


def test_embedding_sim_init_default_dims():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=10)
    assert emb.num_embeddings == 10
    assert emb.embedding_dim > 0
    # Default embeddings is a torch Parameter of size (10, embedding_dim)
    assert emb.embeddings.shape[0] == 10
    assert emb.embeddings.shape[1] == emb.embedding_dim
    # Check embeddings are torch Parameter
    assert hasattr(emb.embeddings, "grad")
    # Embeddings initialized with float32 dtype
    assert emb.embeddings.dtype == torch.float32


def test_embedding_sim_init_custom_dims_and_dtype():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=5, embedding_dim=8, dtype=torch.float64)
    assert emb.num_embeddings == 5
    assert emb.embedding_dim == 8
    assert emb.embeddings.shape == (5, 8)
    assert emb.embeddings.dtype == torch.float64


def test_embedding_sim_forward_with_int_tensor(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=4, embedding_dim=3)
    # Input as LongTensor indices
    indices = torch.tensor([0, 2, 3, 1, 2], dtype=torch.long)
    output = emb(indices)
    # output shape: (len(indices), embedding_dim)
    assert output.shape == (5, 3)
    # Type check
    assert isinstance(output, torch.Tensor)
    # Each row should be the embedding of the corresponding index
    for i, idx in enumerate(indices.tolist()):
        torch.testing.assert_allclose(output[i], emb.embeddings.data[idx], rtol=1e-6, atol=1e-6)


def test_embedding_sim_forward_with_list_of_ints(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=6, embedding_dim=4)
    idx_list = [5, 0, 3, 2]
    output = emb(idx_list)
    assert output.shape == (4, 4)
    for i, idx in enumerate(idx_list):
        torch.testing.assert_allclose(output[i], emb.embeddings.data[idx], rtol=1e-6, atol=1e-6)


def test_embedding_sim_forward_with_tensor_float_raises(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=3)
    float_input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
    with pytest.raises(TypeError):
        _ = emb(float_input)


def test_embedding_sim_forward_with_out_of_range_indices(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=3)
    # Indices out of range should raise an IndexError (typical for embedding layers)
    out_of_range = torch.tensor([0, 2, 3], dtype=torch.long)
    with pytest.raises(IndexError):
        _ = emb(out_of_range)


def test_embedding_sim_forward_empty_input(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=4, embedding_dim=5)
    empty_indices = torch.tensor([], dtype=torch.long)
    output = emb(empty_indices)
    assert output.shape == (0, 5)


def test_embedding_sim_cosine_similarity(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=3, embedding_dim=4)
    emb.embeddings.data = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ], dtype=torch.float32)

    # Similarity between embedding 0 and 1 expected 0
    sim_0_1 = emb.similarity(0, 1)
    assert math.isclose(sim_0_1.item(), 0.0, abs_tol=1e-6)
    # Similarity between embedding 0 and 2 expected 1/sqrt(2) =~ 0.7071
    sim_0_2 = emb.similarity(0, 2)
    assert math.isclose(sim_0_2.item(), 1 / math.sqrt(2), rel_tol=1e-5, abs_tol=1e-6)
    # Similarity between embedding 2 and 2 is 1.0
    sim_2_2 = emb.similarity(2, 2)
    assert math.isclose(sim_2_2.item(), 1.0, abs_tol=1e-6)


def test_embedding_sim_similarity_invalid_index(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=2)
    with pytest.raises(IndexError):
        emb.similarity(0, 2)
    with pytest.raises(IndexError):
        emb.similarity(-1, 1)


def test_embedding_sim_similarity_returns_scalar_tensor(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=5)
    sim = emb.similarity(1, 3)
    # Should return a torch scalar tensor (0-dim tensor)
    assert isinstance(sim, torch.Tensor)
    assert sim.dim() == 0


def test_embedding_sim_embedding_dim_property(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(7, embedding_dim=13)
    assert emb.embedding_dim == 13


def test_embedding_sim_num_embeddings_property(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(11)
    assert emb.num_embeddings == 11


def test_embedding_sim_state_dict_keys(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(3)
    sd = emb.state_dict()
    # Expect at least "embeddings.weight" key or similar according to standard patterns
    # Since source is unknown, just check keys are strings and have embeddings tensor inside
    keys = list(sd.keys())
    for k in keys:
        assert isinstance(k, str)
    # The state dict values corresponding to keys are tensors
    for v in sd.values():
        assert isinstance(v, torch.Tensor)


@pytest.mark.parametrize("indices", [
    torch.tensor([0], dtype=torch.long),
    torch.tensor([1, 2], dtype=torch.long),
    torch.tensor([], dtype=torch.long),
])
def test_embedding_sim_forward_various_sizes(indices):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=5, embedding_dim=7)
    out = emb(indices)
    assert out.shape == (indices.shape[0], 7)


def test_embedding_sim_forward_with_duplicate_indices():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    emb = EmbeddingSim(num_embeddings=5, embedding_dim=3)
    indices = torch.tensor([1, 1, 2, 2, 3], dtype=torch.long)
    output = emb(indices)
    for i, idx in enumerate(indices.tolist()):
        torch.testing.assert_allclose(output[i], emb.embeddings.data[idx], rtol=1e-6, atol=1e-6)

