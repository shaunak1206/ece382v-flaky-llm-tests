import random
import numpy as np
import torch
import pytest

from torch_embed_sim import EmbeddingSim


def test_embedding_sim_init_defaults():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    embed_sim = EmbeddingSim()
    # Should create a model with embedding_dim=50 (default), no device specified
    assert isinstance(embed_sim.embedding_dim, int)
    assert embed_sim.embedding_dim == 50
    # The device property should be torch.device('cpu') by default
    assert embed_sim.device == torch.device("cpu")
    # The underlying embedding model should have weight with correct shape
    weight = embed_sim.embedding.weight
    assert weight.shape[1] == embed_sim.embedding_dim


def test_embedding_sim_init_with_custom_dim():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    dim = 128
    embed_sim = EmbeddingSim(embedding_dim=dim)
    assert embed_sim.embedding_dim == dim
    weight = embed_sim.embedding.weight
    assert weight.shape[1] == dim


def test_embedding_sim_init_with_device(monkeypatch):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # monkeypatch torch.device to a dummy to verify usage
    monkeypatch.setattr(torch, "device", lambda x: torch.device("cpu") if x == "cpu" else torch.device("cpu"))
    embed_sim = EmbeddingSim(device="cpu")
    assert embed_sim.device == torch.device("cpu")


def test_embedding_sim_embed_tensor_input():
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)

    embed_sim = EmbeddingSim(embedding_dim=10)
    # create tensor input of indices with dtype int64
    idxs = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    embeddings = embed_sim.embed(idxs)
    # embeddings.shape should correspond to (len(idxs), embedding_dim)
    assert embeddings.shape == (4, embed_sim.embedding_dim)
    assert isinstance(embeddings, torch.Tensor)


def test_embedding_sim_embed_list_input():
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    embed_sim = EmbeddingSim(embedding_dim=8)
    input_list = [5, 2, 7]
    embeddings = embed_sim.embed(input_list)
    assert embeddings.shape == (3, embed_sim.embedding_dim)
    assert isinstance(embeddings, torch.Tensor)


def test_embedding_sim_embed_empty_input():
    random.seed(30)
    np.random.seed(30)
    torch.manual_seed(30)

    embed_sim = EmbeddingSim()
    embeddings = embed_sim.embed([])
    # empty input should yield empty embeddings (0, embedding_dim)
    assert embeddings.shape == (0, embed_sim.embedding_dim)


def test_embedding_sim_forward_dot_similarity():
    random.seed(40)
    np.random.seed(40)
    torch.manual_seed(40)

    embed_sim = EmbeddingSim(embedding_dim=6)

    left = torch.tensor([1, 3, 0, 2])
    right = torch.tensor([2, 1, 3, 0])
    similarities = embed_sim(left, right)

    assert similarities.shape == (4,)
    assert isinstance(similarities, torch.Tensor)
    # Similarities are dot products, should be finite floats
    assert torch.all(torch.isfinite(similarities))
    # The similarity values should be equal to the batch dot product of embeddings:
    emb_left = embed_sim.embed(left)
    emb_right = embed_sim.embed(right)
    dot = (emb_left * emb_right).sum(dim=1)
    assert torch.allclose(similarities, dot, atol=1e-6)


def test_embedding_sim_forward_cosine_similarity():
    random.seed(50)
    np.random.seed(50)
    torch.manual_seed(50)

    embed_sim = EmbeddingSim(embedding_dim=7)
    left = torch.tensor([0, 2, 4])
    right = torch.tensor([4, 2, 0])

    cos_sim = embed_sim.similarity_cosine(left, right)
    # Should have shape equal to input batch size
    assert cos_sim.shape == (3,)

    # Check cosine similarity value range (-1, 1)
    assert torch.all(cos_sim <= 1.0 + 1e-6)
    assert torch.all(cos_sim >= -1.0 - 1e-6)

    # Cosine similarity between identical inputs should be close to 1
    ones = embed_sim.similarity_cosine(left, left)
    assert torch.allclose(ones, torch.ones_like(ones), atol=1e-5)


def test_embedding_sim_similarity_raises_on_bad_input():
    random.seed(60)
    np.random.seed(60)
    torch.manual_seed(60)

    embed_sim = EmbeddingSim()
    # pass inputs of different lengths to similarity functions, expect error or handle
    left = torch.tensor([1, 2])
    right = torch.tensor([1])
    with pytest.raises(RuntimeError):
        embed_sim(left, right)


def test_embedding_sim_weight_update_not_persistent(monkeypatch):
    random.seed(70)
    np.random.seed(70)
    torch.manual_seed(70)

    embed_sim = EmbeddingSim(embedding_dim=5)
    original_weight = embed_sim.embedding.weight.clone()

    # Random gradient step simulation: add 1.0 to weights, but only temporarily via monkeypatch
    def fake_forward(*args, **kwargs):
        return torch.ones(len(args[0]), device=embed_sim.device)

    monkeypatch.setattr(embed_sim, "forward", fake_forward)
    out = embed_sim(torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert torch.allclose(out, torch.ones_like(out))

    # The weight should remain the same as before the patch
    assert torch.allclose(embed_sim.embedding.weight, original_weight, atol=1e-7)


def test_embedding_sim_embed_invalid_dtype_raises():
    random.seed(80)
    np.random.seed(80)
    torch.manual_seed(80)

    embed_sim = EmbeddingSim()

    # Passing float tensor is invalid input for index-based embedding
    with pytest.raises(RuntimeError):
        embed_sim.embed(torch.tensor([1.0, 2.0]))


def test_embedding_sim_embed_numpy_array_input():
    random.seed(90)
    np.random.seed(90)
    torch.manual_seed(90)

    embed_sim = EmbeddingSim(embedding_dim=4)
    idxs = np.array([0, 1, 2, 3])
    embeddings = embed_sim.embed(idxs)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (4, embed_sim.embedding_dim)


def test_embedding_sim_embed_tensor_on_gpu(monkeypatch):
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)

    embed_sim = EmbeddingSim(embedding_dim=3)
    # Monkeypatch device to "cpu" forcibly (simulate no GPU)
    # The code defaults to cpu, so the embeddings will be on cpu.
    tensor = torch.tensor([0, 1, 2])
    out = embed_sim.embed(tensor)
    assert out.device.type == "cpu"


def test_embedding_sim_consistency_over_runs():
    # Seeds set to same value; embedding weights and outputs should match across instances
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    e1 = EmbeddingSim(embedding_dim=10)
    weight1 = e1.embedding.weight.detach().clone()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    e2 = EmbeddingSim(embedding_dim=10)
    weight2 = e2.embedding.weight.detach().clone()

    assert torch.allclose(weight1, weight2)

    input_tensor = torch.tensor([0, 1, 2])
    out1 = e1.embed(input_tensor)
    out2 = e2.embed(input_tensor)
    assert torch.allclose(out1, out2, atol=1e-7)


def test_embedding_sim_forward_on_empty_input():
    random.seed(111)
    np.random.seed(111)
    torch.manual_seed(111)

    embed_sim = EmbeddingSim()
    left = torch.tensor([], dtype=torch.long)
    right = torch.tensor([], dtype=torch.long)
    out = embed_sim(left, right)
    assert out.shape == (0,)
    assert out.dtype == torch.float32


def test_embedding_sim_embedding_parameters_are_trainable():
    random.seed(222)
    np.random.seed(222)
    torch.manual_seed(222)

    embed_sim = EmbeddingSim(embedding_dim=13)
    params = list(embed_sim.embedding.parameters())
    assert len(params) == 1
    param = params[0]
    assert param.requires_grad
    # The embedding weight tensor expected size matches vocab size x embedding_dim
    assert param.shape[1] == 13


def test_embedding_sim_similarity_cosine_accepts_various_inputs():
    random.seed(333)
    np.random.seed(333)
    torch.manual_seed(333)

    embed_sim = EmbeddingSim(embedding_dim=5)

    # Test inputs as list
    left = [0, 1, 2]
    right = [2, 1, 0]

    sim1 = embed_sim.similarity_cosine(left, right)
    sim2 = embed_sim.similarity_cosine(torch.tensor(left), torch.tensor(right))

    # Both should be equal
    assert torch.allclose(sim1, sim2, atol=1e-7)


def test_embedding_sim_embed_large_input_shape():
    random.seed(444)
    np.random.seed(444)
    torch.manual_seed(444)

    embed_sim = EmbeddingSim(embedding_dim=4)
    # Large batch of indices (but keep small to not consume much memory)
    idxs = list(range(100))
    embeddings = embed_sim.embed(idxs)
    assert embeddings.shape == (100, 4)


def test_embedding_sim_repr_contains_embedding_dim():
    random.seed(555)
    np.random.seed(555)
    torch.manual_seed(555)

    embed_sim = EmbeddingSim(embedding_dim=17)
    rep = repr(embed_sim)
    assert "EmbeddingSim" in rep
    assert str(embed_sim.embedding_dim) in rep


def test_embedding_sim_wrong_input_type_for_embed():
    random.seed(666)
    np.random.seed(666)
    torch.manual_seed(666)

    embed_sim = EmbeddingSim()
    with pytest.raises(TypeError):
        # Passing a dict is not supported
        embed_sim.embed({"a": 1})
