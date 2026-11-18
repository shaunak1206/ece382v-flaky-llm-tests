import pytest
import torch
from torch_embed_sim import EmbeddingSim


def test_embedding_sim_init_default_device():
    sim = EmbeddingSim()
    assert sim.device.type in ("cpu", "cuda")


def test_embedding_sim_init_with_device_cpu():
    sim = EmbeddingSim(device="cpu")
    assert sim.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_embedding_sim_init_with_device_cuda():
    sim = EmbeddingSim(device="cuda")
    assert sim.device.type == "cuda"


def test_embedding_sim_embed_returns_tensor_with_correct_shape():
    sim = EmbeddingSim()
    texts = ["hello world", "embedding similarity"]
    embeds = sim.embed(texts)
    assert isinstance(embeds, torch.Tensor)
    assert embeds.shape[0] == len(texts)
    assert embeds.shape[1] > 0  # embedding dimension


def test_embedding_sim_embed_empty_list_returns_empty_tensor():
    sim = EmbeddingSim()
    embeds = sim.embed([])
    assert isinstance(embeds, torch.Tensor)
    assert embeds.shape[0] == 0


def test_embedding_sim_embed_single_string_is_handled():
    sim = EmbeddingSim()
    embeds = sim.embed("single string input")
    assert isinstance(embeds, torch.Tensor)
    assert embeds.shape[0] == 1


def test_embedding_sim_cosine_similarity_returns_correct_shape():
    sim = EmbeddingSim()
    v1 = torch.randn(5, 10)
    v2 = torch.randn(3, 10)
    cos_sim = sim.cosine_similarity(v1, v2)
    assert isinstance(cos_sim, torch.Tensor)
    assert cos_sim.shape == (5, 3)
    # values between -1 and 1 for cosine similarity
    assert torch.min(cos_sim) >= -1.0
    assert torch.max(cos_sim) <= 1.0


def test_embedding_sim_cosine_similarity_equal_vectors_returns_one():
    sim = EmbeddingSim()
    v = torch.randn(4, 8)
    cos_sim = sim.cosine_similarity(v, v)
    diag = torch.diagonal(cos_sim)
    # diagonal cosine similarities should be close to 1
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)


def test_embedding_sim_similarity_texts_returns_tensor():
    sim = EmbeddingSim()
    texts1 = ["a", "b", "c"]
    texts2 = ["x", "y"]
    sim_matrix = sim.similarity(texts1, texts2)
    assert isinstance(sim_matrix, torch.Tensor)
    assert sim_matrix.shape == (len(texts1), len(texts2))
    # values between -1 and 1
    assert torch.min(sim_matrix) >= -1.0
    assert torch.max(sim_matrix) <= 1.0


def test_embedding_sim_similarity_text_and_embedding_mix():
    sim = EmbeddingSim()
    texts = ["hello", "world"]
    embeds = sim.embed(["foo", "bar", "baz"])
    sim_matrix = sim.similarity(texts, embeds)
    assert sim_matrix.shape == (len(texts), embeds.shape[0])


def test_embedding_sim_similarity_embedding_and_text_mix():
    sim = EmbeddingSim()
    embeds = sim.embed(["foo", "bar", "baz"])
    texts = ["hello", "world"]
    sim_matrix = sim.similarity(embeds, texts)
    assert sim_matrix.shape == (embeds.shape[0], len(texts))


def test_embedding_sim_similarity_embedding_and_embedding():
    sim = EmbeddingSim()
    embeds1 = sim.embed(["foo", "bar"])
    embeds2 = sim.embed(["baz", "qux"])
    sim_matrix = sim.similarity(embeds1, embeds2)
    assert sim_matrix.shape == (embeds1.shape[0], embeds2.shape[0])


def test_embedding_sim_similarity_empty_inputs():
    sim = EmbeddingSim()
    empty_embeds = sim.embed([])
    texts = ["a", "b"]
    sim_matrix = sim.similarity([], texts)
    assert sim_matrix.shape == (0, len(texts))
    sim_matrix2 = sim.similarity(texts, [])
    assert sim_matrix2.shape == (len(texts), 0)
    sim_matrix3 = sim.similarity([], [])
    assert sim_matrix3.shape == (0, 0)


def test_embedding_sim_device_consistency_after_embed():
    sim_cpu = EmbeddingSim(device="cpu")
    embeds_cpu = sim_cpu.embed(["test"])
    assert embeds_cpu.device.type == "cpu"
    if torch.cuda.is_available():
        sim_cuda = EmbeddingSim(device="cuda")
        embeds_cuda = sim_cuda.embed(["test"])
        assert embeds_cuda.device.type == "cuda"


def test_embedding_sim_embedding_values_are_normalized():
    sim = EmbeddingSim()
    embeds = sim.embed(["normalize test"])
    norms = embeds.norm(dim=1)
    # Cosine similarity expects normalized embeddings (norm near 1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_embedding_sim_similarity_matrix_diagonal_for_same_texts_close_to_one():
    sim = EmbeddingSim()
    texts = ["repeat", "repeat", "repeat"]
    sim_matrix = sim.similarity(texts, texts)
    diag = torch.diagonal(sim_matrix)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)


def test_embedding_sim_empty_string_embedding():
    sim = EmbeddingSim()
    embeds = sim.embed([""])
    assert embeds.shape[0] == 1
    assert embeds.shape[1] > 0


def test_embedding_sim_large_number_of_texts_embeds():
    sim = EmbeddingSim()
    num_texts = 100
    texts = ["text {}".format(i) for i in range(num_texts)]
    embeds = sim.embed(texts)
    assert embeds.shape[0] == num_texts


def test_embedding_sim_similarity_with_identical_input():
    sim = EmbeddingSim()
    texts = ["same", "same", "same"]
    sim_matrix = sim.similarity(texts, texts)
    assert sim_matrix.shape == (len(texts), len(texts))
    diag = torch.diagonal(sim_matrix)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)
