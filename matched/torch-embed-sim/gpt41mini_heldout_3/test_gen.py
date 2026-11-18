import pytest
import torch
from torch_embed_sim import EmbeddingSim

@pytest.fixture
def es_model():
    return EmbeddingSim()

def test_default_init(es_model):
    assert hasattr(es_model, 'device')
    assert hasattr(es_model, 'normalize_embeddings')
    assert hasattr(es_model, 'model')
    assert hasattr(es_model, 'tokenizer')
    # model and tokenizer should be set
    assert es_model.model is not None
    assert es_model.tokenizer is not None

def test_forward_returns_tensor_shape(es_model):
    texts = ["hello world", "test string"]
    embeddings = es_model(texts)
    assert isinstance(embeddings, torch.Tensor)
    # embeddings shape should be (batch_size, embedding_dim)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.dim() == 2

def test_forward_single_text_string(es_model):
    text = "a single string"
    embedding = es_model(text)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.dim() == 1

def test_forward_empty_list_returns_empty_tensor(es_model):
    embeddings = es_model([])
    # Should return empty tensor with shape (0, embedding_dim)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == 0

def test_forward_empty_string(es_model):
    embedding = es_model([""])
    # Should produce valid embedding vector
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape[0] == 1
    assert embedding.dim() == 2 or embedding.dim() == 1

def test_similarity_between_same_text(es_model):
    texts = ["the quick brown fox"] * 2
    embedding = es_model(texts)
    sim = es_model.similarity(embedding[0], embedding[1])
    # similarity with itself or duplicates should be close to 1 if normalized
    if es_model.normalize_embeddings:
        assert pytest.approx(sim.item(), 0.001) == 1.0
    else:
        # similarity might be > 0 but not necessarily 1
        assert sim.item() > 0

def test_similarity_different_texts(es_model):
    emb1 = es_model("dog")
    emb2 = es_model("cat")
    sim = es_model.similarity(emb1, emb2)
    # similarity is a scalar tensor
    assert isinstance(sim, torch.Tensor)
    assert sim.dim() == 0

def test_cosine_sim_equals_similarity_when_normalized():
    model = EmbeddingSim(normalize_embeddings=True)
    emb1 = model("text one")
    emb2 = model("text two")
    # similarity should be cosine similarity for normalized embeddings
    sim = model.similarity(emb1, emb2)
    cos_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    assert pytest.approx(sim.item(), 0.001) == cos_sim.item()

def test_euclidean_sim_when_not_normalized():
    model = EmbeddingSim(normalize_embeddings=False)
    emb1 = model("text one")
    emb2 = model("text two")
    sim = model.similarity(emb1, emb2)
    # similarity should be negative euclidean distance
    dist = torch.norm(emb1 - emb2, p=2)
    assert pytest.approx(sim.item(), 1e-4) == -dist.item()

def test_similarity_batch_inputs(es_model):
    texts1 = ["a"] * 3
    texts2 = ["b"] * 3
    emb1 = es_model(texts1)
    emb2 = es_model(texts2)
    # compute pairwise similarity matrix
    sims = es_model.similarity(emb1.unsqueeze(1), emb2.unsqueeze(0))
    # Should broadcast and return matrix of shape (3,3)
    assert sims.shape == (3, 3)

def test_embedding_dim_matches_model_output(es_model):
    texts = ["hello world"]
    emb = es_model(texts)
    # embedding dimension must match underlying model output size
    embedding_dim = emb.shape[-1]
    # Since model is from HF transformer, dims > 0
    assert embedding_dim > 0

def test_call_with_non_string_raises_type_error(es_model):
    with pytest.raises(TypeError):
        _ = es_model(123)

def test_call_with_mixed_list_raises_type_error(es_model):
    with pytest.raises(TypeError):
        _ = es_model(["valid", 123])

def test_embedding_is_deterministic(es_model):
    text = "deterministic test"
    emb1 = es_model(text)
    emb2 = es_model(text)
    # Embeddings for the same input should be close
    assert torch.allclose(emb1, emb2, atol=1e-5)

def test_call_tensor_input_raises_type_error(es_model):
    with pytest.raises(TypeError):
        _ = es_model(torch.tensor([1, 2, 3]))

def test_normalize_parameter_change_affects_output():
    model_norm = EmbeddingSim(normalize_embeddings=True)
    model_no_norm = EmbeddingSim(normalize_embeddings=False)
    text = "normalize test"
    emb_norm = model_norm(text)
    emb_no_norm = model_no_norm(text)
    # When normalized embeddings are requested, outputs should be unit vectors
    norms = torch.norm(emb_norm, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # When not normalized, norms should not be all close to 1
    norms_no_norm = torch.norm(emb_no_norm, dim=-1)
    assert not torch.allclose(norms_no_norm, torch.ones_like(norms_no_norm), atol=1e-2)

def test_similarity_with_different_devices(es_model):
    # if cuda is available test similarity on cuda, else skip
    if torch.cuda.is_available():
        model_cuda = EmbeddingSim()
        model_cuda.model.to('cuda')
        model_cuda.device = torch.device('cuda')
        emb_cpu = es_model("cpu text")
        emb_cuda = model_cuda("cuda text")
        # similarity between tensors on different devices should error or be handled
        with pytest.raises(RuntimeError):
            _ = model_cuda.similarity(emb_cpu, emb_cuda)
    else:
        pytest.skip("CUDA device not available")

def test_batch_input_with_empty_string(es_model):
    texts = ["valid text", ""]
    emb = es_model(texts)
    assert emb.shape[0] == 2
    assert isinstance(emb, torch.Tensor)

def test_call_with_none_raises_type_error(es_model):
    with pytest.raises(TypeError):
        _ = es_model(None)
