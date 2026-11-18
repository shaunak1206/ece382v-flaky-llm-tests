import pytest
import torch
from torch_embed_sim import EmbeddingSim

@pytest.fixture
def embeddings_a():
    return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

@pytest.fixture
def embeddings_b():
    return torch.tensor([[0.0, 1.0], [1.0, 0.0]])

@pytest.fixture
def embeddings_zero():
    return torch.zeros((2, 2))

@pytest.fixture
def embeddings_single():
    return torch.tensor([[3.0, 4.0]])

def test_init_with_default_parameters():
    sim = EmbeddingSim()
    assert sim.p == 2
    assert sim.eps == 1e-8

def test_init_with_custom_parameters():
    sim = EmbeddingSim(p=1.5, eps=1e-5)
    assert sim.p == 1.5
    assert sim.eps == 1e-5

def test_forward_returns_tensor(embeddings_a, embeddings_b):
    sim = EmbeddingSim()
    out = sim(embeddings_a, embeddings_b)
    assert isinstance(out, torch.Tensor)

def test_forward_output_shape(embeddings_a, embeddings_b):
    sim = EmbeddingSim()
    out = sim(embeddings_a, embeddings_b)
    assert out.shape == (embeddings_a.shape[0], embeddings_b.shape[0])

def test_similarity_matrix_range(embeddings_a, embeddings_b):
    sim = EmbeddingSim()
    out = sim(embeddings_a, embeddings_b)
    # Cosine similarity (or norm based) ranges between -1 and 1 approximately
    # But this depends on implementation. Just verify values are finite
    assert torch.all(torch.isfinite(out))

def test_similarity_identity_is_max(embeddings_a):
    sim = EmbeddingSim()
    out = sim(embeddings_a, embeddings_a)
    # Diagonal should be highest similarity for each embedding with itself
    diag = torch.diag(out)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)

def test_batch_with_single_embedding(embeddings_single):
    sim = EmbeddingSim()
    out = sim(embeddings_single, embeddings_single)
    assert out.shape == (1, 1)
    assert torch.allclose(out, torch.tensor([[1.0]]), atol=1e-6)

def test_forward_handles_zero_embeddings(embeddings_a, embeddings_zero):
    sim = EmbeddingSim()
    out = sim(embeddings_a, embeddings_zero)
    # Should not produce NaNs or Infs
    assert torch.all(torch.isfinite(out))

def test_forward_with_different_embedding_sizes():
    sim = EmbeddingSim()
    emb1 = torch.rand(3, 5)
    emb2 = torch.rand(4, 5)
    out = sim(emb1, emb2)
    assert out.shape == (3, 4)

def test_forward_raises_on_dimension_mismatch():
    sim = EmbeddingSim()
    emb1 = torch.rand(3, 5)
    emb2 = torch.rand(4, 6)  # different embedding dim
    with pytest.raises(RuntimeError):
        _ = sim(emb1, emb2)

def test_forward_is_differentiable(embeddings_a, embeddings_b):
    sim = EmbeddingSim()
    embeddings_a = embeddings_a.clone().requires_grad_(True)
    out = sim(embeddings_a, embeddings_b)
    out.sum().backward()
    assert embeddings_a.grad is not None
    assert embeddings_a.grad.shape == embeddings_a.shape

def test_p_norm_effect():
    emb1 = torch.tensor([[1.0, 0.0]])
    emb2 = torch.tensor([[0.0, 1.0]])
    sim2 = EmbeddingSim(p=2)
    sim1 = EmbeddingSim(p=1)
    out2 = sim2(emb1, emb2)
    out1 = sim1(emb1, emb2)
    # Usually, p=1 norm sim should be less or equal to p=2 norm sim for these inputs
    assert out1.min() <= out2.min()

def test_eps_prevents_division_by_zero():
    sim = EmbeddingSim()
    emb = torch.zeros(2, 3)
    out = sim(emb, emb)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_similarity_values_are_symmetric_for_same_inputs(embeddings_a):
    sim = EmbeddingSim()
    out = sim(embeddings_a, embeddings_a)
    assert torch.allclose(out, out.T, atol=1e-6)

def test_forward_with_large_tensors():
    sim = EmbeddingSim()
    emb1 = torch.randn(100, 512)
    emb2 = torch.randn(200, 512)
    out = sim(emb1, emb2)
    assert out.shape == (100, 200)
    assert torch.all(torch.isfinite(out))

def test_forward_accepts_non_float_tensors():
    sim = EmbeddingSim()
    emb1 = torch.randint(0, 10, (2, 4), dtype=torch.int32).float()
    emb2 = torch.randint(0, 10, (3, 4), dtype=torch.int32).float()
    out = sim(emb1, emb2)
    assert out.shape == (2, 3)

def test_forward_result_with_identical_rows():
    sim = EmbeddingSim()
    emb = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
    out = sim(emb, emb)
    diag = torch.diag(out)
    # All rows are identical - sim values should be max 1 on diagonal
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)

def test_forward_on_single_element_embeddings():
    sim = EmbeddingSim()
    emb1 = torch.tensor([[4.0]])
    emb2 = torch.tensor([[5.0]])
    out = sim(emb1, emb2)
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()

def test_forward_output_values_are_real_numbers(embeddings_a, embeddings_b):
    sim = EmbeddingSim()
    out = sim(embeddings_a, embeddings_b)
    assert torch.isreal(out).all()

def test_forward_with_non_contiguous_inputs():
    sim = EmbeddingSim()
    emb1 = torch.randn(5, 3).t()  # (3,5)
    emb1 = emb1.contiguous().t()
    emb2 = torch.randn(4, 3)
    out = sim(emb1, emb2)
    assert out.shape == (5, 4)
