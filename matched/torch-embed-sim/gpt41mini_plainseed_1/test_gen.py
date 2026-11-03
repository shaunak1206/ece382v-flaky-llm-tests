import pytest
import torch
from torch_embed_sim import EmbeddingSim

@pytest.fixture
def device():
    return "cpu"

@pytest.fixture
def sim():
    return EmbeddingSim()

def test_default_initialization(sim):
    # Test default instantiation has expected defaults
    assert isinstance(sim, EmbeddingSim)

def test_similarity_between_identical_vectors_is_one(sim):
    vec = torch.tensor([[1., 2., 3.]], dtype=torch.float32)
    sim_score = sim(vec, vec)
    # since vectors are identical similarity should be 1 or very close to it (cosine similarity)
    assert torch.allclose(sim_score, torch.ones(1, 1), atol=1e-6)

def test_similarity_between_orthogonal_vectors_is_zero_or_close(sim):
    # Create two orthogonal vectors in 2D
    v1 = torch.tensor([[1., 0.]], dtype=torch.float32)
    v2 = torch.tensor([[0., 1.]], dtype=torch.float32)
    sim_score = sim(v1, v2)
    # Cosine similarity of orthogonal vectors should be 0
    assert torch.allclose(sim_score, torch.zeros(1, 1), atol=1e-6)

def test_similarity_shape_with_batch_vectors(sim):
    v1 = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float32)
    v2 = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float32)
    sim_score = sim(v1, v2)
    # Output shape should be [batch_size_v1, batch_size_v2]
    assert sim_score.shape == (2, 2)

def test_similarity_range(sim):
    v1 = torch.randn(3, 5)
    v2 = torch.randn(4, 5)
    scores = sim(v1, v2)
    # Cosine similarity range is [-1, 1], check if scores in this range
    assert torch.all(scores >= -1.0) and torch.all(scores <= 1.0)

def test_similarity_differs_for_different_vectors(sim):
    # Different vectors should have differing similarity scores, not all equal
    v1 = torch.tensor([[1., 0.]])
    v2 = torch.tensor([[1., 1.], [0., 1.]])
    scores = sim(v1, v2)
    assert (scores[0, 0] != scores[0, 1]) or (scores[0, 0] != 1.0)

def test_empty_input_raises(sim):
    with pytest.raises(RuntimeError):
        sim(torch.tensor([]), torch.tensor([[1., 2.]]))

    with pytest.raises(RuntimeError):
        sim(torch.tensor([[1., 2.]]), torch.tensor([]))

def test_incompatible_dimension_raises(sim):
    v1 = torch.randn(2, 5)
    v2 = torch.randn(3, 6)  # different feature dim
    with pytest.raises(RuntimeError):
        sim(v1, v2)

def test_single_vector_similarity_works(sim):
    v1 = torch.tensor([1., 2., 3.])
    v2 = torch.tensor([1., 0., 0.])
    score = sim(v1, v2)
    # score shape should be (1, 1)
    assert score.shape == (1, 1)
    # Value should match manual cosine similarity approx
    cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    assert torch.allclose(score.squeeze(), cos_sim.squeeze(), atol=1e-6)

def test_supports_float64_and_float32(sim):
    v1_64 = torch.randn(2, 4, dtype=torch.float64)
    v2_64 = torch.randn(3, 4, dtype=torch.float64)
    score64 = sim(v1_64, v2_64)
    assert score64.dtype == torch.float64

    v1_32 = torch.randn(2, 4, dtype=torch.float32)
    v2_32 = torch.randn(3, 4, dtype=torch.float32)
    score32 = sim(v1_32, v2_32)
    assert score32.dtype == torch.float32

def test_similarity_with_cuda_if_available(sim):
    if torch.cuda.is_available():
        v1 = torch.randn(2, 3).cuda()
        v2 = torch.randn(2, 3).cuda()
        score = sim(v1, v2)
        assert score.is_cuda
        assert score.shape == (2, 2)
    else:
        pytest.skip("CUDA not available")

def test_similarity_normalizes_input_vectors(sim):
    # Manually normalize inputs and compare similarity to non-normalized inputs
    v1 = torch.randn(2, 5)
    v2 = torch.randn(2, 5)
    sim_clean = sim(v1 / v1.norm(dim=1, keepdim=True), v2 / v2.norm(dim=1, keepdim=True))
    sim_raw = sim(v1, v2)
    # The similarity scores with normalized inputs should match raw inputs similarity
    # because EmbeddingSim should handle normalization internally
    assert torch.allclose(sim_clean, sim_raw, atol=1e-5)

def test_similarity_of_zero_vectors_handled(sim):
    v1 = torch.zeros(1, 4)
    v2 = torch.zeros(1, 4)
    with pytest.raises(RuntimeError):
        sim(v1, v2)

def test_similarity_consistent_with_multiple_calls(sim):
    v1 = torch.randn(3, 7)
    v2 = torch.randn(4, 7)
    score1 = sim(v1, v2)
    score2 = sim(v1, v2)
    assert torch.allclose(score1, score2)

def test_similarity_produces_gradient(sim):
    v1 = torch.randn(2, 6, requires_grad=True)
    v2 = torch.randn(2, 6, requires_grad=True)
    score = sim(v1, v2)
    loss = score.sum()
    loss.backward()
    assert v1.grad is not None
    assert v2.grad is not None

def test_similarity_with_non_float_input_raises(sim):
    v1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    v2 = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    with pytest.raises(RuntimeError):
        sim(v1, v2)

def test_similarity_batch_vs_single_vector(sim):
    batch = torch.randn(5, 8)
    single = torch.randn(8)
    scores_batch = sim(batch, single)
    scores_single = sim(single, batch)
    assert scores_batch.shape == (5, 1)
    assert scores_single.shape == (1, 5)

def test_similarity_identity_matrix(sim):
    size = 4
    eye = torch.eye(size)
    result = sim(eye, eye)
    # The diagonal of result should be 1, off-diagonal less than 1
    diag = torch.diagonal(result)
    assert torch.allclose(diag, torch.ones(size), atol=1e-6)
    off_diag = result - torch.diag(diag)
    assert torch.all(off_diag <= 1.0)
