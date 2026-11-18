import pytest
import torch
from torch_embed_sim import EmbeddingSim


def test_embedding_sim_init_default_device():
    sim = EmbeddingSim()
    assert isinstance(sim, EmbeddingSim)
    # Default device should be CPU or CUDA if available
    assert sim.device.type in ['cpu', 'cuda']


def test_embedding_sim_init_with_device_str():
    sim = EmbeddingSim(device='cpu')
    assert sim.device.type == 'cpu'
    if torch.cuda.is_available():
        sim_cuda = EmbeddingSim(device='cuda')
        assert sim_cuda.device.type == 'cuda'


def test_embedding_sim_call_with_normalized_embeddings():
    sim = EmbeddingSim()
    emb1 = torch.randn(10, 16)
    emb2 = torch.randn(10, 16)
    # Ensure embeddings not normalized input
    sim_norm_false = EmbeddingSim(normalize_embeddings=False)
    sim_norm_true = EmbeddingSim(normalize_embeddings=True)

    # Should work without error
    out1 = sim_norm_false(emb1, emb2)
    out2 = sim_norm_true(emb1, emb2)
    assert isinstance(out1, torch.Tensor)
    assert out1.shape == (10,)
    assert isinstance(out2, torch.Tensor)
    assert out2.shape == (10,)


def test_embedding_sim_call_with_1d_embeddings():
    sim = EmbeddingSim(normalize_embeddings=False)
    emb1 = torch.randn(16)
    emb2 = torch.randn(16)
    result = sim(emb1, emb2)
    assert isinstance(result, torch.Tensor)
    # For single embeddings, output is scalar tensor
    assert result.shape == ()


def test_embedding_sim_call_with_different_shapes_raises():
    sim = EmbeddingSim()
    emb1 = torch.randn(10, 16)
    emb2 = torch.randn(8, 16)
    with pytest.raises(RuntimeError):
        sim(emb1, emb2)


def test_embedding_sim_compatibility_with_different_dtypes():
    sim = EmbeddingSim()
    emb1 = torch.randn(10, 16).float()
    emb2 = torch.randn(10, 16).double()
    result = sim(emb1.to(sim.device), emb2.to(sim.device))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (10,)


def test_embedding_sim_on_different_devices():
    if torch.cuda.is_available():
        sim = EmbeddingSim(device='cuda')
        emb1 = torch.randn(10, 16).to('cuda')
        emb2 = torch.randn(10, 16).to('cuda')
        result = sim(emb1, emb2)
        assert result.device.type == 'cuda'
    else:
        pytest.skip("CUDA not available")


def test_embedding_sim_normalize_embeddings_effect():
    emb1 = torch.randn(5, 20)
    emb2 = torch.randn(5, 20)
    sim = EmbeddingSim(normalize_embeddings=True)
    out_norm = sim(emb1, emb2)
    sim = EmbeddingSim(normalize_embeddings=False)
    out_non_norm = sim(emb1, emb2)
    # Cosine similarity values should be between -1 and 1 when normalized
    assert torch.all((out_norm >= -1) & (out_norm <= 1))
    # The non-normalized should be likely outside this range (dot products)
    assert (out_non_norm.max() > 1 or out_non_norm.min() < -1) or torch.allclose(out_norm, out_non_norm)


def test_embedding_sim_with_batch_size_1():
    sim = EmbeddingSim(normalize_embeddings=True)
    emb1 = torch.randn(1, 32)
    emb2 = torch.randn(1, 32)
    out = sim(emb1, emb2)
    assert out.shape == (1,)
    assert -1 <= out.item() <= 1


def test_embedding_sim_with_large_embedding_dim():
    sim = EmbeddingSim()
    emb1 = torch.randn(100, 4096)
    emb2 = torch.randn(100, 4096)
    out = sim(emb1, emb2)
    assert out.shape == (100,)
    assert isinstance(out, torch.Tensor)


def test_embedding_sim_with_cpu_and_empty_tensor():
    sim = EmbeddingSim()
    emb1 = torch.empty((0, 16))
    emb2 = torch.empty((0, 16))
    out = sim(emb1, emb2)
    assert out.shape == (0,)
    assert out.numel() == 0


def test_embedding_sim_call_requires_tensor_arguments():
    sim = EmbeddingSim()
    emb1 = [[1, 2, 3]]
    emb2 = [[4, 5, 6]]
    with pytest.raises(AttributeError):
        sim(emb1, emb2)


def test_embedding_sim_call_different_last_dim_raises():
    sim = EmbeddingSim()
    emb1 = torch.randn(10, 16)
    emb2 = torch.randn(10, 20)
    with pytest.raises(RuntimeError):
        sim(emb1, emb2)


def test_embedding_sim_call_with_non_2d_but_batch_inputs():
    sim = EmbeddingSim()
    emb1 = torch.randn(4, 5, 16)
    emb2 = torch.randn(4, 5, 16)
    with pytest.raises(RuntimeError):
        sim(emb1, emb2)


def test_embedding_sim_device_moves_tensors_if_needed():
    sim = EmbeddingSim(device='cpu')
    emb1 = torch.randn(10, 16).to('cpu')
    emb2 = torch.randn(10, 16).to('cpu')
    result_cpu = sim(emb1, emb2)
    if torch.cuda.is_available():
        sim_cuda = EmbeddingSim(device='cuda')
        emb1_cuda = emb1.to('cuda')
        emb2_cuda = emb2.to('cuda')
        result_cuda = sim_cuda(emb1_cuda, emb2_cuda)
        assert result_cuda.device.type == 'cuda'
        # Results should be roughly similar between cpu and cuda
        assert torch.allclose(result_cpu.to('cpu'), result_cuda.to('cpu'), atol=1e-5)


def test_embedding_sim_different_batch_sizes_raise():
    sim = EmbeddingSim()
    emb1 = torch.randn(10, 16)
    emb2 = torch.randn(20, 16)
    with pytest.raises(RuntimeError):
        sim(emb1, emb2)


def test_embedding_sim_call_single_vector_like_batch():
    sim = EmbeddingSim()
    emb1 = torch.randn(16)
    emb2 = torch.randn(16)
    res = sim(emb1, emb2)
    assert torch.is_tensor(res)
    assert res.dim() == 0  # scalar


def test_embedding_sim_call_with_float16_and_float32():
    sim = EmbeddingSim()
    emb1 = torch.randn(4, 16).to(torch.float16)
    emb2 = torch.randn(4, 16).to(torch.float32)
    out = sim(emb1, emb2)
    assert out.shape == (4,)


def test_embedding_sim_repr_contains_params():
    sim = EmbeddingSim(normalize_embeddings=True, device='cpu')
    s = repr(sim)
    assert 'normalize_embeddings=True' in s
    assert 'device' in s
    assert 'cpu' in s
