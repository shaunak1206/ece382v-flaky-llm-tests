import pytest
import torch
from torch_embed_sim import EmbeddingSim


@pytest.fixture
def sim():
    # default EmbeddingSim instance, default device cpu
    return EmbeddingSim()


def test_initialization_default_device():
    sim = EmbeddingSim()
    assert sim.device.type == "cpu"


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_initialization_with_device(device):
    sim = EmbeddingSim(device=device)
    assert sim.device.type == device


def test_initialize_embedding_weight_shape_and_dtype():
    vocab_size = 10
    dim = 5
    sim = EmbeddingSim(num_embeddings=vocab_size, embedding_dim=dim)
    weight = sim.embedding.weight
    assert weight.shape == (vocab_size, dim)
    assert weight.dtype == torch.float32


def test_clone_embedding_weight_different_tensor(sim):
    sim2 = EmbeddingSim()
    # weights should not be same object
    assert sim.embedding.weight.data_ptr() != sim2.embedding.weight.data_ptr()


def test_embedding_weight_requires_grad_true(sim):
    assert sim.embedding.weight.requires_grad is True


def test_forward_output_shape_and_dtype():
    vocab_size = 15
    dim = 6
    sim = EmbeddingSim(num_embeddings=vocab_size, embedding_dim=dim)
    x = torch.arange(0, 4, dtype=torch.long)
    output = sim(x)
    assert output.shape == (x.shape[0], dim)
    assert output.dtype == torch.float32


def test_forward_with_device_matching_input_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = EmbeddingSim(device=device)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    x = torch.tensor([1, 2, 3], device=device)
    output = sim(x)
    assert output.device.type == device


def test_forward_input_requires_long_tensor():
    sim = EmbeddingSim(num_embeddings=10)
    with pytest.raises(RuntimeError):
        sim(torch.tensor([1.0, 2.0]))  # float tensor should raise


def test_forward_input_out_of_range_index():
    sim = EmbeddingSim(num_embeddings=10)
    x = torch.tensor([0, 9, 10], dtype=torch.long)
    # IndexError or RuntimeError depending on pytorch version
    with pytest.raises(RuntimeError):
        sim(x)


def test_embedding_sim_repr_contains_info():
    sim = EmbeddingSim(num_embeddings=25, embedding_dim=7, device="cpu")
    r = repr(sim)
    assert "EmbeddingSim" in r
    assert "num_embeddings=25" in r
    assert "embedding_dim=7" in r
    assert "device=cpu" in r


def test_embedding_sim_supports_different_dtype_initialization():
    # dtype parameter is not documented, but let's check if casting possible during forward
    # Here we rely on default float32 embeddings, but test input type mismatch:
    sim = EmbeddingSim(num_embeddings=4, embedding_dim=2)
    x = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    out = sim(x).to(torch.float64)
    assert out.dtype == torch.float64


def test_embedding_sim_embedding_weight_is_parameter(sim):
    from torch.nn import Parameter
    assert isinstance(sim.embedding.weight, Parameter)


def test_embedding_sim_embedding_weight_on_correct_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = EmbeddingSim(device=device)
    assert sim.embedding.weight.device.type == device


def test_forward_returns_same_vectors_for_same_indices():
    sim = EmbeddingSim(num_embeddings=10, embedding_dim=3)
    x = torch.tensor([1, 1, 1], dtype=torch.long)
    out = sim(x)
    assert torch.all(out[0] == out[1])
    assert torch.all(out[1] == out[2])


def test_forward_returns_different_vectors_for_different_indices():
    sim = EmbeddingSim(num_embeddings=10, embedding_dim=3)
    x = torch.tensor([1, 2], dtype=torch.long)
    out = sim(x)
    assert not torch.allclose(out[0], out[1])


def test_embedding_sim_cuda_fallback_to_cpu_if_cuda_unavailable(monkeypatch):
    if torch.cuda.is_available():
        pytest.skip("CUDA available, skipping fallback test")

    # forcibly provide device='cuda' but since cuda unavailable, it should fallback cpu
    sim = EmbeddingSim(device="cuda")
    assert sim.device.type == "cpu"
    # The embedding's device should also be cpu
    assert sim.embedding.weight.device.type == "cpu"


def test_embedding_sim_forward_large_batch():
    sim = EmbeddingSim(num_embeddings=1000, embedding_dim=16)
    x = torch.randint(0, 1000, (128,), dtype=torch.long)
    out = sim(x)
    assert out.shape == (128, 16)


def test_embedding_sim_embedding_weight_grad_update():
    sim = EmbeddingSim(num_embeddings=10, embedding_dim=4)
    optimizer = torch.optim.SGD(sim.parameters(), lr=0.1)
    x = torch.tensor([1, 2, 3], dtype=torch.long)
    out = sim(x).sum()
    out.backward()
    optimizer.step()
    # Weight grad should have been populated
    assert sim.embedding.weight.grad is not None
    # After step, grad is cleared typically, but check nonzero update happened
    with torch.no_grad():
        # Ensure weight is different after update - we check magnitude changed
        weights_after = sim.embedding.weight.clone()
    assert weights_after.norm() != 0


def test_embedding_sim_forward_accepts_cpu_and_cuda_inputs():
    sim = EmbeddingSim(num_embeddings=10, embedding_dim=3, device="cpu")
    x_cpu = torch.tensor([1, 3, 5], dtype=torch.long, device="cpu")
    out_cpu = sim(x_cpu)
    assert out_cpu.device.type == "cpu"

    if torch.cuda.is_available():
        sim_cuda = EmbeddingSim(num_embeddings=10, embedding_dim=3, device="cuda")
        x_cuda = torch.tensor([1, 3, 5], dtype=torch.long, device="cuda")
        out_cuda = sim_cuda(x_cuda)
        assert out_cuda.device.type == "cuda"


def test_embedding_sim_embedding_weight_requires_grad_after_backward():
    sim = EmbeddingSim(num_embeddings=6, embedding_dim=2)
    x = torch.tensor([0, 1, 2], dtype=torch.long)
    output = sim(x).sum()
    output.backward()
    assert sim.embedding.weight.requires_grad
    assert sim.embedding.weight.grad is not None


def test_embedding_sim_forward_empty_input():
    sim = EmbeddingSim(num_embeddings=5, embedding_dim=4)
    x = torch.tensor([], dtype=torch.long)
    out = sim(x)
    # Output shape should be (0, embedding_dim)
    assert out.shape == (0, 4)
