import pytest
import torch
from torch_embed_sim import EmbeddingSim


@pytest.fixture
def model_default():
    return EmbeddingSim()


@pytest.fixture
def model_cosine():
    return EmbeddingSim(similarity='cosine')


@pytest.fixture
def model_dot():
    return EmbeddingSim(similarity='dot')


@pytest.fixture
def input_embeddings():
    torch.manual_seed(0)
    return torch.randn(4, 8)


@pytest.fixture
def input_embeddings_2():
    torch.manual_seed(1)
    return torch.randn(5, 8)


def test_default_similarity_is_cosine(model_default):
    assert model_default.similarity == 'cosine'


def test_can_set_similarity_dot():
    model = EmbeddingSim(similarity='dot')
    assert model.similarity == 'dot'


def test_forward_output_shape_cosine(model_cosine, input_embeddings, input_embeddings_2):
    output = model_cosine(input_embeddings, input_embeddings_2)
    assert output.shape == (4, 5)


def test_forward_output_shape_dot(model_dot, input_embeddings, input_embeddings_2):
    output = model_dot(input_embeddings, input_embeddings_2)
    assert output.shape == (4, 5)


def test_cosine_similarity_values_in_range(model_cosine, input_embeddings):
    out = model_cosine(input_embeddings, input_embeddings)
    # cosine values should be in [-1, 1]
    assert torch.all(out <= 1.0)
    assert torch.all(out >= -1.0)


def test_dot_similarity_values(model_dot, input_embeddings):
    out = model_dot(input_embeddings, input_embeddings)
    # Values can vary, just check output shape and type
    assert out.shape == (4, 4)
    assert isinstance(out, torch.Tensor)


def test_forward_supports_no_y(model_cosine, input_embeddings):
    out1 = model_cosine(input_embeddings)
    out2 = model_cosine(input_embeddings, input_embeddings)
    # By default y is None, so y=x equivalence
    assert torch.allclose(out1, out2, atol=1e-6)


def test_invalid_similarity_value_raises():
    with pytest.raises(ValueError):
        EmbeddingSim(similarity='invalid')


def test_device_forward_cpu(model_default, input_embeddings):
    out = model_default(input_embeddings)
    assert out.device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_on_cuda(model_default, input_embeddings):
    model = EmbeddingSim().to('cuda')
    x = input_embeddings.to('cuda')
    out = model(x)
    assert out.device.type == 'cuda'


def test_forward_empty_tensors(model_default):
    x = torch.empty(0, 8)
    y = torch.empty(0, 8)
    out = model_default(x, y)
    assert out.shape == (0, 0)


def test_forward_mismatched_embedding_dim_raises(model_default):
    x = torch.randn(3, 8)
    y = torch.randn(2, 7)
    with pytest.raises(RuntimeError):
        model_default(x, y)


def test_forward_with_batch_size_1(model_default):
    x = torch.randn(1, 16)
    y = torch.randn(1, 16)
    out = model_default(x, y)
    assert out.shape == (1, 1)


def test_similarity_method_cosine_equivalence(model_cosine, input_embeddings):
    # Manually calculate cosine similarity for first pair and compare
    x = input_embeddings[0]
    y = input_embeddings[1]
    out = model_cosine(input_embeddings, input_embeddings)
    dot = torch.dot(x, y)
    norm_x = x.norm()
    norm_y = y.norm()
    expected = dot / (norm_x * norm_y)
    assert abs(out[0, 1].item() - expected.item()) < 1e-5


def test_similarity_method_dot_equivalence(model_dot, input_embeddings):
    x = input_embeddings[0]
    y = input_embeddings[1]
    out = model_dot(input_embeddings, input_embeddings)
    expected = torch.dot(x, y)
    assert abs(out[0, 1].item() - expected.item()) < 1e-5


def test_repr_contains_similarity(model_dot):
    r = repr(model_dot)
    assert 'dot' in r


def test_repr_contains_similarity_cosine(model_cosine):
    r = repr(model_cosine)
    assert 'cosine' in r


def test_forward_with_single_input_returns_square_matrix(model_default, input_embeddings):
    out = model_default(input_embeddings)
    n = input_embeddings.size(0)
    assert out.size(0) == n
    assert out.size(1) == n


def test_forward_large_batch(model_default):
    x = torch.randn(100, 32)
    y = torch.randn(100, 32)
    out = model_default(x, y)
    assert out.shape == (100, 100)
    # values in [-1, 1] for cosine (default)
    assert torch.all(out <= 1.0) and torch.all(out >= -1.0)
