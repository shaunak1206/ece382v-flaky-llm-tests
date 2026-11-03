import torch
import torch.nn as nn

from torch_embed_sim import EmbeddingSim


def test_init_with_bias_creates_parameter():
    sim = EmbeddingSim(num_embeddings=7)
    assert isinstance(sim.bias, nn.Parameter)
    assert sim.bias.shape == (7,)
    assert sim.num_embeddings == 7


def test_init_without_bias_is_none():
    sim = EmbeddingSim(num_embeddings=7, bias=False)
    assert sim.bias is None
    assert len(list(sim.parameters())) == 0


def test_bias_initialized_to_zeros():
    sim = EmbeddingSim(num_embeddings=5)
    assert torch.all(sim.bias == 0)


def test_reset_parameters_zeroes_bias():
    sim = EmbeddingSim(num_embeddings=5)
    with torch.no_grad():
        sim.bias.fill_(3.0)
    sim.reset_parameters()
    assert torch.all(sim.bias == 0)


def test_reset_parameters_without_bias_does_not_fail():
    sim = EmbeddingSim(num_embeddings=5, bias=False)
    sim.reset_parameters()
    assert sim.bias is None


def test_forward_output_shape_2d():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=6)
    x = torch.randn(4, 3)
    weight = torch.randn(6, 3)
    out = sim(x, weight)
    assert out.shape == (4, 6)


def test_forward_output_shape_3d():
    torch.manual_seed(0)
    sim = EmbeddingSim(num_embeddings=6)
    x = torch.randn(2, 5, 3)
    weight = torch.randn(6, 3)
    out = sim(x, weight)
    assert out.shape == (2, 5, 6)


def test_forward_rows_sum_to_one():
    torch.manual_seed(1)
    sim = EmbeddingSim(num_embeddings=8)
    x = torch.randn(5, 4)
    weight = torch.randn(8, 4)
    out = sim(x, weight)
    assert torch.allclose(out.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_forward_outputs_are_probabilities():
    torch.manual_seed(2)
    sim = EmbeddingSim(num_embeddings=8)
    out = sim(torch.randn(5, 4), torch.randn(8, 4))
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_forward_matches_manual_softmax_without_bias():
    torch.manual_seed(3)
    sim = EmbeddingSim(num_embeddings=5, bias=False)
    x = torch.randn(3, 4)
    weight = torch.randn(5, 4)
    expected = torch.softmax(x.matmul(weight.t()), dim=-1)
    assert torch.allclose(sim(x, weight), expected, atol=1e-6)


def test_forward_matches_manual_softmax_with_bias():
    torch.manual_seed(4)
    sim = EmbeddingSim(num_embeddings=5)
    with torch.no_grad():
        sim.bias.copy_(torch.tensor([0.5, -1.0, 2.0, 0.0, 1.5]))
    x = torch.randn(3, 4)
    weight = torch.randn(5, 4)
    expected = torch.softmax(x.matmul(weight.t()) + sim.bias.detach(), dim=-1)
    assert torch.allclose(sim(x, weight), expected, atol=1e-6)


def test_zero_bias_equals_no_bias():
    torch.manual_seed(5)
    with_bias = EmbeddingSim(num_embeddings=5)
    without_bias = EmbeddingSim(num_embeddings=5, bias=False)
    x = torch.randn(3, 4)
    weight = torch.randn(5, 4)
    assert torch.allclose(with_bias(x, weight), without_bias(x, weight), atol=1e-6)


def test_large_bias_dominates_output():
    sim = EmbeddingSim(num_embeddings=4)
    with torch.no_grad():
        sim.bias.copy_(torch.tensor([0.0, 0.0, 100.0, 0.0]))
    x = torch.zeros(2, 3)
    weight = torch.zeros(4, 3)
    out = sim(x, weight)
    assert torch.all(out.argmax(dim=-1) == 2)
    assert torch.allclose(out[:, 2], torch.ones(2), atol=1e-5)


def test_identical_embeddings_give_uniform_distribution():
    sim = EmbeddingSim(num_embeddings=4, bias=False)
    x = torch.ones(2, 3)
    weight = torch.ones(4, 3)
    out = sim(x, weight)
    assert torch.allclose(out, torch.full((2, 4), 0.25), atol=1e-6)


def test_weight_receives_no_gradient():
    torch.manual_seed(6)
    sim = EmbeddingSim(num_embeddings=5)
    x = torch.randn(3, 4)
    weight = torch.randn(5, 4, requires_grad=True)
    out = sim(x, weight)
    out[:, 0].sum().backward()
    assert weight.grad is None


def test_bias_receives_gradient():
    torch.manual_seed(7)
    sim = EmbeddingSim(num_embeddings=5)
    x = torch.randn(3, 4)
    weight = torch.randn(5, 4)
    out = sim(x, weight)
    (out * torch.arange(5, dtype=torch.float32)).sum().backward()
    assert sim.bias.grad is not None
    assert sim.bias.grad.shape == (5,)


def test_input_receives_gradient():
    torch.manual_seed(8)
    sim = EmbeddingSim(num_embeddings=5)
    x = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(5, 4)
    out = sim(x, weight)
    (out * torch.arange(5, dtype=torch.float32)).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_forward_does_not_modify_inputs():
    torch.manual_seed(9)
    sim = EmbeddingSim(num_embeddings=5)
    x = torch.randn(3, 4)
    weight = torch.randn(5, 4)
    x_copy = x.clone()
    weight_copy = weight.clone()
    sim(x, weight)
    assert torch.equal(x, x_copy)
    assert torch.equal(weight, weight_copy)


def test_extra_repr_with_bias():
    sim = EmbeddingSim(num_embeddings=10)
    assert sim.extra_repr() == 'num_embeddings=10, bias=True'
    assert 'num_embeddings=10, bias=True' in repr(sim)


def test_extra_repr_without_bias():
    sim = EmbeddingSim(num_embeddings=10, bias=False)
    assert sim.extra_repr() == 'num_embeddings=10, bias=False'


def test_works_with_embedding_layer():
    torch.manual_seed(10)
    embed = nn.Embedding(num_embeddings=10, embedding_dim=20)
    sim = EmbeddingSim(num_embeddings=10)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = sim(embed(tokens), embed.weight)
    assert out.shape == (2, 3, 10)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2, 3), atol=1e-5)
