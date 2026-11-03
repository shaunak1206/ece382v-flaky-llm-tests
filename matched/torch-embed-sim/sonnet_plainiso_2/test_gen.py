import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_embed_sim import EmbeddingSim


def test_init_with_bias_creates_parameter():
    layer = EmbeddingSim(5)
    assert isinstance(layer.bias, nn.Parameter)
    assert layer.bias.shape == (5,)
    assert layer.num_embeddings == 5


def test_init_without_bias_is_none():
    layer = EmbeddingSim(5, bias=False)
    assert layer.bias is None
    assert len(list(layer.parameters())) == 0


def test_bias_initialized_to_zeros():
    layer = EmbeddingSim(7)
    assert torch.equal(layer.bias.data, torch.zeros(7))


def test_reset_parameters_zeros_modified_bias():
    layer = EmbeddingSim(4)
    layer.bias.data.fill_(3.0)
    layer.reset_parameters()
    assert torch.equal(layer.bias.data, torch.zeros(4))


def test_reset_parameters_without_bias_is_noop():
    layer = EmbeddingSim(4, bias=False)
    layer.reset_parameters()
    assert layer.bias is None


def test_forward_output_shape_2d():
    layer = EmbeddingSim(6)
    x = torch.ones(3, 4)
    weight = torch.ones(6, 4)
    out = layer(x, weight)
    assert out.shape == (3, 6)


def test_forward_output_shape_3d():
    layer = EmbeddingSim(6)
    x = torch.ones(2, 3, 4)
    weight = torch.ones(6, 4)
    out = layer(x, weight)
    assert out.shape == (2, 3, 6)


def test_forward_rows_sum_to_one():
    layer = EmbeddingSim(5)
    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    weight = torch.tensor([
        [0.1, 0.2, 0.3],
        [1.0, 0.0, -1.0],
        [0.5, 0.5, 0.5],
        [-0.3, 0.8, 0.2],
        [2.0, -1.0, 0.0],
    ])
    out = layer(x, weight)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-6)
    assert (out >= 0).all()


def test_forward_matches_manual_softmax_without_bias():
    layer = EmbeddingSim(3, bias=False)
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = F.softmax(x.matmul(weight.t()), dim=-1)
    out = layer(x, weight)
    assert torch.allclose(out, expected, atol=1e-6)


def test_forward_with_zero_bias_equals_no_bias():
    x = torch.tensor([[0.3, -0.7, 1.2]])
    weight = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    with_bias = EmbeddingSim(2, bias=True)
    without_bias = EmbeddingSim(2, bias=False)
    assert torch.allclose(with_bias(x, weight), without_bias(x, weight), atol=1e-6)


def test_forward_applies_bias_before_softmax():
    layer = EmbeddingSim(3)
    layer.bias.data = torch.tensor([0.0, 1.0, -2.0])
    x = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    logits = x.matmul(weight.t()) + torch.tensor([0.0, 1.0, -2.0])
    expected = F.softmax(logits, dim=-1)
    out = layer(x, weight)
    assert torch.allclose(out, expected, atol=1e-6)


def test_forward_large_bias_dominates_distribution():
    layer = EmbeddingSim(3)
    layer.bias.data = torch.tensor([0.0, 100.0, 0.0])
    x = torch.zeros(1, 2)
    weight = torch.zeros(3, 2)
    out = layer(x, weight)
    assert out[0, 1] > 0.999
    assert out.argmax(dim=-1).item() == 1


def test_forward_uniform_when_all_scores_equal():
    layer = EmbeddingSim(4)
    x = torch.zeros(2, 3)
    weight = torch.randn(4, 3)
    out = layer(x, weight)
    assert torch.allclose(out, torch.full((2, 4), 0.25), atol=1e-6)


def test_forward_does_not_propagate_gradient_to_weight():
    layer = EmbeddingSim(3)
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    weight = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    out = layer(x, weight)
    out[0, 0].backward()
    assert weight.grad is None
    assert x.grad is not None


def test_forward_gradient_reaches_bias_and_input():
    layer = EmbeddingSim(3)
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = layer(x, weight)
    out[0, 0].backward()
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (3,)
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_forward_does_not_modify_inputs():
    layer = EmbeddingSim(3)
    layer.bias.data = torch.tensor([1.0, 2.0, 3.0])
    x = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    x_copy = x.clone()
    w_copy = weight.clone()
    layer(x, weight)
    assert torch.equal(x, x_copy)
    assert torch.equal(weight, w_copy)


def test_forward_with_embedding_weight_argmax_matches_similar_vector():
    embedding = nn.Embedding(4, 3)
    embedding.weight.data = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0],
    ])
    layer = EmbeddingSim(4)
    idx = torch.tensor([2, 0, 1])
    x = embedding(idx) * 10
    out = layer(x, embedding.weight)
    assert out.argmax(dim=-1).tolist() == [2, 0, 1]


def test_extra_repr_with_bias():
    layer = EmbeddingSim(10)
    assert layer.extra_repr() == 'num_embeddings=10, bias=True'
    assert 'num_embeddings=10, bias=True' in repr(layer)


def test_extra_repr_without_bias():
    layer = EmbeddingSim(10, bias=False)
    assert layer.extra_repr() == 'num_embeddings=10, bias=False'
    assert 'bias=False' in repr(layer)


def test_state_dict_contains_bias_only_when_enabled():
    assert 'bias' in EmbeddingSim(3).state_dict()
    assert 'bias' not in EmbeddingSim(3, bias=False).state_dict()
