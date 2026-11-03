import pytest
import torch
import torch.nn.functional as F

from torch_embed_sim import EmbeddingSim


def _inputs(batch=4, seq=3, dim=5, n=7, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq, dim, generator=gen)
    w = torch.randn(n, dim, generator=gen)
    return x, w


def test_default_bias_is_parameter_with_num_embeddings_shape():
    layer = EmbeddingSim(7)
    assert isinstance(layer.bias, torch.nn.Parameter)
    assert layer.bias.shape == (7,)
    assert layer.bias.requires_grad


def test_bias_initialized_to_zeros():
    layer = EmbeddingSim(11)
    assert (layer.bias == 0).all().item()


def test_bias_false_sets_bias_to_none_and_registers_no_parameters():
    layer = EmbeddingSim(5, bias=False)
    assert layer.bias is None
    assert 'bias' not in dict(layer.named_parameters())
    assert list(layer.parameters()) == []


def test_num_embeddings_is_stored():
    assert EmbeddingSim(9).num_embeddings == 9


def test_reset_parameters_zeros_bias():
    layer = EmbeddingSim(4)
    with torch.no_grad():
        layer.bias.fill_(3.5)
    layer.reset_parameters()
    assert (layer.bias == 0).all().item()


def test_reset_parameters_without_bias_is_noop():
    layer = EmbeddingSim(4, bias=False)
    layer.reset_parameters()
    assert layer.bias is None


def test_forward_output_shape_for_batched_input():
    x, w = _inputs(batch=4, seq=3, dim=5, n=7)
    out = EmbeddingSim(7)(x, w)
    assert out.shape == (4, 3, 7)


def test_forward_output_shape_for_vector_input():
    x = torch.randn(5)
    w = torch.randn(7, 5)
    out = EmbeddingSim(7)(x, w)
    assert out.shape == (7,)


def test_forward_rows_sum_to_one():
    x, w = _inputs()
    out = EmbeddingSim(7)(x, w)
    assert torch.allclose(out.sum(dim=-1), torch.ones(4, 3))


def test_forward_outputs_are_nonnegative_and_finite():
    x, w = _inputs()
    out = EmbeddingSim(7)(x, w)
    assert (out >= 0).all().item()
    assert torch.isfinite(out).all().item()


def test_forward_matches_manual_softmax_without_bias():
    x, w = _inputs()
    layer = EmbeddingSim(7, bias=False)
    expected = F.softmax(x.matmul(w.t()), dim=-1)
    assert torch.allclose(layer(x, w), expected, atol=1e-6)


def test_forward_matches_manual_softmax_with_bias():
    x, w = _inputs()
    layer = EmbeddingSim(7)
    with torch.no_grad():
        layer.bias.copy_(torch.linspace(-1.0, 1.0, 7))
    expected = F.softmax(x.matmul(w.t()) + layer.bias.detach(), dim=-1)
    assert torch.allclose(layer(x, w), expected, atol=1e-6)


def test_large_bias_entry_concentrates_distribution_on_that_index():
    x, w = _inputs()
    layer = EmbeddingSim(7)
    base = layer(x, w)
    with torch.no_grad():
        layer.bias[2] = 50.0
    biased = layer(x, w)
    assert (biased.argmax(dim=-1) == 2).all().item()
    assert not torch.allclose(base, biased)


def test_argmax_matches_largest_dot_product_without_bias():
    x, w = _inputs()
    out = EmbeddingSim(7, bias=False)(x, w)
    scores = x.matmul(w.t())
    assert torch.equal(out.argmax(dim=-1), scores.argmax(dim=-1))


def test_forward_does_not_modify_bias_weight_or_inputs():
    x, w = _inputs()
    layer = EmbeddingSim(7)
    with torch.no_grad():
        layer.bias.copy_(torch.arange(7.0) * 0.1)
    bias_before = layer.bias.detach().clone()
    x_before = x.clone()
    w_before = w.clone()
    layer(x, w)
    assert torch.equal(layer.bias.detach(), bias_before)
    assert torch.equal(x, x_before)
    assert torch.equal(w, w_before)


def test_weight_receives_no_gradient_because_it_is_detached():
    x, _ = _inputs()
    w = torch.randn(7, 5, requires_grad=True)
    layer = EmbeddingSim(7)
    out = layer(x, w)
    (out * torch.arange(7.0)).sum().backward()
    assert w.grad is None


def test_bias_and_input_receive_gradients():
    x = torch.randn(4, 3, 5, requires_grad=True)
    w = torch.randn(7, 5)
    layer = EmbeddingSim(7)
    (layer(x, w) * torch.arange(7.0)).sum().backward()
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (7,)
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_large_logits_remain_numerically_stable():
    x = torch.full((2, 5), 1000.0)
    w = torch.ones(3, 5)
    out = EmbeddingSim(3, bias=False)(x, w)
    assert torch.isfinite(out).all().item()
    assert torch.allclose(out, torch.full((2, 3), 1.0 / 3.0), atol=1e-6)


def test_float64_inputs_preserve_dtype():
    x, w = _inputs()
    x = x.double()
    w = w.double()
    layer = EmbeddingSim(7).double()
    out = layer(x, w)
    assert out.dtype == torch.float64


def test_extra_repr_and_repr_report_bias_flag():
    assert EmbeddingSim(3).extra_repr() == 'num_embeddings=3, bias=True'
    layer = EmbeddingSim(3, bias=False)
    assert layer.extra_repr() == 'num_embeddings=3, bias=False'
    assert 'num_embeddings=3, bias=False' in repr(layer)


def test_state_dict_keys_follow_bias_flag():
    assert list(EmbeddingSim(3).state_dict().keys()) == ['bias']
    assert list(EmbeddingSim(3, bias=False).state_dict().keys()) == []


def test_package_exports_embedding_sim():
    import torch_embed_sim
    assert torch_embed_sim.EmbeddingSim is EmbeddingSim


def test_single_embedding_gives_probability_one():
    x = torch.randn(2, 5)
    w = torch.randn(1, 5)
    out = EmbeddingSim(1)(x, w)
    assert torch.allclose(out, torch.ones(2, 1))


def test_bias_size_mismatch_raises_runtime_error():
    layer = EmbeddingSim(3)
    x = torch.randn(2, 5)
    w = torch.randn(7, 5)
    with pytest.raises(RuntimeError):
        layer(x, w)
