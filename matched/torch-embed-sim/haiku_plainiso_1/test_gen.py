import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_embed_sim import EmbeddingSim
from torch_embed_sim import embedding_sim as embedding_sim_module


def _softmax_of_matmul(x, w, bias=None):
    logits = torch.matmul(x, w.t())
    if bias is not None:
        logits = logits + bias
    return F.softmax(logits, dim=-1)


def test_bias_parameter_has_expected_shape_type_and_zero_values():
    sim = EmbeddingSim(5)
    assert isinstance(sim.bias, nn.Parameter)
    assert tuple(sim.bias.shape) == (5,)
    assert torch.all(sim.bias == 0).item()


def test_num_embeddings_is_stored():
    sim = EmbeddingSim(7)
    assert sim.num_embeddings == 7


def test_bias_false_sets_bias_none_and_reset_is_noop():
    sim = EmbeddingSim(3, bias=False)
    assert sim.bias is None
    assert len(list(sim.parameters())) == 0
    sim.reset_parameters()
    assert sim.bias is None


def test_bias_true_bias_is_only_parameter():
    sim = EmbeddingSim(4)
    params = list(sim.parameters())
    assert len(params) == 1
    assert params[0] is sim.bias


def test_state_dict_keys_follow_bias_flag():
    assert list(EmbeddingSim(6).state_dict().keys()) == ['bias']
    assert list(EmbeddingSim(6, bias=False).state_dict().keys()) == []


def test_reset_parameters_zeroes_nonzero_bias():
    sim = EmbeddingSim(5)
    with torch.no_grad():
        sim.bias.fill_(2.5)
    sim.reset_parameters()
    assert torch.all(sim.bias == 0).item()


def test_forward_output_shape():
    torch.manual_seed(0)
    sim = EmbeddingSim(6)
    x = torch.randn(2, 4)
    w = torch.randn(6, 4)
    out = sim(x, w)
    assert tuple(out.shape) == (2, 6)


def test_forward_rows_are_probability_distributions():
    torch.manual_seed(1)
    sim = EmbeddingSim(6)
    with torch.no_grad():
        sim.bias.copy_(torch.randn(6))
    out = sim(torch.randn(3, 4), torch.randn(6, 4))
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert bool((out >= 0).all())


def test_forward_matches_softmax_of_matmul_without_bias():
    torch.manual_seed(2)
    sim = EmbeddingSim(6, bias=False)
    x = torch.randn(2, 4)
    w = torch.randn(6, 4)
    out = sim(x, w)
    expected = _softmax_of_matmul(x, w)
    assert torch.allclose(out, expected, atol=1e-6)


def test_forward_adds_bias_before_softmax():
    torch.manual_seed(3)
    sim = EmbeddingSim(6)
    bias_values = torch.randn(6)
    with torch.no_grad():
        sim.bias.copy_(bias_values)
    x = torch.randn(2, 4)
    w = torch.randn(6, 4)
    out = sim(x, w)
    expected = _softmax_of_matmul(x, w, bias_values)
    assert torch.allclose(out, expected, atol=1e-6)
    assert not torch.allclose(out, _softmax_of_matmul(x, w), atol=1e-6)


def test_large_bias_dominates_argmax():
    torch.manual_seed(4)
    sim = EmbeddingSim(4)
    with torch.no_grad():
        sim.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 100.0]))
    out = sim(torch.randn(5, 4), torch.randn(4, 4))
    assert torch.all(out.argmax(dim=-1) == 3).item()
    assert torch.all(out[:, 3] > 0.99).item()


def test_forward_stable_with_large_logits():
    sim = EmbeddingSim(2, bias=False)
    x = torch.tensor([[1000.0, -1000.0]])
    w = torch.eye(2)
    out = sim(x, w)
    assert torch.isfinite(out).all().item()
    assert torch.allclose(out, torch.tensor([[1.0, 0.0]]), atol=1e-6)


def test_forward_batched_3d_input():
    torch.manual_seed(5)
    sim = EmbeddingSim(6)
    x = torch.randn(2, 3, 4)
    w = torch.randn(6, 4)
    out = sim(x, w)
    assert tuple(out.shape) == (2, 3, 6)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2, 3), atol=1e-6)
    expected = _softmax_of_matmul(x, w, sim.bias)
    assert torch.allclose(out, expected, atol=1e-6)


def test_weight_is_detached_from_graph():
    torch.manual_seed(6)
    sim = EmbeddingSim(6)
    x = torch.randn(2, 4, requires_grad=True)
    w = torch.randn(6, 4, requires_grad=True)
    out = sim(x, w)
    out[:, 0].sum().backward()
    assert w.grad is None
    assert x.grad is not None


def test_input_gradient_flows():
    torch.manual_seed(7)
    sim = EmbeddingSim(6)
    x = torch.randn(2, 4, requires_grad=True)
    w = torch.randn(6, 4)
    out = sim(x, w)
    out[:, 0].sum().backward()
    assert tuple(x.grad.shape) == (2, 4)
    assert torch.any(x.grad != 0).item()


def test_bias_receives_gradient():
    torch.manual_seed(8)
    sim = EmbeddingSim(6)
    with torch.no_grad():
        sim.bias.copy_(torch.randn(6))
    x = torch.randn(2, 4)
    w = torch.randn(6, 4)
    out = sim(x, w)
    out[:, 0].sum().backward()
    assert sim.bias.grad is not None
    assert tuple(sim.bias.grad.shape) == (6,)
    assert torch.any(sim.bias.grad != 0).item()


def test_inputs_are_not_modified_in_place():
    torch.manual_seed(9)
    sim = EmbeddingSim(6)
    with torch.no_grad():
        sim.bias.copy_(torch.randn(6))
    x = torch.randn(2, 4)
    w = torch.randn(6, 4)
    x_before = x.clone()
    w_before = w.clone()
    sim(x, w)
    assert torch.equal(x, x_before)
    assert torch.equal(w, w_before)


def test_gradcheck_on_input_in_double_precision():
    torch.manual_seed(10)
    sim = EmbeddingSim(3).double()
    with torch.no_grad():
        sim.bias.copy_(torch.randn(3, dtype=torch.float64))
    w = torch.randn(3, 4, dtype=torch.float64)
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: sim(inp, w), (x,))


def test_extra_repr_reflects_bias_flag():
    assert EmbeddingSim(5).extra_repr() == 'num_embeddings=5, bias=True'
    assert EmbeddingSim(5, bias=False).extra_repr() == 'num_embeddings=5, bias=False'


def test_repr_uses_class_name_and_extra_repr():
    assert repr(EmbeddingSim(5, bias=False)) == 'EmbeddingSim(num_embeddings=5, bias=False)'


def test_package_exports_embedding_sim_as_nn_module():
    import torch_embed_sim

    assert torch_embed_sim.EmbeddingSim is EmbeddingSim
    assert issubclass(EmbeddingSim, nn.Module)
    assert 'EmbeddingSim' in embedding_sim_module.__all__
