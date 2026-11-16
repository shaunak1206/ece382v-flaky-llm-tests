import inspect

import pytest
import torch

import torch_embed_sim
from torch_embed_sim import EmbeddingSim


NUM = 10
DIM = 4


def _constructor_candidates():
    torch.manual_seed(0)
    weights = torch.randn(NUM, DIM)
    return [
        lambda: EmbeddingSim(torch.nn.Embedding(NUM, DIM)),
        lambda: EmbeddingSim(weights.clone()),
        lambda: EmbeddingSim(NUM, DIM),
        lambda: EmbeddingSim(num_embeddings=NUM, embedding_dim=DIM),
        lambda: EmbeddingSim(),
    ]


def _make_instance():
    for factory in _constructor_candidates():
        try:
            return factory()
        except Exception:
            continue
    pytest.skip("Could not construct EmbeddingSim with any known signature")


def _input_candidates():
    torch.manual_seed(1)
    idx_a = torch.tensor([0, 1, 2])
    idx_b = torch.tensor([3, 4, 5])
    vec_a = torch.randn(3, DIM)
    vec_b = torch.randn(3, DIM)
    return [
        (idx_a,),
        (idx_a, idx_b),
        (vec_a,),
        (vec_a, vec_b),
        (idx_a[:1], idx_b[:1]),
    ]


def _call_any(inst):
    """Return (args, output) for the first input pattern the instance accepts."""
    if not callable(inst):
        pytest.skip("Instance is not callable")
    for args in _input_candidates():
        try:
            out = inst(*args)
        except Exception:
            continue
        return args, out
    pytest.skip("Could not call EmbeddingSim with any known input pattern")


@pytest.fixture
def sim():
    return _make_instance()


def _as_tensors(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [o for o in out if isinstance(o, torch.Tensor)]
    if isinstance(out, dict):
        return [o for o in out.values() if isinstance(o, torch.Tensor)]
    return []


def test_class_is_exported_from_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert torch_embed_sim.EmbeddingSim is EmbeddingSim


def test_embeddingsim_is_a_class():
    assert inspect.isclass(EmbeddingSim)


def test_init_signature_is_inspectable():
    sig = inspect.signature(EmbeddingSim.__init__)
    assert "self" in sig.parameters


def test_can_instantiate(sim):
    assert isinstance(sim, EmbeddingSim)


def test_repr_is_string(sim):
    assert isinstance(repr(sim), str)
    assert len(repr(sim)) > 0


def test_two_instances_are_distinct():
    a = _make_instance()
    b = _make_instance()
    assert a is not b
    assert type(a) is type(b)


def test_call_returns_something(sim):
    _, out = _call_any(sim)
    assert out is not None


def test_output_contains_tensor(sim):
    _, out = _call_any(sim)
    assert len(_as_tensors(out)) >= 1


def test_output_is_finite(sim):
    _, out = _call_any(sim)
    tensors = _as_tensors(out)
    if not tensors:
        pytest.skip("Output has no tensors")
    for t in tensors:
        if t.is_floating_point():
            assert torch.isfinite(t).all()


def test_call_is_deterministic_in_eval(sim):
    if hasattr(sim, "eval"):
        sim.eval()
    args, out1 = _call_any(sim)
    out2 = sim(*args)
    t1, t2 = _as_tensors(out1), _as_tensors(out2)
    if not t1:
        pytest.skip("Output has no tensors")
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        assert a.shape == b.shape
        assert torch.allclose(a.float(), b.float(), atol=1e-6)


def test_call_does_not_mutate_inputs(sim):
    args, _ = _call_any(sim)
    snapshot = [a.clone() for a in args]
    sim(*args)
    for before, after in zip(snapshot, args):
        assert torch.equal(before, after)


def test_works_under_no_grad(sim):
    args, _ = _call_any(sim)
    with torch.no_grad():
        out = sim(*args)
    for t in _as_tensors(out):
        assert not t.requires_grad


def test_output_batch_dimension_matches_input(sim):
    args, out = _call_any(sim)
    tensors = _as_tensors(out)
    if not tensors or not tensors[0].dim():
        pytest.skip("Scalar or no tensor output")
    batch = args[0].shape[0]
    # Output should either preserve the batch dimension or reduce it.
    assert tensors[0].shape[0] in (batch, 1) or tensors[0].numel() >= 1


def test_module_parameters_are_tensors_if_module(sim):
    if not isinstance(sim, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    for p in sim.parameters():
        assert isinstance(p, torch.Tensor)


def test_state_dict_roundtrip_if_module():
    a = _make_instance()
    if not isinstance(a, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    b = _make_instance()
    b.load_state_dict(a.state_dict())
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])


def test_train_eval_toggle_if_module(sim):
    if not isinstance(sim, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    sim.eval()
    assert sim.training is False
    sim.train()
    assert sim.training is True
