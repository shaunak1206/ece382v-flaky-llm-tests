"""
Tests for EmbeddingSim from torch_embed_sim.

NOTE: only `torch_embed_sim/__init__.py` (which re-exports everything from
`torch_embed_sim/embedding_sim.py`) was provided, so the exact constructor and
method signatures of EmbeddingSim were not visible. These tests therefore
discover the API at runtime. They assert behaviour that should hold for any
reasonable embedding-similarity class (importability, construction,
determinism, finite outputs, no mutation of inputs or parameters). Tests that
need a working instance or a recognizable query method skip if one cannot be
built.
"""
import inspect

import pytest
import torch

import torch_embed_sim
from torch_embed_sim import EmbeddingSim

QUERY_METHOD_NAMES = (
    "forward",
    "similarity",
    "similarities",
    "sim",
    "most_similar",
    "top_k",
    "topk",
    "query",
    "search",
    "compute",
)


def _embedding_weights(n=8, d=4):
    gen = torch.Generator().manual_seed(0)
    return torch.randn(n, d, generator=gen)


def _try_build():
    """Try several plausible constructor styles; return an instance or None."""
    weights = _embedding_weights()
    candidates = [
        lambda: EmbeddingSim(torch.nn.Embedding.from_pretrained(weights.clone())),
        lambda: EmbeddingSim(weights.clone()),
        lambda: EmbeddingSim(torch.nn.Embedding(8, 4)),
        lambda: EmbeddingSim(8, 4),
        lambda: EmbeddingSim(),
    ]
    for make in candidates:
        try:
            return make()
        except Exception:
            continue
    return None


@pytest.fixture
def sim():
    obj = _try_build()
    if obj is None:
        pytest.skip("Could not construct EmbeddingSim with any common signature")
    return obj


def _call_query(obj):
    """Try to call a query-like method with common inputs. Returns (result, name, args)
    or skips."""
    ids = torch.tensor([0, 1, 2])
    vecs = _embedding_weights()[:3].clone()
    arg_options = [
        (ids,),
        (vecs,),
        (ids, ids),
        (vecs, vecs),
        (torch.tensor(0),),
        (0,),
    ]
    for name in QUERY_METHOD_NAMES:
        method = getattr(obj, name, None)
        if not callable(method):
            continue
        for args in arg_options:
            try:
                with torch.no_grad():
                    return method(*args), name, args
            except Exception:
                continue
    pytest.skip("No recognizable query method accepted common inputs")


def _tensors_in(result):
    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, (tuple, list)):
        out = []
        for r in result:
            out.extend(_tensors_in(r))
        return out
    if isinstance(result, dict):
        out = []
        for r in result.values():
            out.extend(_tensors_in(r))
        return out
    return []


def test_class_is_importable_from_package():
    assert EmbeddingSim is not None


def test_embeddingsim_is_a_class():
    assert inspect.isclass(EmbeddingSim)


def test_package_attribute_matches_import():
    assert torch_embed_sim.EmbeddingSim is EmbeddingSim


def test_class_defined_in_embedding_sim_module():
    assert EmbeddingSim.__module__.startswith("torch_embed_sim")


def test_constructor_signature_is_inspectable():
    sig = inspect.signature(EmbeddingSim)
    assert isinstance(sig, inspect.Signature)


def test_instance_can_be_created(sim):
    assert isinstance(sim, EmbeddingSim)


def test_module_subclass_supports_parameters_and_eval(sim):
    if not isinstance(sim, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    sim.eval()
    assert not sim.training
    sim.train()
    assert sim.training
    assert isinstance(list(sim.parameters()), list)


def test_module_state_dict_is_dict(sim):
    if not isinstance(sim, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    assert isinstance(sim.state_dict(), dict)


def test_has_at_least_one_public_callable(sim):
    public = [
        n for n in dir(sim)
        if not n.startswith("_") and callable(getattr(sim, n))
    ]
    assert len(public) > 0


def test_query_returns_something(sim):
    result, _, _ = _call_query(sim)
    assert result is not None


def test_query_tensor_outputs_are_finite(sim):
    result, _, _ = _call_query(sim)
    tensors = _tensors_in(result)
    if not tensors:
        pytest.skip("Query result contains no tensors")
    for t in tensors:
        if t.is_floating_point():
            assert torch.isfinite(t).all()


def test_query_is_deterministic(sim):
    if isinstance(sim, torch.nn.Module):
        sim.eval()
    first, name, args = _call_query(sim)
    with torch.no_grad():
        second = getattr(sim, name)(*args)
    t1, t2 = _tensors_in(first), _tensors_in(second)
    if not t1:
        pytest.skip("Query result contains no tensors")
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        assert a.shape == b.shape
        assert torch.allclose(a.float(), b.float(), equal_nan=True)


def test_query_does_not_modify_input_tensor(sim):
    _, name, args = _call_query(sim)
    tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
    if not tensor_args:
        pytest.skip("Query args contain no tensors")
    snapshots = [a.clone() for a in tensor_args]
    with torch.no_grad():
        getattr(sim, name)(*args)
    for orig, snap in zip(tensor_args, snapshots):
        assert torch.equal(orig, snap)


def test_query_does_not_change_parameters(sim):
    if not isinstance(sim, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    before = {k: v.clone() for k, v in sim.state_dict().items()}
    _call_query(sim)
    after = sim.state_dict()
    assert before.keys() == after.keys()
    for k in before:
        assert torch.equal(before[k], after[k])


def test_two_instances_are_independent():
    a, b = _try_build(), _try_build()
    if a is None or b is None:
        pytest.skip("Could not construct EmbeddingSim")
    assert a is not b
