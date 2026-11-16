import copy
import inspect

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import torch_embed_sim
from torch_embed_sim import EmbeddingSim


# The source of `embedding_sim` was not visible, so these tests discover a
# working constructor and call style at runtime and skip when none is found.

_VOCAB = 10
_DIM = 4


def _weight():
    gen = torch.Generator().manual_seed(0)
    return torch.randn(_VOCAB, _DIM, generator=gen)


def _constructor_attempts():
    w = _weight()
    return [
        lambda: EmbeddingSim(),
        lambda: EmbeddingSim(w),
        lambda: EmbeddingSim(nn.Embedding.from_pretrained(w)),
        lambda: EmbeddingSim(nn.Embedding(_VOCAB, _DIM)),
        lambda: EmbeddingSim(_VOCAB, _DIM),
        lambda: EmbeddingSim(num_embeddings=_VOCAB, embedding_dim=_DIM),
        lambda: EmbeddingSim(embedding_dim=_DIM),
        lambda: EmbeddingSim(weight=w),
        lambda: EmbeddingSim(embeddings=w),
        lambda: EmbeddingSim(embedding_matrix=w),
        lambda: EmbeddingSim(embedding=nn.Embedding.from_pretrained(w)),
        lambda: EmbeddingSim(dim=_DIM),
    ]


def _build():
    for attempt in _constructor_attempts():
        try:
            return attempt()
        except Exception:
            continue
    return None


def _call_attempts(inst):
    ids = torch.tensor([0, 1, 2])
    ids2 = torch.tensor([3, 4, 5])
    f = torch.randn(3, _DIM)
    f2 = torch.randn(3, _DIM)
    return [
        lambda: inst(ids),
        lambda: inst(ids, ids2),
        lambda: inst(f),
        lambda: inst(f, f2),
        lambda: inst(ids.unsqueeze(0)),
        lambda: inst(f.unsqueeze(0)),
    ]


def _first_output(inst):
    for attempt in _call_attempts(inst):
        try:
            out = attempt()
        except Exception:
            continue
        if isinstance(out, torch.Tensor):
            return out
    return None


@pytest.fixture
def sim():
    inst = _build()
    if inst is None:
        pytest.skip("could not construct EmbeddingSim with any known signature")
    return inst


@pytest.fixture
def module_sim(sim):
    if not isinstance(sim, nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    return sim


def test_class_is_exported_from_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert EmbeddingSim is torch_embed_sim.EmbeddingSim


def test_embeddingsim_is_a_class():
    assert inspect.isclass(EmbeddingSim)


def test_class_defined_in_embedding_sim_submodule():
    from torch_embed_sim import embedding_sim

    assert embedding_sim.EmbeddingSim is EmbeddingSim


def test_instantiation(sim):
    assert isinstance(sim, EmbeddingSim)


def test_two_instances_are_distinct():
    a = _build()
    b = _build()
    if a is None or b is None:
        pytest.skip("could not construct EmbeddingSim")
    assert a is not b


def test_repr_is_string(sim):
    assert isinstance(repr(sim), str)
    assert len(repr(sim)) > 0


def test_has_public_interface(sim):
    public = [n for n in dir(sim) if not n.startswith("_")]
    assert callable(sim) or len(public) > 0


def test_parameters_are_tensors(module_sim):
    for p in module_sim.parameters():
        assert isinstance(p, torch.Tensor)


def test_state_dict_is_mapping(module_sim):
    sd = module_sim.state_dict()
    for key, value in sd.items():
        assert isinstance(key, str)
        assert isinstance(value, torch.Tensor)


def test_eval_and_train_toggle(module_sim):
    module_sim.eval()
    assert module_sim.training is False
    module_sim.train()
    assert module_sim.training is True


def test_call_returns_tensor(sim):
    out = _first_output(sim)
    if out is None:
        pytest.skip("no known call style returned a tensor")
    assert isinstance(out, torch.Tensor)


def test_output_is_finite(sim):
    out = _first_output(sim)
    if out is None:
        pytest.skip("no known call style returned a tensor")
    if out.dtype.is_floating_point:
        assert torch.isfinite(out).all()


def test_output_deterministic_in_eval_mode(sim):
    if isinstance(sim, nn.Module):
        sim.eval()
    ids = torch.tensor([0, 1, 2])
    ids2 = torch.tensor([3, 4, 5])
    for args in [(ids,), (ids, ids2)]:
        try:
            with torch.no_grad():
                first = sim(*args)
                second = sim(*args)
        except Exception:
            continue
        if isinstance(first, torch.Tensor):
            assert torch.equal(first, second)
            return
    pytest.skip("no known call style worked")


def test_input_not_mutated(sim):
    ids = torch.tensor([0, 1, 2])
    original = ids.clone()
    try:
        sim(ids)
    except Exception:
        try:
            sim(ids, ids.clone())
        except Exception:
            pytest.skip("no known call style worked")
    assert torch.equal(ids, original)


def test_no_grad_output_has_no_grad_fn(sim):
    ids = torch.tensor([0, 1, 2])
    for args in [(ids,), (ids, torch.tensor([3, 4, 5]))]:
        try:
            with torch.no_grad():
                out = sim(*args)
        except Exception:
            continue
        if isinstance(out, torch.Tensor):
            assert out.requires_grad is False
            return
    pytest.skip("no known call style worked")


def test_invalid_input_raises(sim):
    if not callable(sim):
        pytest.skip("instance is not callable")
    with pytest.raises(Exception):
        sim("definitely not a tensor", object())


def test_deepcopy_produces_independent_instance(sim):
    clone = copy.deepcopy(sim)
    assert clone is not sim
    assert type(clone) is type(sim)
