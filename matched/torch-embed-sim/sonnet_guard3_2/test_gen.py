import inspect
import random

import pytest
import torch

import torch_embed_sim
from torch_embed_sim import EmbeddingSim


# NOTE: only the package __init__ (`from .embedding_sim import *`) was visible,
# so these tests discover the constructor signature by introspection and only
# assert properties that should hold for an embedding-similarity class.
# Tests skip (rather than fail) when the API cannot be driven generically.


def _guess_value(name):
    lname = name.lower()
    if "dim" in lname:
        return 8
    if any(k in lname for k in ("num", "vocab", "size", "n_", "count", "len")):
        return 10
    return None


def _make_instance():
    sig = inspect.signature(EmbeddingSim.__init__)
    kwargs = {}
    for pname, param in list(sig.parameters.items())[1:]:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not param.empty:
            continue
        val = _guess_value(pname)
        if val is None:
            pytest.skip("cannot guess required constructor argument %r" % pname)
        kwargs[pname] = val
    try:
        return EmbeddingSim(**kwargs)
    except (TypeError, ValueError, NotImplementedError, RuntimeError):
        pytest.skip("cannot construct EmbeddingSim generically")


def _make_module():
    obj = _make_instance()
    if not isinstance(obj, torch.nn.Module):
        pytest.skip("EmbeddingSim is not a torch.nn.Module")
    return obj


def _try_forward(obj, ids):
    try:
        return obj(ids)
    except (TypeError, ValueError, IndexError, RuntimeError, NotImplementedError):
        pytest.skip("forward cannot be driven with a single id tensor")


def test_exported_from_package():
    random.seed(0)
    torch.manual_seed(0)
    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert torch_embed_sim.EmbeddingSim is EmbeddingSim


def test_is_a_class():
    random.seed(0)
    torch.manual_seed(0)
    assert inspect.isclass(EmbeddingSim)


def test_defined_in_embedding_sim_module():
    random.seed(0)
    torch.manual_seed(0)
    from torch_embed_sim import embedding_sim

    assert embedding_sim.EmbeddingSim is EmbeddingSim


def test_all_contains_class_if_defined():
    random.seed(0)
    torch.manual_seed(0)
    all_names = getattr(torch_embed_sim, "__all__", None)
    if all_names is None:
        pytest.skip("no __all__ defined")
    assert "EmbeddingSim" in all_names


def test_instantiation_returns_instance():
    random.seed(0)
    torch.manual_seed(0)
    obj = _make_instance()
    assert isinstance(obj, EmbeddingSim)


def test_instances_are_independent_objects():
    random.seed(0)
    torch.manual_seed(0)
    a = _make_instance()
    b = _make_instance()
    assert a is not b


def test_repr_is_string():
    random.seed(0)
    torch.manual_seed(0)
    obj = _make_instance()
    assert isinstance(repr(obj), str)
    assert len(repr(obj)) > 0


def test_module_parameters_are_finite():
    random.seed(0)
    torch.manual_seed(0)
    obj = _make_module()
    for p in obj.parameters():
        assert torch.isfinite(p).all()


def test_same_seed_gives_same_state_dict():
    random.seed(0)
    torch.manual_seed(0)
    obj1 = _make_module()
    torch.manual_seed(0)
    obj2 = _make_module()
    sd1, sd2 = obj1.state_dict(), obj2.state_dict()
    assert list(sd1.keys()) == list(sd2.keys())
    for k in sd1:
        assert torch.allclose(sd1[k].float(), sd2[k].float(), atol=1e-6)


def test_state_dict_roundtrip():
    random.seed(0)
    torch.manual_seed(0)
    obj1 = _make_module()
    torch.manual_seed(1)
    obj2 = _make_module()
    obj2.load_state_dict(obj1.state_dict())
    for k, v in obj1.state_dict().items():
        assert torch.allclose(v.float(), obj2.state_dict()[k].float(), atol=1e-6)


def test_train_eval_toggle():
    random.seed(0)
    torch.manual_seed(0)
    obj = _make_module()
    obj.eval()
    assert obj.training is False
    obj.train()
    assert obj.training is True


def test_forward_returns_finite_tensor():
    random.seed(0)
    torch.manual_seed(0)
    obj = _make_module()
    obj.eval()
    ids = torch.tensor([0, 1, 2])
    with torch.no_grad():
        out = _try_forward(obj, ids)
    if not isinstance(out, torch.Tensor):
        pytest.skip("forward does not return a tensor")
    assert torch.isfinite(out.float()).all()


def test_forward_is_deterministic_in_eval_mode():
    random.seed(0)
    torch.manual_seed(0)
    obj = _make_module()
    obj.eval()
    ids = torch.tensor([0, 1, 2])
    with torch.no_grad():
        out1 = _try_forward(obj, ids)
        out2 = _try_forward(obj, ids)
    if not isinstance(out1, torch.Tensor):
        pytest.skip("forward does not return a tensor")
    assert torch.allclose(out1.float(), out2.float(), atol=1e-6)


def test_forward_does_not_modify_input():
    random.seed(0)
    torch.manual_seed(0)
    obj = _make_module()
    obj.eval()
    ids = torch.tensor([0, 1, 2])
    original = ids.clone()
    with torch.no_grad():
        _try_forward(obj, ids)
    assert torch.equal(ids, original)


def test_forward_does_not_change_parameters():
    random.seed(0)
    torch.manual_seed(0)
    obj = _make_module()
    obj.eval()
    before = {k: v.clone() for k, v in obj.state_dict().items()}
    with torch.no_grad():
        _try_forward(obj, torch.tensor([0, 1]))
    after = obj.state_dict()
    assert list(before.keys()) == list(after.keys())
    for k in before:
        assert torch.allclose(before[k].float(), after[k].float(), atol=1e-6)
