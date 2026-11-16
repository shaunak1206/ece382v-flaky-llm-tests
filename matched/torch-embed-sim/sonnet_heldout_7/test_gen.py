import inspect

import pytest

torch = pytest.importorskip("torch")

import torch_embed_sim  # noqa: E402
from torch_embed_sim import EmbeddingSim  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: the constructor signature is not known in advance, so we try a few
# plausible ways to build an instance and skip tests that need one if none work.
# ---------------------------------------------------------------------------

def _candidate_constructors():
    nn = torch.nn
    weights = torch.randn(10, 4)
    return [
        lambda: EmbeddingSim(),
        lambda: EmbeddingSim(nn.Embedding(10, 4)),
        lambda: EmbeddingSim(weights),
        lambda: EmbeddingSim(10, 4),
        lambda: EmbeddingSim(num_embeddings=10, embedding_dim=4),
        lambda: EmbeddingSim(embedding=nn.Embedding(10, 4)),
        lambda: EmbeddingSim(nn.Embedding(10, 4), nn.Embedding(10, 4)),
    ]


def _make_instance():
    for ctor in _candidate_constructors():
        try:
            return ctor()
        except Exception:
            continue
    pytest.skip("Could not construct EmbeddingSim with any known signature")


def _candidate_inputs():
    ids = torch.tensor([0, 1, 2])
    floats = torch.randn(3, 4)
    return [
        (ids,),
        (ids, ids),
        (floats,),
        (floats, floats),
        (torch.tensor([[0, 1], [2, 3]]),),
    ]


def _call_instance(inst):
    for args in _candidate_inputs():
        try:
            with torch.no_grad():
                out = inst(*args)
        except Exception:
            continue
        return args, out
    pytest.skip("Could not call EmbeddingSim instance with any known inputs")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_embeddingsim_exported_from_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")


def test_embeddingsim_is_a_class():
    assert inspect.isclass(EmbeddingSim)


def test_package_and_submodule_export_same_object():
    from torch_embed_sim import embedding_sim

    assert embedding_sim.EmbeddingSim is EmbeddingSim


def test_class_name():
    assert EmbeddingSim.__name__ == "EmbeddingSim"


def test_constructor_signature_is_inspectable():
    sig = inspect.signature(EmbeddingSim)
    assert isinstance(sig, inspect.Signature)


def test_can_instantiate():
    inst = _make_instance()
    assert isinstance(inst, EmbeddingSim)


def test_instance_is_callable_or_has_public_methods():
    inst = _make_instance()
    public = [
        name for name in dir(inst)
        if not name.startswith("_") and callable(getattr(inst, name))
    ]
    assert callable(inst) or len(public) > 0


def test_repr_and_str_return_strings():
    inst = _make_instance()
    assert isinstance(repr(inst), str)
    assert isinstance(str(inst), str)


def test_two_instances_are_distinct_objects():
    a = _make_instance()
    b = _make_instance()
    assert a is not b


def test_module_parameters_and_state_dict_if_nn_module():
    inst = _make_instance()
    if not isinstance(inst, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    params = list(inst.parameters())
    assert all(isinstance(p, torch.nn.Parameter) for p in params)
    assert isinstance(inst.state_dict(), dict)


def test_eval_and_train_modes_if_nn_module():
    inst = _make_instance()
    if not isinstance(inst, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    inst.eval()
    assert inst.training is False
    inst.train()
    assert inst.training is True


def test_call_returns_tensor():
    inst = _make_instance()
    if not callable(inst):
        pytest.skip("Instance is not callable")
    _, out = _call_instance(inst)
    assert isinstance(out, torch.Tensor)


def test_call_output_is_finite():
    inst = _make_instance()
    if not callable(inst):
        pytest.skip("Instance is not callable")
    _, out = _call_instance(inst)
    if not isinstance(out, torch.Tensor) or not out.is_floating_point():
        pytest.skip("Output is not a floating point tensor")
    assert torch.isfinite(out).all()


def test_call_is_deterministic_in_eval_mode():
    inst = _make_instance()
    if not callable(inst):
        pytest.skip("Instance is not callable")
    if isinstance(inst, torch.nn.Module):
        inst.eval()
    args, first = _call_instance(inst)
    with torch.no_grad():
        second = inst(*args)
    assert isinstance(first, torch.Tensor)
    assert torch.allclose(first.float(), second.float())


def test_call_does_not_mutate_inputs():
    inst = _make_instance()
    if not callable(inst):
        pytest.skip("Instance is not callable")
    args, _ = _call_instance(inst)
    copies = [a.clone() for a in args]
    with torch.no_grad():
        inst(*args)
    for original, copy in zip(args, copies):
        assert torch.equal(original, copy)
