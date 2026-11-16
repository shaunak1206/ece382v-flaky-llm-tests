"""
Unit tests for EmbeddingSim exported from torch_embed_sim.

The source of torch_embed_sim/embedding_sim.py was not provided, so these tests
rely only on the public import path and on introspection. Tests that need a
constructor argument they cannot infer from the signature are skipped rather
than guessed.
"""
import copy
import inspect

import pytest
import torch

from torch_embed_sim import EmbeddingSim


def _build(**overrides):
    """Instantiate EmbeddingSim, supplying overrides and relying on defaults.

    Skips the test if a required constructor argument has no override.
    """
    try:
        sig = inspect.signature(EmbeddingSim.__init__)
    except (TypeError, ValueError):
        pytest.skip("EmbeddingSim.__init__ signature is not introspectable")

    kwargs = {}
    params = list(sig.parameters.values())[1:]  # drop self
    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.name in overrides:
            kwargs[p.name] = overrides[p.name]
        elif p.default is not inspect.Parameter.empty:
            continue
        else:
            pytest.skip("required constructor argument {!r} not known".format(p.name))
    return EmbeddingSim(**kwargs)


def _is_module(obj):
    return isinstance(obj, torch.nn.Module)


def test_embedding_sim_is_a_class():
    assert inspect.isclass(EmbeddingSim)


def test_embedding_sim_is_the_same_object_in_submodule():
    import importlib

    submodule = importlib.import_module("torch_embed_sim.embedding_sim")
    assert getattr(submodule, "EmbeddingSim") is EmbeddingSim


def test_torch_dependency_is_available():
    major = int(torch.__version__.split(".")[0])
    assert major >= 1


def test_constructor_signature_is_introspectable():
    sig = inspect.signature(EmbeddingSim)
    assert isinstance(sig, inspect.Signature)


def test_instance_built_with_defaults_is_of_class():
    obj = _build()
    assert isinstance(obj, EmbeddingSim)


def test_repr_and_str_return_strings():
    obj = _build()
    assert isinstance(repr(obj), str)
    assert isinstance(str(obj), str)


def test_two_instances_are_distinct_objects():
    a = _build()
    b = _build()
    assert a is not b


def test_instance_is_deepcopyable():
    obj = _build()
    clone = copy.deepcopy(obj)
    assert isinstance(clone, EmbeddingSim)
    assert clone is not obj


def test_module_parameters_are_finite_when_torch_module():
    obj = _build()
    if not _is_module(obj):
        pytest.skip("EmbeddingSim is not a torch.nn.Module")
    for name, param in obj.named_parameters():
        assert torch.isfinite(param).all(), "non-finite values in {}".format(name)


def test_module_parameters_require_grad_by_default_when_torch_module():
    obj = _build()
    if not _is_module(obj):
        pytest.skip("EmbeddingSim is not a torch.nn.Module")
    params = list(obj.parameters())
    if not params:
        pytest.skip("module exposes no parameters")
    assert all(p.requires_grad for p in params)


def test_seeded_construction_is_reproducible_when_torch_module():
    obj_probe = _build()
    if not _is_module(obj_probe):
        pytest.skip("EmbeddingSim is not a torch.nn.Module")

    torch.manual_seed(1234)
    first = _build()
    torch.manual_seed(1234)
    second = _build()

    sd1 = first.state_dict()
    sd2 = second.state_dict()
    assert sd1.keys() == sd2.keys()
    for key in sd1:
        assert torch.equal(sd1[key], sd2[key]), "mismatch for {}".format(key)


def test_state_dict_roundtrip_when_torch_module():
    obj = _build()
    if not _is_module(obj):
        pytest.skip("EmbeddingSim is not a torch.nn.Module")

    fresh = _build()
    result = fresh.load_state_dict(obj.state_dict())
    assert not result.missing_keys
    assert not result.unexpected_keys
    for key, tensor in obj.state_dict().items():
        assert torch.equal(tensor, fresh.state_dict()[key])


def test_state_dict_is_dict_of_tensors_when_torch_module():
    obj = _build()
    if not _is_module(obj):
        pytest.skip("EmbeddingSim is not a torch.nn.Module")
    sd = obj.state_dict()
    assert isinstance(sd, dict)
    assert all(isinstance(v, torch.Tensor) for v in sd.values())


def test_constructor_rejects_unknown_keyword():
    with pytest.raises(TypeError):
        EmbeddingSim(__definitely_not_a_real_argument__=1)
