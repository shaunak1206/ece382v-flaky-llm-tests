import copy
import inspect

import pytest
import torch

import torch_embed_sim
from torch_embed_sim import EmbeddingSim


# These tests are written against the public surface of EmbeddingSim as exposed
# by the package. Tests that depend on optional behavior (for example, whether
# EmbeddingSim subclasses torch.nn.Module, or whether it can be built with no
# arguments) are skipped rather than failed when that behavior is absent.


@pytest.fixture
def sim():
    try:
        return EmbeddingSim()
    except TypeError as exc:
        pytest.skip("EmbeddingSim requires constructor arguments: {}".format(exc))


def _requires_module(cls):
    if not issubclass(cls, torch.nn.Module):
        pytest.skip("EmbeddingSim is not a torch.nn.Module")


def test_package_exposes_embedding_sim():
    assert hasattr(torch_embed_sim, "EmbeddingSim")


def test_embedding_sim_is_a_class():
    assert inspect.isclass(EmbeddingSim)


def test_package_attribute_is_submodule_class():
    from torch_embed_sim import embedding_sim

    assert torch_embed_sim.EmbeddingSim is embedding_sim.EmbeddingSim


def test_star_import_exports_embedding_sim():
    namespace = {}
    exec("from torch_embed_sim import *", namespace)
    assert "EmbeddingSim" in namespace
    assert namespace["EmbeddingSim"] is EmbeddingSim


def test_constructor_signature_is_inspectable():
    signature = inspect.signature(EmbeddingSim)
    assert signature is not None


def test_instantiation_with_defaults_returns_instance(sim):
    assert isinstance(sim, EmbeddingSim)


def test_two_instances_are_distinct_objects(sim):
    try:
        other = EmbeddingSim()
    except TypeError:
        pytest.skip("EmbeddingSim requires constructor arguments")
    assert sim is not other


def test_repr_and_str_do_not_raise(sim):
    assert isinstance(repr(sim), str)
    assert isinstance(str(sim), str)


def test_deepcopy_produces_equivalent_type(sim):
    clone = copy.deepcopy(sim)
    assert type(clone) is type(sim)
    assert clone is not sim


def test_module_is_torch_module_when_applicable(sim):
    _requires_module(EmbeddingSim)
    assert isinstance(sim, torch.nn.Module)


def test_parameters_are_torch_parameters(sim):
    _requires_module(EmbeddingSim)
    for param in sim.parameters():
        assert isinstance(param, torch.nn.Parameter)


def test_state_dict_round_trip_into_new_instance(sim):
    _requires_module(EmbeddingSim)
    try:
        other = EmbeddingSim()
    except TypeError:
        pytest.skip("EmbeddingSim requires constructor arguments")
    state = sim.state_dict()
    result = other.load_state_dict(state)
    assert not result.missing_keys
    assert not result.unexpected_keys
    for key, tensor in state.items():
        assert torch.equal(other.state_dict()[key], tensor)


def test_train_and_eval_toggle_training_flag(sim):
    _requires_module(EmbeddingSim)
    sim.eval()
    assert sim.training is False
    sim.train()
    assert sim.training is True


def test_eval_mode_returns_self(sim):
    _requires_module(EmbeddingSim)
    assert sim.eval() is sim


def test_zero_grad_clears_gradients(sim):
    _requires_module(EmbeddingSim)
    params = list(sim.parameters())
    if not params:
        pytest.skip("EmbeddingSim has no parameters")
    for p in params:
        p.grad = torch.ones_like(p)
    sim.zero_grad()
    for p in params:
        assert p.grad is None or torch.all(p.grad == 0)


def test_parameters_can_be_cast_to_float64(sim):
    _requires_module(EmbeddingSim)
    sim.double()
    for param in sim.parameters():
        assert param.dtype == torch.float64
</br>
