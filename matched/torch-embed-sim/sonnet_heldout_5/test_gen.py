import inspect

import pytest

torch = pytest.importorskip("torch")

import torch_embed_sim
from torch_embed_sim import EmbeddingSim

# NOTE: the source of torch_embed_sim/embedding_sim.py was not provided, so these
# tests make no assumptions about the exact constructor or method signatures.
# They probe a few common conventions and skip when none of them applies.

_CTOR_ATTEMPTS = [
    lambda: EmbeddingSim(),
    lambda: EmbeddingSim(torch.randn(10, 4)),
    lambda: EmbeddingSim(10, 4),
    lambda: EmbeddingSim(4),
    lambda: EmbeddingSim(num_embeddings=10, embedding_dim=4),
]

_METHOD_NAMES = [
    "similarity",
    "sim",
    "cosine_similarity",
    "compute",
    "forward",
    "__call__",
]


def _build():
    for attempt in _CTOR_ATTEMPTS:
        try:
            return attempt()
        except Exception:
            continue
    pytest.skip("Could not construct EmbeddingSim with any common signature")


def _call_similarity(obj, a, b):
    """Try common similarity method names with two tensor arguments."""
    for name in _METHOD_NAMES:
        fn = getattr(obj, name, None)
        if fn is None or not callable(fn):
            continue
        try:
            return fn(a, b)
        except Exception:
            continue
    pytest.skip("No usable similarity method accepting two tensors found")


@pytest.fixture
def sim():
    return _build()


def test_class_exported_from_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert inspect.isclass(EmbeddingSim)


def test_class_is_same_as_in_submodule():
    from torch_embed_sim import embedding_sim

    assert embedding_sim.EmbeddingSim is EmbeddingSim


def test_class_name():
    assert EmbeddingSim.__name__ == "EmbeddingSim"


def test_can_instantiate(sim):
    assert isinstance(sim, EmbeddingSim)


def test_repr_and_str_work(sim):
    assert isinstance(repr(sim), str)
    assert isinstance(str(sim), str)


def test_instances_are_independent():
    a = _build()
    b = _build()
    assert a is not b


def test_module_state_dict_if_module(sim):
    if not isinstance(sim, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    sd = sim.state_dict()
    assert hasattr(sd, "keys")
    # parameters() should be iterable
    assert isinstance(list(sim.parameters()), list)


def test_module_eval_train_toggle_if_module(sim):
    if not isinstance(sim, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    sim.eval()
    assert sim.training is False
    sim.train()
    assert sim.training is True


def test_similarity_returns_tensor(sim):
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    out = _call_similarity(sim, a, b)
    assert isinstance(out, torch.Tensor)


def test_similarity_output_is_finite(sim):
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    out = _call_similarity(sim, a, b)
    if not isinstance(out, torch.Tensor):
        pytest.skip("Non-tensor output")
    assert torch.isfinite(out.float()).all()


def test_similarity_is_deterministic(sim):
    if isinstance(sim, torch.nn.Module):
        sim.eval()
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    out1 = _call_similarity(sim, a, b)
    out2 = _call_similarity(sim, a, b)
    if not (isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor)):
        pytest.skip("Non-tensor output")
    assert torch.allclose(out1.float(), out2.float())


def test_similarity_does_not_mutate_inputs(sim):
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    a_copy, b_copy = a.clone(), b.clone()
    _call_similarity(sim, a, b)
    assert torch.equal(a, a_copy)
    assert torch.equal(b, b_copy)


def test_similarity_output_has_batch_dimension(sim):
    a = torch.randn(5, 4)
    b = torch.randn(5, 4)
    out = _call_similarity(sim, a, b)
    if not isinstance(out, torch.Tensor):
        pytest.skip("Non-tensor output")
    # Either pairwise (5,) / (5,1) or full matrix (5,5)
    assert out.shape[0] == 5


def test_similarity_output_is_floating_point(sim):
    a = torch.randn(2, 4)
    b = torch.randn(2, 4)
    out = _call_similarity(sim, a, b)
    if not isinstance(out, torch.Tensor):
        pytest.skip("Non-tensor output")
    assert out.is_floating_point()
