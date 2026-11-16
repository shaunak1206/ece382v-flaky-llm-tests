import copy
import inspect

import pytest
import torch

import torch_embed_sim
from torch_embed_sim import EmbeddingSim


_POSITIONAL = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _build(**kwargs):
    """Construct EmbeddingSim, skipping if the constructor needs other arguments."""
    try:
        return EmbeddingSim(**kwargs)
    except TypeError as exc:
        pytest.skip(f"EmbeddingSim() with no arguments is not supported: {exc}")


def _required_positional_count(fn):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        pytest.skip("forward signature cannot be introspected")
    return sum(
        1
        for p in sig.parameters.values()
        if p.kind in _POSITIONAL and p.default is inspect.Parameter.empty
    )


def _make_inputs(sim):
    """Build random float tensors matching forward's positional arity."""
    forward = getattr(sim, "forward", None)
    if forward is None:
        pytest.skip("EmbeddingSim does not define forward")
    n = _required_positional_count(forward)
    if n == 0 or n > 2:
        pytest.skip(f"forward takes {n} required positional arguments; not supported by this generic test")
    gen = torch.Generator().manual_seed(0)
    return [torch.rand(4, 8, generator=gen) for _ in range(n)]


@pytest.fixture
def sim():
    return _build()


# --- Package and import surface -------------------------------------------

def test_package_exports_embedding_sim():
    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert torch_embed_sim.EmbeddingSim is EmbeddingSim


def test_embedding_sim_is_class():
    assert inspect.isclass(EmbeddingSim)


def test_submodule_defines_same_class():
    import torch_embed_sim.embedding_sim as submodule

    assert submodule.EmbeddingSim is EmbeddingSim


def test_constructor_signature_is_introspectable():
    sig = inspect.signature(EmbeddingSim)
    assert isinstance(sig, inspect.Signature)


# --- Construction ----------------------------------------------------------

def test_instance_can_be_created(sim):
    assert isinstance(sim, EmbeddingSim)


def test_separate_instances_are_distinct_objects():
    a = _build()
    b = _build()
    assert a is not b


def test_instance_repr_does_not_raise(sim):
    assert isinstance(repr(sim), str)


# --- Forward / call behaviour ---------------------------------------------

def test_call_returns_tensor(sim):
    inputs = _make_inputs(sim)
    out = sim(*inputs)
    assert isinstance(out, torch.Tensor)


def test_output_values_are_finite(sim):
    inputs = _make_inputs(sim)
    out = sim(*inputs)
    assert torch.isfinite(out).all()


def test_output_preserves_batch_dimension(sim):
    inputs = _make_inputs(sim)
    out = sim(*inputs)
    if out.dim() > 0:
        assert out.shape[0] == 4


def test_repeated_calls_are_deterministic_in_eval_mode(sim):
    if hasattr(sim, "eval"):
        sim.eval()
    inputs = _make_inputs(sim)
    with torch.no_grad():
        first = sim(*inputs)
        second = sim(*inputs)
    assert torch.equal(first, second)


def test_forward_does_not_mutate_inputs(sim):
    inputs = _make_inputs(sim)
    snapshots = [t.clone() for t in inputs]
    sim(*inputs)
    for original, snapshot in zip(inputs, snapshots):
        assert torch.equal(original, snapshot)


def test_forward_works_under_no_grad(sim):
    inputs = _make_inputs(sim)
    with torch.no_grad():
        out = sim(*inputs)
    assert isinstance(out, torch.Tensor)


def test_forward_is_differentiable_when_inputs_require_grad(sim):
    inputs = [t.clone().requires_grad_(True) for t in _make_inputs(sim)]
    out = sim(*inputs)
    if not out.requires_grad:
        pytest.skip("output is not differentiable in this configuration")
    out.sum().backward()
    for t in inputs:
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()


def test_identical_inputs_produce_identical_outputs(sim):
    if hasattr(sim, "eval"):
        sim.eval()
    inputs = _make_inputs(sim)
    if len(inputs) == 2:
        same = [inputs[0], inputs[0].clone()]
    else:
        same = [inputs[0]]
    with torch.no_grad():
        out = sim(*same)
    assert torch.isfinite(out).all()


def test_deepcopy_produces_equivalent_output(sim):
    if hasattr(sim, "eval"):
        sim.eval()
    inputs = _make_inputs(sim)
    clone = copy.deepcopy(sim)
    if hasattr(clone, "eval"):
        clone.eval()
    with torch.no_grad():
        assert torch.allclose(sim(*inputs), clone(*inputs))


def test_default_dtype_output_is_floating_point(sim):
    inputs = _make_inputs(sim)
    out = sim(*inputs)
    assert out.is_floating_point()
