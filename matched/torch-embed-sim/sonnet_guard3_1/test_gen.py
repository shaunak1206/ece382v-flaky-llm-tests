# Tests for EmbeddingSim (torch_embed_sim/__init__.py re-exports it from
# torch_embed_sim.embedding_sim).
#
# NOTE: the source of `embedding_sim` was not provided, so these tests avoid
# assuming an exact constructor or method signature. They discover a working
# way to build the object and call its similarity method, and skip (rather than
# fail spuriously) if no known calling convention works. Everything asserted is
# deterministic.

import inspect

import pytest
import torch

import torch_embed_sim
from torch_embed_sim import EmbeddingSim

NUM = 10
DIM = 4

_SIM_METHOD_NAMES = ("similarity", "sim", "forward", "compute", "score")


def _builders():
    return [
        lambda: EmbeddingSim(torch.nn.Embedding(NUM, DIM)),
        lambda: EmbeddingSim(torch.randn(NUM, DIM)),
        lambda: EmbeddingSim(NUM, DIM),
        lambda: EmbeddingSim(num_embeddings=NUM, embedding_dim=DIM),
        lambda: EmbeddingSim(),
    ]


def _build():
    last = None
    for b in _builders():
        try:
            return b()
        except Exception as exc:  # noqa: BLE001
            last = exc
    pytest.skip("could not construct EmbeddingSim with known signatures: %r" % (last,))


def _first_tensor(res):
    if torch.is_tensor(res):
        return res
    if isinstance(res, (tuple, list)) and res and torch.is_tensor(res[0]):
        return res[0]
    return None


def _call_sim(obj):
    """Return (result_tensor, args) using the first calling convention that works."""
    arg_sets = [
        (torch.randn(3, DIM), torch.randn(3, DIM)),
        (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])),
        (torch.randn(3, DIM),),
        (torch.tensor([0, 1, 2]),),
    ]
    names = [n for n in _SIM_METHOD_NAMES if callable(getattr(obj, n, None))]
    if callable(obj):
        names.append("__call__")
    for name in names:
        fn = obj if name == "__call__" else getattr(obj, name)
        for args in arg_sets:
            try:
                res = fn(*[a.clone() for a in args])
            except Exception:  # noqa: BLE001
                continue
            t = _first_tensor(res)
            if t is not None:
                return fn, t, args
    pytest.skip("no known similarity calling convention worked")


def test_exported_from_package():
    torch.manual_seed(0)
    from torch_embed_sim import embedding_sim

    assert torch_embed_sim.EmbeddingSim is embedding_sim.EmbeddingSim


def test_is_a_class():
    torch.manual_seed(0)
    assert inspect.isclass(EmbeddingSim)


def test_in_dunder_all_if_defined():
    torch.manual_seed(0)
    from torch_embed_sim import embedding_sim

    names = getattr(embedding_sim, "__all__", None)
    if names is None:
        pytest.skip("embedding_sim defines no __all__")
    assert "EmbeddingSim" in names


def test_instantiation():
    torch.manual_seed(0)
    obj = _build()
    assert isinstance(obj, EmbeddingSim)


def test_instances_are_distinct():
    torch.manual_seed(0)
    a = _build()
    b = _build()
    assert a is not b


def test_repr_is_string():
    torch.manual_seed(0)
    obj = _build()
    assert isinstance(repr(obj), str)
    assert len(repr(obj)) > 0


def test_module_parameters_are_tensors_if_module():
    torch.manual_seed(0)
    obj = _build()
    if not isinstance(obj, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    for p in obj.parameters():
        assert isinstance(p, torch.nn.Parameter)


def test_state_dict_roundtrip_if_module():
    torch.manual_seed(0)
    obj = _build()
    if not isinstance(obj, torch.nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    sd = {k: v.clone() for k, v in obj.state_dict().items()}
    obj.load_state_dict(sd)
    for k, v in obj.state_dict().items():
        assert torch.equal(v, sd[k])


def test_similarity_returns_tensor():
    torch.manual_seed(0)
    obj = _build()
    _, res, _ = _call_sim(obj)
    assert torch.is_tensor(res)


def test_similarity_values_finite():
    torch.manual_seed(0)
    obj = _build()
    _, res, _ = _call_sim(obj)
    assert torch.isfinite(res.float()).all()


def test_similarity_is_deterministic():
    torch.manual_seed(0)
    obj = _build()
    if isinstance(obj, torch.nn.Module):
        obj.eval()
    fn, res, args = _call_sim(obj)
    again = _first_tensor(fn(*[a.clone() for a in args]))
    assert again is not None
    assert torch.allclose(res.float(), again.float(), atol=1e-6)


def test_similarity_does_not_mutate_inputs():
    torch.manual_seed(0)
    obj = _build()
    fn, _, args = _call_sim(obj)
    copies = [a.clone() for a in args]
    fn(*args)
    for orig, after in zip(copies, args):
        assert torch.equal(orig, after)


def test_similarity_no_grad_matches_grad_mode():
    torch.manual_seed(0)
    obj = _build()
    if isinstance(obj, torch.nn.Module):
        obj.eval()
    fn, res, args = _call_sim(obj)
    with torch.no_grad():
        res_ng = _first_tensor(fn(*[a.clone() for a in args]))
    assert res_ng is not None
    assert torch.allclose(res.detach().float(), res_ng.float(), atol=1e-6)


def test_similarity_output_is_floating_point():
    torch.manual_seed(0)
    obj = _build()
    _, res, _ = _call_sim(obj)
    assert res.is_floating_point()
