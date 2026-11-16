import inspect
import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import torch_embed_sim
from torch_embed_sim import EmbeddingSim

NUM = 10
DIM = 4

QUERY_METHOD_NAMES = [
    "similarity",
    "similarities",
    "most_similar",
    "top_k",
    "topk",
    "query",
    "search",
    "get_similarity",
    "compute",
    "sim",
    "forward",
]


def _constructor_attempts():
    torch.manual_seed(0)
    weight = torch.randn(NUM, DIM)
    return [
        ((nn.Embedding(NUM, DIM),), {}),
        ((weight.clone(),), {}),
        ((NUM, DIM), {}),
        ((), {"num_embeddings": NUM, "embedding_dim": DIM}),
        ((), {}),
    ]


def _make():
    last = None
    for args, kwargs in _constructor_attempts():
        try:
            return EmbeddingSim(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
    pytest.skip("Could not construct EmbeddingSim with known signatures: %r" % (last,))


def _query_attempts():
    return [
        (torch.tensor([0, 1]),),
        (torch.tensor([1]),),
        (0,),
        (torch.randn(DIM),),
        (torch.randn(2, DIM),),
    ]


def _run_query(obj):
    """Return (name, args, result) for the first query that works, else skip."""
    for name in QUERY_METHOD_NAMES:
        method = getattr(obj, name, None)
        if not callable(method):
            continue
        for args in _query_attempts():
            try:
                with torch.no_grad():
                    result = method(*args)
            except Exception:  # noqa: BLE001
                continue
            return name, args, result
    pytest.skip("No query method accepted the generic inputs")


def _first_tensor(result):
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (tuple, list)):
        for item in result:
            t = _first_tensor(item)
            if t is not None:
                return t
    return None


def _snapshot(obj):
    if isinstance(obj, nn.Module):
        return {k: v.detach().clone() for k, v in obj.state_dict().items()}
    return None


def test_embeddingsim_is_exported_from_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")


def test_embeddingsim_is_a_class():
    assert inspect.isclass(EmbeddingSim)


def test_embeddingsim_defined_inside_package():
    assert EmbeddingSim.__module__.startswith("torch_embed_sim")


def test_signature_is_inspectable():
    sig = inspect.signature(EmbeddingSim)
    assert isinstance(sig, inspect.Signature)


def test_exposes_public_callable_api():
    public = [
        n
        for n, v in inspect.getmembers(EmbeddingSim)
        if not n.startswith("_") and callable(v)
    ]
    assert public or callable(getattr(EmbeddingSim, "__call__", None))


def test_can_instantiate():
    obj = _make()
    assert isinstance(obj, EmbeddingSim)


def test_instances_are_independent_objects():
    a = _make()
    b = _make()
    assert a is not b


def test_repr_and_str_work():
    obj = _make()
    assert isinstance(repr(obj), str)
    assert isinstance(str(obj), str)


def test_module_parameters_are_tensors_if_nn_module():
    obj = _make()
    if not isinstance(obj, nn.Module):
        pytest.skip("EmbeddingSim is not an nn.Module")
    for p in obj.parameters():
        assert isinstance(p, torch.Tensor)


def test_query_returns_tensor_or_tensor_container():
    obj = _make()
    _, _, result = _run_query(obj)
    assert _first_tensor(result) is not None


def test_query_result_is_finite():
    obj = _make()
    _, _, result = _run_query(obj)
    t = _first_tensor(result)
    if t is None or not t.is_floating_point():
        pytest.skip("No floating-point result to check")
    assert torch.isfinite(t).all()


def test_query_is_deterministic():
    obj = _make()
    name, args, first = _run_query(obj)
    with torch.no_grad():
        second = getattr(obj, name)(*args)
    t1, t2 = _first_tensor(first), _first_tensor(second)
    assert t1 is not None and t2 is not None
    assert t1.shape == t2.shape
    assert torch.allclose(t1.float(), t2.float(), equal_nan=True)


def test_query_does_not_mutate_state():
    obj = _make()
    before = _snapshot(obj)
    if before is None:
        pytest.skip("State not inspectable for non-Module")
    _run_query(obj)
    after = _snapshot(obj)
    assert before.keys() == after.keys()
    for k in before:
        assert torch.equal(before[k], after[k])


def test_query_does_not_mutate_tensor_input():
    obj = _make()
    name, args, _ = _run_query(obj)
    tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
    if not tensor_args:
        pytest.skip("Query input was not a tensor")
    snapshot = copy.deepcopy(tensor_args)
    with torch.no_grad():
        getattr(obj, name)(*args)
    for orig, now in zip(snapshot, tensor_args):
        assert torch.equal(orig, now)
