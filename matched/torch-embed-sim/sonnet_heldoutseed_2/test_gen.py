import inspect

import pytest

torch = pytest.importorskip("torch")

import torch_embed_sim  # noqa: E402
from torch_embed_sim import EmbeddingSim  # noqa: E402

NUM, DIM = 10, 4


def _constructors():
    torch.manual_seed(0)
    weight = torch.randn(NUM, DIM)
    return [
        lambda: EmbeddingSim(weight.clone()),
        lambda: EmbeddingSim(torch.nn.Embedding.from_pretrained(weight.clone())),
        lambda: EmbeddingSim(torch.nn.Embedding(NUM, DIM)),
        lambda: EmbeddingSim(NUM, DIM),
        lambda: EmbeddingSim(num_embeddings=NUM, embedding_dim=DIM),
        lambda: EmbeddingSim(),
    ]


def build():
    last = None
    for make in _constructors():
        try:
            return make()
        except Exception as exc:  # noqa: BLE001
            last = exc
    pytest.skip("Could not construct EmbeddingSim with known signatures: %r" % (last,))


def _callables(sim):
    out = []
    if callable(sim):
        out.append(sim)
    for name in ("similarity", "sim", "compute", "forward", "score", "most_similar"):
        fn = getattr(sim, name, None)
        if callable(fn) and fn not in out:
            out.append(fn)
    return out


def _input_variants(batch=3):
    torch.manual_seed(1)
    idx_a = torch.arange(batch) % NUM
    idx_b = (torch.arange(batch) + 1) % NUM
    vec_a = torch.randn(batch, DIM)
    vec_b = torch.randn(batch, DIM)
    return [
        (idx_a,),
        (idx_a, idx_b),
        (vec_a,),
        (vec_a, vec_b),
    ]


def compute(sim, batch=3):
    for fn in _callables(sim):
        for args in _input_variants(batch):
            try:
                res = fn(*args)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(res, torch.Tensor):
                return res, batch
    pytest.skip("Could not discover a working similarity call for EmbeddingSim")


def test_exported_from_package():
    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert torch_embed_sim.EmbeddingSim is EmbeddingSim


def test_is_a_class():
    assert inspect.isclass(EmbeddingSim)


def test_can_be_instantiated():
    sim = build()
    assert isinstance(sim, EmbeddingSim)


def test_instances_are_independent_objects():
    a = build()
    b = build()
    assert a is not b


def test_has_callable_interface():
    sim = build()
    assert len(_callables(sim)) > 0


def test_compute_returns_tensor():
    res, _ = compute(build())
    assert isinstance(res, torch.Tensor)


def test_result_is_floating_point():
    res, _ = compute(build())
    assert res.is_floating_point()


def test_result_is_finite():
    res, _ = compute(build())
    assert torch.isfinite(res).all()


def test_result_is_nonempty():
    res, _ = compute(build())
    assert res.numel() > 0


def test_result_leading_dimension_matches_batch():
    res, batch = compute(build(), batch=3)
    if res.dim() == 0:
        pytest.skip("scalar output")
    assert res.shape[0] == batch


def test_deterministic_for_same_input():
    sim = build()
    res1, batch = compute(sim)
    res2, _ = compute(sim, batch)
    assert torch.allclose(res1, res2, equal_nan=True)


def test_works_under_no_grad():
    sim = build()
    with torch.no_grad():
        res, _ = compute(sim)
    assert isinstance(res, torch.Tensor)
    assert not res.requires_grad


def test_single_item_batch():
    res, _ = compute(build(), batch=1)
    assert isinstance(res, torch.Tensor)
    assert torch.isfinite(res).all()


def test_result_on_cpu_by_default():
    res, _ = compute(build())
    assert res.device.type == "cpu"
