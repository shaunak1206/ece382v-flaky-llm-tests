"""Unit tests for EmbeddingSim exported from torch_embed_sim.

Note: only torch_embed_sim/__init__.py was provided. These tests therefore
introspect EmbeddingSim's constructor signature and optional methods rather
than assuming a fixed API. Tests skip when the API they need is not present.
"""
import inspect

import pytest
import torch

import torch_embed_sim
import torch_embed_sim.embedding_sim as embedding_sim_module
from torch_embed_sim import EmbeddingSim


_VAR_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _required_params():
    """Return the list of required constructor parameters of EmbeddingSim."""
    sig = inspect.signature(EmbeddingSim)
    return [
        p for p in sig.parameters.values()
        if p.kind not in _VAR_KINDS and p.default is inspect.Parameter.empty
    ]


def _infer_kwargs():
    """Build plausible keyword arguments for required constructor parameters."""
    kwargs = {}
    for p in _required_params():
        lname = p.name.lower()
        if "dim" in lname:
            kwargs[p.name] = 4
        elif any(tok in lname for tok in ("num", "size", "count", "vocab")):
            kwargs[p.name] = 10
        else:
            pytest.skip("cannot infer a value for required parameter %r" % p.name)
    return kwargs


def _make_sim(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return EmbeddingSim(**_infer_kwargs())


def _require_module(obj):
    if not isinstance(obj, torch.nn.Module):
        pytest.skip("EmbeddingSim is not a torch.nn.Module")


@pytest.fixture
def sim():
    return _make_sim(seed=0)


# --- Import and export surface -------------------------------------------------

def test_package_reexports_embedding_sim():
    assert hasattr(torch_embed_sim, "EmbeddingSim")
    assert torch_embed_sim.EmbeddingSim is EmbeddingSim


def test_embedding_sim_module_defines_embedding_sim():
    assert hasattr(embedding_sim_module, "EmbeddingSim")
    assert embedding_sim_module.EmbeddingSim is EmbeddingSim


def test_embedding_sim_is_a_class():
    assert inspect.isclass(EmbeddingSim)


# --- Construction --------------------------------------------------------------

def test_constructor_accepts_inferred_arguments():
    instance = _make_sim()
    assert isinstance(instance, EmbeddingSim)


def test_repr_is_non_empty_string(sim):
    text = repr(sim)
    assert isinstance(text, str)
    assert text


def test_missing_required_arguments_raise_type_error():
    if not _required_params():
        pytest.skip("constructor has no required parameters")
    with pytest.raises(TypeError):
        EmbeddingSim()


def test_unknown_keyword_argument_raises_type_error():
    sig = inspect.signature(EmbeddingSim)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kw:
        pytest.skip("constructor accepts **kwargs")
    kwargs = _infer_kwargs()
    kwargs["definitely_not_a_real_argument"] = 1
    with pytest.raises(TypeError):
        EmbeddingSim(**kwargs)


def test_two_instances_are_independent_objects():
    a = _make_sim(seed=1)
    b = _make_sim(seed=1)
    assert a is not b


# --- torch.nn.Module behaviour (only if EmbeddingSim is a Module) ---------------

def test_parameters_are_torch_parameters(sim):
    _require_module(sim)
    params = list(sim.parameters())
    for p in params:
        assert isinstance(p, torch.nn.Parameter)


def test_parameters_are_finite(sim):
    _require_module(sim)
    for name, p in sim.named_parameters():
        assert torch.isfinite(p).all(), "non-finite values in %s" % name


def test_same_seed_produces_identical_parameters():
    a = _make_sim(seed=42)
    b = _make_sim(seed=42)
    _require_module(a)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb), "parameter %s differs across same seed" % na


def test_different_seeds_produce_different_parameters():
    a = _make_sim(seed=1)
    b = _make_sim(seed=2)
    _require_module(a)
    params_a = list(a.parameters())
    params_b = list(b.parameters())
    if not params_a:
        pytest.skip("module has no parameters")
    assert any(not torch.equal(pa, pb) for pa, pb in zip(params_a, params_b))


def test_state_dict_round_trip_restores_parameters():
    src = _make_sim(seed=3)
    _require_module(src)
    dst = _make_sim(seed=99)
    dst.load_state_dict(src.state_dict())
    for (n1, p1), (n2, p2) in zip(src.named_parameters(), dst.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_state_dict_keys_match_named_parameters(sim):
    _require_module(sim)
    state_keys = set(sim.state_dict().keys())
    param_keys = {name for name, _ in sim.named_parameters()}
    assert param_keys.issubset(state_keys)


def test_module_can_be_moved_to_double_precision(sim):
    _require_module(sim)
    sim.double()
    for p in sim.parameters():
        assert p.dtype == torch.float64


# --- Optional similarity / forward behaviour -----------------------------------

def _similarity_method(obj):
    for name in ("similarity", "cosine_similarity", "sim"):
        method = getattr(obj, name, None)
        if callable(method):
            return method
    pytest.skip("EmbeddingSim has no similarity method")


def test_similarity_of_identical_inputs_is_maximal(sim):
    method = _similarity_method(sim)
    dim = _infer_kwargs().get("embedding_dim", _infer_kwargs().get("dim", 4))
    x = torch.randn(1, dim)
    try:
        score = method(x, x)
    except TypeError:
        pytest.skip("similarity signature not compatible with (tensor, tensor)")
    score = torch.as_tensor(score).flatten()
    assert torch.allclose(score, torch.ones_like(score), atol=1e-5)


def test_similarity_is_symmetric(sim):
    method = _similarity_method(sim)
    x = torch.randn(1, 4)
    y = torch.randn(1, 4)
    try:
        xy = method(x, y)
        yx = method(y, x)
    except TypeError:
        pytest.skip("similarity signature not compatible with (tensor, tensor)")
    assert torch.allclose(torch.as_tensor(xy), torch.as_tensor(yx), atol=1e-6)


def test_forward_output_does_not_require_grad_under_no_grad(sim):
    _require_module(sim)
    if not callable(getattr(sim, "forward", None)):
        pytest.skip("no forward method")
    idx = torch.zeros(2, dtype=torch.long)
    try:
        with torch.no_grad():
            out = sim(idx)
    except TypeError:
        pytest.skip("forward signature not compatible with index tensor input")
    if isinstance(out, torch.Tensor):
        assert not out.requires_grad
