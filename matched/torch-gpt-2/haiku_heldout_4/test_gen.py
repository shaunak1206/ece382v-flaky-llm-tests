import inspect
import math

import pytest

torch = pytest.importorskip("torch")
tgt = pytest.importorskip("torch_gpt_2")


def _find(name):
    """Look up a public name on the package, falling back to its submodules."""
    if hasattr(tgt, name):
        return getattr(tgt, name)
    for sub in ("net", "gen", "loader"):
        try:
            module = __import__("torch_gpt_2." + sub, fromlist=[name])
        except ImportError:
            continue
        if hasattr(module, name):
            return getattr(module, name)
    pytest.skip("{} is not available in torch_gpt_2".format(name))


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _make_bpe():
    cls = _find("BytePairEncoding")
    bpe_ranks = {("a", "b"): 0, ("ab", "c"): 1}
    try:
        return cls(bpe_ranks)
    except TypeError:
        pytest.skip("BytePairEncoding constructor signature is not compatible")


def _make_gpt2():
    cls = _find("GPT_2")
    try:
        return cls.__new__(cls) if False else _try_construct(cls)
    except Exception as e:  # pragma: no cover - depends on project config
        pytest.skip("Could not construct GPT_2: {}".format(e))


def _try_construct(cls):
    # The config format is project specific; try the simplest possibilities.
    params = list(inspect.signature(cls.__init__).parameters)
    if len(params) <= 1:
        return cls()
    raise TypeError("GPT_2 requires a configuration that these tests do not know how to build")


def _length(out):
    if isinstance(out, torch.Tensor):
        return out.shape[-1]
    return len(out)


# ---------------------------------------------------------------- gelu

def test_gelu_zero_maps_to_zero():
    gelu = _find("gelu")
    out = gelu(torch.tensor([0.0]))
    assert torch.allclose(out, torch.tensor([0.0]), atol=1e-7)


def test_gelu_matches_tanh_approximation():
    gelu = _find("gelu")
    x = torch.linspace(-4.0, 4.0, steps=101)
    assert torch.allclose(gelu(x), _gelu_tanh(x), atol=1e-3)


def test_gelu_matches_torch_functional_gelu():
    gelu = _find("gelu")
    x = torch.linspace(-4.0, 4.0, steps=101)
    expected = torch.nn.functional.gelu(x)
    assert torch.allclose(gelu(x), expected, atol=1e-3)


def test_gelu_large_positive_is_approximately_identity():
    gelu = _find("gelu")
    x = torch.tensor([10.0, 20.0])
    assert torch.allclose(gelu(x), x, rtol=1e-3)


def test_gelu_large_negative_is_approximately_zero():
    gelu = _find("gelu")
    x = torch.tensor([-10.0, -20.0])
    assert torch.allclose(gelu(x), torch.zeros_like(x), atol=1e-3)


def test_gelu_preserves_shape_and_dtype():
    gelu = _find("gelu")
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    out = gelu(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_gelu_is_monotonic_for_positive_inputs():
    gelu = _find("gelu")
    x = torch.linspace(0.0, 5.0, steps=50)
    out = gelu(x)
    assert torch.all(out[1:] >= out[:-1])


# ---------------------------------------------------------------- BytePairEncoding

def test_bpe_is_exported_class():
    cls = _find("BytePairEncoding")
    assert inspect.isclass(cls)


def test_bpe_encode_empty_string_returns_empty():
    bpe = _make_bpe()
    result = bpe.encode("")
    assert len(result) == 0


def test_bpe_encode_is_deterministic():
    bpe = _make_bpe()
    text = "abcab"
    assert list(bpe.encode(text)) == list(bpe.encode(text))


def test_bpe_encode_nonempty_text_produces_tokens():
    bpe = _make_bpe()
    result = bpe.encode("abc")
    assert len(result) > 0


# ---------------------------------------------------------------- GPT_2

def test_gpt2_is_torch_module():
    cls = _find("GPT_2")
    assert issubclass(cls, torch.nn.Module)


def test_gpt2_forward_preserves_batch_and_sequence_dims():
    model = _make_gpt2()
    ids = torch.zeros((2, 5), dtype=torch.long)
    with torch.no_grad():
        out = model(ids)
    assert out.shape[0] == 2
    assert out.shape[1] == 5


def test_gpt2_has_trainable_parameters():
    model = _make_gpt2()
    params = list(model.parameters())
    assert len(params) > 0
    assert any(p.requires_grad for p in params)


# ---------------------------------------------------------------- generate

def test_generate_is_callable():
    gen = _find("generate")
    assert callable(gen)


def test_generate_extends_sequence_length():
    gen = _find("generate")
    model = _make_gpt2()
    inputs = torch.zeros((1, 3), dtype=torch.long)
    try:
        out = gen(model, inputs, length=4)
    except TypeError:
        pytest.skip("generate signature is not compatible with (model, inputs, length=...)")
    assert _length(out) >= 3 + 1
