import inspect

import pytest

torch = pytest.importorskip("torch")

import torch_gpt_2 as tg  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_attr(name):
    """Fetch a public symbol from the package, skipping if it is unavailable."""
    obj = getattr(tg, name, None)
    if obj is None:
        pytest.skip("{} is not exposed by torch_gpt_2".format(name))
    return obj


def _get_gelu():
    gelu = getattr(tg, "gelu", None)
    if gelu is None:
        try:
            from torch_gpt_2.net import gelu  # type: ignore
        except ImportError:
            pytest.skip("gelu not found")
    return gelu


def _build_gpt2():
    """
    Construct a tiny GPT_2 model. The constructor signature is not known
    to this test file, so only parameters that match common names are
    passed. If construction is not possible with those, the test is skipped.
    """
    GPT_2 = _get_attr("GPT_2")
    sig = inspect.signature(GPT_2)
    candidates = {
        "vocab_size": 50,
        "n_vocab": 50,
        "n_ctx": 8,
        "context_length": 8,
        "max_len": 8,
        "max_seq_len": 8,
        "n_embd": 16,
        "embedding_dim": 16,
        "d_model": 16,
        "dim": 16,
        "n_head": 2,
        "num_heads": 2,
        "n_layer": 1,
        "num_layers": 1,
    }
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}
    try:
        return GPT_2(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on project API
        pytest.skip("Could not construct GPT_2 with inferred kwargs: {}".format(exc))


def _length_kwarg_name(func):
    params = inspect.signature(func).parameters
    for name in ("length", "max_len", "steps", "max_new_tokens", "n_tokens", "max_tokens"):
        if name in params:
            return name
    return None


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_package_exposes_expected_names():
    for name in ("BytePairEncoding", "GPT_2", "gelu", "generate"):
        assert hasattr(tg, name), "torch_gpt_2 should expose {}".format(name)


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

def test_gelu_of_zero_is_zero():
    gelu = _get_gelu()
    out = gelu(torch.zeros(1))
    assert torch.allclose(out, torch.zeros(1), atol=1e-7)


def test_gelu_matches_tanh_approximation():
    gelu = _get_gelu()
    x = torch.linspace(-3.0, 3.0, steps=61)
    expected = 0.5 * x * (1.0 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi if hasattr(torch, "pi") else 2.0 / 3.141592653589793))
        * (x + 0.044715 * x ** 3)))
    assert torch.allclose(gelu(x), expected, atol=1e-3)


def test_gelu_large_positive_inputs_approximately_identity():
    gelu = _get_gelu()
    x = torch.tensor([6.0, 8.0, 10.0])
    assert torch.allclose(gelu(x), x, atol=1e-4)


def test_gelu_large_negative_inputs_approximately_zero():
    gelu = _get_gelu()
    x = torch.tensor([-8.0, -10.0])
    assert torch.allclose(gelu(x), torch.zeros_like(x), atol=1e-4)


def test_gelu_is_monotonic_on_positive_inputs():
    gelu = _get_gelu()
    x = torch.linspace(0.0, 5.0, steps=50)
    y = gelu(x)
    assert torch.all(y[1:] >= y[:-1])


def test_gelu_preserves_shape():
    gelu = _get_gelu()
    x = torch.randn(2, 3, 4)
    assert gelu(x).shape == x.shape


def test_gelu_preserves_dtype():
    gelu = _get_gelu()
    x = torch.randn(5, dtype=torch.float64)
    assert gelu(x).dtype == torch.float64


def test_gelu_is_elementwise():
    gelu = _get_gelu()
    x = torch.randn(4, 6)
    batched = gelu(x)
    rowwise = torch.stack([gelu(row) for row in x])
    assert torch.allclose(batched, rowwise, atol=1e-6)


def test_gelu_gradient_is_finite():
    gelu = _get_gelu()
    x = torch.randn(10, requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# BytePairEncoding
# ---------------------------------------------------------------------------

def test_bpe_can_be_constructed_from_ranks():
    BytePairEncoding = _get_attr("BytePairEncoding")
    try:
        bpe = BytePairEncoding({("a", "b"): 0})
    except TypeError as exc:  # pragma: no cover - depends on project API
        pytest.skip("BytePairEncoding constructor differs: {}".format(exc))
    assert bpe is not None


def test_bpe_encode_returns_list_for_text():
    BytePairEncoding = _get_attr("BytePairEncoding")
    try:
        bpe = BytePairEncoding({("a", "b"): 0})
    except TypeError as exc:  # pragma: no cover
        pytest.skip("BytePairEncoding constructor differs: {}".format(exc))
    if not hasattr(bpe, "encode"):
        pytest.skip("BytePairEncoding has no encode method")
    result = bpe.encode("ab ab")
    assert isinstance(result, list)


def test_bpe_encode_empty_string_gives_empty_result():
    BytePairEncoding = _get_attr("BytePairEncoding")
    try:
        bpe = BytePairEncoding({("a", "b"): 0})
    except TypeError as exc:  # pragma: no cover
        pytest.skip("BytePairEncoding constructor differs: {}".format(exc))
    if not hasattr(bpe, "encode"):
        pytest.skip("BytePairEncoding has no encode method")
    assert list(bpe.encode("")) == []


def test_bpe_encoding_is_deterministic():
    BytePairEncoding = _get_attr("BytePairEncoding")
    try:
        bpe = BytePairEncoding({("a", "b"): 0})
    except TypeError as exc:  # pragma: no cover
        pytest.skip("BytePairEncoding constructor differs: {}".format(exc))
    if not hasattr(bpe, "encode"):
        pytest.skip("BytePairEncoding has no encode method")
    text = "hello world hello"
    assert bpe.encode(text) == bpe.encode(text)


# ---------------------------------------------------------------------------
# GPT_2
# ---------------------------------------------------------------------------

def test_gpt2_is_torch_module():
    GPT_2 = _get_attr("GPT_2")
    assert issubclass(GPT_2, torch.nn.Module)


def test_gpt2_parameters_are_trainable():
    model = _build_gpt2()
    params = list(model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_gpt2_forward_preserves_batch_and_sequence_dims():
    model = _build_gpt2()
    model.eval()
    ids = torch.zeros(2, 4, dtype=torch.long)
    with torch.no_grad():
        out = model(ids)
    if isinstance(out, (tuple, list)):
        out = out[0]
    assert out.dim() == 3
    assert out.shape[0] == 2
    assert out.shape[1] == 4


def test_gpt2_forward_is_deterministic_in_eval_mode():
    model = _build_gpt2()
    model.eval()
    ids = torch.arange(6, dtype=torch.long).reshape(1, 6) % 10
    with torch.no_grad():
        a = model(ids)
        b = model(ids)
    if isinstance(a, (tuple, list)):
        a, b = a[0], b[0]
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_is_callable():
    generate = _get_attr("generate")
    assert callable(generate)


def test_generate_extends_prompt_by_requested_length():
    generate = _get_attr("generate")
    model = _build_gpt2()
    model.eval()
    length_name = _length_kwarg_name(generate)
    if length_name is None:
        pytest.skip("generate does not expose a recognised length argument")
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    torch.manual_seed(0)
    out = generate(model, prompt, **{length_name: 3})
    if hasattr(out, "shape"):
        assert out.shape[-1] == prompt.shape[-1] + 3
    else:
        assert len(out[0]) == prompt.shape[-1] + 3


def test_generate_is_reproducible_with_same_seed():
    generate = _get_attr("generate")
    model = _build_gpt2()
    model.eval()
    length_name = _length_kwarg_name(generate)
    if length_name is None:
        pytest.skip("generate does not expose a recognised length argument")
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    kwargs = {length_name: 3}
    torch.manual_seed(42)
    first = generate(model, prompt, **kwargs)
    torch.manual_seed(42)
    second = generate(model, prompt, **kwargs)
    if hasattr(first, "shape"):
        assert torch.equal(first, second)
    else:
        assert first == second
