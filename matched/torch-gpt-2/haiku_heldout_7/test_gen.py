import inspect
import math
import os

import pytest

torch = pytest.importorskip("torch")

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate, get_bpe_from_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_to_unicode():
    """Standard GPT-2 byte-to-unicode mapping used by BPE vocabularies."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _write_bpe_files(tmp_path):
    """Create a tiny encoder.json / vocab.bpe pair with one merge: 'h' + 'e' -> 'he'."""
    byte_chars = sorted(set(_bytes_to_unicode().values()))
    encoder = {ch: i for i, ch in enumerate(byte_chars)}
    encoder["he"] = len(encoder)
    encoder["<|endoftext|>"] = len(encoder)

    encoder_path = tmp_path / "encoder.json"
    vocab_path = tmp_path / "vocab.bpe"

    import json
    encoder_path.write_text(json.dumps(encoder), encoding="utf-8")
    vocab_path.write_text("#version: 0.2\nh e\n", encoding="utf-8")
    return str(encoder_path), str(vocab_path), encoder


def _to_list(x):
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if x and isinstance(x[0], list):
        return x[0]
    return list(x)


def _make_model():
    config = {
        "n_vocab": 32,
        "n_ctx": 16,
        "n_embd": 8,
        "n_head": 2,
        "n_layer": 1,
    }
    try:
        model = GPT_2(config)
    except TypeError:
        try:
            model = GPT_2(**config)
        except TypeError:
            pytest.skip("GPT_2 constructor signature not recognized by this test")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_package_exposes_public_names():
    for name in ("GPT_2", "gelu", "generate", "BytePairEncoding", "get_bpe_from_files"):
        assert hasattr(torch_gpt_2, name), f"missing export: {name}"


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

def test_gelu_is_zero_at_origin():
    out = gelu(torch.zeros(1))
    assert torch.allclose(out, torch.zeros(1), atol=1e-7)


def test_gelu_matches_tanh_approximation():
    x = torch.linspace(-4.0, 4.0, steps=41)
    expected = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    out = gelu(x)
    assert torch.allclose(out, expected, atol=1e-5)


def test_gelu_approximates_identity_for_large_positive_inputs():
    x = torch.tensor([6.0, 8.0, 10.0])
    out = gelu(x)
    assert torch.allclose(out, x, atol=1e-3)


def test_gelu_approaches_zero_for_large_negative_inputs():
    x = torch.tensor([-8.0, -10.0])
    out = gelu(x)
    assert torch.all(out.abs() < 1e-3)


def test_gelu_preserves_shape_and_dtype():
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    out = gelu(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_gelu_is_monotonic_on_positive_range():
    x = torch.linspace(0.0, 5.0, steps=50)
    out = gelu(x)
    diffs = out[1:] - out[:-1]
    assert torch.all(diffs >= -1e-7)


# ---------------------------------------------------------------------------
# BytePairEncoding
# ---------------------------------------------------------------------------

def test_get_bpe_from_files_returns_bpe_instance(tmp_path):
    encoder_path, vocab_path, _ = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert isinstance(bpe, BytePairEncoding)


def test_bpe_encode_returns_list_of_ints(tmp_path):
    encoder_path, vocab_path, _ = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    tokens = bpe.encode("hello")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)


def test_bpe_encode_empty_string_gives_empty_list(tmp_path):
    encoder_path, vocab_path, _ = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert bpe.encode("") == []


def test_bpe_applies_merge_rule(tmp_path):
    encoder_path, vocab_path, encoder = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert bpe.encode("he") == [encoder["he"]]


def test_bpe_round_trip_preserves_text(tmp_path):
    encoder_path, vocab_path, _ = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    text = "hello world, he said!"
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_round_trip_handles_non_ascii(tmp_path):
    encoder_path, vocab_path, _ = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    text = "café 😀"
    assert bpe.decode(bpe.encode(text)) == text


# ---------------------------------------------------------------------------
# GPT_2
# ---------------------------------------------------------------------------

def test_gpt2_is_torch_module():
    assert issubclass(GPT_2, torch.nn.Module)


def test_gpt2_forward_output_shape():
    model = _make_model()
    ids = torch.zeros(1, 5, dtype=torch.long)
    with torch.no_grad():
        out = model(ids)
    assert out.shape[0] == 1
    assert out.shape[1] == 5
    assert out.shape[2] == 32


def test_gpt2_has_trainable_parameters():
    model = _make_model()
    params = list(model.parameters())
    assert len(params) > 0
    assert any(p.requires_grad for p in params)


def test_gpt2_eval_forward_is_deterministic():
    model = _make_model()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        first = model(ids)
        second = model(ids)
    assert torch.allclose(first, second)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def _call_generate(model, context, length):
    params = inspect.signature(generate).parameters
    kwargs = {}
    if "length" in params:
        kwargs["length"] = length
    elif "max_len" in params:
        kwargs["max_len"] = length
    else:
        pytest.skip("generate signature does not expose a length argument")
    return generate(model, context, **kwargs)


def test_generate_returns_longer_sequence():
    model = _make_model()
    context = [1, 2, 3]
    result = _to_list(_call_generate(model, context, length=4))
    assert len(result) > len(context)


def test_generate_keeps_context_prefix():
    model = _make_model()
    context = [5, 6, 7]
    result = _to_list(_call_generate(model, context, length=3))
    assert result[: len(context)] == context


def test_generate_produces_valid_token_ids():
    model = _make_model()
    context = [1, 2]
    result = _to_list(_call_generate(model, context, length=5))
    assert all(0 <= t < 32 for t in result)
