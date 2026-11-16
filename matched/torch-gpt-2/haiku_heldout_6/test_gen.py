import math
import json

import pytest

torch = pytest.importorskip("torch")

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate, get_bpe_from_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_to_unicode():
    """GPT-2 byte <-> printable unicode mapping (reimplemented for test vocab)."""
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
    return {b: chr(c) for b, c in zip(bs, cs)}


def _write_bpe_files(tmp_path):
    byte_chars = _bytes_to_unicode()
    token_dict = {}
    for b in range(256):
        token_dict[byte_chars[b]] = b
    token_dict["<|endoftext|>"] = 256
    token_dict["he"] = 257
    encoder_path = tmp_path / "encoder.json"
    vocab_path = tmp_path / "vocab.bpe"
    encoder_path.write_text(json.dumps(token_dict), encoding="utf-8")
    vocab_path.write_text("#version: 0.2\nh e\n", encoding="utf-8")
    return str(encoder_path), str(vocab_path), token_dict


def _make_model(n_vocab=50, n_ctx=16, n_embd=16, n_head=2, n_layer=1):
    torch.manual_seed(0)
    model = GPT_2(
        n_vocab=n_vocab,
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
    )
    model.eval()
    return model


def _tanh_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _exact_gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

def test_gelu_zero_at_zero():
    out = gelu(torch.tensor([0.0]))
    assert torch.allclose(out, torch.tensor([0.0]), atol=1e-7)


def test_gelu_matches_reference_formula():
    x = torch.linspace(-5.0, 5.0, steps=101)
    out = gelu(x)
    ref_tanh = _tanh_gelu(x)
    ref_exact = _exact_gelu(x)
    close_tanh = torch.allclose(out, ref_tanh, atol=1e-3)
    close_exact = torch.allclose(out, ref_exact, atol=1e-3)
    assert close_tanh or close_exact


def test_gelu_preserves_shape_and_dtype():
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    out = gelu(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_gelu_saturation_limits():
    big_pos = torch.tensor([10.0])
    big_neg = torch.tensor([-10.0])
    assert torch.allclose(gelu(big_pos), big_pos, atol=1e-4)
    assert torch.allclose(gelu(big_neg), torch.zeros_like(big_neg), atol=1e-4)


def test_gelu_monotonic_for_positive_inputs():
    x = torch.linspace(0.0, 6.0, steps=200)
    out = gelu(x)
    diffs = out[1:] - out[:-1]
    assert torch.all(diffs >= -1e-6)


def test_gelu_supports_backward():
    x = torch.randn(5, requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# BytePairEncoding
# ---------------------------------------------------------------------------

def test_bpe_is_reexported_from_keras_gpt_2():
    import keras_gpt_2

    assert BytePairEncoding is keras_gpt_2.BytePairEncoding
    assert torch_gpt_2.BytePairEncoding is BytePairEncoding


def test_get_bpe_from_files_returns_bpe_instance(tmp_path):
    encoder_path, vocab_path, _ = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert isinstance(bpe, BytePairEncoding)


def test_bpe_round_trip_ascii(tmp_path):
    encoder_path, vocab_path, _ = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    text = "hello world"
    tokens = bpe.encode(text)
    assert bpe.decode(tokens) == text


def test_bpe_round_trip_non_ascii(tmp_path):
    encoder_path, vocab_path, _ = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    text = "h\u00e9llo w\u00f6rld \u4f60\u597d"
    tokens = bpe.encode(text)
    assert bpe.decode(tokens) == text


def test_bpe_encode_returns_ids_within_vocab(tmp_path):
    encoder_path, vocab_path, token_dict = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    tokens = bpe.encode("some text to encode")
    assert len(tokens) > 0
    max_id = max(token_dict.values())
    assert all(isinstance(t, int) for t in tokens)
    assert all(0 <= t <= max_id for t in tokens)


def test_bpe_applies_learned_merge(tmp_path):
    encoder_path, vocab_path, _ = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert bpe.encode("he") == [257]


def test_bpe_empty_string_round_trip(tmp_path):
    encoder_path, vocab_path, _ = _write_bpe_files(tmp_path)
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert bpe.decode(bpe.encode("")) == ""


# ---------------------------------------------------------------------------
# GPT_2
# ---------------------------------------------------------------------------

def test_gpt2_is_exported_module_class():
    assert isinstance(GPT_2, type)
    assert torch_gpt_2.GPT_2 is GPT_2


def test_gpt2_forward_output_shape():
    model = _make_model(n_vocab=50, n_ctx=16)
    inputs = torch.randint(0, 50, (2, 8), dtype=torch.long)
    with torch.no_grad():
        logits = model(inputs)
    assert logits.shape == (2, 8, 50)


def test_gpt2_is_causal():
    model = _make_model(n_vocab=50, n_ctx=16)
    base = torch.randint(0, 50, (1, 10), dtype=torch.long)
    changed = base.clone()
    changed[0, 7:] = (changed[0, 7:] + 1) % 50
    with torch.no_grad():
        out_base = model(base)
        out_changed = model(changed)
    assert torch.allclose(out_base[:, :7], out_changed[:, :7], atol=1e-5)


def test_gpt2_eval_is_deterministic():
    model = _make_model(n_vocab=50, n_ctx=16)
    inputs = torch.randint(0, 50, (1, 6), dtype=torch.long)
    with torch.no_grad():
        first = model(inputs)
        second = model(inputs)
    assert torch.allclose(first, second)


def test_gpt2_parameters_require_grad():
    model = _make_model()
    params = list(model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_output_length():
    model = _make_model(n_vocab=50, n_ctx=16)
    context = torch.randint(0, 50, (1, 3), dtype=torch.long)
    out = generate(model, context, 5)
    assert out.shape[-1] == 3 + 5


def test_generate_preserves_context_prefix():
    model = _make_model(n_vocab=50, n_ctx=16)
    context = torch.randint(0, 50, (1, 4), dtype=torch.long)
    out = generate(model, context, 4)
    assert torch.equal(out[:, :4], context)


def test_generate_tokens_within_vocab():
    model = _make_model(n_vocab=50, n_ctx=16)
    context = torch.randint(0, 50, (1, 2), dtype=torch.long)
    out = generate(model, context, 6)
    assert int(out.min()) >= 0
    assert int(out.max()) < 50
