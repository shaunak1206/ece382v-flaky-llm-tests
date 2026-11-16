import math

import pytest
import torch

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _make_bpe(with_merge=False):
    chars = [c for _, c in sorted(_bytes_to_unicode().items())]
    token_dict = {c: i for i, c in enumerate(chars)}
    bpe_rank = {}
    if with_merge:
        token_dict["ab"] = len(token_dict)
        bpe_rank[("a", "b")] = 0
    return BytePairEncoding(token_dict, bpe_rank)


def _make_model(n_vocab=256):
    torch.manual_seed(0)
    model = GPT_2(
        n_vocab=n_vocab,
        n_ctx=32,
        n_embd=16,
        n_head=2,
        n_layer=2,
    )
    model.eval()
    return model


def _logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


# ---------------------------- gelu ----------------------------

def test_gelu_zero():
    out = gelu(torch.zeros(3))
    assert torch.allclose(out, torch.zeros(3), atol=1e-7)


def test_gelu_matches_tanh_formula():
    x = torch.linspace(-4, 4, 41)
    expected = 0.5 * x * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3))
    )
    assert torch.allclose(gelu(x), expected, atol=1e-5)


def test_gelu_large_positive_is_identity_like():
    x = torch.tensor([10.0, 20.0])
    assert torch.allclose(gelu(x), x, atol=1e-4)


def test_gelu_large_negative_is_near_zero():
    x = torch.tensor([-10.0, -20.0])
    assert torch.allclose(gelu(x), torch.zeros(2), atol=1e-4)


def test_gelu_preserves_shape():
    x = torch.randn(2, 3, 4)
    assert gelu(x).shape == x.shape


def test_gelu_gradient_flows():
    x = torch.randn(5, requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------- BytePairEncoding ----------------------------

def test_bpe_encode_returns_list_of_ints():
    bpe = _make_bpe()
    ids = bpe.encode("hello")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_bpe_roundtrip_ascii():
    bpe = _make_bpe()
    text = "Hello, world! 123"
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_roundtrip_unicode():
    bpe = _make_bpe()
    text = "héllo 世界"
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_merge_is_applied():
    bpe = _make_bpe(with_merge=True)
    ids = bpe.encode("ab")
    assert len(ids) == 1
    assert bpe.decode(ids) == "ab"


def test_bpe_without_merge_gives_per_char_tokens():
    bpe = _make_bpe()
    assert len(bpe.encode("ab")) == 2


# ---------------------------- GPT_2 ----------------------------

def test_gpt2_output_shape():
    model = _make_model()
    x = torch.randint(0, 256, (2, 7))
    with torch.no_grad():
        logits = _logits(model(x))
    assert logits.shape[0] == 2
    assert logits.shape[-1] == 256
    assert logits.shape[-2] == 7


def test_gpt2_deterministic_in_eval_mode():
    model = _make_model()
    x = torch.randint(0, 256, (1, 5))
    with torch.no_grad():
        a = _logits(model(x))
        b = _logits(model(x))
    assert torch.allclose(a, b)


def test_gpt2_is_causal():
    model = _make_model()
    x = torch.randint(0, 256, (1, 6))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 256
    with torch.no_grad():
        a = _logits(model(x))
        b = _logits(model(y))
    # Changing the last token must not affect earlier positions.
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


# ---------------------------- generate ----------------------------

def test_generate_returns_one_string_per_input():
    model = _make_model()
    bpe = _make_bpe()
    outputs = generate(model, bpe, ["abc", "hello world"], length=4, top_k=1)
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


def test_generate_greedy_is_deterministic():
    model = _make_model()
    bpe = _make_bpe()
    a = generate(model, bpe, ["hello"], length=5, top_k=1)
    b = generate(model, bpe, ["hello"], length=5, top_k=1)
    assert a == b
