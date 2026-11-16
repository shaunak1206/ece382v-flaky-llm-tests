import inspect

import numpy as np
import pytest
import torch

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate

N_VOCAB = 16
N_CTX = 32


def _make_model(**overrides):
    wanted = dict(n_vocab=N_VOCAB, n_ctx=N_CTX, n_embd=8, n_head=2, n_layer=2)
    wanted.update(overrides)
    params = inspect.signature(GPT_2.__init__).parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = {k: v for k, v in wanted.items() if has_var_kw or k in params}
    model = GPT_2(**kwargs)
    model.eval()
    return model


def _logits(model, ids):
    with torch.no_grad():
        out = model(torch.tensor(ids, dtype=torch.long))
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


class FakeBPE(object):
    """Character-level tokenizer that records what it is asked to decode."""

    def __init__(self):
        self.decoded = []

    def encode(self, text):
        return [ord(c) % N_VOCAB for c in text]

    def decode(self, ids):
        ids = [int(i) for i in ids]
        self.decoded.append(ids)
        return ''.join(chr(ord('a') + i) for i in ids)


def _make_real_bpe():
    token_dict = {'a': 0, 'b': 1, 'c': 2, 'ab': 3, '\u0120': 4}
    bpe_rank = {('a', 'b'): 0}
    return BytePairEncoding(token_dict, bpe_rank)


# ---------------------------- gelu ----------------------------

def test_gelu_zero_is_zero():
    torch.manual_seed(0)
    out = gelu(torch.zeros(3))
    assert torch.allclose(out, torch.zeros(3), atol=1e-6)


def test_gelu_matches_tanh_formula():
    torch.manual_seed(0)
    x = torch.linspace(-4.0, 4.0, steps=33)
    expected = 0.5 * x * (1.0 + torch.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
    assert torch.allclose(gelu(x), expected, atol=1e-3)


def test_gelu_large_positive_and_negative():
    torch.manual_seed(0)
    out = gelu(torch.tensor([10.0, -10.0]))
    assert abs(out[0].item() - 10.0) < 1e-3
    assert abs(out[1].item()) < 1e-3


def test_gelu_preserves_shape():
    torch.manual_seed(0)
    x = torch.linspace(-1.0, 1.0, steps=24).reshape(2, 3, 4)
    assert gelu(x).shape == x.shape


def test_gelu_odd_symmetry_identity():
    torch.manual_seed(0)
    x = torch.linspace(-3.0, 3.0, steps=25)
    # gelu(x) - gelu(-x) == x
    assert torch.allclose(gelu(x) - gelu(-x), x, atol=1e-4)


def test_gelu_known_value_at_one():
    torch.manual_seed(0)
    out = gelu(torch.tensor([1.0]))
    assert abs(out.item() - 0.8412) < 2e-3


# ------------------------ BytePairEncoding ------------------------

def test_bpe_encode_merges_pair():
    bpe = _make_real_bpe()
    assert bpe.encode('ab') == [3]


def test_bpe_encode_unmerged_order():
    bpe = _make_real_bpe()
    assert bpe.encode('ba') == [1, 0]


def test_bpe_encode_single_chars():
    bpe = _make_real_bpe()
    assert bpe.encode('c') == [2]


def test_bpe_decode_basic():
    bpe = _make_real_bpe()
    assert bpe.decode([3, 2]) == 'abc'


def test_bpe_round_trip_with_space():
    bpe = _make_real_bpe()
    text = 'ab c'
    assert bpe.decode(bpe.encode(text)) == text


# ----------------------------- GPT_2 -----------------------------

def test_gpt2_output_shape():
    torch.manual_seed(0)
    model = _make_model()
    logits = _logits(model, [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    assert tuple(logits.shape) == (2, 5, N_VOCAB)


def test_gpt2_output_is_finite():
    torch.manual_seed(0)
    model = _make_model()
    logits = _logits(model, [[1, 2, 3]])
    assert torch.isfinite(logits).all()


def test_gpt2_eval_forward_deterministic():
    torch.manual_seed(0)
    model = _make_model()
    ids = [[3, 1, 4, 1, 5]]
    first = _logits(model, ids)
    second = _logits(model, ids)
    assert torch.allclose(first, second, atol=1e-6)


def test_gpt2_is_causal():
    torch.manual_seed(0)
    model = _make_model()
    a = _logits(model, [[1, 2, 3, 4]])
    b = _logits(model, [[1, 2, 3, 9]])
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-4)


def test_gpt2_has_parameters():
    torch.manual_seed(0)
    model = _make_model()
    assert sum(p.numel() for p in model.parameters()) > 0


# ----------------------------- generate -----------------------------

def test_generate_returns_one_string_per_text():
    torch.manual_seed(0)
    model = _make_model()
    bpe = FakeBPE()
    outputs = generate(model, bpe, ['abc', 'bcd'], length=3, top_k=1)
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


def test_generate_greedy_is_deterministic():
    torch.manual_seed(0)
    model = _make_model()
    first = generate(model, FakeBPE(), ['abc'], length=4, top_k=1)
    second = generate(model, FakeBPE(), ['abc'], length=4, top_k=1)
    assert first == second


def test_generate_token_count_and_range():
    torch.manual_seed(0)
    model = _make_model()
    bpe = FakeBPE()
    prompt = 'abc'
    length = 4
    generate(model, bpe, [prompt], length=length, top_k=1)
    assert len(bpe.decoded) >= 1
    ids = bpe.decoded[0]
    assert len(ids) in (length, len(prompt) + length)
    assert all(0 <= i < N_VOCAB for i in ids)
