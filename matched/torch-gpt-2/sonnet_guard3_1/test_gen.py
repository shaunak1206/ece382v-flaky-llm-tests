import inspect
import math

import numpy as np
import pytest
import torch

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


VOCAB_SIZE = 32


def _build_bpe(with_merge=True):
    token_dict = {}
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
        token_dict[ch] = i
    token_dict[chr(288)] = 26  # byte-level representation of the space
    bpe_rank = {}
    if with_merge:
        token_dict["he"] = 27
        bpe_rank[("h", "e")] = 0
    return BytePairEncoding(token_dict=token_dict, bpe_rank=bpe_rank)


def _build_model():
    wanted = {
        "vocab_size": VOCAB_SIZE,
        "pos_num": 16,
        "embed_dim": 8,
        "encoder_num": 1,
        "head_num": 2,
        "feed_forward_dim": 16,
    }
    params = inspect.signature(GPT_2.__init__).parameters
    kwargs = {k: v for k, v in wanted.items() if k in params}
    model = GPT_2(**kwargs)
    model.eval()
    return model


def _logits(model, ids):
    out = model(ids)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _generate(model, bpe, texts, length):
    params = inspect.signature(generate).parameters
    kwargs = {}
    if "length" in params:
        kwargs["length"] = length
    if "top_k" in params:
        kwargs["top_k"] = 1
    return generate(model, bpe, texts, **kwargs)


# ---------------- gelu ----------------

def test_gelu_zero_is_zero():
    torch.manual_seed(0)
    np.random.seed(0)
    out = gelu(torch.zeros(3))
    assert torch.allclose(out, torch.zeros(3), atol=1e-7)


def test_gelu_large_positive_is_identity():
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.tensor([10.0, 20.0, 50.0])
    assert torch.allclose(gelu(x), x, atol=1e-4)


def test_gelu_large_negative_is_zero():
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.tensor([-10.0, -20.0, -50.0])
    assert torch.allclose(gelu(x), torch.zeros(3), atol=1e-4)


def test_gelu_matches_tanh_formula():
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.linspace(-4, 4, 17)
    expected = torch.tensor([
        0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)))
        for v in x.tolist()
    ])
    assert torch.allclose(gelu(x), expected, atol=1e-4)


def test_gelu_preserves_shape_and_known_value():
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.ones(2, 3, 4)
    out = gelu(x)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.full((2, 3, 4), 0.8412), atol=1e-3)


def test_gelu_supports_backward():
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # derivative at 0 is 0.5
    assert abs(x.grad[1].item() - 0.5) < 1e-4


# ---------------- BytePairEncoding ----------------

def test_bpe_roundtrip_simple_text():
    torch.manual_seed(0)
    np.random.seed(0)
    bpe = _build_bpe()
    text = "hello world"
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_roundtrip_without_merges():
    torch.manual_seed(0)
    np.random.seed(0)
    bpe = _build_bpe(with_merge=False)
    text = "abc def"
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_merge_reduces_token_count():
    torch.manual_seed(0)
    np.random.seed(0)
    merged = _build_bpe(with_merge=True)
    unmerged = _build_bpe(with_merge=False)
    assert len(unmerged.encode("he")) == 2
    assert len(merged.encode("he")) == 1


def test_bpe_encode_is_deterministic():
    torch.manual_seed(0)
    np.random.seed(0)
    bpe = _build_bpe()
    assert bpe.encode("the hen") == bpe.encode("the hen")


# ---------------- GPT_2 ----------------

def test_gpt2_output_shape_and_finite():
    torch.manual_seed(0)
    np.random.seed(0)
    model = _build_model()
    ids = torch.randint(0, VOCAB_SIZE, (2, 5))
    with torch.no_grad():
        out = _logits(model, ids)
    assert out.shape[0] == 2
    assert out.shape[1] == 5
    assert torch.isfinite(out).all()


def test_gpt2_eval_forward_is_deterministic():
    torch.manual_seed(0)
    np.random.seed(0)
    model = _build_model()
    ids = torch.randint(0, VOCAB_SIZE, (1, 6))
    with torch.no_grad():
        a = _logits(model, ids)
        b = _logits(model, ids)
    assert torch.allclose(a, b, atol=1e-6)


def test_gpt2_is_causal():
    torch.manual_seed(0)
    np.random.seed(0)
    model = _build_model()
    ids1 = torch.randint(0, VOCAB_SIZE, (1, 6))
    ids2 = ids1.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % VOCAB_SIZE
    with torch.no_grad():
        out1 = _logits(model, ids1)
        out2 = _logits(model, ids2)
    assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-4)


def test_gpt2_has_trainable_parameters():
    torch.manual_seed(0)
    np.random.seed(0)
    model = _build_model()
    params = list(model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


# ---------------- generate ----------------

def test_generate_returns_one_string_per_input():
    torch.manual_seed(0)
    np.random.seed(0)
    model = _build_model()
    bpe = _build_bpe()
    outputs = _generate(model, bpe, ["he", "abc"], 3)
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


def test_generate_is_deterministic_with_top_k_one():
    torch.manual_seed(0)
    np.random.seed(0)
    model = _build_model()
    bpe = _build_bpe()
    first = _generate(model, bpe, ["he"], 4)
    second = _generate(model, bpe, ["he"], 4)
    assert first == second


def test_generate_is_exported_from_package():
    torch.manual_seed(0)
    np.random.seed(0)
    assert callable(torch_gpt_2.generate)
    assert callable(torch_gpt_2.gelu)
    assert torch_gpt_2.GPT_2 is GPT_2
