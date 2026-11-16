import math

import pytest
import torch

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


def _build_token_dict(extra_tokens=()):
    chars = [chr(c) for c in range(33, 127)] + ['\u0120']  # 'Ġ' is the byte-level space
    token_dict = {}
    for ch in chars:
        token_dict[ch] = len(token_dict)
    for tok in extra_tokens:
        if tok not in token_dict:
            token_dict[tok] = len(token_dict)
    return token_dict


@pytest.fixture
def plain_bpe():
    return BytePairEncoding(_build_token_dict(), {})


@pytest.fixture
def merge_bpe():
    token_dict = _build_token_dict(extra_tokens=['ab', 'abc'])
    bpe_rank = {('a', 'b'): 0, ('ab', 'c'): 1}
    return BytePairEncoding(token_dict, bpe_rank)


def _small_model(n_vocab):
    torch.manual_seed(0)
    model = GPT_2(n_vocab=n_vocab, n_ctx=32, n_embd=16, n_head=2, n_layer=2)
    model.eval()
    return model


def _logits(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


# ---------------------------------------------------------------- gelu

def test_gelu_zero_is_zero():
    out = gelu(torch.zeros(3))
    assert torch.allclose(out, torch.zeros(3), atol=1e-7)


def test_gelu_matches_tanh_formula():
    x = torch.linspace(-4, 4, 41)
    expected = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    assert torch.allclose(gelu(x), expected, atol=1e-5)


def test_gelu_large_positive_is_identity_like():
    x = torch.tensor([10.0, 20.0])
    assert torch.allclose(gelu(x), x, atol=1e-4)


def test_gelu_large_negative_is_zero_like():
    x = torch.tensor([-10.0, -20.0])
    assert torch.allclose(gelu(x), torch.zeros(2), atol=1e-4)


def test_gelu_preserves_shape_and_dtype():
    x = torch.randn(2, 3, 4)
    out = gelu(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_gelu_is_differentiable():
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------- BytePairEncoding

def test_bpe_encode_returns_int_list(plain_bpe):
    ids = plain_bpe.encode('abc')
    assert isinstance(ids, list)
    assert len(ids) == 3
    assert all(isinstance(i, int) for i in ids)


def test_bpe_roundtrip_with_space(plain_bpe):
    text = 'hello world'
    assert plain_bpe.decode(plain_bpe.encode(text)) == text


def test_bpe_merges_reduce_token_count(plain_bpe, merge_bpe):
    assert len(plain_bpe.encode('abc')) == 3
    assert len(merge_bpe.encode('abc')) == 1
    assert len(merge_bpe.encode('ab')) == 1


def test_bpe_merged_roundtrip(merge_bpe):
    text = 'abc ab ba'
    assert merge_bpe.decode(merge_bpe.encode(text)) == text


# ---------------------------------------------------------------- GPT_2

def test_gpt2_forward_output_shape():
    model = _small_model(50)
    x = torch.randint(0, 50, (2, 7))
    with torch.no_grad():
        logits = _logits(model(x))
    assert logits.shape[0] == 2
    assert logits.shape[-1] == 50
    assert logits.shape[-2] == 7


def test_gpt2_forward_is_deterministic_in_eval():
    model = _small_model(50)
    x = torch.randint(0, 50, (1, 5))
    with torch.no_grad():
        a = _logits(model(x))
        b = _logits(model(x))
    assert torch.allclose(a, b)
    assert torch.isfinite(a).all()


def test_gpt2_is_causal():
    model = _small_model(50)
    x = torch.randint(0, 50, (1, 6))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 50
    with torch.no_grad():
        a = _logits(model(x))
        b = _logits(model(y))
    # changing the last token must not affect earlier positions
    assert torch.allclose(a[0, :5], b[0, :5], atol=1e-5)


def test_gpt2_backward_produces_gradients():
    model = _small_model(50)
    model.train()
    x = torch.randint(0, 50, (2, 4))
    _logits(model(x)).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ---------------------------------------------------------------- generate

def test_generate_returns_strings_for_list_input(plain_bpe):
    model = _small_model(len(plain_bpe.token_dict))
    with torch.no_grad():
        outputs = generate(model, plain_bpe, ['ab', 'cd ef'], length=4, top_k=1)
    assert isinstance(outputs, list)
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


def test_generate_greedy_is_deterministic(plain_bpe):
    model = _small_model(len(plain_bpe.token_dict))
    with torch.no_grad():
        first = generate(model, plain_bpe, ['abc'], length=5, top_k=1)
        second = generate(model, plain_bpe, ['abc'], length=5, top_k=1)
    assert first == second


def test_generate_single_string_input(plain_bpe):
    model = _small_model(len(plain_bpe.token_dict))
    with torch.no_grad():
        output = generate(model, plain_bpe, 'abc', length=3, top_k=1)
    if isinstance(output, list):
        assert len(output) == 1
        output = output[0]
    assert isinstance(output, str)
