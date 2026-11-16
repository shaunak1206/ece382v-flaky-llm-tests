import inspect
import math

import pytest
import torch

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_vocab(with_merge=False):
    tokens = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    tokens.append('\u0120')  # byte-level representation of the space character
    if with_merge:
        tokens.append('ll')
    return {token: i for i, token in enumerate(tokens)}


def _make_bpe(with_merge=False):
    token_dict = _make_vocab(with_merge)
    bpe_rank = {('l', 'l'): 0} if with_merge else {}
    return BytePairEncoding(token_dict, bpe_rank)


def _make_model(n_vocab, n_ctx=16, n_embd=8, n_head=2, n_layer=2, seed=0):
    torch.manual_seed(seed)
    wanted = {
        'n_vocab': n_vocab,
        'n_ctx': n_ctx,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
    }
    params = inspect.signature(GPT_2.__init__).parameters
    kwargs = {k: v for k, v in wanted.items() if k in params}
    model = GPT_2(**kwargs)
    model.eval()
    return model


def _logits(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


# --------------------------------------------------------------------------
# gelu
# --------------------------------------------------------------------------

def test_gelu_exported():
    assert callable(gelu)
    assert hasattr(torch_gpt_2, 'gelu')


def test_gelu_zero():
    out = gelu(torch.zeros(5))
    assert torch.allclose(out, torch.zeros(5), atol=1e-6)


def test_gelu_matches_reference_formula():
    x = torch.linspace(-4.0, 4.0, 41)
    expected = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    assert torch.allclose(gelu(x), expected, atol=1e-2)


def test_gelu_asymptotics():
    big = torch.tensor([10.0, 20.0])
    assert torch.allclose(gelu(big), big, atol=1e-3)
    small = torch.tensor([-10.0, -20.0])
    assert torch.allclose(gelu(small), torch.zeros(2), atol=1e-3)


def test_gelu_preserves_shape_and_dtype():
    x = torch.randn(2, 3, 4)
    out = gelu(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_gelu_is_differentiable():
    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


# --------------------------------------------------------------------------
# BytePairEncoding
# --------------------------------------------------------------------------

def test_bpe_encode_returns_list_of_ints():
    bpe = _make_bpe()
    ids = bpe.encode('hello')
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_bpe_roundtrip_without_merges():
    bpe = _make_bpe()
    text = 'hello world'
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_roundtrip_with_merges():
    bpe = _make_bpe(with_merge=True)
    text = 'hello all'
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_merge_reduces_token_count():
    plain = _make_bpe()
    merged = _make_bpe(with_merge=True)
    assert len(plain.encode('ll')) == 2
    ids = merged.encode('ll')
    assert ids == [_make_vocab(with_merge=True)['ll']]


# --------------------------------------------------------------------------
# GPT_2
# --------------------------------------------------------------------------

def test_gpt2_output_shape():
    n_vocab = len(_make_vocab())
    model = _make_model(n_vocab)
    x = torch.randint(0, n_vocab, (3, 7))
    with torch.no_grad():
        out = _logits(model(x))
    assert out.shape == (3, 7, n_vocab)


def test_gpt2_is_torch_module_with_parameters():
    model = _make_model(len(_make_vocab()))
    assert isinstance(model, torch.nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_gpt2_eval_is_deterministic():
    n_vocab = len(_make_vocab())
    model = _make_model(n_vocab)
    x = torch.randint(0, n_vocab, (2, 6))
    with torch.no_grad():
        a = _logits(model(x))
        b = _logits(model(x))
    assert torch.allclose(a, b)
    assert torch.isfinite(a).all()


def test_gpt2_is_causal():
    n_vocab = len(_make_vocab())
    model = _make_model(n_vocab)
    x1 = torch.randint(0, n_vocab, (1, 8))
    x2 = x1.clone()
    x2[0, 5:] = (x2[0, 5:] + 1) % n_vocab
    with torch.no_grad():
        o1 = _logits(model(x1))
        o2 = _logits(model(x2))
    assert torch.allclose(o1[:, :5], o2[:, :5], atol=1e-5)


def test_gpt2_batch_items_independent():
    n_vocab = len(_make_vocab())
    model = _make_model(n_vocab)
    x = torch.randint(0, n_vocab, (2, 6))
    with torch.no_grad():
        both = _logits(model(x))
        single = _logits(model(x[:1]))
    assert torch.allclose(both[:1], single, atol=1e-5)


def test_gpt2_backward_produces_gradients():
    n_vocab = len(_make_vocab())
    model = _make_model(n_vocab)
    model.train()
    x = torch.randint(0, n_vocab, (2, 5))
    _logits(model(x)).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------

def test_generate_returns_list_of_strings():
    bpe = _make_bpe()
    model = _make_model(len(_make_vocab()))
    out = generate(model, bpe, ['hello world'], length=4, top_k=1)
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], str)


def test_generate_greedy_is_deterministic():
    bpe = _make_bpe()
    model = _make_model(len(_make_vocab()))
    a = generate(model, bpe, ['hello world'], length=5, top_k=1)
    b = generate(model, bpe, ['hello world'], length=5, top_k=1)
    assert a == b


def test_generate_longer_length_gives_no_shorter_output():
    bpe = _make_bpe()
    model = _make_model(len(_make_vocab()))
    short = generate(model, bpe, ['hello world'], length=2, top_k=1)[0]
    long = generate(model, bpe, ['hello world'], length=6, top_k=1)[0]
    assert len(long) >= len(short)


def test_generate_multiple_prompts():
    bpe = _make_bpe()
    model = _make_model(len(_make_vocab()))
    out = generate(model, bpe, ['abc def', 'xyz uvw'], length=3, top_k=1)
    assert len(out) == 2
    assert all(isinstance(s, str) for s in out)
