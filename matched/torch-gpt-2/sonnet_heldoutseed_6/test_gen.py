import inspect
import math

import pytest
import torch

import torch_gpt_2
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


MERGES = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]


def _make_bpe():
    chars = sorted(_bytes_to_unicode().values())
    token_dict = {ch: i for i, ch in enumerate(chars)}
    for a, b in MERGES:
        token_dict[a + b] = len(token_dict)
    # Fix merges so that each intermediate token exists
    bpe_rank = {pair: i for i, pair in enumerate(MERGES)}
    return BytePairEncoding(token_dict, bpe_rank), token_dict


def _make_model(n_vocab, n_ctx=32, n_embd=16, n_head=2, n_layer=2):
    candidates = {
        "n_vocab": n_vocab,
        "n_ctx": n_ctx,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
    }
    params = inspect.signature(GPT_2.__init__).parameters
    kwargs = {k: v for k, v in candidates.items() if k in params}
    model = GPT_2(**kwargs)
    model.eval()
    return model


def _logits(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


# ---------------------------------------------------------------- gelu

def test_gelu_zero():
    out = gelu(torch.zeros(3))
    assert torch.allclose(out, torch.zeros(3), atol=1e-7)


def test_gelu_matches_tanh_formula():
    x = torch.linspace(-4, 4, 41)
    expected = 0.5 * x * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3))
    )
    assert torch.allclose(gelu(x), expected, atol=1e-4)


def test_gelu_asymptotics():
    big = torch.tensor([10.0, 20.0])
    assert torch.allclose(gelu(big), big, atol=1e-3)
    neg = torch.tensor([-10.0, -20.0])
    assert torch.allclose(gelu(neg), torch.zeros(2), atol=1e-3)


def test_gelu_preserves_shape():
    x = torch.randn(2, 3, 4)
    assert gelu(x).shape == x.shape


def test_gelu_is_differentiable():
    x = torch.randn(5, requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------- BytePairEncoding

def test_bpe_encode_returns_ints():
    bpe, token_dict = _make_bpe()
    ids = bpe.encode("hello world")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < len(token_dict) for i in ids)


def test_bpe_roundtrip_ascii():
    bpe, _ = _make_bpe()
    text = "hello world, this is a test."
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_roundtrip_unicode():
    bpe, _ = _make_bpe()
    text = "héllo wörld 你好"
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_merges_reduce_token_count():
    bpe, token_dict = _make_bpe()
    ids = bpe.encode("hello")
    assert ids == [token_dict["hello"]]


def test_bpe_unmerged_text_is_character_level():
    bpe, token_dict = _make_bpe()
    ids = bpe.encode("xyz")
    assert ids == [token_dict["x"], token_dict["y"], token_dict["z"]]


def test_bpe_empty_text():
    bpe, _ = _make_bpe()
    assert list(bpe.encode("")) == []
    assert bpe.decode([]) == ""


# ---------------------------------------------------------------- GPT_2

def test_gpt2_is_torch_module_with_parameters():
    model = _make_model(n_vocab=300)
    assert isinstance(model, torch.nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_gpt2_output_shape():
    n_vocab = 300
    model = _make_model(n_vocab=n_vocab)
    x = torch.randint(0, n_vocab, (2, 7))
    with torch.no_grad():
        out = _logits(model(x))
    assert out.shape[0] == 2
    assert out.shape[-1] == n_vocab
    assert torch.isfinite(out).all()


def test_gpt2_eval_is_deterministic():
    n_vocab = 300
    model = _make_model(n_vocab=n_vocab)
    x = torch.randint(0, n_vocab, (1, 6))
    with torch.no_grad():
        a = _logits(model(x))
        b = _logits(model(x))
    assert torch.allclose(a, b)


def test_gpt2_batch_rows_independent():
    n_vocab = 300
    model = _make_model(n_vocab=n_vocab)
    row = torch.randint(0, n_vocab, (1, 5))
    x = torch.cat([row, row], dim=0)
    with torch.no_grad():
        out = _logits(model(x))
    assert torch.allclose(out[0], out[1], atol=1e-5)


def test_gpt2_causal_attention():
    n_vocab = 300
    model = _make_model(n_vocab=n_vocab)
    x1 = torch.randint(0, n_vocab, (1, 6))
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % n_vocab
    with torch.no_grad():
        o1 = _logits(model(x1))
        o2 = _logits(model(x2))
    if o1.dim() < 3 or o1.shape[-2] != 6:
        pytest.skip("model output does not expose every position")
    assert torch.allclose(o1[0, :-1], o2[0, :-1], atol=1e-5)


def test_gpt2_backward_produces_gradients():
    n_vocab = 300
    model = _make_model(n_vocab=n_vocab)
    model.train()
    x = torch.randint(0, n_vocab, (2, 5))
    out = _logits(model(x))
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ---------------------------------------------------------------- generate

def test_generate_returns_one_string_per_prompt():
    bpe, token_dict = _make_bpe()
    model = _make_model(n_vocab=len(token_dict))
    out = generate(model, bpe, ["hello", "world"], length=4, top_k=1)
    assert len(out) == 2
    assert all(isinstance(s, str) for s in out)


def test_generate_greedy_is_deterministic():
    bpe, token_dict = _make_bpe()
    model = _make_model(n_vocab=len(token_dict))
    a = generate(model, bpe, ["hello"], length=5, top_k=1)
    b = generate(model, bpe, ["hello"], length=5, top_k=1)
    assert a == b


def test_package_exports():
    for name in ("BytePairEncoding", "GPT_2", "gelu", "generate"):
        assert hasattr(torch_gpt_2, name)
