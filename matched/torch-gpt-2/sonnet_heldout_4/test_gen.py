import inspect
import json

import pytest
import torch

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate, get_bpe_from_files


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


MERGES = [("h", "e"), ("l", "l")]


def _build_token_dict():
    token_dict = {}
    for ch in _bytes_to_unicode().values():
        token_dict[ch] = len(token_dict)
    for a, b in MERGES:
        token_dict[a + b] = len(token_dict)
    return token_dict


@pytest.fixture
def token_dict():
    return _build_token_dict()


@pytest.fixture
def bpe(token_dict):
    rank = {pair: i for i, pair in enumerate(MERGES)}
    return BytePairEncoding(token_dict, rank)


@pytest.fixture
def bpe_no_merges(token_dict):
    return BytePairEncoding(token_dict, {})


def _make_model(n_vocab, n_ctx=32, n_embd=16, n_head=2, n_layer=2):
    values = {
        "n_vocab": n_vocab,
        "vocab_size": n_vocab,
        "n_ctx": n_ctx,
        "n_positions": n_ctx,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
    }
    sig = inspect.signature(GPT_2.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in values:
            kwargs[name] = values[name]
        elif param.default is inspect.Parameter.empty:
            pytest.skip("Cannot construct GPT_2: unknown required argument %r" % name)
    model = GPT_2(**kwargs)
    model.eval()
    return model


def _logits(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


# ---------------------------------------------------------------- gelu

def test_gelu_zero():
    out = gelu(torch.zeros(3))
    assert torch.allclose(out, torch.zeros(3), atol=1e-7)


def test_gelu_large_positive_is_identity():
    x = torch.tensor([10.0, 20.0])
    assert torch.allclose(gelu(x), x, atol=1e-4)


def test_gelu_large_negative_is_zero():
    x = torch.tensor([-10.0, -20.0])
    assert torch.allclose(gelu(x), torch.zeros(2), atol=1e-4)


def test_gelu_matches_tanh_approximation():
    import math

    x = torch.linspace(-3, 3, 25)
    expected = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    assert torch.allclose(gelu(x), expected, atol=1e-4)


def test_gelu_preserves_shape():
    x = torch.randn(2, 3, 4)
    assert gelu(x).shape == x.shape


def test_gelu_known_value_and_gradient():
    x = torch.tensor([1.0], requires_grad=True)
    y = gelu(x)
    assert abs(y.item() - 0.8412) < 1e-2
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------- BytePairEncoding

def test_bpe_encode_returns_int_list(bpe):
    ids = bpe.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_bpe_merges_applied(bpe, token_dict):
    ids = bpe.encode("hello")
    assert ids == [token_dict["he"], token_dict["ll"], token_dict["o"]]


def test_bpe_without_merges_is_bytewise(bpe_no_merges, token_dict):
    ids = bpe_no_merges.encode("hello")
    assert ids == [token_dict[c] for c in "hello"]


def test_bpe_roundtrip_ascii(bpe):
    text = "hello world, hello again!"
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_roundtrip_unicode(bpe):
    text = "héllo 世界"
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_empty_text(bpe):
    assert bpe.encode("") == []
    assert bpe.decode([]) == ""


def test_get_bpe_from_files(tmp_path, token_dict):
    encoder_path = tmp_path / "encoder.json"
    vocab_path = tmp_path / "vocab.bpe"
    encoder_path.write_text(json.dumps(token_dict), encoding="utf-8")
    lines = ["#version: 0.2"] + ["%s %s" % pair for pair in MERGES]
    vocab_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loaded = get_bpe_from_files(str(encoder_path), str(vocab_path))
    assert isinstance(loaded, BytePairEncoding)
    assert loaded.encode("hello") == [token_dict["he"], token_dict["ll"], token_dict["o"]]
    assert loaded.decode(loaded.encode("hello there")) == "hello there"


# ---------------------------------------------------------------- GPT_2

def test_gpt2_is_torch_module():
    model = _make_model(n_vocab=30)
    assert isinstance(model, torch.nn.Module)
    assert len(list(model.parameters())) > 0


def test_gpt2_forward_shape_and_finite():
    torch.manual_seed(0)
    n_vocab = 30
    model = _make_model(n_vocab=n_vocab)
    x = torch.randint(0, n_vocab, (3, 7))
    with torch.no_grad():
        out = _logits(model(x))
    assert out.shape[0] == 3
    assert out.shape[1] == 7
    assert out.shape[-1] == n_vocab
    assert torch.isfinite(out).all()


def test_gpt2_causal_prefix_independent_of_future():
    torch.manual_seed(1)
    n_vocab = 30
    model = _make_model(n_vocab=n_vocab)
    x1 = torch.randint(0, n_vocab, (1, 6))
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % n_vocab
    with torch.no_grad():
        o1 = _logits(model(x1))
        o2 = _logits(model(x2))
    assert torch.allclose(o1[:, :5], o2[:, :5], atol=1e-4)


# ---------------------------------------------------------------- generate

def test_generate_returns_list_of_strings(bpe, token_dict):
    torch.manual_seed(0)
    model = _make_model(n_vocab=len(token_dict))
    outputs = generate(model, bpe, ["hello"], length=4, top_k=1)
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)


def test_generate_batch_and_deterministic_top1(bpe, token_dict):
    torch.manual_seed(0)
    model = _make_model(n_vocab=len(token_dict))
    texts = ["hello", "he"]
    first = generate(model, bpe, texts, length=3, top_k=1)
    second = generate(model, bpe, texts, length=3, top_k=1)
    assert len(first) == 2
    assert first == second
