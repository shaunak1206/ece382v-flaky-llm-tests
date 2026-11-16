import inspect

import pytest
import torch

from torch_gpt_2 import BytePairEncoding, GPT_2, generate

try:
    from torch_gpt_2 import gelu
except ImportError:  # pragma: no cover
    from torch_gpt_2.net import gelu


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
    return dict(zip(bs, map(chr, cs)))


def _make_bpe():
    chars = sorted(_bytes_to_unicode().values())
    token_dict = {ch: i for i, ch in enumerate(chars)}
    for tok in ['he', 'll', 'hell']:
        token_dict[tok] = len(token_dict)
    bpe_rank = {('h', 'e'): 0, ('l', 'l'): 1, ('he', 'll'): 2}
    return BytePairEncoding(token_dict, bpe_rank), token_dict


_NAME_CANDIDATES = {
    'vocab': ['vocab_size', 'n_vocab'],
    'ctx': ['n_ctx', 'n_positions', 'max_len', 'seq_len'],
    'embd': ['n_embd', 'n_embed', 'embed_dim', 'hidden_dim', 'd_model', 'n_embedding'],
    'head': ['n_head', 'num_heads', 'head_num', 'n_heads'],
    'layer': ['n_layer', 'num_layers', 'n_layers', 'layer_num'],
}


def _make_model(vocab_size, n_ctx=32, n_embd=8, n_head=2, n_layer=2):
    values = {'vocab': vocab_size, 'ctx': n_ctx, 'embd': n_embd,
              'head': n_head, 'layer': n_layer}
    params = inspect.signature(GPT_2.__init__).parameters
    kwargs = {}
    matched = set()
    for key, names in _NAME_CANDIDATES.items():
        for name in names:
            if name in params:
                kwargs[name] = values[key]
                matched.add(name)
                break
    for name, p in params.items():
        if name == 'self' or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name not in matched and p.default is inspect.Parameter.empty:
            pytest.skip('Unrecognized required constructor argument: %s' % name)
    torch.manual_seed(0)
    model = GPT_2(**kwargs)
    model.eval()
    return model


def _logits(model, ids):
    with torch.no_grad():
        out = model(ids)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


@pytest.fixture(scope='module')
def bpe_and_dict():
    return _make_bpe()


@pytest.fixture(scope='module')
def model(bpe_and_dict):
    _, token_dict = bpe_and_dict
    return _make_model(len(token_dict))


# ---------------- gelu ----------------

def test_gelu_zero():
    out = gelu(torch.zeros(3))
    assert torch.allclose(out, torch.zeros(3), atol=1e-7)


def test_gelu_shape_preserved():
    x = torch.randn(2, 3, 4)
    assert gelu(x).shape == x.shape


def test_gelu_large_values():
    x = torch.tensor([10.0, -10.0])
    out = gelu(x)
    assert out[0].item() == pytest.approx(10.0, abs=1e-4)
    assert out[1].item() == pytest.approx(0.0, abs=1e-4)


def test_gelu_matches_tanh_formula():
    x = torch.linspace(-3, 3, 25)
    expected = 0.5 * x * (1 + torch.tanh(
        (2 / torch.pi if hasattr(torch, 'pi') else 0.6366197723675814) ** 0.5
        * (x + 0.044715 * x ** 3)))
    assert torch.allclose(gelu(x), expected, atol=1e-4)


def test_gelu_known_value_and_gradient():
    x = torch.tensor([1.0], requires_grad=True)
    y = gelu(x)
    assert y.item() == pytest.approx(0.8412, abs=1e-3)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------- BytePairEncoding ----------------

def test_bpe_encode_applies_merges(bpe_and_dict):
    bpe, td = bpe_and_dict
    assert bpe.encode('hello') == [td['hell'], td['o']]


def test_bpe_encode_without_merge_is_per_char(bpe_and_dict):
    bpe, td = bpe_and_dict
    assert bpe.encode('abc') == [td['a'], td['b'], td['c']]


def test_bpe_empty_string(bpe_and_dict):
    bpe, _ = bpe_and_dict
    assert bpe.encode('') == []
    assert bpe.decode([]) == ''


@pytest.mark.parametrize('text', [
    'hello',
    'hello world',
    'Hello, World! 123',
    'héllo wörld',
    '你好 world',
    'line1\nline2\ttab',
])
def test_bpe_roundtrip(bpe_and_dict, text):
    bpe, _ = bpe_and_dict
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_indices_are_valid_ints(bpe_and_dict):
    bpe, td = bpe_and_dict
    ids = bpe.encode('hello there')
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < len(td) for i in ids)


# ---------------- GPT_2 ----------------

def test_gpt2_is_torch_module_with_parameters(model):
    assert isinstance(model, torch.nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_gpt2_output_shape(model, bpe_and_dict):
    _, td = bpe_and_dict
    ids = torch.randint(0, len(td), (3, 7))
    out = _logits(model, ids)
    assert out.shape[0] == 3
    assert out.shape[-1] == len(td)
    assert torch.isfinite(out).all()


def test_gpt2_deterministic_in_eval(model, bpe_and_dict):
    _, td = bpe_and_dict
    ids = torch.randint(0, len(td), (2, 5))
    assert torch.allclose(_logits(model, ids), _logits(model, ids))


def test_gpt2_causal(model, bpe_and_dict):
    _, td = bpe_and_dict
    a = torch.randint(0, len(td), (1, 6))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % len(td)
    out_a = _logits(model, a)
    out_b = _logits(model, b)
    if out_a.dim() != 3 or out_a.shape[1] != 6:
        pytest.skip('Model does not return per-position logits')
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)


# ---------------- generate ----------------

def test_generate_returns_string(model, bpe_and_dict):
    bpe, _ = bpe_and_dict
    result = generate(model, bpe, 'hello', length=3)
    assert isinstance(result, str)


def test_generate_greedy_is_deterministic(model, bpe_and_dict):
    bpe, _ = bpe_and_dict
    r1 = generate(model, bpe, 'hello', length=4, top_k=1)
    r2 = generate(model, bpe, 'hello', length=4, top_k=1)
    assert r1 == r2


def test_generate_batch_returns_list(model, bpe_and_dict):
    bpe, _ = bpe_and_dict
    result = generate(model, bpe, ['hello', 'he'], length=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(r, str) for r in result)
