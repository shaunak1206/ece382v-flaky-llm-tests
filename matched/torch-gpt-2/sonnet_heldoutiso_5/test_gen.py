import inspect
import json

import pytest
import torch

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate, get_bpe_from_files

SPACE = chr(288)  # byte-level BPE representation of ' '
LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def build_token_dict():
    token_dict = {c: i for i, c in enumerate(LETTERS)}
    token_dict[SPACE] = len(token_dict)
    token_dict['he'] = len(token_dict)
    token_dict['ll'] = len(token_dict)
    return token_dict


def build_bpe():
    token_dict = build_token_dict()
    bpe_rank = {('h', 'e'): 0, ('l', 'l'): 1}
    return BytePairEncoding(token_dict, bpe_rank)


def build_model(n_vocab, **extra):
    wanted = dict(n_vocab=n_vocab, n_ctx=32, n_embd=16, n_head=2, n_layer=2)
    wanted.update(extra)
    params = inspect.signature(GPT_2.__init__).parameters
    kwargs = {k: v for k, v in wanted.items() if k in params}
    torch.manual_seed(0)
    model = GPT_2(**kwargs)
    model.eval()
    return model


def logits_of(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


# ---------------------------------------------------------------- gelu

def test_gelu_zero():
    out = gelu(torch.zeros(3))
    assert torch.allclose(out, torch.zeros(3), atol=1e-6)


def test_gelu_known_value():
    out = gelu(torch.tensor([1.0]))
    assert out.item() == pytest.approx(0.8412, abs=1e-3)


def test_gelu_asymptotics():
    out = gelu(torch.tensor([10.0, -10.0]))
    assert out[0].item() == pytest.approx(10.0, abs=1e-3)
    assert out[1].item() == pytest.approx(0.0, abs=1e-3)


def test_gelu_matches_tanh_formula_and_shape():
    x = torch.linspace(-3, 3, 24).reshape(2, 3, 4)
    expected = 0.5 * x * (1.0 + torch.tanh(
        (2.0 / 3.141592653589793) ** 0.5 * (x + 0.044715 * x ** 3)))
    out = gelu(x)
    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=1e-3)


def test_gelu_negative_inputs_bounded_below():
    x = torch.linspace(-5, 5, 101)
    out = gelu(x)
    assert out.min().item() > -0.2
    assert torch.all(out[x > 0] > 0)


# ---------------------------------------------------------------- BytePairEncoding

def test_bpe_encode_merges():
    bpe = build_bpe()
    token_dict = build_token_dict()
    assert list(bpe.encode('hello')) == [token_dict['he'], token_dict['ll'], token_dict['o']]


def test_bpe_decode_roundtrip():
    bpe = build_bpe()
    for text in ['hello', 'hello world', 'a b c', 'he llo']:
        assert bpe.decode(bpe.encode(text)) == text


def test_bpe_encode_with_space():
    bpe = build_bpe()
    token_dict = build_token_dict()
    ids = list(bpe.encode('a b'))
    assert ids == [token_dict['a'], token_dict[SPACE], token_dict['b']] or \
        ids == [token_dict['a'], token_dict[SPACE + 'b']] if (SPACE + 'b') in token_dict else \
        ids == [token_dict['a'], token_dict[SPACE], token_dict['b']]


def test_get_bpe_from_files(tmp_path):
    encoder_path = tmp_path / 'encoder.json'
    vocab_path = tmp_path / 'vocab.bpe'
    token_dict = build_token_dict()
    encoder_path.write_text(json.dumps(token_dict), encoding='utf-8')
    vocab_path.write_text('#version: 0.2\nh e\nl l\n', encoding='utf-8')
    bpe = get_bpe_from_files(str(encoder_path), str(vocab_path))
    assert list(bpe.encode('hello')) == [token_dict['he'], token_dict['ll'], token_dict['o']]
    assert bpe.decode(bpe.encode('hello')) == 'hello'


# ---------------------------------------------------------------- GPT_2

def test_gpt2_is_module_with_parameters():
    model = build_model(len(build_token_dict()))
    assert isinstance(model, torch.nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_gpt2_forward_shape():
    n_vocab = len(build_token_dict())
    model = build_model(n_vocab)
    x = torch.randint(0, n_vocab, (2, 5))
    with torch.no_grad():
        out = logits_of(model(x))
    assert out.shape[0] == 2
    assert out.shape[-1] == n_vocab
    assert torch.isfinite(out).all()


def test_gpt2_forward_deterministic_in_eval():
    n_vocab = len(build_token_dict())
    model = build_model(n_vocab)
    x = torch.randint(0, n_vocab, (1, 6))
    with torch.no_grad():
        a = logits_of(model(x))
        b = logits_of(model(x))
    assert torch.allclose(a, b)


def test_gpt2_backward_produces_gradients():
    n_vocab = len(build_token_dict())
    model = build_model(n_vocab)
    model.train()
    x = torch.randint(0, n_vocab, (2, 4))
    out = logits_of(model(x))
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0 for g in grads)


# ---------------------------------------------------------------- generate

def test_generate_returns_one_string_per_input():
    bpe = build_bpe()
    model = build_model(len(build_token_dict()))
    outputs = generate(model, bpe, ['hello', 'he'], length=3, top_k=1)
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


def test_generate_is_deterministic_with_top_k_1():
    bpe = build_bpe()
    model = build_model(len(build_token_dict()))
    first = generate(model, bpe, ['hello'], length=4, top_k=1)
    second = generate(model, bpe, ['hello'], length=4, top_k=1)
    assert first == second


def test_generate_keeps_prompt_prefix():
    bpe = build_bpe()
    model = build_model(len(build_token_dict()))
    outputs = generate(model, bpe, ['hello'], length=3, top_k=1)
    assert outputs[0].startswith('hello')
