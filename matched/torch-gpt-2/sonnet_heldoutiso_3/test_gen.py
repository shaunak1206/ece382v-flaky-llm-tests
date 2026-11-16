import math

import pytest
import torch

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate, get_bpe_from_files


def _make_bpe():
    token_dict = {
        'a': 0,
        'b': 1,
        'c': 2,
        'ab': 3,
        '\u0120': 4,  # byte-level BPE representation of a space
    }
    bpe_rank = {('a', 'b'): 0}
    return BytePairEncoding(token_dict, bpe_rank)


def _make_model(n_vocab=5, n_ctx=16, n_embd=8, n_head=2, n_layer=2):
    torch.manual_seed(0)
    model = GPT_2(n_vocab=n_vocab, n_ctx=n_ctx, n_embd=n_embd,
                  n_head=n_head, n_layer=n_layer)
    model.eval()
    return model


def _logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


# ---------------------------------------------------------------- gelu

def test_gelu_zero():
    out = gelu(torch.tensor([0.0]))
    assert abs(out.item()) < 1e-7


def test_gelu_matches_tanh_formula():
    x = torch.linspace(-4, 4, 41)
    expected = 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    assert torch.allclose(gelu(x), expected, atol=1e-5)


def test_gelu_large_positive_is_identity_like():
    x = torch.tensor([10.0, 20.0])
    assert torch.allclose(gelu(x), x, atol=1e-4)


def test_gelu_large_negative_is_near_zero():
    x = torch.tensor([-10.0, -20.0])
    assert torch.allclose(gelu(x), torch.zeros(2), atol=1e-4)


def test_gelu_preserves_shape_and_has_gradient():
    x = torch.randn(3, 4, 5, requires_grad=True)
    out = gelu(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


# ------------------------------------------------------ BytePairEncoding

def test_bpe_encode_merges_pair():
    bpe = _make_bpe()
    assert bpe.encode('ab') == [3]


def test_bpe_encode_without_merge():
    bpe = _make_bpe()
    assert bpe.encode('ac') == [0, 2]


def test_bpe_decode_inverse_of_encode():
    bpe = _make_bpe()
    for text in ['ab', 'abc', 'cab', 'ab ab']:
        assert bpe.decode(bpe.encode(text)) == text


def test_bpe_encode_returns_list_of_ints():
    bpe = _make_bpe()
    tokens = bpe.encode('abc')
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)


def test_get_bpe_from_files(tmp_path):
    import json
    encoder_path = tmp_path / 'encoder.json'
    vocab_path = tmp_path / 'vocab.bpe'
    encoder_path.write_text(json.dumps({'a': 0, 'b': 1, 'ab': 2}))
    vocab_path.write_text('#version: 0.2\na b\n')
    bpe = get_bpe_from_files(str(encoder_path), str(vocab_path))
    assert bpe.encode('ab') == [2]
    assert bpe.decode([2]) == 'ab'


# --------------------------------------------------------------- GPT_2

def test_gpt2_output_shape():
    model = _make_model(n_vocab=5)
    x = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]], dtype=torch.long)
    with torch.no_grad():
        logits = _logits(model(x))
    assert tuple(logits.shape) == (2, 4, 5)


def test_gpt2_deterministic_in_eval_mode():
    model = _make_model()
    x = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        first = _logits(model(x))
        second = _logits(model(x))
    assert torch.allclose(first, second)


def test_gpt2_is_causal():
    model = _make_model()
    x1 = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    x2 = torch.tensor([[0, 1, 2, 4]], dtype=torch.long)
    with torch.no_grad():
        out1 = _logits(model(x1))
        out2 = _logits(model(x2))
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5)


def test_gpt2_backward_produces_gradients():
    model = _make_model()
    model.train()
    x = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    logits = _logits(model(x))
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_gpt2_batch_items_are_independent():
    model = _make_model()
    a = torch.tensor([[0, 1, 2]], dtype=torch.long)
    b = torch.tensor([[3, 4, 1]], dtype=torch.long)
    with torch.no_grad():
        single = _logits(model(a))
        batched = _logits(model(torch.cat([a, b], dim=0)))
    assert torch.allclose(single[0], batched[0], atol=1e-5)


# ------------------------------------------------------------ generate

def test_generate_returns_one_string_per_input():
    bpe = _make_bpe()
    model = _make_model(n_vocab=5)
    outputs = generate(model, bpe, ['ab', 'abc'], length=3, top_k=1)
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


def test_generate_greedy_is_deterministic():
    bpe = _make_bpe()
    model = _make_model(n_vocab=5)
    first = generate(model, bpe, ['ab'], length=4, top_k=1)
    second = generate(model, bpe, ['ab'], length=4, top_k=1)
    assert first == second
