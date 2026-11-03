import inspect
import math

import pytest
import torch

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _byte_map():
    """The standard GPT-2 byte to printable unicode character mapping."""
    bs = (list(range(ord('!'), ord('~') + 1)) +
          list(range(0xA1, 0xAC + 1)) +
          list(range(0xAE, 0xFF + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _make_bpe(with_merges=True):
    chars = [_byte_map()[b] for b in range(256)]
    token_dict = {ch: i for i, ch in enumerate(chars)}
    bpe_rank = {}
    if with_merges:
        token_dict['he'] = 256
        token_dict['hel'] = 257
        bpe_rank[('h', 'e')] = 0
        bpe_rank[('he', 'l')] = 1
    return BytePairEncoding(token_dict, bpe_rank)


class DigitBPE(object):
    """A trivial tokenizer: every digit character is a token."""

    def encode(self, text):
        return [int(c) for c in text]

    def decode(self, tokens):
        return ''.join(str(t) for t in tokens)


class IncrementNet(object):
    """Fake net that always predicts (token + 1) mod vocab at each position."""

    def __init__(self, vocab=10):
        self.vocab = vocab
        self.shapes = []

    def __call__(self, x):
        self.shapes.append(tuple(x.shape))
        out = torch.zeros(x.shape[0], x.shape[1], self.vocab)
        nxt = (x + 1) % self.vocab
        out.scatter_(2, nxt.unsqueeze(-1), 1.0)
        return out


class TwoChoiceNet(object):
    """Fake net: 0.7 on (token + 1), 0.3 on (token + 2), modulo vocab."""

    def __init__(self, vocab=10):
        self.vocab = vocab

    def __call__(self, x):
        out = torch.zeros(x.shape[0], x.shape[1], self.vocab)
        out.scatter_(2, ((x + 1) % self.vocab).unsqueeze(-1), 0.7)
        out.scatter_(2, ((x + 2) % self.vocab).unsqueeze(-1), 0.3)
        return out


def _small_net(n_vocab=11, n_ctx=8, n_embd=8, n_head=2, n_layer=2):
    torch.manual_seed(0)
    net = GPT_2(n_vocab=n_vocab, n_ctx=n_ctx, n_embd=n_embd, n_head=n_head, n_layer=n_layer)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

def test_gelu_zero_is_zero():
    assert gelu(torch.tensor(0.0)).item() == 0.0


def test_gelu_matches_tanh_formula():
    x = torch.tensor([-3.0, -1.0, -0.5, 0.5, 1.0, 2.5])
    expected = [0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)))
                for v in x.tolist()]
    assert gelu(x).tolist() == pytest.approx(expected, abs=1e-5)


def test_gelu_asymptotics():
    big = torch.tensor([10.0, 20.0])
    assert gelu(big).tolist() == pytest.approx(big.tolist(), abs=1e-4)
    small = torch.tensor([-10.0, -20.0])
    assert gelu(small).tolist() == pytest.approx([0.0, 0.0], abs=1e-4)


def test_gelu_preserves_shape_and_is_elementwise():
    x = torch.randn(2, 3, 4)
    y = gelu(x)
    assert y.shape == x.shape
    assert torch.allclose(y[1, 2, 3], gelu(x[1, 2, 3]))


def test_gelu_odd_symmetry_identity():
    # gelu(x) - gelu(-x) == x holds exactly for the tanh approximation.
    x = torch.linspace(-4.0, 4.0, 17)
    assert torch.allclose(gelu(x) - gelu(-x), x, atol=1e-5)


# ---------------------------------------------------------------------------
# BytePairEncoding
# ---------------------------------------------------------------------------

def test_package_exports_bpe_class():
    assert torch_gpt_2.BytePairEncoding is BytePairEncoding
    assert inspect.isclass(BytePairEncoding)


def test_bpe_roundtrip_ascii():
    bpe = _make_bpe()
    for text in ['hello world', 'The quick brown fox.', 'a  b\tc\nd', '12345 !?']:
        assert bpe.decode(bpe.encode(text)) == text


def test_bpe_roundtrip_unicode():
    bpe = _make_bpe()
    text = 'café ☃ 中文'
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_empty_text():
    bpe = _make_bpe()
    assert bpe.encode('') == []
    assert bpe.decode([]) == ''


def test_bpe_merges_reduce_token_count():
    merged = _make_bpe(with_merges=True)
    plain = _make_bpe(with_merges=False)
    assert len(plain.encode('hel')) == 3
    assert merged.encode('hel') == [257]
    assert len(merged.encode('hello')) < len(plain.encode('hello'))


def test_bpe_encode_returns_known_integer_ids():
    bpe = _make_bpe()
    ids = bpe.encode('hello world')
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i <= 257 for i in ids)


# ---------------------------------------------------------------------------
# GPT_2
# ---------------------------------------------------------------------------

def test_gpt2_default_activation_is_gelu():
    default = inspect.signature(GPT_2.__init__).parameters['feed_forward_activation'].default
    assert default is gelu


def test_gpt2_output_shape_is_logits_over_vocab():
    net = _small_net(n_vocab=11, n_ctx=8)
    x = torch.randint(0, 11, (3, 8))
    with torch.no_grad():
        out = net(x)
    assert tuple(out.shape) == (3, 8, 11)


def test_gpt2_structure_follows_hyperparameters():
    net = _small_net(n_vocab=13, n_embd=8, n_layer=3)
    assert tuple(net.embedding.weight.shape) == (13, 8)
    assert len(net.encoder.components) == 3


def test_gpt2_eval_forward_is_deterministic():
    net = _small_net()
    x = torch.randint(0, 11, (2, 8))
    with torch.no_grad():
        first = net(x)
        second = net(x)
    assert torch.equal(first, second)


def test_gpt2_is_causal():
    net = _small_net()
    x = torch.randint(0, 11, (1, 8))
    y = x.clone()
    y[0, 5:] = (y[0, 5:] + 3) % 11
    with torch.no_grad():
        out_x = net(x)
        out_y = net(y)
    assert torch.allclose(out_x[:, :5], out_y[:, :5], atol=1e-4)
    assert not torch.allclose(out_x[:, 5:], out_y[:, 5:], atol=1e-4)


def test_gpt2_batch_items_are_independent():
    net = _small_net()
    x = torch.randint(0, 11, (3, 8))
    with torch.no_grad():
        batched = net(x)
        single = torch.cat([net(x[i:i + 1]) for i in range(3)], dim=0)
    assert torch.allclose(batched, single, atol=1e-4)


def test_gpt2_backward_produces_gradients():
    net = _small_net()
    net.train()
    x = torch.randint(0, 11, (2, 8))
    net(x).sum().backward()
    assert net.embedding.weight.grad is not None
    assert torch.isfinite(net.embedding.weight.grad).all()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_single_text_appends_predicted_tokens():
    net = IncrementNet()
    assert generate(net, DigitBPE(), ['123'], length=3) == ['123456']


def test_generate_calls_net_once_per_new_token():
    net = IncrementNet()
    generate(net, DigitBPE(), ['12'], length=4)
    assert len(net.shapes) == 4
    assert [shape[1] for shape in net.shapes] == [2, 3, 4, 5]


def test_generate_handles_texts_of_different_lengths():
    net = IncrementNet()
    outputs = generate(net, DigitBPE(), ['1', '78'], length=2)
    assert outputs == ['123', '7890']


def test_generate_zero_length_returns_original_texts():
    net = IncrementNet()
    outputs = generate(net, DigitBPE(), ['42', '7'], length=0)
    assert outputs == ['42', '7']
    assert net.shapes == []


def test_generate_top_k_one_is_deterministic():
    net = TwoChoiceNet()
    first = generate(net, DigitBPE(), ['5'], length=6, top_k=1)
    second = generate(net, DigitBPE(), ['5'], length=6, top_k=1)
    assert first == second == ['5678901']


def test_generate_top_k_two_only_samples_from_best_two():
    torch.manual_seed(0)
    net = TwoChoiceNet()
    out = generate(net, DigitBPE(), ['0'], length=30, top_k=2)[0]
    digits = [int(c) for c in out]
    assert len(digits) == 31
    for prev, cur in zip(digits, digits[1:]):
        assert (cur - prev) % 10 in (1, 2)


def test_generate_with_real_gpt2_returns_valid_tokens():
    net = _small_net(n_vocab=10, n_ctx=8)
    bpe = DigitBPE()
    outputs = generate(net, bpe, ['123', '45'], length=3, top_k=1)
    assert len(outputs) == 2
    assert outputs[0].startswith('123') and len(outputs[0]) == 6
    assert outputs[1].startswith('45') and len(outputs[1]) == 5
    assert outputs == generate(net, bpe, ['123', '45'], length=3, top_k=1)
