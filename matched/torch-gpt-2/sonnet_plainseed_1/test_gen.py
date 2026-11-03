import inspect
import math

import pytest
import torch

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class CharBPE(object):
    """Tiny stand-in encoder: letters a-z map to ids 1-26."""

    def encode(self, text):
        return [ord(c) - ord('a') + 1 for c in text]

    def decode(self, tokens):
        return ''.join(chr(t - 1 + ord('a')) for t in tokens)


class RecordingBPE(object):
    """Encoder that keeps the token lists given to decode."""

    def __init__(self, table):
        self.table = table
        self.decoded = []

    def encode(self, text):
        return list(self.table[text])

    def decode(self, tokens):
        self.decoded.append(list(tokens))
        return list(tokens)


class SuccessorNet(object):
    """Fake net: after token x, the most likely next token is (x + 1) % n_vocab."""

    def __init__(self, n_vocab):
        self.n_vocab = n_vocab
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        out = torch.zeros(x.size(0), x.size(1), self.n_vocab)
        nxt = (x + 1) % self.n_vocab
        out.scatter_(2, nxt.unsqueeze(-1), 1.0)
        return out


class TwoChoiceNet(object):
    """Fake net: next token is either (x + 1) or (x + 2) modulo n_vocab."""

    def __init__(self, n_vocab):
        self.n_vocab = n_vocab

    def __call__(self, x):
        out = torch.zeros(x.size(0), x.size(1), self.n_vocab)
        out.scatter_(2, ((x + 1) % self.n_vocab).unsqueeze(-1), 0.6)
        out.scatter_(2, ((x + 2) % self.n_vocab).unsqueeze(-1), 0.4)
        return out


def make_bpe(merges=None):
    token_dict = {
        'a': 0, 'b': 1, 'c': 2, 'ab': 3, 'abc': 4, 'Ġ': 5,
    }
    bpe_rank = {}
    if merges:
        for rank, pair in enumerate(merges):
            bpe_rank[pair] = rank
    return BytePairEncoding(token_dict, bpe_rank)


def make_small_net(n_vocab=11, n_ctx=8, n_embd=8, n_head=2, n_layer=2):
    torch.manual_seed(0)
    net = GPT_2(n_vocab=n_vocab, n_ctx=n_ctx, n_embd=n_embd, n_head=n_head, n_layer=n_layer)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# BytePairEncoding
# ---------------------------------------------------------------------------

def test_bpe_encode_without_merges_is_per_character():
    bpe = make_bpe()
    assert bpe.encode('abc') == [0, 1, 2]


def test_bpe_encode_applies_merge_rules():
    bpe = make_bpe(merges=[('a', 'b')])
    assert bpe.encode('ab') == [3]
    assert bpe.encode('abc') == [3, 2]


def test_bpe_decode_inverts_encode():
    bpe = make_bpe(merges=[('a', 'b')])
    for text in ['a', 'ab', 'abc', 'cba']:
        assert bpe.decode(bpe.encode(text)) == text


def test_bpe_decode_maps_space_marker_back_to_space():
    bpe = make_bpe()
    assert bpe.decode([0, 5, 1]) == 'a b'


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

def test_gelu_zero_is_zero():
    assert gelu(torch.tensor(0.0)).item() == pytest.approx(0.0, abs=1e-7)


def test_gelu_known_values():
    out = gelu(torch.tensor([1.0, -1.0]))
    assert out[0].item() == pytest.approx(0.8412, abs=1e-3)
    assert out[1].item() == pytest.approx(-0.1588, abs=1e-3)


def test_gelu_matches_tanh_formula():
    x = torch.linspace(-4.0, 4.0, 41)
    expected = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    assert torch.allclose(gelu(x), expected, atol=1e-6)


def test_gelu_asymptotics():
    big = torch.tensor([10.0, 20.0])
    assert torch.allclose(gelu(big), big, atol=1e-4)
    neg = torch.tensor([-10.0, -20.0])
    assert torch.allclose(gelu(neg), torch.zeros(2), atol=1e-4)


def test_gelu_preserves_shape_and_supports_backward():
    x = torch.randn(3, 4, 5, requires_grad=True)
    y = gelu(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# GPT_2
# ---------------------------------------------------------------------------

def test_gpt2_output_shape():
    net = make_small_net(n_vocab=11)
    x = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    out = net(x)
    assert tuple(out.shape) == (2, 4, 11)


def test_gpt2_builds_requested_number_of_layers():
    net = make_small_net(n_layer=3)
    assert len(net.encoder.components) == 3


def test_gpt2_embedding_weight_shape():
    net = make_small_net(n_vocab=11, n_embd=8)
    assert tuple(net.embedding.weight.shape) == (11, 8)


def test_gpt2_default_feed_forward_activation_is_gelu():
    default = inspect.signature(GPT_2.__init__).parameters['feed_forward_activation'].default
    assert default is gelu


def test_gpt2_eval_forward_is_deterministic():
    net = make_small_net()
    x = torch.LongTensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        first = net(x)
        second = net(x)
    assert torch.allclose(first, second)


def test_gpt2_is_causal():
    net = make_small_net()
    a = torch.LongTensor([[1, 2, 3, 4, 5]])
    b = torch.LongTensor([[1, 2, 3, 9, 10]])
    with torch.no_grad():
        out_a = net(a)
        out_b = net(b)
    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-5)
    assert not torch.allclose(out_a[:, 3:], out_b[:, 3:], atol=1e-5)


def test_gpt2_backward_populates_gradients():
    net = make_small_net()
    x = torch.LongTensor([[1, 2, 3, 4]])
    net.train()
    net(x).sum().backward()
    assert net.embedding.weight.grad is not None
    assert torch.isfinite(net.embedding.weight.grad).all()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_greedy_follows_net_argmax():
    net = SuccessorNet(30)
    out = generate(net, CharBPE(), ['abc'], length=3, top_k=1)
    # a,b,c = 1,2,3 -> next tokens 4,5,6 -> d,e,f
    assert out == ['abcdef']


def test_generate_length_zero_returns_prompt():
    net = SuccessorNet(30)
    out = generate(net, CharBPE(), ['abc', 'de'], length=0)
    assert out == ['abc', 'de']
    assert net.calls == 0


def test_generate_calls_net_once_per_step():
    net = SuccessorNet(30)
    generate(net, CharBPE(), ['ab'], length=5)
    assert net.calls == 5


def test_generate_returns_one_output_per_text():
    net = SuccessorNet(30)
    out = generate(net, CharBPE(), ['a', 'bb', 'ccc'], length=2)
    assert len(out) == 3


def test_generate_batch_with_different_lengths_is_independent():
    net = SuccessorNet(30)
    out = generate(net, CharBPE(), ['a', 'abc'], length=2)
    # 'a'(1) -> 2,3 ; 'abc' last token 3 -> 4,5
    assert out[0] == 'abc'
    assert out[1] == 'abcde'


def test_generate_output_has_prompt_length_plus_generated_length():
    net = SuccessorNet(30)
    bpe = RecordingBPE({'p': [1, 2, 3], 'q': [4]})
    out = generate(net, bpe, ['p', 'q'], length=4)
    assert len(out[0]) == 3 + 4
    assert len(out[1]) == 1 + 4
    assert out[0][:3] == [1, 2, 3]
    assert out[1][:1] == [4]


def test_generate_top_k_samples_only_from_top_candidates():
    net = TwoChoiceNet(30)
    bpe = RecordingBPE({'p': [5]})
    for _ in range(5):
        out = generate(net, bpe, ['p'], length=1, top_k=2)
        assert out[0][0] == 5
        assert out[0][1] in (6, 7)


def test_generate_with_real_gpt2_produces_valid_tokens():
    net = make_small_net(n_vocab=11, n_ctx=16)
    bpe = RecordingBPE({'p': [1, 2, 3]})
    out = generate(net, bpe, ['p'], length=3, top_k=1)
    assert len(out[0]) == 6
    assert out[0][:3] == [1, 2, 3]
    assert all(0 <= t < 11 for t in out[0])
