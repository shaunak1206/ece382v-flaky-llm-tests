import json
import math

import torch

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate, get_bpe_from_files


ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


class _CharBPE(object):
    """Character-level stand-in for a BPE object with the encode/decode interface."""

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.index = {c: i for i, c in enumerate(alphabet)}

    def encode(self, text):
        return [self.index[c] for c in text]

    def decode(self, ids):
        return ''.join(self.alphabet[i] for i in ids)


class _NextTokenNet(torch.nn.Module):
    """Puts all probability mass on (last token + 1) mod vocab_size."""

    def __init__(self, vocab_size):
        super(_NextTokenNet, self).__init__()
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        nxt = (x + 1) % self.vocab_size
        logits = torch.zeros(x.shape[0], x.shape[1], self.vocab_size)
        logits.scatter_(2, nxt.unsqueeze(-1), 1.0)
        return logits


class _FixedLogitsNet(torch.nn.Module):
    """Returns the same logits vector at every position of every row."""

    def __init__(self, logits):
        super(_FixedLogitsNet, self).__init__()
        self.register_buffer('logits', torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        batch, seq = x.shape
        return self.logits.view(1, 1, -1).expand(batch, seq, -1).clone()


def _small_gpt2(n_vocab=26, n_ctx=8):
    torch.manual_seed(0)
    net = GPT_2(n_vocab=n_vocab, n_ctx=n_ctx, n_embd=8, n_head=2, n_layer=2)
    net.eval()
    return net


def _make_bpe(tmp_path):
    token_dict = {c: i for i, c in enumerate(ALPHABET)}
    token_dict['he'] = len(ALPHABET)
    encoder_path = tmp_path / 'encoder.json'
    vocab_path = tmp_path / 'vocab.bpe'
    encoder_path.write_text(json.dumps(token_dict), encoding='utf-8')
    vocab_path.write_text('#version: 0.2\nh e\n', encoding='utf-8')
    return get_bpe_from_files(str(encoder_path), str(vocab_path))


# ---------------------------------------------------------------- gelu

def test_gelu_is_zero_at_origin():
    assert float(gelu(torch.tensor(0.0))) == 0.0


def test_gelu_matches_tanh_reference_formula():
    for x in [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5]:
        expected = 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
        actual = float(gelu(torch.tensor(x, dtype=torch.float64)))
        assert abs(actual - expected) < 1e-9


def test_gelu_close_to_exact_erf_definition():
    xs = torch.linspace(-5.0, 5.0, 201, dtype=torch.float64)
    exact = 0.5 * xs * (1.0 + torch.erf(xs / math.sqrt(2.0)))
    assert torch.max(torch.abs(gelu(xs) - exact)).item() < 1e-3


def test_gelu_approaches_identity_for_large_positive_inputs():
    assert abs(float(gelu(torch.tensor(10.0, dtype=torch.float64))) - 10.0) < 1e-6


def test_gelu_approaches_zero_for_large_negative_inputs():
    assert abs(float(gelu(torch.tensor(-10.0, dtype=torch.float64)))) < 1e-6


def test_gelu_preserves_shape_and_acts_elementwise():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    out = gelu(x)
    assert out.shape == x.shape
    assert torch.allclose(out.reshape(-1), gelu(x.reshape(-1)), atol=1e-6)


def test_gelu_gradient_is_finite():
    x = torch.linspace(-3.0, 3.0, 13, requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------- BytePairEncoding

def test_bpe_unmerged_letters_map_to_token_ids(tmp_path):
    bpe = _make_bpe(tmp_path)
    assert isinstance(bpe, BytePairEncoding)
    ids = bpe.encode('ab')
    assert isinstance(ids, list)
    assert ids == [0, 1]


def test_bpe_applies_learned_merge(tmp_path):
    bpe = _make_bpe(tmp_path)
    assert bpe.encode('he') == [len(ALPHABET)]
    assert bpe.encode('hel') == [len(ALPHABET), ALPHABET.index('l')]


def test_bpe_round_trip_on_letters(tmp_path):
    bpe = _make_bpe(tmp_path)
    assert bpe.decode(bpe.encode('hello')) == 'hello'


def test_bpe_encode_is_deterministic(tmp_path):
    bpe = _make_bpe(tmp_path)
    assert bpe.encode('hello') == bpe.encode('hello')


# ---------------------------------------------------------------- GPT_2

def test_gpt2_output_shape_is_batch_seq_vocab():
    net = _small_gpt2()
    x = torch.LongTensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    with torch.no_grad():
        out = net(x)
    assert tuple(out.shape) == (2, 5, 26)


def test_gpt2_builds_requested_number_of_layers():
    net = _small_gpt2()
    assert len(net.encoder.components) == 2


def test_gpt2_is_causal():
    net = _small_gpt2()
    x = torch.LongTensor([[1, 2, 3, 4, 5]])
    y = torch.LongTensor([[1, 2, 3, 4, 9]])
    with torch.no_grad():
        out_x = net(x)
        out_y = net(y)
    # Changing the last token must not change any earlier position.
    assert torch.allclose(out_x[:, :4], out_y[:, :4], atol=1e-5)
    assert not torch.allclose(out_x[:, 4], out_y[:, 4])


def test_gpt2_batch_rows_are_independent():
    net = _small_gpt2()
    batch = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
    with torch.no_grad():
        together = net(batch)
        alone = net(batch[1:2])
    assert torch.allclose(together[1:2], alone, atol=1e-5)


def test_gpt2_backward_populates_embedding_gradient():
    net = _small_gpt2()
    net.train()
    x = torch.LongTensor([[1, 2, 3]])
    net(x).sum().backward()
    grad = net.embedding.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


# ---------------------------------------------------------------- generate

def test_generate_greedy_extends_prompt():
    net = _NextTokenNet(len(ALPHABET))
    bpe = _CharBPE(ALPHABET)
    assert generate(net, bpe, ['ab'], length=3, top_k=1) == ['abcde']


def test_generate_zero_length_returns_prompt_unchanged():
    net = _NextTokenNet(len(ALPHABET))
    bpe = _CharBPE(ALPHABET)
    assert generate(net, bpe, ['ab'], length=0) == ['ab']


def test_generate_batch_with_padding_matches_single_prompts():
    net = _NextTokenNet(len(ALPHABET))
    bpe = _CharBPE(ALPHABET)
    together = generate(net, bpe, ['ab', 'xyz'], length=2)
    alone = [generate(net, bpe, [text], length=2)[0] for text in ['ab', 'xyz']]
    assert together == alone == ['abcd', 'xyzab']


def test_generate_wraps_around_vocabulary():
    net = _NextTokenNet(len(ALPHABET))
    bpe = _CharBPE(ALPHABET)
    assert generate(net, bpe, ['xyz'], length=2) == ['xyzab']


def test_generate_calls_net_once_per_new_token():
    net = _NextTokenNet(len(ALPHABET))
    generate(net, _CharBPE(ALPHABET), ['ab', 'cd'], length=4)
    assert net.calls == 4


def test_generate_top_k_only_samples_from_top_candidates():
    logits = [0.0] * 10
    logits[3] = 0.9
    logits[7] = 0.1
    net = _FixedLogitsNet(logits)
    bpe = _CharBPE('abcdefghij')
    torch.manual_seed(1)
    out = generate(net, bpe, ['a'], length=30, top_k=2)[0]
    assert len(out) == 31
    assert set(out[1:]) <= {'d', 'h'}


def test_generate_top_k_one_is_independent_of_random_seed():
    logits = [0.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    net = _FixedLogitsNet(logits)
    bpe = _CharBPE('abcdefghij')
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        assert generate(net, bpe, ['a'], length=4, top_k=1) == ['abbbb']


def test_generate_with_real_gpt2_extends_prompt():
    net = _small_gpt2(n_vocab=26, n_ctx=8)
    out = generate(net, _CharBPE(ALPHABET), ['ab'], length=3, top_k=1)
    assert len(out) == 1
    assert len(out[0]) == 5
    assert out[0].startswith('ab')
    assert set(out[0]) <= set(ALPHABET)
