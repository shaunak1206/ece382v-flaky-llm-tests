import json
import math

import pytest
import torch

from torch_gpt_2 import GPT_2, BytePairEncoding, gelu, generate, get_bpe_from_files


def _byte_chars():
    """Printable stand-ins for the 256 byte values, in GPT-2 byte-level BPE order."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return [chr(c) for c in cs]


def _build_bpe(tmp_path, tokens, merges=()):
    """Write encoder.json and vocab.bpe to disk and load them with the public loader."""
    encoder = {token: index for index, token in enumerate(tokens)}
    encoder_path = tmp_path / "encoder.json"
    encoder_path.write_text(json.dumps(encoder), encoding="utf-8")
    vocab_path = tmp_path / "vocab.bpe"
    lines = ["#version: 0.2"] + ["%s %s" % pair for pair in merges]
    vocab_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bpe = get_bpe_from_files(str(encoder_path), str(vocab_path))
    return bpe, len(tokens)


@pytest.fixture
def byte_bpe(tmp_path):
    return _build_bpe(tmp_path, _byte_chars())


@pytest.fixture
def ascii_bpe(tmp_path):
    return _build_bpe(tmp_path, [chr(i) for i in range(33, 127)])


def _small_net(n_vocab=20, n_ctx=16, **kwargs):
    return GPT_2(n_vocab=n_vocab, n_ctx=n_ctx, n_embd=8, n_head=2, n_layer=2, **kwargs)


def _shift_net(vocab_size, calls=None):
    """Fake net: at every position p the largest logit is for token (input[p] + 1) % vocab_size."""
    def net(x):
        if calls is not None:
            calls.append((tuple(x.shape), x.dtype))
        logits = torch.zeros(x.shape[0], x.shape[1], vocab_size)
        logits.scatter_(2, ((x + 1) % vocab_size).unsqueeze(-1), 1.0)
        return logits
    return net


def _chain(prompt_ids, steps, vocab_size):
    ids = list(prompt_ids)
    for _ in range(steps):
        ids.append((ids[-1] + 1) % vocab_size)
    return ids


# BytePairEncoding (re-exported from keras_gpt_2)

def test_bpe_is_keras_class_and_encode_returns_int_list(byte_bpe):
    bpe, _ = byte_bpe
    assert isinstance(bpe, BytePairEncoding)
    ids = bpe.encode("hi")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) == 2  # no merges: one token per byte


def test_encode_decode_roundtrip_ascii(byte_bpe):
    bpe, _ = byte_bpe
    text = "Hello, world! 123\n\tdone"
    assert bpe.decode(bpe.encode(text)) == text


def test_encode_decode_roundtrip_non_ascii(byte_bpe):
    bpe, _ = byte_bpe
    text = "café ☕ naïve"
    assert bpe.decode(bpe.encode(text)) == text


def test_merges_collapse_pairs_into_single_token(tmp_path):
    bpe, _ = _build_bpe(tmp_path, _byte_chars() + ["ab", "abc"],
                        merges=[("a", "b"), ("ab", "c")])
    ids = bpe.encode("abc")
    assert len(ids) == 1
    assert ids == [257]  # "ab" is id 256, "abc" is id 257
    assert bpe.decode(ids) == "abc"


def test_lower_rank_merge_is_applied_first(tmp_path):
    bpe, _ = _build_bpe(tmp_path, _byte_chars() + ["ab", "bc"],
                        merges=[("b", "c"), ("a", "b")])
    ids = bpe.encode("abc")
    assert ids == bpe.encode("a") + bpe.encode("bc")
    assert ids != bpe.encode("ab") + bpe.encode("c")
    assert bpe.decode(ids) == "abc"


# GPT_2 network

def test_forward_returns_logits_per_position():
    net = _small_net(n_vocab=11)
    x = torch.arange(15).reshape(3, 5) % 11
    out = net(x)
    assert out.shape == (3, 5, 11)


def test_forward_accepts_single_token_sequence():
    net = _small_net(n_vocab=11).eval()
    with torch.no_grad():
        out = net(torch.tensor([[4]]))
    assert out.shape == (1, 1, 11)


def test_forward_is_causal():
    net = _small_net(n_vocab=20).eval()
    x = torch.tensor([[3, 1, 4, 1, 5, 9]])
    y = x.clone()
    y[0, -1] = 7
    with torch.no_grad():
        a = net(x)
        b = net(y)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_forward_is_deterministic_in_eval_mode():
    net = _small_net().eval()
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.equal(net(x), net(x))


def test_gradients_reach_embedding_weights():
    net = _small_net(n_vocab=20)
    out = net(torch.tensor([[1, 2, 3]]))
    out.sum().backward()
    grad = net.embedding.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_feed_forward_activation_is_passed_through():
    x = torch.tensor([[1, 2, 3, 4, 5]])
    base = _small_net(n_vocab=20).eval()
    explicit_gelu = _small_net(n_vocab=20, feed_forward_activation=gelu).eval()
    relu = _small_net(n_vocab=20, feed_forward_activation=torch.relu).eval()
    explicit_gelu.load_state_dict(base.state_dict())
    relu.load_state_dict(base.state_dict())
    with torch.no_grad():
        assert torch.allclose(base(x), explicit_gelu(x), atol=1e-6)
        assert not torch.allclose(base(x), relu(x), atol=1e-6)


def test_sequence_longer_than_n_ctx_raises():
    net = _small_net(n_ctx=4).eval()
    with pytest.raises((RuntimeError, IndexError, ValueError)):
        net(torch.zeros((1, 5), dtype=torch.long))


# gelu

def test_gelu_of_zero_is_zero():
    assert gelu(torch.tensor(0.0)).item() == 0.0


def test_gelu_close_to_exact_erf_form():
    x = torch.linspace(-6, 6, 241)
    exact = 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
    assert torch.allclose(gelu(x), exact, atol=1e-3)


def test_gelu_saturates_for_large_magnitudes():
    out = gelu(torch.tensor([10.0, -10.0]))
    assert torch.allclose(out, torch.tensor([10.0, 0.0]), atol=1e-4)


def test_gelu_preserves_shape_and_dtype():
    x = torch.linspace(-2, 2, 24, dtype=torch.float64).reshape(2, 3, 4)
    out = gelu(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float64


def test_gelu_is_differentiable():
    x = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    gelu(x).sum().backward()
    assert torch.isfinite(x.grad).all()


# generate

def test_generate_returns_one_string_per_prompt(byte_bpe):
    bpe, size = byte_bpe
    out = generate(_shift_net(size), bpe, ["ab", "xyz"], length=1)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(s, str) for s in out)


def test_generate_appends_predicted_tokens(byte_bpe):
    bpe, size = byte_bpe
    out = generate(_shift_net(size), bpe, ["ab"], length=3)
    assert out == ["abcde"]


def test_generate_length_zero_returns_prompt_without_calling_net(byte_bpe):
    bpe, size = byte_bpe
    calls = []
    out = generate(_shift_net(size, calls), bpe, ["hello"], length=0)
    assert out == ["hello"]
    assert calls == []


def test_generate_batch_uses_each_prompts_own_last_token(byte_bpe):
    bpe, size = byte_bpe
    prompts = ["a", "xyz"]
    out = generate(_shift_net(size), bpe, prompts, length=2)
    expected = [bpe.decode(_chain(bpe.encode(p), 2, size)) for p in prompts]
    assert out == expected
    assert out[0] == "abc"


def test_generate_feeds_growing_batch_each_step(byte_bpe):
    bpe, size = byte_bpe
    calls = []
    generate(_shift_net(size, calls), bpe, ["ab", "c"], length=3)
    assert [shape for shape, _ in calls] == [(2, 2), (2, 3), (2, 4)]
    assert all(dtype == torch.int64 for _, dtype in calls)


def test_generate_with_real_model_is_repeatable_for_top_k_1(ascii_bpe):
    bpe, size = ascii_bpe
    torch.manual_seed(0)
    net = _small_net(n_vocab=size, n_ctx=32).eval()
    first = generate(net, bpe, ["hello"], length=5)
    second = generate(net, bpe, ["hello"], length=5)
    assert first == second
    assert len(first) == 1
    assert first[0].startswith("hello")
    assert first[0].isascii()


def test_generate_top_k_two_matches_top_k_one_when_one_token_dominates(byte_bpe):
    bpe, size = byte_bpe
    net = _shift_net(size)
    assert generate(net, bpe, ["ab"], length=3, top_k=2) == ["abcde"]
    assert generate(net, bpe, ["ab"], length=3, top_k=1) == ["abcde"]
