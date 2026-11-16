"""Unit tests for BytePairEncoding, GPT_2, gelu and generate in torch_gpt_2.

Assumed interfaces (the sources of net.py, gen.py and keras_gpt_2 were not provided):
  * GPT_2(config) takes a config with n_vocab, n_ctx, n_embd, n_head, n_layer,
    and is a torch.nn.Module whose forward maps (batch, seq) int ids to
    (batch, seq, n_vocab) logits.
  * gelu(tensor) is the tanh approximation used by GPT-2.
  * generate(model, context, length) takes a list of ints and returns the
    context followed by `length` new token ids.
  * get_bpe_from_files(encoder_path, vocab_path) reads an encoder.json (token -> id)
    and a vocab.bpe (merge lines after a "#version" header).
Only torch is seeded, since it is the only RNG used by the code under test.
"""
import json
import math

import torch
from torch import nn

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate, get_bpe_from_files


class _Config(dict):
    """Dict that also supports attribute access, so both config styles work."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def _small_config():
    return _Config(n_vocab=11, n_ctx=8, n_embd=8, n_head=2, n_layer=1)


def _write_bpe_files(tmp_path, encoder, merges):
    encoder_path = tmp_path / "encoder.json"
    encoder_path.write_text(json.dumps(encoder), encoding="utf-8")
    vocab_path = tmp_path / "vocab.bpe"
    body = "".join("{} {}\n".format(a, b) for a, b in merges)
    vocab_path.write_text("#version: 0.2\n" + body, encoding="utf-8")
    return str(encoder_path), str(vocab_path)


def _as_list(out):
    if isinstance(out, torch.Tensor):
        return [int(v) for v in out.flatten().tolist()]
    return [int(v) for v in out]


class _PeakedLM(nn.Module):
    """Stub LM whose logits put essentially all mass on token 3 at every position.

    The logit gap is 1000, so the probability of any other token underflows to
    exactly 0 and sampling always returns token 3, whatever the RNG state.
    """

    VOCAB = 11

    def forward(self, x, *args, **kwargs):
        x = torch.as_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch, seq = x.shape
        logits = torch.zeros(batch, seq, self.VOCAB)
        logits[..., 3] = 1000.0
        return logits


# ---------------------------------------------------------------- BytePairEncoding

def test_get_bpe_from_files_returns_bytepairencoding(tmp_path):
    encoder_path, vocab_path = _write_bpe_files(
        tmp_path, {"a": 0, "b": 1, "ab": 2}, [("a", "b")]
    )
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert isinstance(bpe, BytePairEncoding)


def test_bpe_encode_applies_single_merge(tmp_path):
    encoder_path, vocab_path = _write_bpe_files(
        tmp_path, {"a": 0, "b": 1, "ab": 2}, [("a", "b")]
    )
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert _as_list(bpe.encode("ab")) == [2]


def test_bpe_encode_without_applicable_merge_keeps_characters(tmp_path):
    encoder_path, vocab_path = _write_bpe_files(
        tmp_path, {"a": 0, "b": 1, "c": 2, "ab": 3}, [("a", "b")]
    )
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert _as_list(bpe.encode("ac")) == [0, 2]


def test_bpe_merge_priority_follows_merge_rank(tmp_path):
    # "b c" has rank 0 and "a b" has rank 1, so "bc" must be merged first,
    # giving [a, bc] rather than [ab, c].
    encoder_path, vocab_path = _write_bpe_files(
        tmp_path,
        {"a": 0, "b": 1, "c": 2, "bc": 3, "ab": 4, "abc": 5},
        [("b", "c"), ("a", "b")],
    )
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert _as_list(bpe.encode("abc")) == [0, 3]


def test_bpe_encode_empty_string_gives_empty_list(tmp_path):
    encoder_path, vocab_path = _write_bpe_files(
        tmp_path, {"a": 0, "b": 1, "ab": 2}, [("a", "b")]
    )
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert _as_list(bpe.encode("")) == []


def test_bpe_decode_round_trip(tmp_path):
    encoder_path, vocab_path = _write_bpe_files(
        tmp_path, {"a": 0, "b": 1, "c": 2, "ab": 3}, [("a", "b")]
    )
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert bpe.decode(bpe.encode("abc")) == "abc"
    assert bpe.decode(bpe.encode("ab")) == "ab"


def test_bpe_encode_is_deterministic(tmp_path):
    encoder_path, vocab_path = _write_bpe_files(
        tmp_path, {"a": 0, "b": 1, "c": 2, "bc": 3, "ab": 4, "abc": 5},
        [("b", "c"), ("a", "b")],
    )
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert _as_list(bpe.encode("abcab")) == _as_list(bpe.encode("abcab"))


# ---------------------------------------------------------------------------- gelu

def test_gelu_of_zero_is_zero():
    out = gelu(torch.tensor(0.0))
    assert abs(float(out)) < 1e-7


def test_gelu_matches_tanh_formula():
    values = [-3.0, -1.5, -0.5, 0.25, 1.0, 2.0, 4.0]
    x = torch.tensor(values, dtype=torch.float32)
    out = gelu(x).tolist()
    for xi, yi in zip(values, out):
        expected = 0.5 * xi * (1.0 + math.tanh(
            math.sqrt(2.0 / math.pi) * (xi + 0.044715 * xi ** 3)))
        assert abs(yi - expected) < 1e-5


def test_gelu_large_positive_is_approximately_identity():
    out = float(gelu(torch.tensor(10.0)))
    assert abs(out - 10.0) < 1e-4


def test_gelu_large_negative_is_approximately_zero():
    out = float(gelu(torch.tensor(-10.0)))
    assert abs(out) < 1e-5


# ------------------------------------------------------------------------- GPT_2

def test_gpt2_forward_output_shape():
    torch.manual_seed(0)
    model = GPT_2(_small_config())
    model.eval()
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    assert tuple(logits.shape) == (2, 4, 11)


def test_gpt2_eval_forward_is_deterministic():
    torch.manual_seed(0)
    model = GPT_2(_small_config())
    model.eval()
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        first = model(x)
        second = model(x)
    assert torch.allclose(first, second, atol=0.0, rtol=0.0)


def test_gpt2_same_seed_gives_identical_weights():
    torch.manual_seed(123)
    model_a = GPT_2(_small_config())
    torch.manual_seed(123)
    model_b = GPT_2(_small_config())
    params_a = list(model_a.state_dict().values())
    params_b = list(model_b.state_dict().values())
    assert len(params_a) == len(params_b) > 0
    for pa, pb in zip(params_a, params_b):
        assert torch.equal(pa, pb)


def test_gpt2_batched_output_matches_single_samples():
    torch.manual_seed(0)
    model = GPT_2(_small_config())
    model.eval()
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        batched = model(x)
        first = model(x[:1])
        second = model(x[1:])
    assert torch.allclose(batched[:1], first, atol=1e-5)
    assert torch.allclose(batched[1:], second, atol=1e-5)


def test_gpt2_is_causal():
    # Changing the last token must not change logits at earlier positions.
    torch.manual_seed(0)
    model = GPT_2(_small_config())
    model.eval()
    x1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    x2 = torch.tensor([[1, 2, 3, 9]], dtype=torch.long)
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5)


# ---------------------------------------------------------------------- generate

def test_generate_returns_context_plus_requested_length():
    torch.manual_seed(0)
    out = _as_list(generate(_PeakedLM(), [1, 2], 5))
    assert len(out) == 2 + 5


def test_generate_keeps_context_prefix():
    torch.manual_seed(0)
    out = _as_list(generate(_PeakedLM(), [1, 2], 4))
    assert out[:2] == [1, 2]


def test_generate_with_peaked_model_picks_dominant_token():
    torch.manual_seed(0)
    out = _as_list(generate(_PeakedLM(), [1, 2], 4))
    assert out[2:] == [3, 3, 3, 3]


def test_generate_does_not_mutate_context_list():
    torch.manual_seed(0)
    context = [1, 2]
    generate(_PeakedLM(), context, 3)
    assert context == [1, 2]


def test_generate_tokens_are_within_vocabulary():
    torch.manual_seed(0)
    model = GPT_2(_small_config())
    model.eval()
    out = _as_list(generate(model, [1, 2, 3], 4))
    assert len(out) == 3 + 4
    assert all(0 <= t < 11 for t in out)


def test_generate_is_reproducible_with_same_seed():
    torch.manual_seed(7)
    model = GPT_2(_small_config())
    model.eval()
    torch.manual_seed(7)
    first = _as_list(generate(model, [1, 2, 3], 5))
    torch.manual_seed(7)
    second = _as_list(generate(model, [1, 2, 3], 5))
    assert first == second
