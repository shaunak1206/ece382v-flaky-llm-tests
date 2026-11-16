import math

import pytest
import torch
import torch.nn as nn

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_gpt_2():
    """Build a tiny GPT_2 model. Seeds torch before construction so weights are fixed."""
    torch.manual_seed(0)
    return GPT_2(n_vocab=11, n_ctx=8, n_embd=8, n_head=2, n_layer=1)


class _FakeLM(nn.Module):
    """Deterministic language model: the logit for token (last + 1) % vocab is the largest."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # Dummy parameter so the module has a device/dtype and is a proper nn.Module.
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        batch, length = inputs.shape
        logits = torch.zeros(batch, length, self.vocab_size)
        next_ids = (inputs + 1) % self.vocab_size
        logits.scatter_(2, next_ids.unsqueeze(-1), 10.0)
        return logits


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

def test_gelu_zero_maps_to_zero():
    torch.manual_seed(0)
    x = torch.zeros(3)
    y = gelu(x)
    assert torch.allclose(y, torch.zeros(3), atol=1e-7)


def test_gelu_matches_reference_values():
    torch.manual_seed(0)
    x = torch.tensor([-1.0, 1.0, 2.0])
    y = gelu(x)
    # Reference values of x * Phi(x); tolerance covers exact and tanh approximations.
    expected = [-0.158655, 0.841345, 1.954500]
    for got, want in zip(y.tolist(), expected):
        assert got == pytest.approx(want, abs=1e-3)


def test_gelu_large_positive_is_near_identity():
    torch.manual_seed(0)
    x = torch.tensor([10.0, 20.0])
    y = gelu(x)
    assert y.tolist() == pytest.approx([10.0, 20.0], abs=1e-4)


def test_gelu_large_negative_is_near_zero():
    torch.manual_seed(0)
    x = torch.tensor([-10.0, -20.0])
    y = gelu(x)
    assert y.tolist() == pytest.approx([0.0, 0.0], abs=1e-4)


def test_gelu_preserves_shape():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    assert gelu(x).shape == (2, 3, 4)


def test_gelu_is_nondecreasing_on_positive_range():
    torch.manual_seed(0)
    x = torch.linspace(0.0, 5.0, steps=101)
    y = gelu(x)
    diffs = y[1:] - y[:-1]
    assert bool((diffs >= -1e-6).all())


def test_gelu_gradient_is_finite_and_backprop_works():
    torch.manual_seed(0)
    x = torch.randn(5, requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# BytePairEncoding
# ---------------------------------------------------------------------------

def _tiny_bpe():
    token_dict = {"<unk>": 0, "l": 1, "o": 2, "w": 3, "lo": 4}
    bpe_ranks = {("l", "o"): 0}
    return BytePairEncoding(token_dict, bpe_ranks)


def test_bpe_tokenize_empty_string_returns_empty_list():
    bpe = _tiny_bpe()
    assert bpe.tokenize("") == []


def test_bpe_tokenize_returns_list_of_strings():
    bpe = _tiny_bpe()
    tokens = bpe.tokenize("low low")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_bpe_tokenize_is_deterministic_across_calls():
    bpe = _tiny_bpe()
    first = bpe.tokenize("lowest low")
    second = bpe.tokenize("lowest low")
    assert first == second


def test_bpe_applies_available_merge():
    bpe = _tiny_bpe()
    tokens = bpe.tokenize("lo")
    # The only merge ('l', 'o') should be applied, so some token contains "lo"
    # and no bare "l" followed by a bare "o" remains as separate pieces.
    assert any("lo" in t for t in tokens)
    assert not any(t == "l" for t in tokens)


# ---------------------------------------------------------------------------
# GPT_2
# ---------------------------------------------------------------------------

def test_gpt_2_forward_output_shape():
    model = _small_gpt_2()
    model.eval()
    inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    with torch.no_grad():
        logits = model(inputs)
    assert logits.shape == (2, 4, 11)


def test_gpt_2_forward_is_deterministic_in_eval_mode():
    model = _small_gpt_2()
    model.eval()
    inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        first = model(inputs)
        second = model(inputs)
    assert torch.allclose(first, second, atol=1e-6)


def test_gpt_2_same_seed_gives_same_weights():
    torch.manual_seed(0)
    a = GPT_2(n_vocab=11, n_ctx=8, n_embd=8, n_head=2, n_layer=1)
    torch.manual_seed(0)
    b = GPT_2(n_vocab=11, n_ctx=8, n_embd=8, n_head=2, n_layer=1)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.allclose(sa[key].float(), sb[key].float(), atol=1e-7)


def test_gpt_2_is_causal():
    # Changing a future token must not change logits at earlier positions.
    torch.manual_seed(0)
    model = _small_gpt_2()
    model.eval()
    base = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    changed = base.clone()
    changed[0, -1] = 9
    with torch.no_grad():
        out_base = model(base)
        out_changed = model(changed)
    assert torch.allclose(out_base[:, :-1], out_changed[:, :-1], atol=1e-5)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_greedy_follows_fake_model_rule(monkeypatch):
    torch.manual_seed(0)
    vocab = 6
    model = _FakeLM(vocab)
    model.eval()
    outputs = generate(model, [[0]], eos_id=vocab - 1, length=3, top_k=1)
    # Greedy decoding with the fake rule should produce 0 -> 1 -> 2 -> 3 style sequences.
    seq = outputs[0] if isinstance(outputs[0], list) else outputs[0].tolist()
    assert seq[:4] == [0, 1, 2, 3]


def test_generate_stops_at_eos_token():
    torch.manual_seed(0)
    vocab = 4
    model = _FakeLM(vocab)
    model.eval()
    # Starting at 1, the rule produces 2, then 3 (the eos token), which should end generation.
    outputs = generate(model, [[1]], eos_id=3, length=10, top_k=1)
    seq = outputs[0] if isinstance(outputs[0], list) else outputs[0].tolist()
    assert seq[-1] == 3
    assert 3 not in seq[:-1]


def test_generate_is_reproducible_with_same_seed():
    model = _FakeLM(7)
    model.eval()
    torch.manual_seed(42)
    first = generate(model, [[0]], eos_id=6, length=4, top_k=0)
    torch.manual_seed(42)
    second = generate(model, [[0]], eos_id=6, length=4, top_k=0)
    assert first == second
