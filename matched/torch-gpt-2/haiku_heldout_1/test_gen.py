import inspect

import numpy as np
import pytest
import torch

import torch_gpt_2
import keras_gpt_2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 11
FAVORED = 3


class _ConstantLM(torch.nn.Module):
    """Tiny deterministic LM: always strongly prefers token FAVORED."""

    def __init__(self, vocab=VOCAB, favored=FAVORED):
        super().__init__()
        self.vocab = vocab
        self.favored = favored
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, seq = x.shape[0], x.shape[1]
        logits = torch.zeros(batch, seq, self.vocab, dtype=torch.float32)
        logits[..., self.favored] = 100.0
        return logits


def _as_token_list(result):
    """Normalize generate() output to a flat list of ints for the first batch row."""
    if torch.is_tensor(result):
        result = result.tolist()
    if len(result) > 0 and isinstance(result[0], (list, tuple)):
        result = result[0]
    return [int(t) for t in result]


def _reference_gelu(x):
    """Tanh approximation of GELU (as used by GPT-2)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _get_gelu():
    fn = getattr(torch_gpt_2, "gelu", None)
    if fn is None:
        pytest.skip("gelu is not exposed by torch_gpt_2")
    return fn


def _get_generate():
    fn = getattr(torch_gpt_2, "generate", None)
    if fn is None:
        pytest.skip("generate is not exposed by torch_gpt_2")
    return fn


def _build_gpt2():
    cls = getattr(torch_gpt_2, "GPT_2", None)
    if cls is None:
        pytest.skip("GPT_2 is not exposed by torch_gpt_2")
    hparams = dict(n_vocab=50, n_ctx=16, n_embd=8, n_head=2, n_layer=1)
    try:
        return cls(hparams)
    except TypeError:
        pass
    try:
        return cls(**hparams)
    except TypeError:
        pytest.skip("GPT_2 constructor signature not supported by these tests")


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------

def test_package_exports_expected_names():
    assert hasattr(torch_gpt_2, "BytePairEncoding")
    assert hasattr(torch_gpt_2, "get_bpe_from_files")
    assert hasattr(torch_gpt_2, "generate")
    assert hasattr(torch_gpt_2, "GPT_2")


def test_bpe_is_reexported_from_keras_gpt_2():
    assert torch_gpt_2.BytePairEncoding is keras_gpt_2.BytePairEncoding


def test_get_bpe_from_files_is_reexported_from_keras_gpt_2():
    assert torch_gpt_2.get_bpe_from_files is keras_gpt_2.get_bpe_from_files


def test_get_bpe_from_files_missing_files_raises(tmp_path):
    missing_encoder = tmp_path / "does_not_exist_encoder.json"
    missing_vocab = tmp_path / "does_not_exist_vocab.bpe"
    with pytest.raises(OSError):
        torch_gpt_2.get_bpe_from_files(str(missing_encoder), str(missing_vocab))


def test_bpe_class_is_a_class():
    assert inspect.isclass(torch_gpt_2.BytePairEncoding)


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

def test_gelu_at_zero_is_zero():
    gelu = _get_gelu()
    out = gelu(torch.tensor([0.0]))
    assert torch.allclose(out, torch.tensor([0.0]), atol=1e-7)


def test_gelu_matches_reference_tanh_approximation():
    gelu = _get_gelu()
    xs = np.linspace(-5.0, 5.0, 101).astype(np.float32)
    out = gelu(torch.from_numpy(xs)).numpy()
    expected = _reference_gelu(xs.astype(np.float64))
    # Tolerance covers small differences between tanh and erf-based variants.
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_gelu_large_positive_inputs_are_close_to_identity():
    gelu = _get_gelu()
    x = torch.tensor([6.0, 8.0, 10.0])
    out = gelu(x)
    assert torch.allclose(out, x, atol=1e-3)


def test_gelu_large_negative_inputs_are_close_to_zero():
    gelu = _get_gelu()
    x = torch.tensor([-8.0, -10.0])
    out = gelu(x)
    assert torch.all(out.abs() < 1e-3)


def test_gelu_preserves_shape_and_dtype():
    gelu = _get_gelu()
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    out = gelu(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float32


def test_gelu_gradient_is_finite_and_defined():
    gelu = _get_gelu()
    x = torch.linspace(-3.0, 3.0, 25, requires_grad=True)
    y = gelu(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# GPT_2
# ---------------------------------------------------------------------------

def test_gpt2_is_torch_module():
    model = _build_gpt2()
    assert isinstance(model, torch.nn.Module)


def test_gpt2_forward_returns_vocab_sized_logits():
    torch.manual_seed(0)
    model = _build_gpt2()
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape[0] == 1
    assert logits.shape[1] == 4
    assert logits.shape[2] == 50


def test_gpt2_eval_forward_is_deterministic():
    torch.manual_seed(0)
    model = _build_gpt2()
    model.eval()
    tokens = torch.tensor([[5, 6, 7]], dtype=torch.long)
    with torch.no_grad():
        first = model(tokens)
        second = model(tokens)
    assert torch.allclose(first, second)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_appends_requested_number_of_tokens():
    generate = _get_generate()
    model = _ConstantLM()
    context = [1, 2]
    out = generate(model, context, 4)
    tokens = _as_token_list(out)
    assert len(tokens) == len(context) + 4


def test_generate_preserves_context_prefix():
    generate = _get_generate()
    model = _ConstantLM()
    context = [1, 2, 5]
    out = generate(model, context, 3)
    tokens = _as_token_list(out)
    assert tokens[:len(context)] == context


def test_generate_picks_strongly_favored_token():
    generate = _get_generate()
    model = _ConstantLM(favored=FAVORED)
    context = [1, 2]
    out = generate(model, context, 5)
    tokens = _as_token_list(out)
    assert tokens[len(context):] == [FAVORED] * 5


def test_generate_zero_length_returns_context_only():
    generate = _get_generate()
    model = _ConstantLM()
    context = [4, 5, 6]
    out = generate(model, context, 0)
    tokens = _as_token_list(out)
    assert tokens == context
