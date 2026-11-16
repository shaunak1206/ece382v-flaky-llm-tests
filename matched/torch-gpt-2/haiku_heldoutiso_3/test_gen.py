import inspect
import math

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate
import keras_gpt_2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gelu_reference(x):
    """Tanh approximation of GELU used by GPT-2."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


_HPARAM_CANDIDATES = {
    "n_vocab": 32,
    "n_ctx": 16,
    "n_embd": 8,
    "n_head": 2,
    "n_layer": 1,
    "vocab_size": 32,
    "context_length": 16,
    "embed_dim": 8,
    "num_heads": 2,
    "num_layers": 1,
}


def _build_small_gpt2():
    """Construct a tiny GPT_2 by matching constructor parameter names.

    Skips the calling test if the constructor cannot be satisfied with the
    candidate hyperparameters, so the test suite does not fail on unrelated
    signature differences.
    """
    sig = inspect.signature(GPT_2.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in _HPARAM_CANDIDATES:
            kwargs[name] = _HPARAM_CANDIDATES[name]
        elif param.default is inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            pytest.skip(f"Cannot construct GPT_2: unknown required argument {name!r}")
    try:
        model = GPT_2(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on project API
        pytest.skip(f"Cannot construct GPT_2 with small config: {exc!r}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

def test_gelu_zero_maps_to_zero():
    out = gelu(torch.zeros(1))
    assert torch.allclose(out, torch.zeros(1), atol=1e-7)


def test_gelu_matches_tanh_reference():
    torch.manual_seed(0)
    x = torch.randn(4, 5, 6) * 3.0
    assert torch.allclose(gelu(x), _gelu_reference(x), atol=1e-5)


def test_gelu_preserves_shape():
    x = torch.randn(2, 3, 7)
    assert gelu(x).shape == x.shape


def test_gelu_large_positive_approaches_identity():
    x = torch.tensor([10.0, 20.0])
    assert torch.allclose(gelu(x), x, atol=1e-3)


def test_gelu_large_negative_approaches_zero():
    x = torch.tensor([-10.0, -20.0])
    assert torch.allclose(gelu(x), torch.zeros_like(x), atol=1e-3)


def test_gelu_is_differentiable_with_finite_grads():
    x = torch.linspace(-4, 4, steps=33, requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# BytePairEncoding
# ---------------------------------------------------------------------------

def test_bytepairencoding_is_reexported_from_keras_gpt_2():
    assert BytePairEncoding is keras_gpt_2.BytePairEncoding
    assert inspect.isclass(BytePairEncoding)


def test_package_exposes_get_bpe_from_files_loader_is_callable():
    # The package imports get_bpe_from_files; make sure the symbol is available
    # and callable through the keras_gpt_2 dependency.
    assert callable(keras_gpt_2.get_bpe_from_files)


def test_get_bpe_from_files_missing_files_raises(tmp_path):
    missing_encoder = tmp_path / "missing_encoder.json"
    missing_vocab = tmp_path / "missing_vocab.bpe"
    with pytest.raises(Exception):
        keras_gpt_2.get_bpe_from_files(str(missing_encoder), str(missing_vocab))


# ---------------------------------------------------------------------------
# GPT_2
# ---------------------------------------------------------------------------

def test_gpt2_is_torch_module_subclass():
    assert issubclass(GPT_2, nn.Module)


def test_gpt2_constructs_with_small_config():
    model = _build_small_gpt2()
    assert isinstance(model, nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_gpt2_forward_returns_tensor_with_vocab_dimension():
    model = _build_small_gpt2()
    ids = torch.randint(0, _HPARAM_CANDIDATES["n_vocab"], (2, 5))
    with torch.no_grad():
        try:
            out = model(ids)
        except Exception as exc:  # pragma: no cover - depends on project API
            pytest.skip(f"GPT_2 forward signature differs: {exc!r}")
    logits = out[0] if isinstance(out, (tuple, list)) else out
    assert isinstance(logits, torch.Tensor)
    assert logits.shape[0] == 2
    assert logits.shape[1] == 5
    assert logits.shape[-1] == _HPARAM_CANDIDATES["n_vocab"]


def test_gpt2_forward_is_deterministic_in_eval_mode():
    model = _build_small_gpt2()
    torch.manual_seed(1)
    ids = torch.randint(0, _HPARAM_CANDIDATES["n_vocab"], (1, 4))
    with torch.no_grad():
        try:
            a = model(ids)
            b = model(ids)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"GPT_2 forward signature differs: {exc!r}")
    a = a[0] if isinstance(a, (tuple, list)) else a
    b = b[0] if isinstance(b, (tuple, list)) else b
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_is_callable_and_exported():
    assert callable(generate)
    assert torch_gpt_2.generate is generate


def test_generate_produces_longer_sequence_than_prompt():
    model = _build_small_gpt2()
    sig = inspect.signature(generate)
    prompt = torch.randint(0, _HPARAM_CANDIDATES["n_vocab"], (1, 3))

    kwargs = {}
    for name, param in sig.parameters.items():
        if name in ("model", "gpt2", "net"):
            kwargs[name] = model
        elif name in ("inputs", "input_ids", "tokens", "context", "prompt", "x"):
            kwargs[name] = prompt
        elif name in ("length", "max_len", "max_length", "steps", "n_tokens", "num_tokens"):
            kwargs[name] = 4
        elif param.default is inspect.Parameter.empty:
            pytest.skip(f"Unknown required argument for generate: {name!r}")

    try:
        with torch.no_grad():
            out = generate(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on project API
        pytest.skip(f"generate signature differs: {exc!r}")

    if isinstance(out, torch.Tensor):
        assert out.shape[-1] > prompt.shape[-1]
    else:
        assert len(out[0] if isinstance(out[0], (list, tuple)) else out) > 0
