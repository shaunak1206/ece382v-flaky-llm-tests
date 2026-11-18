import inspect
import json
import math
from pathlib import Path

import pytest
import torch

import torch_gpt_2


def _call_with_supported_kwargs(callable_obj, candidates):
    """
    Build kwargs from `candidates` matching the callable's signature.
    Returns (result, used_kwargs) when callable succeeds.
    """
    sig = inspect.signature(callable_obj)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in candidates:
            kwargs[name] = candidates[name]
        elif param.default is inspect._empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(f"Missing required parameter: {name}")
    return callable_obj(**kwargs), kwargs


def _maybe_instantiate_gpt2():
    cls = torch_gpt_2.GPT_2
    sig = inspect.signature(cls)

    candidates = {
        # common GPT-2 constructor names
        "vocab_size": 32,
        "n_vocab": 32,
        "vocab": 32,
        "context_size": 8,
        "ctx": 8,
        "n_ctx": 8,
        "n_positions": 8,
        "max_seq_len": 8,
        "block_size": 8,
        "sequence_length": 8,
        "n_layer": 2,
        "layers": 2,
        "num_layers": 2,
        "n_head": 2,
        "heads": 2,
        "num_heads": 2,
        "n_embd": 16,
        "embd_dim": 16,
        "embedding_dim": 16,
        "hidden_size": 16,
        "dropout": 0.0,
    }

    kwargs = {}
    for name, param in sig.parameters.items():
        if name in candidates:
            kwargs[name] = candidates[name]
        elif param.default is inspect._empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(f"Unsupported required parameter: {name}")

    return cls(**kwargs)


def _make_bpe():
    """
    Attempt to construct BytePairEncoding using common constructor names.
    """
    cls = torch_gpt_2.BytePairEncoding
    sig = inspect.signature(cls)

    # Minimal GPT-2-like toy vocab and merges.
    encoder = {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "lo": 6, "hello": 7}
    merges = [("h", "e"), ("l", "l"), ("l", "o"), ("he", "ll"), ("hell", "o")]

    candidates = {
        "encoder": encoder,
        "vocab": encoder,
        "token_to_id": encoder,
        "bpe_merges": merges,
        "merges": merges,
    }

    kwargs = {}
    for name, param in sig.parameters.items():
        if name in candidates:
            kwargs[name] = candidates[name]
        elif param.default is inspect._empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(f"Unsupported required parameter: {name}")

    return cls(**kwargs)


def test_module_exports_generate():
    assert hasattr(torch_gpt_2, "generate")
    assert callable(torch_gpt_2.generate)


def test_module_exports_bpe_and_get_bpe_from_files():
    assert hasattr(torch_gpt_2, "BytePairEncoding")
    assert hasattr(torch_gpt_2, "get_bpe_from_files")
    assert callable(torch_gpt_2.get_bpe_from_files)


def test_gelu_zero_is_zero():
    x = torch.tensor([0.0])
    y = torch_gpt_2.gelu(x)
    assert torch.isclose(y, torch.tensor([0.0])).all()


def test_gelu_preserves_shape_and_dtype():
    x = torch.randn(2, 3, dtype=torch.float32)
    y = torch_gpt_2.gelu(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_gelu_matches_reference_formula():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    y = torch_gpt_2.gelu(x)
    ref = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    assert torch.allclose(y, ref, atol=1e-7, rtol=1e-7)


def test_gelu_is_odd_like_for_small_values_signs():
    x = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    y = torch_gpt_2.gelu(x)
    assert y[0] < 0
    assert y[1] > 0


def test_byte_pair_encoding_construction():
    bpe = _make_bpe()
    assert isinstance(bpe, torch_gpt_2.BytePairEncoding)


def test_byte_pair_encoding_has_encode_and_decode():
    bpe = _make_bpe()
    assert hasattr(bpe, "encode")
    assert hasattr(bpe, "decode")
    assert callable(bpe.encode)
    assert callable(bpe.decode)


def test_byte_pair_encoding_encode_decode_roundtrip_simple():
    bpe = _make_bpe()

    encoded = bpe.encode("hello")
    assert isinstance(encoded, (list, tuple))
    assert len(encoded) > 0
    assert all(isinstance(i, int) for i in encoded)

    decoded = bpe.decode(encoded)
    assert isinstance(decoded, str)
    assert decoded == "hello"


def test_byte_pair_encoding_encode_is_deterministic():
    bpe = _make_bpe()
    text = "hello hello"
    assert bpe.encode(text) == bpe.encode(text)


def test_get_bpe_from_files_reads_files(tmp_path):
    vocab = {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "lo": 6, "hello": 7}
    merges = ["#version: 0.2", "h e", "l l", "l o", "he ll", "hell o"]

    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
    merges_path.write_text("\n".join(merges), encoding="utf-8")

    bpe = torch_gpt_2.get_bpe_from_files(str(vocab_path), str(merges_path))
    assert isinstance(bpe, torch_gpt_2.BytePairEncoding)
    assert bpe.encode("hello") == bpe.encode("hello")


def test_gpt2_can_be_instantiated_with_small_dimensions():
    model = _maybe_instantiate_gpt2()
    assert isinstance(model, torch_gpt_2.GPT_2)


def test_gpt2_has_forward_method():
    model = _maybe_instantiate_gpt2()
    assert callable(getattr(model, "forward"))


def test_gpt2_forward_returns_tensor_with_batch_and_sequence_dims():
    model = _maybe_instantiate_gpt2()
    model.eval()

    # Find likely vocab size from common attributes; fall back to 32.
    vocab_size = None
    for attr in ("vocab_size", "n_vocab"):
        if hasattr(model, attr):
            vocab_size = int(getattr(model, attr))
            break
    if vocab_size is None:
        vocab_size = 32

    x = torch.randint(0, vocab_size, (2, 5), dtype=torch.long)

    with torch.no_grad():
        out = model(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2
    assert out.shape[1] == 5


def test_gpt2_forward_is_deterministic_in_eval_mode():
    torch.manual_seed(0)
    model = _maybe_instantiate_gpt2()
    model.eval()

    vocab_size = None
    for attr in ("vocab_size", "n_vocab"):
        if hasattr(model, attr):
            vocab_size = int(getattr(model, attr))
            break
    if vocab_size is None:
        vocab_size = 32

    x = torch.randint(0, vocab_size, (1, 4), dtype=torch.long)

    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)

    assert torch.allclose(y1, y2)


def test_generate_is_callable_and_returns_string_or_sequence(monkeypatch):
    # Patch the imported generate function's globals so we can call it without
    # relying on heavy model loading or sampling internals.
    gen_fn = torch_gpt_2.generate
    if not callable(gen_fn):
        pytest.skip("generate is not callable")

    # Try to make a minimal smoke test by monkeypatching common global dependencies.
    # If the implementation doesn't use these names, the test still checks importability.
    g = getattr(gen_fn, "__globals__", {})
    original = {}

    def _patch(name, value):
        if name in g:
            original[name] = g[name]
            g[name] = value

    try:
        # Common dependencies in sampling/generation code.
        _patch("torch", torch)
        _patch("np", __import__("numpy") if "numpy" in globals() else None)
        # The function should at least be directly callable.
        sig = inspect.signature(gen_fn)
        kwargs = {}
        for name, param in sig.parameters.items():
            if param.default is not inspect._empty:
                continue
            # Provide simple placeholders for the first required args if any.
            if "prompt" in name or "text" in name:
                kwargs[name] = "hello"
            elif "length" in name or "n" == name or "num" in name:
                kwargs[name] = 1
            elif "top" in name:
                kwargs[name] = 1.0
            elif "temperature" in name:
                kwargs[name] = 1.0
            elif "seed" in name:
                kwargs[name] = 0
            elif "model" in name:
                # Best-effort: use a tiny model if needed.
                try:
                    kwargs[name] = _maybe_instantiate_gpt2()
                except Exception:
                    kwargs[name] = None
            elif "bpe" in name or "tokenizer" in name:
                kwargs[name] = _make_bpe()
            else:
                # Unknown required parameter; skip rather than make the test brittle.
                pytest.skip(f"Unsupported required parameter in generate(): {name}")

        result = gen_fn(**kwargs)
        assert isinstance(result, (str, list, tuple, torch.Tensor))
    finally:
        for name, value in original.items():
            g[name] = value
