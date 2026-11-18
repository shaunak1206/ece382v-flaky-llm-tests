import inspect
import json
from pathlib import Path

import pytest
import torch

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate, get_bpe_from_files


def _call_first(obj, names, *args, **kwargs):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)(*args, **kwargs), name
    raise AttributeError(f"None of the methods exist on {type(obj).__name__}: {names}")


def _make_minimal_bpe_files(tmp_path: Path):
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"

    # Small, deterministic vocabulary; the exact format is what GPT-2 BPE loaders
    # typically expect.
    vocab = {
        "a": 0,
        "b": 1,
        "ab": 2,
        "<|endoftext|>": 3,
    }
    vocab_path.write_text(json.dumps(vocab, sort_keys=True), encoding="utf-8")
    merges_path.write_text("#version: 0.2\na b\n", encoding="utf-8")
    return vocab_path, merges_path


def _make_tiny_gpt2():
    sig = inspect.signature(GPT_2)
    kwargs = {}
    # Populate common GPT-2 constructor parameter names with tiny values.
    candidates = {
        "vocab_size": 8,
        "n_vocab": 8,
        "n_ctx": 4,
        "n_positions": 4,
        "n_embd": 8,
        "embd_dim": 8,
        "hidden_size": 8,
        "n_head": 2,
        "n_layer": 2,
        "dropout": 0.0,
        "embd_pdrop": 0.0,
        "resid_pdrop": 0.0,
        "attn_pdrop": 0.0,
    }

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in candidates:
            kwargs[name] = candidates[name]
        elif param.default is inspect._empty:
            # Unknown required argument; cannot safely instantiate.
            raise TypeError(f"Unsupported required GPT_2 constructor arg: {name}")

    return GPT_2(**kwargs)


def test_init_reexports_expected_symbols():
    assert torch_gpt_2.BytePairEncoding is BytePairEncoding
    assert torch_gpt_2.GPT_2 is GPT_2
    assert torch_gpt_2.gelu is gelu
    assert torch_gpt_2.generate is generate
    assert torch_gpt_2.get_bpe_from_files is get_bpe_from_files


def test_gelu_matches_reference_formula_on_known_values():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    actual = gelu(x)
    expected = 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_gelu_preserves_shape_and_dtype():
    x = torch.randn(2, 3, dtype=torch.float32)
    y = gelu(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_gelu_is_zero_at_zero():
    x = torch.tensor(0.0)
    y = gelu(x)
    assert torch.isclose(y, torch.tensor(0.0), atol=0.0, rtol=0.0)


def test_gelu_is_monotonic_on_a_sorted_input():
    x = torch.linspace(-3, 3, steps=25)
    y = gelu(x)
    diffs = y[1:] - y[:-1]
    assert torch.all(diffs >= -1e-7)


def test_generate_reexport_points_to_same_object_as_module_level():
    import torch_gpt_2.gen as gen_module

    assert generate is gen_module.generate


def test_generate_is_callable():
    assert callable(generate)


def test_bytepairencoding_can_be_loaded_from_files(tmp_path):
    vocab_path, merges_path = _make_minimal_bpe_files(tmp_path)
    bpe = get_bpe_from_files(str(vocab_path), str(merges_path))
    assert isinstance(bpe, BytePairEncoding)


def test_bytepairencoding_exposes_common_string_roundtrip_methods(tmp_path):
    vocab_path, merges_path = _make_minimal_bpe_files(tmp_path)
    bpe = get_bpe_from_files(str(vocab_path), str(merges_path))

    encoded, enc_name = _call_first(bpe, ["encode", "tokenize"], "ab")
    assert enc_name in {"encode", "tokenize"}
    assert encoded is not None

    decoded, dec_name = _call_first(bpe, ["decode", "detokenize"], encoded)
    assert dec_name in {"decode", "detokenize"}
    assert isinstance(decoded, str)
    assert decoded


def test_bytepairencoding_encode_is_deterministic(tmp_path):
    vocab_path, merges_path = _make_minimal_bpe_files(tmp_path)
    bpe = get_bpe_from_files(str(vocab_path), str(merges_path))

    encoded1, _ = _call_first(bpe, ["encode", "tokenize"], "ab")
    encoded2, _ = _call_first(bpe, ["encode", "tokenize"], "ab")
    assert encoded1 == encoded2


def test_bytepairencoding_decode_is_deterministic(tmp_path):
    vocab_path, merges_path = _make_minimal_bpe_files(tmp_path)
    bpe = get_bpe_from_files(str(vocab_path), str(merges_path))

    encoded, _ = _call_first(bpe, ["encode", "tokenize"], "ab")
    decoded1, _ = _call_first(bpe, ["decode", "detokenize"], encoded)
    decoded2, _ = _call_first(bpe, ["decode", "detokenize"], encoded)
    assert decoded1 == decoded2


def test_gpt2_constructor_is_inspectable():
    sig = inspect.signature(GPT_2)
    assert isinstance(sig, inspect.Signature)
    assert len(sig.parameters) > 0


def test_gpt2_can_be_instantiated_with_tiny_dimensions_if_supported():
    model = _make_tiny_gpt2()
    assert isinstance(model, GPT_2)


def test_gpt2_forward_accepts_integer_token_tensor_if_supported():
    model = _make_tiny_gpt2()
    model.eval()

    x = torch.zeros((1, 4), dtype=torch.long)
    with torch.no_grad():
        out = model(x)

    # The exact output type varies by implementation, but it should be tensor-like
    # and preserve the batch dimension.
    assert hasattr(out, "shape")
    assert out.shape[0] == 1


def test_gpt2_forward_is_deterministic_in_eval_mode_if_supported():
    model = _make_tiny_gpt2()
    model.eval()

    x = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2, atol=1e-6, rtol=1e-6)
