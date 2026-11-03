import inspect

import pytest
import torch

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


def _can_instantiate_without_required_args(cls):
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is inspect._empty and param.kind not in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD,
        ):
            return False
    return True


def _maybe_instantiate(cls):
    if _can_instantiate_without_required_args(cls):
        return cls()
    pytest.skip(f"{cls.__name__} requires constructor arguments")


def test_package_reexports_byte_pair_encoding():
    import keras_gpt_2

    assert torch_gpt_2.BytePairEncoding is keras_gpt_2.BytePairEncoding


def test_package_reexports_generate():
    import torch_gpt_2.gen as gen_module

    assert torch_gpt_2.generate is gen_module.generate


def test_package_reexports_gelu_from_net():
    import torch_gpt_2.net as net_module

    assert torch_gpt_2.gelu is net_module.gelu


def test_package_reexports_gpt2_class():
    import torch_gpt_2.net as net_module

    assert torch_gpt_2.GPT_2 is net_module.GPT_2


def test_gelu_maps_zero_to_zero():
    x = torch.tensor(0.0)
    y = gelu(x)
    assert torch.is_tensor(y)
    assert y.item() == pytest.approx(0.0)


def test_gelu_matches_reference_formula_on_sample_tensor():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    y = gelu(x)

    expected = 0.5 * x * (
        1.0
        + torch.tanh(
            (torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x.pow(3)))
        )
    )
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_gelu_preserves_shape_and_dtype():
    x = torch.randn(2, 3, dtype=torch.float32)
    y = gelu(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_gelu_accepts_scalar_tensor():
    x = torch.tensor(1.5)
    y = gelu(x)
    assert y.shape == x.shape
    assert y.item() == pytest.approx(float(gelu(torch.tensor([1.5])).item()))


def test_bytepairencoding_is_instantiable_or_skipped():
    bpe = _maybe_instantiate(BytePairEncoding)
    assert isinstance(bpe, BytePairEncoding)


def test_bytepairencoding_exposes_some_public_api():
    public_attrs = [name for name in dir(BytePairEncoding) if not name.startswith("_")]
    assert any(name in public_attrs for name in ("encode", "decode", "tokenize", "vocab_size"))


def test_bytepairencoding_encode_decode_roundtrip_if_available():
    bpe = _maybe_instantiate(BytePairEncoding)

    if not hasattr(bpe, "encode") or not hasattr(bpe, "decode"):
        pytest.skip("BytePairEncoding does not expose encode/decode")

    text = "Hello, world!"
    encoded = bpe.encode(text)
    decoded = bpe.decode(encoded)

    assert isinstance(encoded, (list, tuple))
    assert decoded == text


def test_bytepairencoding_empty_string_roundtrip_if_available():
    bpe = _maybe_instantiate(BytePairEncoding)

    if not hasattr(bpe, "encode") or not hasattr(bpe, "decode"):
        pytest.skip("BytePairEncoding does not expose encode/decode")

    encoded = bpe.encode("")
    decoded = bpe.decode(encoded)

    assert decoded == ""


def test_bytepairencoding_encode_returns_integers_if_available():
    bpe = _maybe_instantiate(BytePairEncoding)

    if not hasattr(bpe, "encode"):
        pytest.skip("BytePairEncoding does not expose encode")

    encoded = bpe.encode("abc")
    assert all(isinstance(i, int) for i in encoded)


def test_gpt2_is_reexported_class():
    assert isinstance(GPT_2, type)


def test_gpt2_can_be_instantiated_or_skipped():
    model = _maybe_instantiate(GPT_2)
    assert isinstance(model, GPT_2)


def test_generate_is_callable():
    assert callable(generate)


def test_generate_signature_has_parameters():
    sig = inspect.signature(generate)
    assert len(sig.parameters) >= 1
