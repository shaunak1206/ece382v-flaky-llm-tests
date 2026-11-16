import importlib
import json
import math

import pytest

torch = pytest.importorskip("torch")

import torch_gpt_2
from torch_gpt_2 import BytePairEncoding, get_bpe_from_files


def _resolve(name, *submodules):
    """Look up a public name on the package, falling back to submodules."""
    obj = getattr(torch_gpt_2, name, None)
    if obj is not None:
        return obj
    for sub in submodules:
        try:
            mod = importlib.import_module(f"torch_gpt_2.{sub}")
        except ImportError:
            continue
        obj = getattr(mod, name, None)
        if obj is not None:
            return obj
    pytest.skip(f"{name} is not available in torch_gpt_2")


def _reference_gelu(x):
    """Tanh approximation of GELU as used by GPT-2."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


@pytest.fixture
def bpe_files(tmp_path):
    encoder = tmp_path / "encoder.json"
    vocab = tmp_path / "vocab.bpe"
    encoder.write_text(json.dumps({"a": 0, "b": 1, "ab": 2}), encoding="utf-8")
    vocab.write_text("#version: 0.2\na b\n", encoding="utf-8")
    return str(encoder), str(vocab)


# ---------------------------------------------------------------------------
# BytePairEncoding
# ---------------------------------------------------------------------------

def test_get_bpe_from_files_returns_bytepairencoding(bpe_files):
    encoder_path, vocab_path = bpe_files
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    assert isinstance(bpe, BytePairEncoding)


def test_get_bpe_from_files_missing_file_raises(tmp_path):
    missing_encoder = str(tmp_path / "missing_encoder.json")
    missing_vocab = str(tmp_path / "missing_vocab.bpe")
    with pytest.raises(OSError):
        get_bpe_from_files(missing_encoder, missing_vocab)


def test_bpe_encode_returns_list(bpe_files):
    bpe = get_bpe_from_files(*bpe_files)
    result = bpe.encode("ab")
    assert isinstance(result, list)
    assert len(result) > 0


def test_bpe_encode_empty_string_returns_empty_list(bpe_files):
    bpe = get_bpe_from_files(*bpe_files)
    assert bpe.encode("") == []


def test_bpe_encode_is_deterministic(bpe_files):
    bpe = get_bpe_from_files(*bpe_files)
    first = bpe.encode("abab ba")
    second = bpe.encode("abab ba")
    assert first == second


def test_bpe_encode_handles_non_ascii_without_error(bpe_files):
    bpe = get_bpe_from_files(*bpe_files)
    result = bpe.encode("héllo ✓ 你好")
    assert isinstance(result, list)


def test_bpe_constructed_directly_from_ranks_is_instance():
    try:
        bpe = BytePairEncoding({("a", "b"): 0})
    except TypeError:
        pytest.skip("BytePairEncoding constructor signature differs")
    assert isinstance(bpe, BytePairEncoding)


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

@pytest.fixture
def gelu():
    return _resolve("gelu", "net")


def test_gelu_of_zero_is_zero(gelu):
    out = gelu(torch.zeros(1))
    assert torch.allclose(out, torch.zeros(1), atol=1e-7)


def test_gelu_matches_tanh_reference(gelu):
    xs = torch.linspace(-5.0, 5.0, 101)
    out = gelu(xs)
    expected = torch.tensor([_reference_gelu(float(x)) for x in xs])
    assert torch.allclose(out, expected, atol=1e-5)


def test_gelu_known_value_at_one(gelu):
    out = gelu(torch.tensor([1.0])).item()
    assert abs(out - 0.8412) < 1e-3


def test_gelu_large_positive_approximates_identity(gelu):
    out = gelu(torch.tensor([10.0])).item()
    assert abs(out - 10.0) < 1e-3


def test_gelu_large_negative_approximates_zero(gelu):
    out = gelu(torch.tensor([-10.0])).item()
    assert abs(out) < 1e-3


def test_gelu_is_monotonic_on_positive_inputs(gelu):
    xs = torch.linspace(0.0, 5.0, 50)
    out = gelu(xs)
    assert torch.all(out[1:] >= out[:-1])


def test_gelu_preserves_shape(gelu):
    x = torch.randn(2, 3, 4)
    assert gelu(x).shape == x.shape


def test_gelu_gradient_is_finite(gelu):
    x = torch.randn(8, requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# GPT_2
# ---------------------------------------------------------------------------

def test_gpt2_is_torch_module_subclass():
    GPT_2 = _resolve("GPT_2", "net")
    assert isinstance(GPT_2, type)
    assert issubclass(GPT_2, torch.nn.Module)


def test_gpt2_defines_forward():
    GPT_2 = _resolve("GPT_2", "net")
    assert callable(getattr(GPT_2, "forward", None))


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_is_callable():
    generate = _resolve("generate", "gen")
    assert callable(generate)


def test_generate_takes_at_least_one_argument():
    generate = _resolve("generate", "gen")
    params = inspect_parameters(generate)
    assert len(params) >= 1


def inspect_parameters(func):
    import inspect
    return list(inspect.signature(func).parameters.values())
