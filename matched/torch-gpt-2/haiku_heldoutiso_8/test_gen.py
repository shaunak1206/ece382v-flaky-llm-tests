import inspect
import json
import math

import pytest
import torch
import torch.nn as nn

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate, get_bpe_from_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY = dict(n_vocab=20, n_ctx=8, n_embd=16, n_head=2, n_layer=1)

# Accepted constructor/function parameter names for each hyperparameter.
ALIASES = {
    "n_vocab": ("n_vocab", "vocab_size"),
    "n_ctx": ("n_ctx", "n_positions", "max_position_embeddings", "context_length", "max_len"),
    "n_embd": ("n_embd", "embed_dim", "hidden_size", "d_model", "n_state"),
    "n_head": ("n_head", "num_heads", "n_heads", "num_attention_heads"),
    "n_layer": ("n_layer", "num_layers", "n_layers", "num_hidden_layers"),
}


def _kwargs_for(fn, values):
    """Map generic hyperparameter names onto the parameter names fn actually accepts."""
    params = inspect.signature(fn).parameters
    kwargs = {}
    for key, names in ALIASES.items():
        for name in names:
            if name in params:
                kwargs[name] = values[key]
                break
    return kwargs


def _build_model():
    model = GPT_2(**_kwargs_for(GPT_2, TINY))
    model.eval()
    return model


def _first_tensor(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _call_generate(model, prompt, steps):
    """Call generate with the step-count keyword the function accepts."""
    params = inspect.signature(generate).parameters
    for name in ("length", "max_len", "steps", "num_tokens", "n_tokens", "max_new_tokens", "gen_len"):
        if name in params:
            return generate(model, prompt, **{name: steps})
    # Fall back to positional third argument.
    return generate(model, prompt, steps)


@pytest.fixture
def bpe(tmp_path):
    encoder = {"a": 0, "b": 1, "ab": 2, "<|endoftext|>": 3}
    enc_path = tmp_path / "encoder.json"
    enc_path.write_text(json.dumps(encoder))
    vocab_path = tmp_path / "vocab.bpe"
    vocab_path.write_text("#version: 0.2\na b\n")
    return get_bpe_from_files(str(enc_path), str(vocab_path))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return _build_model()


# ---------------------------------------------------------------------------
# gelu
# ---------------------------------------------------------------------------

def test_gelu_zero_maps_to_zero():
    out = gelu(torch.zeros(4))
    assert torch.allclose(out, torch.zeros(4))


def test_gelu_matches_tanh_approximation():
    x = torch.linspace(-4.0, 4.0, steps=33)
    expected = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    assert torch.allclose(gelu(x), expected, atol=1e-6)


def test_gelu_large_positive_approximates_identity():
    x = torch.tensor([10.0, 20.0])
    assert torch.allclose(gelu(x), x, atol=1e-4)


def test_gelu_large_negative_approaches_zero():
    x = torch.tensor([-10.0, -20.0])
    assert torch.allclose(gelu(x), torch.zeros(2), atol=1e-4)


def test_gelu_preserves_shape_and_dtype():
    x = torch.randn(2, 3, 5, dtype=torch.float32)
    out = gelu(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_gelu_is_monotonic_on_positive_inputs():
    x = torch.linspace(0.0, 5.0, steps=50)
    y = gelu(x)
    assert torch.all(y[1:] >= y[:-1])


def test_gelu_supports_backward():
    x = torch.randn(6, requires_grad=True)
    gelu(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# BytePairEncoding
# ---------------------------------------------------------------------------

def test_bpe_is_bytepairencoding_instance(bpe):
    assert isinstance(bpe, BytePairEncoding)


def test_bpe_merges_adjacent_pair_into_single_token(bpe):
    assert bpe.encode("ab") == [2]


def test_bpe_unmerged_characters_map_to_their_ids(bpe):
    # "b a" is not a merge rule, so each character keeps its own id.
    assert bpe.encode("ba") == [1, 0]


def test_bpe_encode_returns_list_of_ints(bpe):
    tokens = bpe.encode("abab")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)


def test_bpe_encode_empty_string_returns_empty_list(bpe):
    assert bpe.encode("") == []


def test_bpe_encode_is_deterministic(bpe):
    assert bpe.encode("abba") == bpe.encode("abba")


# ---------------------------------------------------------------------------
# GPT_2
# ---------------------------------------------------------------------------

def test_gpt2_is_nn_module():
    assert issubclass(GPT_2, nn.Module)


def test_gpt2_has_trainable_parameters(model):
    params = list(model.parameters())
    assert len(params) > 0
    assert any(p.requires_grad for p in params)


def test_gpt2_forward_output_shape(model):
    batch, seq = 2, 5
    ids = torch.randint(0, TINY["n_vocab"], (batch, seq))
    with torch.no_grad():
        out = _first_tensor(model(ids))
    assert out.shape[0] == batch
    assert out.shape[1] == seq
    assert out.shape[-1] == TINY["n_vocab"]


def test_gpt2_eval_forward_is_deterministic(model):
    ids = torch.randint(0, TINY["n_vocab"], (1, 4))
    with torch.no_grad():
        first = _first_tensor(model(ids))
        second = _first_tensor(model(ids))
    assert torch.allclose(first, second)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_returns_prompt_plus_requested_steps(model):
    torch.manual_seed(1)
    prompt = torch.randint(0, TINY["n_vocab"], (1, 3))
    steps = 4
    out = torch.as_tensor(_call_generate(model, prompt, steps))
    assert out.shape[-1] == prompt.shape[-1] + steps


def test_generate_preserves_prompt_prefix(model):
    torch.manual_seed(2)
    prompt = torch.randint(0, TINY["n_vocab"], (1, 3))
    out = torch.as_tensor(_call_generate(model, prompt, 3))
    assert torch.equal(out[..., : prompt.shape[-1]], prompt)


def test_generate_tokens_within_vocabulary(model):
    torch.manual_seed(3)
    prompt = torch.randint(0, TINY["n_vocab"], (1, 2))
    out = torch.as_tensor(_call_generate(model, prompt, 5))
    assert int(out.min()) >= 0
    assert int(out.max()) < TINY["n_vocab"]
