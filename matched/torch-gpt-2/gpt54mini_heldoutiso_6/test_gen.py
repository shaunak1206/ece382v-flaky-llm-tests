import inspect
from types import SimpleNamespace

import pytest
import torch

import torch_gpt_2 as tg


def _gelu_reference(x):
    return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))))


def _make_minimal_bpe():
    class DummyBPE:
        def __init__(self):
            self.vocab = {i: chr(ord("a") + i) for i in range(6)}
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

        def encode(self, text):
            return [self.inv_vocab.get(ch, 0) for ch in text]

        def decode(self, tokens):
            return "".join(self.vocab.get(int(t), "?") for t in tokens)

    return DummyBPE()


def _call_gelu(x):
    gelu = tg.gelu
    try:
        return gelu(x)
    except TypeError:
        return gelu(x=x)


def _build_gpt2():
    cls = tg.GPT_2
    sig = inspect.signature(cls)
    kwargs = {}
    common = {
        "vocab_size": 16,
        "n_vocab": 16,
        "n_ctx": 8,
        "context_size": 8,
        "block_size": 8,
        "n_embd": 8,
        "embed_dim": 8,
        "d_model": 8,
        "n_head": 2,
        "num_heads": 2,
        "n_layer": 2,
        "num_layers": 2,
        "dropout": 0.0,
    }
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in common:
            kwargs[name] = common[name]
        elif param.default is not inspect._empty:
            continue
        else:
            # Best-effort fallback for unknown required args.
            # Use a small integer so constructor can proceed if it accepts generic sizes.
            kwargs[name] = 8
    return cls(**kwargs)


def _extract_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    if hasattr(output, "logits") and torch.is_tensor(output.logits):
        return output.logits
    raise AssertionError(f"Could not extract tensor from output type {type(output)!r}")


def _call_generate(*args, **kwargs):
    fn = tg.generate
    try:
        return fn(*args, **kwargs)
    except TypeError:
        return fn(**kwargs)


def test_module_exports_generate_and_bpe():
    assert hasattr(tg, "generate")
    assert hasattr(tg, "BytePairEncoding")
    assert callable(tg.generate)
    assert tg.BytePairEncoding is not None


def test_gelu_returns_zero_for_zero():
    x = torch.zeros(5)
    y = _call_gelu(x)
    assert torch.allclose(y, torch.zeros_like(x))


def test_gelu_matches_reference_formula():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)
    y = _call_gelu(x)
    expected = _gelu_reference(x)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_gelu_preserves_shape_and_dtype():
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    y = _call_gelu(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_gelu_backward_produces_gradients():
    x = torch.randn(7, requires_grad=True)
    y = _call_gelu(x).sum()
    y.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_gelu_is_close_to_identity_for_large_positive_inputs():
    x = torch.tensor([5.0, 8.0, 10.0], dtype=torch.float32)
    y = _call_gelu(x)
    assert torch.allclose(y, x, atol=1e-3, rtol=1e-3)


def test_gelu_is_close_to_zero_for_large_negative_inputs():
    x = torch.tensor([-5.0, -8.0, -10.0], dtype=torch.float32)
    y = _call_gelu(x)
    assert torch.all(y.abs() < 1e-2)


def test_bytepairencoding_class_has_common_public_methods():
    bpe_cls = tg.BytePairEncoding
    method_names = dir(bpe_cls)
    assert any(name in method_names for name in ("encode", "tokenize"))
    assert any(name in method_names for name in ("decode", "detokenize"))


def test_get_bpe_from_files_raises_for_missing_files(tmp_path):
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    with pytest.raises(Exception):
        tg.get_bpe_from_files(str(vocab_file), str(merges_file))


def test_gpt2_can_be_instantiated_if_constructor_uses_common_names():
    model = _build_gpt2()
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_gpt2_forward_returns_tensor_like_output():
    model = _build_gpt2()
    model.eval()
    x = torch.zeros(2, 4, dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    y = _extract_tensor(out)
    assert y.ndim >= 2
    assert y.shape[0] == 2


def test_gpt2_forward_is_deterministic_in_eval_mode():
    torch.manual_seed(0)
    model = _build_gpt2()
    model.eval()
    x = torch.randint(0, 8, (1, 4), dtype=torch.long)
    with torch.no_grad():
        out1 = _extract_tensor(model(x))
        out2 = _extract_tensor(model(x))
    assert torch.allclose(out1, out2)


def test_generate_returns_text_for_stubbed_model_and_bpe():
    bpe = _make_minimal_bpe()

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

        def forward(self, tokens, *args, **kwargs):
            # Produce logits that deterministically select the next token.
            self.calls += 1
            batch, seq_len = tokens.shape
            vocab = 6
            logits = torch.full((batch, seq_len, vocab), -1000.0)
            next_id = min(seq_len % vocab, vocab - 1)
            logits[:, -1, next_id] = 1000.0
            return logits

    model = DummyModel()

    sig = inspect.signature(tg.generate)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in ("model", "net"):
            kwargs[name] = model
        elif name in ("bpe", "encoder", "tokenizer"):
            kwargs[name] = bpe
        elif name in ("prompt", "text", "start", "seed_text"):
            kwargs[name] = "ab"
        elif name in ("length", "n_tokens", "max_tokens", "num_tokens"):
            kwargs[name] = 4
        elif name in ("temperature",):
            kwargs[name] = 1.0
        elif name in ("top_k",):
            kwargs[name] = 1
        elif param.default is not inspect._empty:
            continue
        else:
            # Use a sensible fallback for any unknown required positional arg.
            if param.annotation is int:
                kwargs[name] = 4
            else:
                kwargs[name] = "ab"

    result = _call_generate(**kwargs)
    assert isinstance(result, str)
    assert len(result) >= 2


def test_generate_preserves_prompt_prefix_for_stubbed_model_and_bpe():
    bpe = _make_minimal_bpe()

    class DummyModel(torch.nn.Module):
        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

        def forward(self, tokens, *args, **kwargs):
            batch, seq_len = tokens.shape
            vocab = 6
            logits = torch.full((batch, seq_len, vocab), -1000.0)
            logits[:, -1, 1] = 1000.0
            return logits

    model = DummyModel()

    sig = inspect.signature(tg.generate)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in ("model", "net"):
            kwargs[name] = model
        elif name in ("bpe", "encoder", "tokenizer"):
            kwargs[name] = bpe
        elif name in ("prompt", "text", "start", "seed_text"):
            kwargs[name] = "ab"
        elif name in ("length", "n_tokens", "max_tokens", "num_tokens"):
            kwargs[name] = 3
        elif name in ("temperature",):
            kwargs[name] = 1.0
        elif name in ("top_k",):
            kwargs[name] = 1
        elif param.default is not inspect._empty:
            continue
        else:
            kwargs[name] = "ab"

    result = _call_generate(**kwargs)
    assert result.startswith("ab")
