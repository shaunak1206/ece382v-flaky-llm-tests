import inspect
from types import SimpleNamespace

import pytest
import torch

import torch_gpt_2 as tgpt2


def _generate_call_kwargs(generate_fn, model, bpe, text, length=1, temperature=1.0, top_k=0):
    """
    Build kwargs for generate() based on its signature so tests stay resilient
    to small API differences.
    """
    sig = inspect.signature(generate_fn)
    kwargs = {}
    params = sig.parameters

    # Common positional / keyword names used by GPT-2 generators.
    if "model" in params:
        kwargs["model"] = model
    if "bpe" in params:
        kwargs["bpe"] = bpe
    if "text" in params:
        kwargs["text"] = text
    elif "prompt" in params:
        kwargs["prompt"] = text

    if "length" in params:
        kwargs["length"] = length
    elif "n_tokens" in params:
        kwargs["n_tokens"] = length
    elif "steps" in params:
        kwargs["steps"] = length

    if "temperature" in params:
        kwargs["temperature"] = temperature
    if "top_k" in params:
        kwargs["top_k"] = top_k
    if "k" in params and "top_k" not in params:
        kwargs["k"] = top_k

    return kwargs


class _FakeBPE:
    def __init__(self):
        self.vocab = {
            "<|endoftext|>": 0,
            "hello": 1,
            "world": 2,
            "!": 3,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        return [self.vocab[t] for t in text.split()]

    def decode(self, tokens):
        return " ".join(self.inv_vocab[int(t)] for t in tokens)


class _FakeModel:
    def __init__(self, vocab_size=4, preferred_token=2):
        self.vocab_size = vocab_size
        self.preferred_token = preferred_token
        self.eval_called = False
        self.forward_calls = []

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, input_ids):
        # Record the input shape and return logits with a strong preference for
        # one token at the final position.
        self.forward_calls.append(tuple(input_ids.shape))
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, dtype=torch.float32)
        logits[:, :, self.preferred_token] = 10.0
        return logits


def test_module_reexports_bytepairencoding():
    from keras_gpt_2 import BytePairEncoding as KerasBytePairEncoding

    assert tgpt2.BytePairEncoding is KerasBytePairEncoding


def test_module_reexports_gpt2_class():
    assert hasattr(tgpt2, "GPT_2")
    assert inspect.isclass(tgpt2.GPT_2)


def test_gpt2_is_a_torch_module():
    assert issubclass(tgpt2.GPT_2, torch.nn.Module)


def test_module_reexports_gelu_function():
    assert callable(tgpt2.gelu)


def test_gelu_zero_is_zero():
    x = torch.tensor([0.0], dtype=torch.float32)
    y = tgpt2.gelu(x)
    assert torch.allclose(y, torch.tensor([0.0], dtype=torch.float32), atol=1e-7)


def test_gelu_matches_standard_formula():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    expected = 0.5 * x * (1.0 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x ** 3)
    ))
    actual = tgpt2.gelu(x)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_gelu_preserves_shape_and_dtype():
    x = torch.randn(2, 3, dtype=torch.float32)
    y = tgpt2.gelu(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_gelu_is_non_decreasing_on_sorted_inputs():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)
    y = tgpt2.gelu(x)
    assert torch.all(y[1:] >= y[:-1] - 1e-6)


def test_generate_exists_and_is_callable():
    assert callable(tgpt2.generate)


def test_generate_appends_predicted_token(monkeypatch):
    fake_bpe = _FakeBPE()
    fake_model = _FakeModel(preferred_token=2)

    # Make sampling deterministic in case generate() uses torch.multinomial.
    monkeypatch.setattr(torch, "multinomial", lambda probs, num_samples, replacement=False: torch.zeros((probs.shape[0], num_samples), dtype=torch.long))

    kwargs = _generate_call_kwargs(tgpt2.generate, fake_model, fake_bpe, "hello", length=1, temperature=1.0, top_k=0)
    result = tgpt2.generate(**kwargs)

    assert fake_model.eval_called is True
    assert "hello" in result
    assert "world" in result


def test_generate_length_zero_returns_prompt(monkeypatch):
    fake_bpe = _FakeBPE()
    fake_model = _FakeModel(preferred_token=2)

    monkeypatch.setattr(torch, "multinomial", lambda probs, num_samples, replacement=False: torch.zeros((probs.shape[0], num_samples), dtype=torch.long))

    kwargs = _generate_call_kwargs(tgpt2.generate, fake_model, fake_bpe, "hello", length=0, temperature=1.0, top_k=0)
    result = tgpt2.generate(**kwargs)

    assert result == "hello"


def test_generate_uses_model_forward(monkeypatch):
    fake_bpe = _FakeBPE()
    fake_model = _FakeModel(preferred_token=2)

    monkeypatch.setattr(torch, "multinomial", lambda probs, num_samples, replacement=False: torch.zeros((probs.shape[0], num_samples), dtype=torch.long))

    kwargs = _generate_call_kwargs(tgpt2.generate, fake_model, fake_bpe, "hello", length=2, temperature=1.0, top_k=0)
    _ = tgpt2.generate(**kwargs)

    assert fake_model.forward_calls, "generate() should call the model at least once"
    assert all(len(shape) == 2 for shape in fake_model.forward_calls)


def test_gpt2_class_has_forward_method():
    assert hasattr(tgpt2.GPT_2, "forward")
    assert callable(getattr(tgpt2.GPT_2, "forward"))
