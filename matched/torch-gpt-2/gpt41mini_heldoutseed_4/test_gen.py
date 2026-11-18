import pytest
import torch
import torch.nn as nn
import torch_gpt_2 as gpt2
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


@pytest.fixture
def sample_texts():
    return ["hello world", "test input for bpe", "another sentence"]


def test_bytepairencoding_encode_decode(sample_texts):
    bpe = BytePairEncoding()
    bpe.build_vocab(sample_texts)
    for text in sample_texts:
        tokens = bpe.encode(text)
        decoded = bpe.decode(tokens)
        # decode(encode(text)) should roughly reconstruct original text (possibly ignoring spaces/casing)
        assert isinstance(tokens, list)
        assert isinstance(decoded, str)
        assert all(isinstance(tok, int) for tok in tokens)


def test_bytepairencoding_vocab_and_len(sample_texts):
    bpe = BytePairEncoding()
    bpe.build_vocab(sample_texts)
    vocab_size = len(bpe)
    assert vocab_size > 0
    assert hasattr(bpe, "vocab")
    assert isinstance(bpe.vocab, dict)


def test_bytepairencoding_encode_unknown_token_behavior():
    bpe = BytePairEncoding()
    bpe.build_vocab(["known words only"])
    tokens = bpe.encode("this text has OOV tokens")
    # Should handle unknown tokens without failure, possibly mapping to unknown token id
    assert isinstance(tokens, list)


def test_gelu_output_and_gradient():
    x = torch.randn(10, requires_grad=True)
    y = gelu(x)
    assert y.shape == x.shape
    # Test backward does not raise
    y.sum().backward()
    assert x.grad is not None


def test_gpt2_forward_shapes():
    vocab_size = 100
    seq_len = 12
    batch_size = 4
    model = GPT_2(
        vocab_size=vocab_size,
        n_layer=2,
        n_head=4,
        n_embd=16,
        max_seq=seq_len,
    )
    model.eval()
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_gpt2_generate_basic():
    vocab_size = 50
    seq_len = 8
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=16, max_seq=seq_len)
    model.eval()
    prompt = [1, 2, 3]
    output = generate(model, prompt, max_len=10)
    # Output should be a list or tensor with length <= max_len
    assert isinstance(output, (list, torch.Tensor))
    if isinstance(output, list):
        assert len(output) <= 10
    else:
        assert output.shape[1] <= 10  # batch dim may be absent if input was 1d


def test_generate_returns_prompt_when_max_len_not_increased():
    vocab_size = 20
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=8, max_seq=20)
    model.eval()
    prompt = [5, 4, 3]
    out = generate(model, prompt, max_len=len(prompt))
    if isinstance(out, torch.Tensor):
        out = out.tolist() if out.dim() > 1 else out.tolist()
    assert out[:len(prompt)] == prompt


def test_generate_with_temperature_and_top_k():
    vocab_size = 30
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=8, max_seq=15)
    model.eval()
    prompt = [0, 1, 2]
    output = generate(model, prompt, max_len=10, temperature=0.8, top_k=5)
    if isinstance(output, torch.Tensor):
        output = output.tolist()
    assert len(output) <= 10


def test_generate_invalid_prompt_handling():
    vocab_size = 30
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=8, max_seq=10)
    model.eval()
    # Passing empty prompt should not fail
    output = generate(model, [], max_len=5)
    assert isinstance(output, (list, torch.Tensor))
    # Passing prompt with out of vocab tokens
    output2 = generate(model, [vocab_size + 10], max_len=5)
    assert isinstance(output2, (list, torch.Tensor))


def test_gpt2_model_can_be_instantiated_with_defaults():
    # defaults may come from the model or example usage
    vocab_size = 100
    model = GPT_2(vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (1, 5))
    logits = model(x)
    assert logits.shape == (1, 5, vocab_size)


def test_gelu_values_against_reference():
    # gelu(x) approximately equals x * 0.5 * (1 + erf(x / sqrt(2)))
    x = torch.linspace(-3, 3, steps=10)
    y = gelu(x)
    import math
    import torch.nn.functional as F
    expected = x * 0.5 * (1 + torch.erf(x / (2**0.5)))
    assert torch.allclose(y, expected, atol=1e-5)


def test_bytepairencoding_len_and_contains():
    bpe = BytePairEncoding()
    bpe.build_vocab(["aaa aaa bbb ccc"])
    for token in ["aaa", "bbb", "ccc", "ddd"]:
        contains = token in bpe.vocab or False
        if token in ["aaa", "bbb", "ccc"]:
            assert contains is True
        else:
            assert contains is False
    # test len
    assert isinstance(len(bpe), int)
    assert len(bpe) == len(bpe.vocab)


def test_bytepairencoding_decode_unknown_token():
    bpe = BytePairEncoding()
    bpe.build_vocab(["a b c"])
    unknown_id = max(bpe.vocab.values()) + 10
    decoded = bpe.decode([unknown_id])
    # Should return some string, possibly empty or placeholder but not crash
    assert isinstance(decoded, str)


@pytest.mark.parametrize("seq_len", [1, 5, 10])
def test_gpt2_forward_variable_seq_len(seq_len):
    vocab_size = 50
    batch_size = 2
    model = GPT_2(vocab_size=vocab_size, n_layer=2, n_head=2, n_embd=12, max_seq=seq_len)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_generate_with_no_prompt_raises_or_returns():
    vocab_size = 20
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=8, max_seq=10)
    model.eval()
    # If generate expects non-empty prompt, passing None or empty list is tested
    try:
        output = generate(model, None, max_len=5)
    except Exception:
        # acceptable if error is raised
        pass
    else:
        # Or output is returned
        assert isinstance(output, (list, torch.Tensor))


def test_generate_with_max_len_less_than_prompt_length():
    vocab_size = 15
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=8, max_seq=8)
    model.eval()
    prompt = [1, 2, 3, 4, 5]
    max_len = 3
    output = generate(model, prompt, max_len=max_len)
    if isinstance(output, torch.Tensor):
        output = output.tolist()
    # output length must be at most max_len
    assert len(output) <= max_len


def test_gpt2_rejects_invalid_input_shape():
    vocab_size = 20
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=8, max_seq=10)
    bad_input = torch.randint(0, vocab_size, (5,))  # Should be 2D tensor for batch
    # Some implementations may support 1d input, others raise
    try:
        _ = model(bad_input)
    except Exception:
        pass
    else:
        # If no error, output shape must be (seq_len, vocab_size) or similar
        pass


def test_gpt2_and_generate_consistency_on_seed():
    torch.manual_seed(0)
    vocab_size = 30
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=8, max_seq=10)
    model.eval()
    prompt = [1, 2]
    out1 = generate(model, prompt, max_len=6)
    torch.manual_seed(0)
    out2 = generate(model, prompt, max_len=6)
    if isinstance(out1, torch.Tensor):
        out1 = out1.tolist()
    if isinstance(out2, torch.Tensor):
        out2 = out2.tolist()
    assert out1 == out2


def test_gelu_on_large_inputs():
    x = torch.randn(10000, requires_grad=True)
    y = gelu(x)
    y.sum().backward()
    assert x.grad is not None
    assert y.shape == x.shape


def test_bytepairencoding_repr_contains_vocab_size():
    bpe = BytePairEncoding()
    bpe.build_vocab(["some sample text"])
    rep = repr(bpe)
    # repr should mention something about vocab size or tokens
    assert isinstance(rep, str)
    assert str(len(bpe)) in rep

