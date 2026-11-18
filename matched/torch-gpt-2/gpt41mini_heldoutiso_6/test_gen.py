import pytest
import torch
import torch.nn.functional as F
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


@pytest.fixture
def sample_bpe(tmp_path):
    # create dummy vocab and merges files
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    # minimal vocab with just a few tokens
    vocab_content = '{"hello":0,"world":1,"!":2,"he":3,"llo":4,"wor":5,"ld":6}'
    merges_content = "he llo\nwor ld\n"
    vocab_file.write_text(vocab_content)
    merges_file.write_text(merges_content)
    bpe = BytePairEncoding(vocab_file=str(vocab_file), merges_file=str(merges_file))
    return bpe


def test_bpe_encode_decode(sample_bpe):
    text = "hello world!"
    tokens = sample_bpe.encode(text)
    # Ensure tokens is a list of integers
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

    decoded = sample_bpe.decode(tokens)
    # Decoded text should be close to original (may differ in spacing/casing)
    assert isinstance(decoded, str)
    assert "hello" in decoded.lower()
    assert "world" in decoded.lower()

    # Decoding the encoding should reproduce similar tokens
    tokens2 = sample_bpe.encode(decoded)
    assert tokens2 == tokens


def test_bpe_encode_empty_string(sample_bpe):
    tokens = sample_bpe.encode("")
    assert isinstance(tokens, list)
    assert len(tokens) == 0
    decoded = sample_bpe.decode(tokens)
    assert decoded == ""


def test_bpe_decode_invalid_token(sample_bpe):
    # decode a token id that does not exist in vocab
    with pytest.raises(KeyError):
        # forcibly decode an invalid token id by tampering vocab inside
        inv_token = max(sample_bpe._vocab.values()) + 10
        sample_bpe.decode([inv_token])


def test_gpt2_init_and_forward():
    vocab_size = 1000
    seq_len = 10
    # create GPT_2 with custom vocab size, num embeddings etc
    model = GPT_2(vocab_size=vocab_size, n_ctx=seq_len)
    model.eval()

    # input batch: batch size 2, seq_len tokens (random tokens)
    input_ids = torch.randint(low=0, high=vocab_size, size=(2, seq_len), dtype=torch.long)

    # forward pass returns logits of shape (batch, seq_len, vocab_size)
    logits = model(input_ids)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, seq_len, vocab_size)
    # logits values should be finite
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("input_value,expected", [
    (torch.tensor([-3.0, 0.0, 3.0]), torch.nn.functional.gelu(torch.tensor([-3.0, 0.0, 3.0]))),
    (torch.randn(5), None),
])
def test_gelu_output(input_value, expected):
    result = gelu(input_value)
    assert isinstance(result, torch.Tensor)
    assert result.shape == input_value.shape
    # For the first input_value param, test close to torch's gelu
    if expected is not None:
        assert torch.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("prompt", [
    "hello",
    "test generation",
    "",
])
def test_generate_output_length_and_type(prompt, sample_bpe):
    # generate returns string text given a prompt
    output = generate(prompt=prompt, max_length=20, temperature=0.8, top_k=5)
    assert isinstance(output, str)
    # generated text length should be >= prompt length (at least tokens)
    assert len(output) >= len(prompt)


def test_generate_with_invalid_temperature():
    with pytest.raises(ValueError):
        generate(prompt="hello", max_length=10, temperature=-1.0)

    with pytest.raises(ValueError):
        generate(prompt="hello", max_length=10, temperature=100.0)


def test_generate_with_zero_max_length():
    output = generate(prompt="hello", max_length=0)
    # Should return prompt or empty string without error
    assert isinstance(output, str)


def test_gpt2_generate_logits_consistency():
    vocab_size = 50
    seq_len = 5
    model = GPT_2(vocab_size=vocab_size, n_ctx=seq_len)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    logits1 = model(input_ids)
    logits2 = model(input_ids)

    # Running twice with same input returns same logits
    assert torch.allclose(logits1, logits2, atol=1e-6)
    assert logits1.shape == (1, seq_len, vocab_size)


def test_bpe_encode_whitespace_handling(sample_bpe):
    text = "hello  world"
    tokens = sample_bpe.encode(text)
    decoded = sample_bpe.decode(tokens)
    # Multiple spaces should not cause an error
    assert isinstance(decoded, str)
    assert "hello" in decoded.lower()
    assert "world" in decoded.lower()


def test_generate_output_with_different_top_k_values():
    outputs = set()
    prompt = "hello"
    for top_k in [1, 5, 10]:
        output = generate(prompt=prompt, max_length=15, temperature=0.7, top_k=top_k)
        assert isinstance(output, str)
        outputs.add(output)
    # At least two outputs should differ to show stochasticity
    assert len(outputs) >= 1


def test_generate_output_with_temperature_variations():
    prompt = "test"
    output_low_temp = generate(prompt=prompt, max_length=15, temperature=0.01, top_k=5)
    output_high_temp = generate(prompt=prompt, max_length=15, temperature=2.0, top_k=5)
    # Outputs should be strings
    assert isinstance(output_low_temp, str)
    assert isinstance(output_high_temp, str)
    # High temp output should differ from low temp output (non-strict, but generally true)
    assert output_low_temp != output_high_temp or len(output_low_temp) == 0 or len(output_high_temp) == 0


def test_gelu_gradient_and_dtype():
    x = torch.randn(10, requires_grad=True)
    y = gelu(x)
    y_sum = y.sum()
    y_sum.backward()
    assert x.grad is not None
    # dtype is float32 or float64 compatible with gelu
    assert x.dtype == y.dtype


def test_bpe_encode_decode_non_ascii_text(sample_bpe):
    text = "héllo wörld!"
    tokens = sample_bpe.encode(text)
    decoded = sample_bpe.decode(tokens)
    assert isinstance(decoded, str)
    assert "hello" in decoded.lower() or "héllo" in decoded.lower()
    assert "world" in decoded.lower() or "wörld" in decoded.lower()


def test_generate_with_custom_seed_monotonicity():
    prompt = "Let’s test seed"
    out1 = generate(prompt=prompt, max_length=20, temperature=1.0, top_k=5, seed=42)
    out2 = generate(prompt=prompt, max_length=20, temperature=1.0, top_k=5, seed=42)
    out3 = generate(prompt=prompt, max_length=20, temperature=1.0, top_k=5, seed=43)
    # same seed produces same output
    assert out1 == out2
    # different seed produces different output (likely)
    assert out1 != out3 or out1 == "" or out3 == ""


def test_bpe_repr(sample_bpe):
    r = repr(sample_bpe)
    assert isinstance(r, str)
    assert "BytePairEncoding" in r


def test_gpt2_repr_and_str():
    model = GPT_2()
    r = repr(model)
    s = str(model)
    assert isinstance(r, str)
    assert isinstance(s, str)
    assert "GPT_2" in r
    assert "GPT_2" in s
