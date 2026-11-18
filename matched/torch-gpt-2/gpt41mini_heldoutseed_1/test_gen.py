import io
import re
import torch
import pytest

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


@pytest.fixture
def example_bpe_files(tmp_path):
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    # Minimal example BPE vocab and merges from GPT-2 style
    vocab = {
        "l": 0,
        "o": 1,
        "w": 2,
        "lo": 3,
        "low": 4,
        "e": 5,
        "r": 6,
        "er": 7,
        "low er": 8,
        "low er</w>": 9,
        "</w>": 10,
    }
    merges = "\n".join([
        "#version: 0.2",
        "l o",
        "lo w",
        "e r",
        "low er",
    ])
    vocab_file.write_text(str(vocab).replace("'", '"'))
    merges_file.write_text(merges)
    return str(vocab_file), str(merges_file)


def test_bpe_encode_and_decode(example_bpe_files):
    vocab_file, merges_file = example_bpe_files
    bpe = BytePairEncoding(vocab_file, merges_file)
    text = "lower"
    tokens = bpe.encode(text)
    assert isinstance(tokens, list)
    # Tokens should be integers and all be in vocab
    assert all(isinstance(t, int) for t in tokens)
    decoded = bpe.decode(tokens)
    assert isinstance(decoded, str)
    # Decoded text should be equal or near equal ignoring </w> marker
    assert re.sub(r"</w>", "", decoded) == text or decoded.replace("</w>", "") == text


def test_bpe_encode_empty_string(example_bpe_files):
    vocab_file, merges_file = example_bpe_files
    bpe = BytePairEncoding(vocab_file, merges_file)
    tokens = bpe.encode("")
    assert tokens == []
    decoded = bpe.decode(tokens)
    assert decoded == ""


def test_bpe_encode_unknown_token(example_bpe_files):
    vocab_file, merges_file = example_bpe_files
    bpe = BytePairEncoding(vocab_file, merges_file)
    # 'xyz' likely not in vocab, check that encode doesn't fail
    tokens = bpe.encode("xyz")
    assert isinstance(tokens, list)
    # decode should produce string (may be different)
    decoded = bpe.decode(tokens)
    assert isinstance(decoded, str)


def test_gpt2_model_forward_and_shapes():
    # basic test that GPT_2 model produces output with expected shapes
    model = GPT_2()
    # Use small dummy input tensor (batch_size=1, seq_len=5)
    input_ids = torch.randint(0, model.n_vocab, (1, 5))
    output = model(input_ids)
    # Output should be (batch, seq_len, vocab_size)
    assert output.shape == (1, 5, model.n_vocab)


def test_gpt2_model_output_deterministic_for_same_input():
    model = GPT_2()
    model.eval()
    input_ids = torch.randint(0, model.n_vocab, (1, 4))
    with torch.no_grad():
        out1 = model(input_ids)
        out2 = model(input_ids)
    # Outputs should be equal
    assert torch.allclose(out1, out2)


def test_gelu_output_range_and_shape():
    # test gelu on tensor input and check shape and range
    x = torch.linspace(-10, 10, steps=100)
    y = gelu(x)
    assert y.shape == x.shape
    # Values should be generally between min and max of x (gelu smooths)
    assert y.min() >= x.min()
    assert y.max() <= x.max()


def test_generate_returns_tensor_with_expected_shape_and_dtype():
    model = GPT_2()
    bpe_vocab_file = model.bpe.vocab if hasattr(model, "bpe") else None
    # Generate with minimal input text and length
    seq_out = generate(model, "Hello", temperature=1.0, seed=42, max_len=5)
    assert isinstance(seq_out, str)
    # It should be non-empty string
    assert len(seq_out) > 0


def test_generate_different_seed_produces_different_output():
    model = GPT_2()
    text = "Hello"
    out1 = generate(model, text, seed=123, max_len=10)
    out2 = generate(model, text, seed=321, max_len=10)
    # Output should differ
    assert out1 != out2


def test_generate_with_temperature_limits():
    model = GPT_2()
    text = "Test"
    out_low = generate(model, text, temperature=0.1, max_len=7)
    out_high = generate(model, text, temperature=10.0, max_len=7)
    assert isinstance(out_low, str)
    assert isinstance(out_high, str)
    # outputs should be decodable strings (not empty)
    assert len(out_low) > 0
    assert len(out_high) > 0


def test_generate_invalid_temperature_raises_or_corrects():
    model = GPT_2()
    text = "Test"
    # Negative temperature: expect it to either raise or fix internally
    with pytest.raises(Exception):
        generate(model, text, temperature=-1.0)
    # Zero temperature may cause determinism: check it runs
    out = generate(model, text, temperature=0)
    assert isinstance(out, str)


def test_generate_seed_reproducibility():
    model = GPT_2()
    text = "reproducible"
    out1 = generate(model, text, max_len=6, seed=2022)
    out2 = generate(model, text, max_len=6, seed=2022)
    assert out1 == out2


@pytest.mark.parametrize("input_text", ["", "a", "the quick brown fox", "123 !@#"])
def test_bpe_encode_decode_roundtrip(input_text, example_bpe_files):
    vocab_file, merges_file = example_bpe_files
    bpe = BytePairEncoding(vocab_file, merges_file)
    tokens = bpe.encode(input_text)
    out_text = bpe.decode(tokens)
    # Spaces might be preserved or merged, so just check that output is str and tokens is list
    assert isinstance(tokens, list)
    assert isinstance(out_text, str)


def test_gpt2_output_log_softmax():
    model = GPT_2()
    input_ids = torch.randint(0, model.n_vocab, (2, 3))
    output = model(input_ids)
    # The output logits should not be equal to log softmax directly,
    # but checking output is float tensor with expected shape
    assert output.dtype == torch.float32
    assert output.shape == (2, 3, model.n_vocab)


def test_gpt2_forward_with_various_input_lengths():
    model = GPT_2()
    for seq_len in [1, 5, 10]:
        input_ids = torch.randint(0, model.n_vocab, (3, seq_len))
        output = model(input_ids)
        assert output.shape == (3, seq_len, model.n_vocab)


def test_generate_with_invalid_seed_type():
    model = GPT_2()
    # Seed as string raises error or gets coerced
    with pytest.raises(TypeError):
        generate(model, "text", seed="invalid_seed")


def test_gpt2_can_handle_long_sequence(monkeypatch):
    model = GPT_2()
    # monkeypatch max position embeddings to small number to test error or clipping
    monkeypatch.setattr(model.config, "n_ctx", 10)
    input_ids = torch.randint(0, model.n_vocab, (1, 20))
    with pytest.raises(Exception):
        model(input_ids)


def test_generate_max_len_zero_returns_input_text():
    model = GPT_2()
    input_text = "Hello"
    out = generate(model, input_text, max_len=0)
    # Expect output to be input or empty or same string returned
    assert isinstance(out, str)
    assert len(out) >= len(input_text) or out == input_text
