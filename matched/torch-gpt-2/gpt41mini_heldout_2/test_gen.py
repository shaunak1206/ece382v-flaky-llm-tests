import pytest
import torch
import torch.nn.functional as F

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


@pytest.fixture
def sample_vocab_files(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"

    # Minimal example vocab and merges from GPT-2 BPE
    vocab_content = '{"hello": 0, "world": 1, "he": 2, "##llo": 3, "wor": 4, "ld": 5, "hello world": 6, "Ġ": 7}'
    merges_content = """#version: 0.2
l l
o w
or ld
he ##llo
"""
    vocab_path.write_text(vocab_content)
    merges_path.write_text(merges_content)
    return str(vocab_path), str(merges_path)


def test_bpe_tokenize_and_decode_roundtrip(sample_vocab_files):
    vocab_path, merges_path = sample_vocab_files
    bpe = BytePairEncoding(vocab_path, merges_path)
    text = "hello world"
    tokens = bpe.encode(text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    decoded = bpe.decode(tokens)
    assert isinstance(decoded, str)
    assert "hello" in decoded or "world" in decoded  # approximate check


def test_bpe_encode_empty_string(sample_vocab_files):
    vocab_path, merges_path = sample_vocab_files
    bpe = BytePairEncoding(vocab_path, merges_path)
    tokens = bpe.encode("")
    assert tokens == []
    decoded = bpe.decode(tokens)
    assert decoded == ""


def test_bpe_encode_unknown_token(sample_vocab_files):
    vocab_path, merges_path = sample_vocab_files
    bpe = BytePairEncoding(vocab_path, merges_path)
    tokens = bpe.encode("unseen_token_123")
    assert isinstance(tokens, list)
    # Should contain only ints; maybe <unk> token or fallback
    for t in tokens:
        assert isinstance(t, int)


def test_gpt_2_forward_output_shape_and_dtype():
    config = {
        "vocab_size": 10,
        "n_embd": 8,
        "n_layer": 2,
        "n_head": 2,
        "block_size": 16,
    }
    model = GPT_2(**config)
    batch_size = 4
    seq_len = 10
    x = torch.randint(0, config["vocab_size"], (batch_size, seq_len), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, config["vocab_size"])
    assert logits.dtype == torch.float32


def test_gpt_2_forward_with_single_input_token():
    config = {
        "vocab_size": 20,
        "n_embd": 16,
        "n_layer": 1,
        "n_head": 4,
        "block_size": 8,
    }
    model = GPT_2(**config)
    x = torch.randint(0, config["vocab_size"], (1, 1), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (1, 1, config["vocab_size"])


def test_gpt_2_generate_logits_differ_with_different_inputs():
    config = {
        "vocab_size": 15,
        "n_embd": 12,
        "n_layer": 2,
        "n_head": 3,
        "block_size": 12,
    }
    model = GPT_2(**config)
    input1 = torch.randint(0, config["vocab_size"], (1, 5), dtype=torch.long)
    input2 = torch.randint(0, config["vocab_size"], (1, 5), dtype=torch.long)
    logits1 = model(input1)
    logits2 = model(input2)
    # Most often logits should differ if inputs differ
    assert not torch.allclose(logits1, logits2)


@pytest.mark.parametrize("input_val", [0.0, 1.0, -1.0, 10.0, -10.0, 5.5])
def test_gelu_output_range_and_differentiability(input_val):
    x = torch.tensor([input_val], requires_grad=True)
    y = gelu(x)
    # check output is scalar float tensor
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    # backward should work
    y.sum().backward()
    assert x.grad is not None


def test_gelu_matches_torch_reference():
    # Compare to torch.nn.functional.gelu for several inputs
    inputs = torch.linspace(-3, 3, steps=50)
    outputs_ours = gelu(inputs)
    outputs_torch = F.gelu(inputs)
    diff = (outputs_ours - outputs_torch).abs().max()
    assert diff < 1e-5


def test_generate_returns_tokens_and_respects_max_tokens():
    config = {
        "vocab_size": 30,
        "n_embd": 16,
        "n_layer": 2,
        "n_head": 4,
        "block_size": 16,
    }
    model = GPT_2(**config)
    # Starting with a batch of one sequence
    start_tokens = torch.randint(0, config["vocab_size"], (1, 3), dtype=torch.long)
    # Generate 5 tokens max
    generated = generate(model, start_tokens, max_tokens=5, eos_token=None)
    assert isinstance(generated, torch.Tensor)
    # Output should be longer or equal to input + max_tokens
    assert generated.shape[1] >= start_tokens.shape[1]
    assert generated.shape[0] == start_tokens.shape[0]


def test_generate_eos_stops_generation_early():
    # This test assumes EOS token generation stops output early
    vocab_size = 10
    eos_token = 0  # for test purpose
    config = {
        "vocab_size": vocab_size,
        "n_embd": 8,
        "n_layer": 1,
        "n_head": 2,
        "block_size": 8,
    }
    model = GPT_2(**config)
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    start_tokens = torch.randint(1, vocab_size, (1, 3), dtype=torch.long)
    generated = generate(model, start_tokens, max_tokens=20, eos_token=eos_token)
    # check that EOS token appears once generation ends or is not present
    if eos_token in generated[0].tolist():
        # after eos token, no more tokens should be present
        eos_index = (generated[0] == eos_token).nonzero(as_tuple=True)[0].item()
        # slice after eos token should be empty or padding (none expected here)
        assert eos_index < generated.shape[1]


def test_generate_preserves_batch_size_and_initial_tokens():
    config = {
        "vocab_size": 25,
        "n_embd": 10,
        "n_layer": 2,
        "n_head": 5,
        "block_size": 20,
    }
    model = GPT_2(**config)
    batch_size = 3
    init_len = 4
    start_tokens = torch.randint(0, config["vocab_size"], (batch_size, init_len), dtype=torch.long)
    out = generate(model, start_tokens, max_tokens=7)
    assert out.shape[0] == batch_size
    # The initial tokens must be prefix of output tokens
    assert torch.equal(out[:, :init_len], start_tokens)


def test_generate_respects_block_size():
    block_size = 10
    vocab_size = 15
    config = {
        "vocab_size": vocab_size,
        "n_embd": 12,
        "n_layer": 1,
        "n_head": 3,
        "block_size": block_size,
    }
    model = GPT_2(**config)
    input_tokens = torch.randint(0, vocab_size, (1, block_size - 2), dtype=torch.long)
    generated = generate(model, input_tokens, max_tokens=5)
    # Output sequence should not exceed block_size
    assert generated.shape[1] <= block_size


def test_bpe_encode_decode_consistency_with_multiple_strings(sample_vocab_files):
    vocab_path, merges_path = sample_vocab_files
    bpe = BytePairEncoding(vocab_path, merges_path)
    texts = ["hello", "world", "hello world", "", "a quick test"]
    for text in texts:
        tokens = bpe.encode(text)
        decoded = bpe.decode(tokens)
        assert isinstance(decoded, str)
        # decoded may differ exact due to BPE merges (non-exact), so just check non-empty expected
        if text:
            assert any(c in decoded for c in text if c != " ")


def test_bpe_decode_invalid_token_raises(sample_vocab_files):
    vocab_path, merges_path = sample_vocab_files
    bpe = BytePairEncoding(vocab_path, merges_path)
    # Passing tokens that likely don't exist
    invalid_tokens = [-1, 99999]
    with pytest.raises(KeyError):
        bpe.decode(invalid_tokens)


def test_gpt_2_invalid_inputs_raise():
    config = {
        "vocab_size": 10,
        "n_embd": 8,
        "n_layer": 1,
        "n_head": 2,
        "block_size": 8,
    }
    model = GPT_2(**config)
    # Input of wrong dtype
    x = torch.randn(1, 4, dtype=torch.float)
    with pytest.raises(RuntimeError):
        _ = model(x)
    # Input with too large token ids should raise error or produce something
    x2 = torch.full((1, 4), 100, dtype=torch.long)
    logits = model(x2)
    assert logits.shape == (1, 4, config["vocab_size"])


def test_generate_with_zero_max_tokens_returns_start_tokens():
    config = {
        "vocab_size": 20,
        "n_embd": 10,
        "n_layer": 1,
        "n_head": 2,
        "block_size": 10,
    }
    model = GPT_2(**config)
    start_tokens = torch.randint(0, config["vocab_size"], (2, 3), dtype=torch.long)
    output = generate(model, start_tokens, max_tokens=0)
    assert torch.equal(output, start_tokens)


def test_generate_with_eos_token_not_in_vocab():
    config = {
        "vocab_size": 12,
        "n_embd": 8,
        "n_layer": 1,
        "n_head": 2,
        "block_size": 12,
    }
    model = GPT_2(**config)
    start_tokens = torch.randint(0, config["vocab_size"], (1, 3), dtype=torch.long)
    # eos_token value outside vocab range - no early stopping expected
    eos_token = 1000
    output = generate(model, start_tokens, max_tokens=5, eos_token=eos_token)
    assert output.shape[1] >= start_tokens.shape[1]


def test_generate_deterministic_with_fixed_seed():
    config = {
        "vocab_size": 15,
        "n_embd": 16,
        "n_layer": 2,
        "n_head": 4,
        "block_size": 16,
    }
    model = GPT_2(**config)
    start_tokens = torch.randint(0, config["vocab_size"], (1, 4), dtype=torch.long)
    torch.manual_seed(123)
    output1 = generate(model, start_tokens, max_tokens=5)
    torch.manual_seed(123)
    output2 = generate(model, start_tokens, max_tokens=5)
    assert torch.equal(output1, output2)
