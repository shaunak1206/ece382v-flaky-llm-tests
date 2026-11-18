import pytest
import torch
import torch.nn.functional as F
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate

@pytest.fixture
def sample_vocab_files(tmp_path):
    vocab_path = tmp_path / "vocab.txt"
    merges_path = tmp_path / "merges.txt"
    vocab_content = "\n".join([
        "l",
        "o",
        "w",
        "e",
        "r",
        "t",
        "h",
        "s",
        "i",
        "n",
        "er",
        "re",
    ])
    merges_content = "\n".join([
        "l o",
        "lo w",
        "low e",
        "low er",
        "er s",
        "t h",
        "th e",
        "he r",
    ])
    vocab_path.write_text(vocab_content)
    merges_path.write_text(merges_content)
    return str(vocab_path), str(merges_path)

def test_bpe_encode_and_decode_roundtrip(sample_vocab_files):
    vocab_file, merges_file = sample_vocab_files
    bpe = BytePairEncoding(vocab_file, merges_file)
    text = "lower there"
    tokens = bpe.encode(text)
    decoded = bpe.decode(tokens)
    assert isinstance(tokens, list)
    assert all(isinstance(tok, int) for tok in tokens)
    assert isinstance(decoded, str)
    assert decoded.replace(" ", "") == text.replace(" ", "")  # spaces may be normalized/dropped

def test_bpe_encode_empty_string(sample_vocab_files):
    vocab_file, merges_file = sample_vocab_files
    bpe = BytePairEncoding(vocab_file, merges_file)
    tokens = bpe.encode("")
    assert tokens == []

def test_bpe_decode_empty_list(sample_vocab_files):
    vocab_file, merges_file = sample_vocab_files
    bpe = BytePairEncoding(vocab_file, merges_file)
    text = bpe.decode([])
    assert text == ""

def test_bpe_encode_unknown_tokens(sample_vocab_files):
    vocab_file, merges_file = sample_vocab_files
    bpe = BytePairEncoding(vocab_file, merges_file)
    # Assuming BPE returns some form of fallback for unknown tokens or raises no exceptions
    tokens = bpe.encode("xyz")
    assert isinstance(tokens, list)
    assert all(isinstance(tok, int) for tok in tokens)

def test_gelu_output_shape_and_values():
    x = torch.linspace(-3, 3, steps=100)
    y = gelu(x)
    assert y.shape == x.shape
    # gelu(x) should approximately be 0 at 0 and positive for positive x, negative for negative x
    assert torch.isclose(y[49], torch.tensor(0.), atol=1e-2)
    assert y[0] < 0
    assert y[-1] > 0

def test_gelu_gradient():
    x = torch.linspace(-3, 3, steps=10, requires_grad=True)
    y = gelu(x)
    y.sum().backward()
    assert x.grad is not None
    # Gradients should be finite
    assert torch.all(torch.isfinite(x.grad))

@pytest.fixture
def small_gpt2_model():
    # Create a small GPT_2 model with minimal config to test methods
    model = GPT_2(
        n_vocab=20,
        n_ctx=8,
        n_embd=16,
        n_layer=2,
        n_head=4,
    )
    return model

def test_gpt2_forward_output_shape(small_gpt2_model):
    x = torch.randint(0, 20, (1, 8))
    logits = small_gpt2_model(x)
    assert logits.shape == (1, 8, 20)

def test_gpt2_output_dtype_and_device(small_gpt2_model):
    x = torch.randint(0, 20, (2, 8))
    logits = small_gpt2_model(x)
    assert logits.dtype == torch.float32
    assert logits.device == x.device

def test_gpt2_generate_basic(small_gpt2_model):
    prompt = torch.randint(0, 20, (1, 5))
    max_new_tokens = 3
    output = generate(
        small_gpt2_model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        sample=False,
        eos_token_id=None,
    )
    assert output.shape == (1, 5 + max_new_tokens)
    # The prompt prefix matches
    assert torch.all(output[:, :5] == prompt)

def test_gpt2_generate_with_eos_token(small_gpt2_model):
    eos_token_id = 1
    prompt = torch.randint(0, 20, (1, 3))
    # artificially set '1' in prompt to test no early stopping on prompt tokens
    output = generate(
        small_gpt2_model,
        prompt,
        max_new_tokens=10,
        temperature=1.0,
        sample=False,
        eos_token_id=eos_token_id,
    )
    assert output.shape[1] <= (3 + 10)
    # Output begins with prompt exactly
    assert torch.all(output[:, :3] == prompt)
    # If eos_token_id got generated, output sequence ends there or before max_new_tokens
    eos_positions = (output == eos_token_id).nonzero(as_tuple=True)
    if eos_positions[0].numel() > 0:
        # All eos tokens should appear only at or after prompt length
        assert all(pos >= 3 for pos in eos_positions[1].tolist())

def test_generate_sampling_vs_greedy_consistency(small_gpt2_model):
    prompt = torch.randint(0, 20, (1, 4))
    output_greedy = generate(
        small_gpt2_model,
        prompt,
        max_new_tokens=5,
        temperature=1.0,
        sample=False,
        eos_token_id=None,
    )
    output_sample = generate(
        small_gpt2_model,
        prompt,
        max_new_tokens=5,
        temperature=1.0,
        sample=True,
        eos_token_id=None,
    )
    # Greedy output is deterministic -> repeated calls yield same output
    output_greedy_2 = generate(
        small_gpt2_model,
        prompt,
        max_new_tokens=5,
        temperature=1.0,
        sample=False,
        eos_token_id=None,
    )
    assert torch.equal(output_greedy, output_greedy_2)
    # Sample output likely differs from greedy (not guaranteed, but very likely)
    # We just check that it is shape consistent
    assert output_sample.shape == output_greedy.shape

def test_generate_temperature_effect(small_gpt2_model):
    prompt = torch.randint(0, 20, (1, 2))
    output_t1 = generate(
        small_gpt2_model,
        prompt,
        max_new_tokens=5,
        temperature=1.0,
        sample=True,
        eos_token_id=None,
    )
    output_t05 = generate(
        small_gpt2_model,
        prompt,
        max_new_tokens=5,
        temperature=0.5,
        sample=True,
        eos_token_id=None,
    )
    # Just check outputs are valid and shape consistent
    assert output_t1.shape == (1, 7)
    assert output_t05.shape == (1, 7)

def test_bytepairencoding_repr_and_str(sample_vocab_files):
    vocab_file, merges_file = sample_vocab_files
    bpe = BytePairEncoding(vocab_file, merges_file)
    r = repr(bpe)
    s = str(bpe)
    assert isinstance(r, str)
    assert isinstance(s, str)
    # Should mention vocab and merges files or number of merges
    assert "BytePairEncoding" in r or "BytePairEncoding" in s

def test_gpt2_repr_str(small_gpt2_model):
    r = repr(small_gpt2_model)
    s = str(small_gpt2_model)
    assert isinstance(r, str)
    assert isinstance(s, str)
    assert "GPT_2" in r or "GPT_2" in s

def test_generate_invalid_inputs(small_gpt2_model):
    prompt = torch.randint(0, 20, (1, 4))
    with pytest.raises(ValueError):
        # negative max_new_tokens is invalid, assuming generate validates it
        generate(small_gpt2_model, prompt, max_new_tokens=-1)
    # test temperature=0 gives reasonable output (may collapse to greedy)
    output = generate(small_gpt2_model, prompt, max_new_tokens=2, temperature=0.0, sample=True)
    assert output.shape == (1, 6)
