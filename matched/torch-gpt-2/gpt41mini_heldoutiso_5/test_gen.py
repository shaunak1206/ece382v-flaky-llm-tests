import pytest
import torch
import torch.nn.functional as F
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate

@pytest.fixture
def sample_bpe(tmp_path):
    # Create dummy vocabulary and merges files for BytePairEncoding
    vocab = tmp_path / "vocab.json"
    merges = tmp_path / "merges.txt"
    # Minimal vocab and merges for testing
    vocab.write_text('{"hello":0, "world":1, "h":2, "e":3, "l":4, "o":5, "w":6, "r":7, "d":8, "<unk>":9}')
    merges.write_text("h e\nl l\no w\nw o\nr d\n")
    bpe = BytePairEncoding(str(vocab), str(merges))
    return bpe

def test_bpe_encode_decode_consistency(sample_bpe):
    text = "hello world"
    encoded = sample_bpe.encode(text)
    decoded = sample_bpe.decode(encoded)
    assert isinstance(encoded, list)
    # decoded should be a string and contain original words without unknown tokens
    assert isinstance(decoded, str)
    assert "hello" in decoded
    assert "world" in decoded

def test_bpe_encode_unknown_token(sample_bpe):
    text = "unknown_token"
    encoded = sample_bpe.encode(text)
    # Unknown tokens should be mapped to <unk> index (9 in our vocab)
    assert all(token >= 0 for token in encoded)
    assert 9 in encoded or all(token == 9 for token in encoded)

def test_bpe_decode_unknown_token(sample_bpe):
    # tokens containing unknown index (9)
    tokens = [9, 9, 9]
    decoded = sample_bpe.decode(tokens)
    assert isinstance(decoded, str)
    # Since tokens are unknown, decoded string should not be empty
    assert decoded != ""

def test_gpt_2_forward_output_shape():
    model = GPT_2(n_ctx=10, n_embd=16, n_layer=2, n_head=2, vocab_size=50)
    x = torch.randint(0, 50, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, 50)
    # Output logits should be float tensor
    assert logits.dtype == torch.float32

def test_gpt_2_forward_dtype_and_device():
    model = GPT_2(n_ctx=5, n_embd=8, n_layer=1, n_head=2, vocab_size=20)
    x = torch.randint(0, 20, (2, 5))
    logits = model(x)
    assert logits.device == x.device
    assert logits.dtype == torch.float32

def test_gelu_function_values():
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = gelu(x)
    # Output shape matches input
    assert y.shape == x.shape
    # For 0, gelu(0) ≈ 0
    assert abs(y[1].item()) < 1e-6
    # Values for negatives should be less than 0
    assert y[0].item() < 0
    # Values for positives should be greater than 0
    assert y[2].item() > 0

def test_gelu_gradient():
    x = torch.randn(5, requires_grad=True)
    y = gelu(x)
    y.sum().backward()
    assert x.grad is not None
    # Gradient shape same as input
    assert x.grad.shape == x.shape

@pytest.mark.parametrize("length,temperature,top_k", [
    (5, 1.0, None),
    (10, 0.7, 5),
    (1, 1.5, 10),
])
def test_generate_output_length_and_type(length, temperature, top_k):
    model = GPT_2(n_ctx=10, n_embd=16, n_layer=2, n_head=4, vocab_size=30)
    # Use random prefix tokens within vocab range
    prefix = torch.randint(0, 30, (1, 3))
    generated = generate(
        model,
        prefix=prefix,
        length=length,
        temperature=temperature,
        top_k=top_k,
        sample=True,
        device=prefix.device
    )
    # Generated sequence length should be prefix length + length
    assert isinstance(generated, torch.Tensor)
    assert generated.shape[1] == prefix.shape[1] + length
    # Check dtype and device match
    assert generated.dtype == prefix.dtype
    assert generated.device == prefix.device

def test_generate_without_sampling_returns_greedy():
    model = GPT_2(n_ctx=10, n_embd=16, n_layer=2, n_head=2, vocab_size=20)
    prefix = torch.randint(0, 20, (1, 5))
    generated = generate(model, prefix=prefix, length=3, sample=False, device=prefix.device)
    # Output shape correct
    assert generated.shape == (1, 8)
    # Values should be in vocab range
    assert torch.all(generated >= 0)
    assert torch.all(generated < 20)

def test_generate_with_device_cpu_and_gpu(monkeypatch):
    # This test tries to ensure no device error when device is cpu or cuda (if available)
    model = GPT_2(n_ctx=10, n_embd=8, n_layer=1, n_head=2, vocab_size=15)
    prefix = torch.randint(0, 15, (1, 4))
    for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
        generated = generate(model, prefix=prefix.to(device), length=2, sample=True, device=device)
        assert generated.device.type == device

def test_generate_handles_empty_prefix():
    model = GPT_2(n_ctx=10, n_embd=8, n_layer=1, n_head=2, vocab_size=15)
    prefix = torch.empty((1, 0), dtype=torch.long)
    generated = generate(model, prefix=prefix, length=5, sample=True, device='cpu')
    assert generated.shape == (1, 5)
    assert torch.all(generated >= 0)
    assert torch.all(generated < 15)

def test_generate_returns_input_when_length_zero():
    model = GPT_2(n_ctx=10, n_embd=8, n_layer=1, n_head=2, vocab_size=15)
    prefix = torch.randint(0, 15, (1, 3))
    generated = generate(model, prefix=prefix, length=0, sample=True, device='cpu')
    # Should return prefix unchanged
    assert torch.equal(generated, prefix)

def test_generate_invalid_temperature_raises():
    model = GPT_2(n_ctx=10, n_embd=8, n_layer=1, n_head=2, vocab_size=15)
    prefix = torch.randint(0, 15, (1, 3))
    with pytest.raises(Exception):
        generate(model, prefix=prefix, length=3, temperature=0, sample=True, device='cpu')

def test_bpe_encode_empty_string(sample_bpe):
    encoded = sample_bpe.encode("")
    assert encoded == [] or isinstance(encoded, list)

def test_bpe_decode_empty_list(sample_bpe):
    decoded = sample_bpe.decode([])
    # Decoding empty list returns empty string or some consistent value
    assert isinstance(decoded, str)

def test_gpt_2_invalid_input_size_raises():
    model = GPT_2(n_ctx=8, n_embd=8, n_layer=1, n_head=2, vocab_size=10)
    # Input length > n_ctx should raise or handle gracefully
    x = torch.randint(0, 10, (1, 12))
    with pytest.raises(Exception):
        model(x)

def test_gpt_2_output_logits_probabilities():
    model = GPT_2(n_ctx=6, n_embd=16, n_layer=2, n_head=4, vocab_size=30)
    x = torch.randint(0, 30, (2, 6))
    logits = model(x)
    probs = F.softmax(logits, dim=-1)
    # Probabilities sum to 1 on vocab dim per token
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
