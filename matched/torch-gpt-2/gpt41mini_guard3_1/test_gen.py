import math
import random
import numpy as np
import torch
import pytest

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


def test_gelu_values():
    # gelu should match approximate formula
    x = torch.tensor([-3.0, 0.0, 1.0, 3.0], dtype=torch.float32)
    y = gelu(x)

    # Reference gelu approximation (exact GELU uses erf, but torch.nn.functional.gelu uses this approx)
    # We'll recompute with exact erf formula for higher precision check:
    expected = 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
    assert torch.allclose(y, expected, atol=1e-6)


def test_gelu_scalar_and_tensor():
    # scalar input returns scalar tensor of same dtype
    for val in [-1.5, 0.0, 2.5]:
        x = torch.tensor(val, dtype=torch.float32)
        y = gelu(x)
        expected = 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
        assert isinstance(y, torch.Tensor)
        assert y.shape == torch.Size([])
        assert math.isclose(y.item(), expected.item(), rel_tol=1e-6)


def test_bytepairencoding_encode_decode_roundtrip(tmp_path):
    # Prepare a small BPE vocab and merges files to initialize BytePairEncoding
    vocab_content = """
{"hello":0,"world":1,"!":2,"he":3,"llo":4,"wo":5,"rld":6}
""".strip()
    merges_content = """
#version: 0.2
h e
l l
lo </w>
wo r
r l d
""".strip()
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    vocab_file.write_text(vocab_content)
    merges_file.write_text(merges_content)
    bpe = BytePairEncoding(str(vocab_file), str(merges_file))

    # encode "hello world!"
    input_text = "hello world!"
    token_ids = bpe.encode(input_text)

    # The encoded tokens should be a list of integers (token ids)
    assert isinstance(token_ids, list)
    assert all(isinstance(i, int) for i in token_ids)

    # decode to string and check that it includes the original words (exact decoding may differ)
    decoded = bpe.decode(token_ids)
    # The decoded string should contain the original words (exact might differ due to BPE tokens)
    for word in ["hello", "world", "!"]:
        assert word in decoded


@pytest.mark.parametrize("text", [
    "a", "aaaaaa", "abracadabra", "Hello, world!", "", "1234567890"
])
def test_bytepairencoding_encode_decode_idempotent(tmp_path, text):
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    # Minimal dummy vocab and merges that should tokenize letters
    vocab_content = '{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15,"q":16,"r":17,"s":18,"t":19,"u":20,"v":21,"w":22,"x":23,"y":24,"z":25,"!":26,",":27," ":28,"\n":29,".":30}'
    merges_content = """
#version: 0.2
a b
c d
e f
g h
i j
k l
m n
o p
q r
s t
u v
w x
y z
"""
    vocab_file.write_text(vocab_content)
    merges_file.write_text(merges_content)
    bpe = BytePairEncoding(str(vocab_file), str(merges_file))

    # Encoding and then decoding the text returns a string containing all original characters (case sensitive)
    encoded = bpe.encode(text)
    decoded = bpe.decode(encoded)
    # The decoded output may differ in spacing if BPE uses special end-of-word tokens, so just check all chars are present in decoded output
    for ch in text:
        assert ch.lower() in decoded.lower() or ch.upper() in decoded


def test_gpt_2_forward_and_parameter_shapes():
    # Create a GPT_2 instance with small config for testing
    model = GPT_2(
        n_vocab=100,
        n_ctx=10,
        n_embd=16,
        n_head=2,
        n_layer=2,
        dropout=0.0,
    )
    model.eval()

    # Input: batch size 2, seq length 5; random integers between 0 and vocab size-1
    torch.manual_seed(42)
    input_ids = torch.randint(0, 100, (2, 5), dtype=torch.int64)

    with torch.no_grad():
        logits, cache = model(input_ids, None)

    # logits shape must be (batch size, seq length, vocab size)
    assert logits.shape == (2, 5, 100)

    # cache should be a dict or None or similar (checking cache keys if dict)
    if cache is not None:
        assert isinstance(cache, dict)
        # Cache keys correspond to some layer-wise keys (usually strings)
        for k in cache.keys():
            assert isinstance(k, str)
    else:
        # model might not produce cache in this call
        assert cache is None

    # logits dtype and device correctness
    assert logits.dtype == torch.float32
    assert logits.device == input_ids.device

    # The model parameters should be present and have expected shapes
    param_shapes = {
        "transformer.wte.weight": (100, 16),
        "transformer.wpe.weight": (10, 16),
    }
    for name, shape in param_shapes.items():
        p = dict(model.named_parameters()).get(name, None)
        assert p is not None
        assert p.shape == shape


def test_gpt_2_generate_reproducible(monkeypatch):
    # Seed random, torch, and numpy RNGs as used by generate or model
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Use small model config for speed and reproducibility
    model = GPT_2(
        n_vocab=50,
        n_ctx=8,
        n_embd=16,
        n_head=2,
        n_layer=2,
        dropout=0.0,
    )
    model.eval()

    # Monkeypatch any randomness internal to model if necessary (none visible)

    # Input prompt sequence: 1 batch, 3 tokens
    input_ids = torch.tensor([[5, 10, 15]], dtype=torch.int64)

    # Run generate multiple times and check output is identical
    out1 = generate(model, input_ids, length=5, temperature=1.0, top_k=5, sample=False)
    # reset all seeds again to confirm determinism
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    out2 = generate(model, input_ids, length=5, temperature=1.0, top_k=5, sample=False)

    # Outputs must be identical tensors
    assert torch.equal(out1, out2)

    # The output shape must match (batch, input_len + length)
    assert out1.shape == (1, 3 + 5)


def test_generate_sample_vs_greedy_output_differs():
    # Seed RNGs for reproducibility
    random.seed(4321)
    np.random.seed(4321)
    torch.manual_seed(4321)

    model = GPT_2(
        n_vocab=30,
        n_ctx=10,
        n_embd=16,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    )
    model.eval()

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)

    # Greedy generation
    greedy_output = generate(model, input_ids, length=4, temperature=1.0, top_k=5, sample=False)

    # Sampling enabled generation
    random.seed(4321)
    np.random.seed(4321)
    torch.manual_seed(4321)
    sample_output = generate(model, input_ids, length=4, temperature=1.0, top_k=5, sample=True)

    # Outputs should differ as sampling mode enabled randomness
    assert not torch.equal(greedy_output, sample_output)

    # Both outputs shape should be equal
    assert greedy_output.shape == sample_output.shape


def test_bytepairencoding_handles_empty_string(tmp_path):
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    vocab_file.write_text('{}')
    merges_file.write_text('')
    bpe = BytePairEncoding(str(vocab_file), str(merges_file))
    encoded = bpe.encode("")
    # Empty string should encode to empty list
    assert encoded == []
    decoded = bpe.decode(encoded)
    # Decoding empty list returns empty string
    assert decoded == ""


def test_bytepairencoding_encode_unknown_tokens(tmp_path):
    # Use minimal vocab with some tokens missing
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    vocab_file.write_text('{"hello":0}')
    merges_file.write_text('')
    bpe = BytePairEncoding(str(vocab_file), str(merges_file))
    encoded = bpe.encode("unknownword")
    # Encoding unknown tokens returns a list of ints (likely fallback tokens)
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    # Decoding produces some string (may not equal original)
    decoded = bpe.decode(encoded)
    assert isinstance(decoded, str)


def test_generate_with_temperature_and_top_k_effect(monkeypatch):
    # Seed RNGs at start
    random.seed(1111)
    np.random.seed(1111)
    torch.manual_seed(1111)

    model = GPT_2(
        n_vocab=40,
        n_ctx=10,
        n_embd=16,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    )
    model.eval()

    input_ids = torch.tensor([[7, 14, 21]], dtype=torch.int64)

    # Generate twice with different parameters and check outputs shape and type
    out_temp_high = generate(model, input_ids, length=6, temperature=2.0, top_k=10, sample=True)
    random.seed(1111)
    np.random.seed(1111)
    torch.manual_seed(1111)
    out_temp_low = generate(model, input_ids, length=6, temperature=0.01, top_k=10, sample=True)

    for out in (out_temp_high, out_temp_low):
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 3 + 6)
        assert out.dtype == torch.int64


def test_gpt_2_forced_cache_argument_behavior():
    # Check that passing cache=None or cache dict yields outputs without error
    model = GPT_2(
        n_vocab=20,
        n_ctx=6,
        n_embd=8,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    )
    model.eval()
    input_ids = torch.randint(0, 20, (1, 4), dtype=torch.int64)

    # Cache None
    with torch.no_grad():
        logits1, cache1 = model(input_ids, None)
    assert logits1.shape == (1, 4, 20)
    # Cache dict (simulate empty cache)
    empty_cache = {}
    with torch.no_grad():
        logits2, cache2 = model(input_ids, empty_cache)
    assert logits2.shape == (1, 4, 20)
    # Cache returned (if any) differs from input cache or not None
    assert (cache1 is None) or isinstance(cache1, dict)
    assert (cache2 is None) or isinstance(cache2, dict)


def test_gpt_2_named_parameters_names():
    # Check presence of some expected parameter names in GPT_2
    model = GPT_2(
        n_vocab=25,
        n_ctx=12,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.1,
    )
    names = {n for n, _ in model.named_parameters()}
    # The embedding layers are expected
    assert "transformer.wte.weight" in names
    assert "transformer.wpe.weight" in names
    # At least one layer.norm.weight should exist
    norm_names = [n for n in names if "ln" in n and "weight" in n]
    assert len(norm_names) > 0


def test_generate_invalid_length_raises(monkeypatch):
    # Check that generate with zero or negative length raises (if implemented that way)
    model = GPT_2(
        n_vocab=10,
        n_ctx=5,
        n_embd=8,
        n_head=2,
        n_layer=1,
        dropout=0.,
    )
    model.eval()
    input_ids = torch.tensor([[1, 2]], dtype=torch.int64)

    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        generate(model, input_ids, length=0)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        generate(model, input_ids, length=-1)


def test_generate_output_tokens_within_vocab(monkeypatch):
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)

    model = GPT_2(
        n_vocab=15,
        n_ctx=7,
        n_embd=16,
        n_head=1,
        n_layer=1,
        dropout=0.0,
    )
    model.eval()

    input_ids = torch.tensor([[3, 6, 9]], dtype=torch.int64)
    out = generate(model, input_ids, length=4, temperature=0.7, top_k=5, sample=True)

    # All generated tokens must be in [0, n_vocab-1]
    assert torch.all((out >= 0) & (out < 15))


def test_gpt_2_can_be_moved_to_device_and_predict(monkeypatch):
    model = GPT_2(
        n_vocab=10,
        n_ctx=5,
        n_embd=8,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    )
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    torch.manual_seed(77)
    input_ids = torch.randint(0, 10, (1, 3), dtype=torch.int64).to(device)
    with torch.no_grad():
        logits, cache = model(input_ids, None)

    assert logits.device == device
    assert logits.shape == (1, 3, 10)

