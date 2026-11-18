import pytest
import torch
import torch.nn.functional as F

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


def test_gelu_values():
    # test known values for gelu function (approximate)
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = gelu(x)
    expected = torch.tensor([-0.045402, -0.158807, 0.0, 0.841192, 1.954598])
    # Use almost equal comparison
    assert torch.allclose(y, expected, atol=1e-4)


def test_gelu_gradient():
    x = torch.randn(10, requires_grad=True)
    y = gelu(x)
    y.sum().backward()
    assert x.grad is not None
    # Gradient should not be zero everywhere
    assert (x.grad.abs() > 0).any()


def test_bpe_tokenize_and_decode_roundtrip(tmp_path):
    # Write dummy vocab and merges files to tmp_path
    vocab_content = '{"hello": 0, "world": 1, "he": 2, "llo": 3, "wor": 4, "ld":5}'
    merges_content = "he llo\nwor ld\n"
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    vocab_path.write_text(vocab_content)
    merges_path.write_text(merges_content)

    bpe = BytePairEncoding(str(vocab_path), str(merges_path))
    tokens = bpe.tokenize("hello world")
    # tokens should be list of integer indices
    assert all(isinstance(t, int) for t in tokens)
    # decode(tokens) should reconstruct input without spaces (bpe standard behavior)
    decoded = bpe.decode(tokens).replace(" ", "")
    assert decoded == "helloworld"


def test_bpe_tokenize_unknown_token(tmp_path):
    # Use vocab with limited tokens
    vocab_content = '{"he":0, "llo":1}'
    merges_content = ""
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    vocab_path.write_text(vocab_content)
    merges_path.write_text(merges_content)

    bpe = BytePairEncoding(str(vocab_path), str(merges_path))
    # tokenize string with unknown characters
    tokens = bpe.tokenize("xyz")  # "xyz" tokens unknown
    # token list is non-empty (BPE fallback to chars or something else)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    # decoded tokens should be str and not empty
    decoded = bpe.decode(tokens)
    assert isinstance(decoded, str)
    assert decoded != ""


def test_gpt2_forward_and_shape():
    # Instantiate GPT_2 with minimal config for test
    vocab_size = 50
    seq_len = 10
    batch_size = 2
    model = GPT_2(vocab_size=vocab_size, n_layer=2, n_head=2, n_embd=32, block_size=seq_len)
    model.eval()

    # random input token ids
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        output = model(x)
    # Output shape should be (batch_size, seq_len, vocab_size)
    assert output.shape == (batch_size, seq_len, vocab_size)
    # Output logits should be finite
    assert torch.isfinite(output).all()


def test_gpt2_generate_output_length_and_vocab():
    vocab_size = 40
    seq_len = 8
    model = GPT_2(vocab_size=vocab_size, n_layer=2, n_head=2, n_embd=64, block_size=seq_len)
    model.eval()

    device = torch.device("cpu")

    prompt = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    # generate should produce tensor of shape (1, generated_seq_len)
    max_new_tokens = 5
    output = generate(model, prompt, max_new_tokens=max_new_tokens, temperature=1.0, device=device)
    assert output.shape[0] == 1
    assert output.shape[1] == seq_len + max_new_tokens
    # tokens should be in vocab range
    assert (output >= 0).all() and (output < vocab_size).all()


def test_gpt2_generate_reproducibility_seed(tmp_path):
    vocab_size = 20
    seq_len = 6
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=16, block_size=seq_len)
    model.eval()
    device = torch.device("cpu")
    prompt = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    torch.manual_seed(123)
    out1 = generate(model, prompt, max_new_tokens=4, temperature=0.7, device=device)
    torch.manual_seed(123)
    out2 = generate(model, prompt, max_new_tokens=4, temperature=0.7, device=device)

    # Outputs should be identical given same seed
    assert torch.equal(out1, out2)


def test_generate_with_temperature_extremes():
    vocab_size = 15
    seq_len = 5
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=1, n_embd=16, block_size=seq_len)
    model.eval()
    device = torch.device("cpu")
    prompt = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    output_low_temp = generate(model, prompt, max_new_tokens=3, temperature=1e-4, device=device)
    output_high_temp = generate(model, prompt, max_new_tokens=3, temperature=10.0, device=device)

    # Ensure output shape correctness
    assert output_low_temp.shape == (1, seq_len + 3)
    assert output_high_temp.shape == (1, seq_len + 3)
    # Tokens must be valid indices
    assert (output_low_temp >= 0).all() and (output_low_temp < vocab_size).all()
    assert (output_high_temp >= 0).all() and (output_high_temp < vocab_size).all()


def test_generate_with_top_k_and_top_p():
    vocab_size = 25
    seq_len = 7
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=32, block_size=seq_len)
    model.eval()
    device = torch.device("cpu")

    prompt = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    # Test top_k only
    output_top_k = generate(model, prompt, max_new_tokens=4, temperature=1.0, top_k=5, device=device)
    assert output_top_k.shape == (1, seq_len + 4)
    assert (output_top_k >= 0).all() and (output_top_k < vocab_size).all()

    # Test top_p only (0.9)
    output_top_p = generate(model, prompt, max_new_tokens=4, temperature=1.0, top_p=0.9, device=device)
    assert output_top_p.shape == (1, seq_len + 4)
    assert (output_top_p >= 0).all() and (output_top_p < vocab_size).all()

    # Test both top_k and top_p
    output_both = generate(model, prompt, max_new_tokens=4, temperature=1.0, top_k=5, top_p=0.8, device=device)
    assert output_both.shape == (1, seq_len + 4)
    assert (output_both >= 0).all() and (output_both < vocab_size).all()


def test_bpe_unicode_tokenization(tmp_path):
    vocab_content = '{"😀":0, "😃":1, "😄":2}'
    merges_content = ""
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    vocab_path.write_text(vocab_content)
    merges_path.write_text(merges_content)

    bpe = BytePairEncoding(str(vocab_path), str(merges_path))
    tokens = bpe.tokenize("😀😃😄")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    decoded = bpe.decode(tokens)
    # Decoding should contain original characters (may be concatenated)
    assert all(ch in decoded for ch in "😀😃😄")


def test_gpt2_output_logits_distribution():
    vocab_size = 30
    seq_len = 5
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=3, n_embd=32, block_size=seq_len)
    model.eval()

    x = torch.randint(0, vocab_size, (1, seq_len))
    with torch.no_grad():
        logits = model(x)

    # For each position softmax along vocab dim should sum to 1
    probs = F.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    # sums shape is (1, seq_len)
    assert sums.shape == (1, seq_len)
    # all sums close to 1
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_generate_with_device_consistency():
    vocab_size = 20
    seq_len = 5
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=16, block_size=seq_len)

    prompt = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    # Run on CPU
    cpu_output = generate(model, prompt, max_new_tokens=3, temperature=1.0, device=torch.device("cpu"))
    assert cpu_output.device.type == "cpu"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        prompt_cuda = prompt.to(device)
        cuda_output = generate(model, prompt_cuda, max_new_tokens=3, temperature=1.0, device=device)
        assert cuda_output.device.type == "cuda"
        # Move back to cpu and check shape
        cuda_output_cpu = cuda_output.cpu()
        assert cuda_output_cpu.shape == cpu_output.shape
    else:
        pytest.skip("CUDA not available, skipping CUDA device test.")


def test_bpe_loading_invalid_files(tmp_path):
    # Write invalid json to vocab file
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    vocab_path.write_text("not a json")
    merges_path.write_text("")

    # BytePairEncoding should raise exception on loading invalid vocab json file
    with pytest.raises(Exception):
        _ = BytePairEncoding(str(vocab_path), str(merges_path))


def test_gpt2_forward_batch_consistency():
    vocab_size = 30
    seq_len = 10
    model = GPT_2(vocab_size=vocab_size, n_layer=2, n_head=2, n_embd=64, block_size=seq_len)
    model.eval()

    batch1 = torch.randint(0, vocab_size, (1, seq_len))
    batch2 = torch.cat([batch1, batch1], dim=0)  # batch of 2 identical samples

    with torch.no_grad():
        out1 = model(batch1)
        out2 = model(batch2)

    # The first output in batch2 should be equal to out1
    assert torch.allclose(out1, out2[0:1], atol=1e-6)


def test_generate_output_with_zero_max_new_tokens():
    vocab_size = 10
    seq_len = 4
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=1, n_embd=8, block_size=seq_len)
    model.eval()

    prompt = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    output = generate(model, prompt, max_new_tokens=0, temperature=1.0, device=torch.device("cpu"))
    # Output should be equal to prompt when max_new_tokens=0
    assert torch.equal(output, prompt)


def test_generate_invalid_top_k_and_top_p_behavior():
    vocab_size = 10
    seq_len = 6
    model = GPT_2(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=16, block_size=seq_len)
    model.eval()

    prompt = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    # top_k greater than vocab size should be handled gracefully (should equal no filtering)
    output = generate(model, prompt, max_new_tokens=3, temperature=1.0, top_k=1000, device=torch.device("cpu"))
    assert output.shape == (1, seq_len + 3)

    # top_p greater than 1 should be treated as 1 (no filtering)
    output = generate(model, prompt, max_new_tokens=3, temperature=1.0, top_p=1.5, device=torch.device("cpu"))
    assert output.shape == (1, seq_len + 3)
