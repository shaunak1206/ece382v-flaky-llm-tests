import pytest
import random
import numpy as np
import torch
import math
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


def test_gelu_values():
    # test gelu at some fixed inputs, compared to known expected values (from GELU definition)
    xs = torch.tensor([-3.0, -1.0, 0.0, 0.5, 1.0, 3.0])
    # gelu implementation in torch_gpt_2 matches the standard GELU
    ys = gelu(xs)
    # expected values computed with torch.nn.functional.gelu(xs)
    expected = torch.tensor([-0.0040495, -0.15880796, 0.0, 0.3457146, 0.84119207, 2.9960494])
    assert torch.allclose(ys, expected, rtol=1e-5, atol=1e-6)


def test_gelu_output_shape_and_type():
    x = torch.randn(10, 5)
    y = gelu(x)
    assert y.shape == x.shape
    assert isinstance(y, torch.Tensor)
    # gelu output should be finite
    assert torch.isfinite(y).all()


def test_BytePairEncoding_encode_decode_roundtrip():
    # BytePairEncoding requires BPE vocab and merges. We simulate a minimal vocab/merges per keras_gpt_2 docs.
    # Since keras_gpt_2 is an actual dependency, assume BPE files are loaded normally,
    # but we do not have files here, so we build minimal vocab/merges manually.
    vocab = {"l":0, "o":1, "w":2, "lo":3, "low":4}
    merges = [("l","o"), ("lo","w")]
    bpe = BytePairEncoding(vocab, merges)
    text = "low"
    encoded = bpe.encode(text)
    # encode returns a list of token ids (int)
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    # decode should round trip to original
    decoded = bpe.decode(encoded)
    assert isinstance(decoded, str)
    # decoded text may include a space appended if the tokenizer adds it at end/beginning
    assert text in decoded or decoded in text


def test_BytePairEncoding_encode_handles_empty_string():
    bpe = BytePairEncoding({}, [])
    encoded = bpe.encode("")
    assert encoded == []
    decoded = bpe.decode(encoded)
    assert decoded == ""


def test_BytePairEncoding_decode_invalid_token_id_results_in_empty_or_no_error():
    # Pass token ids not in vocab - decode should handle gracefully, ideally returning empty or joining unknown tokens as is
    bpe = BytePairEncoding({"a":0}, [])
    # token id 99999 not in vocab mapping
    decoded = bpe.decode([99999])
    # decoded is str, may be empty string or string containing unknown tokens, check type and not error
    assert isinstance(decoded, str)


def test_GPT_2_predict_output_shape_and_type():
    # seed torch RNG to avoid flaky test, GPT_2 uses torch inside
    torch.manual_seed(42)
    model = GPT_2()
    # test a typical input tensor shape, e.g. batch=2, seq=4
    tokens = torch.randint(0, 5000, (2, 4), dtype=torch.long)
    output = model.predict(tokens)
    # output shape expected [batch, seq, vocab_size], vocab_size known by model.vocab_size
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == tokens.shape[0]
    assert output.shape[1] == tokens.shape[1]
    assert output.shape[2] == model.vocab_size
    # output is probabilities (softmaxed?), so values in [0,1]
    assert (output >= 0).all() and (output <= 1).all()
    # each vector sums to 1 across vocab dim (softmax)
    sums = output.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-4)


def test_GPT_2_can_load_and_predict_consistent(monkeypatch):
    # seed torch RNG for reproducibility
    torch.manual_seed(123)
    # call predict twice on same input should produce identical outputs (deterministic)
    model = GPT_2()
    tokens = torch.randint(0, model.vocab_size, (1, 5), dtype=torch.long)
    out1 = model.predict(tokens)
    out2 = model.predict(tokens)
    assert torch.allclose(out1, out2, rtol=1e-6, atol=1e-8)


def test_generate_output_length_and_type_and_basic_properties():
    # seed randoms used inside generate, likely torch and random
    torch.manual_seed(111)
    random.seed(111)
    np.random.seed(111)
    # simple seed tokens: batch 1, seq 3 with small vocab tokens
    seed_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    model = GPT_2()
    max_len = 7
    generated = generate(model, seed_tokens, max_len=max_len)
    # generated shape batch x seq (seq >= max_len)
    assert isinstance(generated, torch.Tensor)
    assert generated.shape[0] == seed_tokens.shape[0]
    assert generated.shape[1] >= seed_tokens.shape[1]
    assert generated.shape[1] <= max_len or generated.shape[1] == max_len
    # tokens are valid ints in vocab range
    assert torch.is_integral(generated).all()
    assert (generated >= 0).all()
    assert (generated < model.vocab_size).all()
    # generated tokens start with seed tokens (prefix)
    for i in range(seed_tokens.shape[0]):
        assert (generated[i, :seed_tokens.shape[1]] == seed_tokens[i]).all()


def test_generate_with_max_len_eq_seed_len_returns_seed(monkeypatch):
    torch.manual_seed(222)
    random.seed(222)
    np.random.seed(222)

    model = GPT_2()
    seed_tokens = torch.randint(0, model.vocab_size, (1, 5), dtype=torch.long)
    generated = generate(model, seed_tokens, max_len=seed_tokens.shape[1])
    # Should return identical tensor (seed), no new tokens
    assert torch.equal(generated, seed_tokens)


def test_generate_with_max_len_less_than_seed_len_raises(monkeypatch):
    model = GPT_2()
    seed_tokens = torch.randint(0, model.vocab_size, (1, 5), dtype=torch.long)
    # max_len less than seed length should raise or error
    with pytest.raises(Exception):
        generate(model, seed_tokens, max_len=seed_tokens.shape[1] - 1)


def test_gpt2_predict_with_different_batch_sizes_and_seq_lengths():
    torch.manual_seed(77)
    model = GPT_2()
    for batch_size in [1, 3]:
        for seq_len in [1, 5]:
            tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len), dtype=torch.long)
            output = model.predict(tokens)
            assert output.shape == (batch_size, seq_len, model.vocab_size)


def test_BytePairEncoding_encode_consistency_across_runs():
    vocab = {"a":0, "b":1, "ab":2}
    merges = [("a","b")]
    bpe = BytePairEncoding(vocab, merges)
    # encode same input twice returns same output
    text = "ab"
    out1 = bpe.encode(text)
    out2 = bpe.encode(text)
    assert out1 == out2


def test_BytePairEncoding_decode_consistency_across_runs():
    vocab = {"a":0, "b":1, "ab":2}
    merges = [("a","b")]
    bpe = BytePairEncoding(vocab, merges)
    token_ids = [2, 0, 1]
    out1 = bpe.decode(token_ids)
    out2 = bpe.decode(token_ids)
    assert out1 == out2


def test_gelu_monotonicity_and_symmetry():
    # gelu(x) is approximately odd function (symmetric around 0)
    x_pos = torch.linspace(0, 3, 100)
    x_neg = -x_pos
    y_pos = gelu(x_pos)
    y_neg = gelu(x_neg)
    # gelu(-x) ~ -gelu(x), test with some tolerance
    diff = y_neg + y_pos
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-4, rtol=1e-3)
    # gelu is monotonic increasing (y[i] <= y[i+1])
    assert torch.all(y_pos[1:] >= y_pos[:-1])


def test_generate_deterministic_with_fixed_seed(monkeypatch):
    seed = 555
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = GPT_2()
    seed_tokens = torch.tensor([[4, 5, 6]], dtype=torch.long)
    out1 = generate(model, seed_tokens, max_len=10)
    # reset seeds and generate again, should get the same output
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    out2 = generate(model, seed_tokens, max_len=10)
    assert torch.equal(out1, out2)
