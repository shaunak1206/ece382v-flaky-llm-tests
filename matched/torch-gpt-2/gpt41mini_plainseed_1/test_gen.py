import pytest
import torch
import numpy as np

from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


# ----------------------------
# Tests for BytePairEncoding
# ----------------------------

def test_bpe_encode_decode_consistency(tmp_path):
    # Prepare vocab and merges files
    vocab_content = '{"t":0, "h":1, "th":2, "e":3, "the":4, "r":5, "er":6, "e_":7, "re":8}'
    merges_content = "t h\nth e\nr e\n"

    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"

    vocab_file.write_text(vocab_content)
    merges_file.write_text(merges_content)

    bpe = BytePairEncoding(str(vocab_file), str(merges_file))
    text = "there"

    encoded = bpe.encode(text)
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)

    decoded = bpe.decode(encoded)
    assert decoded == text


def test_bpe_encode_unknown_token(tmp_path):
    vocab_content = '{"a":0, "b":1}'
    merges_content = ""

    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"

    vocab_file.write_text(vocab_content)
    merges_file.write_text(merges_content)

    bpe = BytePairEncoding(str(vocab_file), str(merges_file))

    # Input some characters with unknown tokens
    text = "abc"
    encoded = bpe.encode(text)

    # Expect unknown tokens skipped or handled - cannot be empty
    assert len(encoded) >= 1
    # Decoding should return the known tokens, likely partial
    decoded = bpe.decode(encoded)
    assert isinstance(decoded, str)
    assert decoded in text


def test_bpe_decode_invalid_indices(tmp_path):
    # Setup minimal vocab / merges
    vocab_content = '{"a":0, "b":1}'
    merges_content = ""

    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"

    vocab_file.write_text(vocab_content)
    merges_file.write_text(merges_content)

    bpe = BytePairEncoding(str(vocab_file), str(merges_file))

    # Provide invalid indices (out of range)
    invalid_indices = [999, 1000, -1]

    with pytest.raises(IndexError):
        bpe.decode(invalid_indices)


# ----------------------------
# Tests for gelu
# ----------------------------

def test_gelu_output_type_and_shape():
    x_np = np.linspace(-3, 3, 7).astype(np.float32)
    x = torch.tensor(x_np)

    y = gelu(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape

def test_gelu_output_values():
    x = torch.tensor([-2.0, 0.0, 2.0])
    y = gelu(x)

    # GELU output approx values for reference
    expected = torch.tensor([-0.0454027, 0.0, 1.9545973])

    # Close in values (relative tolerance)
    assert torch.allclose(y, expected, atol=1e-5)


def test_gelu_gradient():
    x = torch.linspace(-3, 3, steps=10, requires_grad=True)
    y = gelu(x)
    y.sum().backward()
    # Gradient must be non-none and have same shape
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ----------------------------
# Tests for GPT_2 class
# ----------------------------

def test_gpt2_forward_output_shape():
    model = GPT_2(
        vocab_size=50,
        n_ctx=10,
        n_embd=16,
        n_layer=2,
        n_head=2
    )

    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, 50, (batch_size, seq_len))

    output = model(input_ids)
    # output shape should be (batch_size, seq_len, vocab_size)
    assert output.shape == (batch_size, seq_len, 50)
    # Output is a tensor
    assert isinstance(output, torch.Tensor)


def test_gpt2_forward_different_batch_seq_length():
    model = GPT_2(
        vocab_size=30,
        n_ctx=15,
        n_embd=32,
        n_layer=1,
        n_head=4
    )

    # batch size 2, shorter seq length 7
    input_ids = torch.randint(0, 30, (2, 7))
    output = model(input_ids)

    assert output.shape == (2, 7, 30)


def test_gpt2_invalid_input_dim():
    model = GPT_2(
        vocab_size=20,
        n_ctx=10,
        n_embd=8,
        n_layer=1,
        n_head=2
    )
    # Provide wrong input shape (1D tensor)
    input_ids = torch.randint(0, 20, (10,))
    with pytest.raises(RuntimeError):
        model(input_ids)


def test_gpt2_generate_simple():
    model = GPT_2(
        vocab_size=40,
        n_ctx=10,
        n_embd=16,
        n_layer=2,
        n_head=4
    )

    input_ids = torch.randint(0, 40, (1, 5))

    # Generate 3 tokens, temperature 1, no top_k filtering
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_tokens=3,
        temperature=1.0,
        top_k=None,
        eos_id=None
    )
    # Output shape: [batch_size, input_seq_len + max_tokens]
    assert isinstance(output_ids, torch.Tensor)
    assert output_ids.shape[0] == 1
    assert output_ids.shape[1] == 5 + 3

    # All token ids in range of vocab
    assert ((output_ids >= 0) & (output_ids < 40)).all()


def test_generate_stops_on_eos():
    # Create a trivial model whose forward returns high prob for eos_id after one step
    class DummyModel:
        def __init__(self, eos_id):
            self.vocab_size = eos_id + 1
            self.eos_id = eos_id

        def __call__(self, input_ids):
            batch_size, seq_len = input_ids.shape
            # Produce random logits but force high logit at eos_id for last token
            logits = torch.zeros((batch_size, seq_len, self.vocab_size))
            logits[..., self.eos_id] = 1000.0
            return logits

    eos_id = 2
    model = DummyModel(eos_id)
    input_ids = torch.zeros((1, 1), dtype=torch.long)

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_tokens=10,
        temperature=1.0,
        top_k=None,
        eos_id=eos_id
    )
    # Should stop generating as soon as eos_id is generated
    assert eos_id in output_ids[0].tolist()
    # Length can be <= 1+max_tokens
    assert output_ids.shape[1] <= 11


def test_generate_top_k_filtering_effect():
    model = GPT_2(
        vocab_size=20,
        n_ctx=15,
        n_embd=8,
        n_layer=1,
        n_head=2
    )

    input_ids = torch.randint(0, 20, (1, 3))
    output_ids_no_topk = generate(
        model=model,
        input_ids=input_ids,
        max_tokens=5,
        temperature=1.0,
        top_k=None,
        eos_id=None
    )
    output_ids_topk = generate(
        model=model,
        input_ids=input_ids,
        max_tokens=5,
        temperature=1.0,
        top_k=3,
        eos_id=None
    )
    # Both outputs shapes are correct
    assert output_ids_no_topk.shape == (1, 3 + 5)
    assert output_ids_topk.shape == (1, 3 + 5)
    # Output token ids are in valid range
    assert ((output_ids_topk >= 0) & (output_ids_topk < 20)).all()
    assert ((output_ids_no_topk >= 0) & (output_ids_no_topk < 20)).all()


def test_generate_temperature_variation():
    model = GPT_2(
        vocab_size=15,
        n_ctx=7,
        n_embd=8,
        n_layer=1,
        n_head=1
    )
    input_ids = torch.randint(0, 15, (1, 4))

    for temp in [0.1, 0.5, 1.0, 2.0]:
        output_ids = generate(
            model=model,
            input_ids=input_ids,
            max_tokens=3,
            temperature=temp,
            top_k=None,
            eos_id=None
        )
        assert output_ids.shape == (1, 4 + 3)
        assert ((output_ids >= 0) & (output_ids < 15)).all()


def test_generate_with_empty_input_ids():
    model = GPT_2(
        vocab_size=10,
        n_ctx=5,
        n_embd=8,
        n_layer=1,
        n_head=1
    )
    # Empty sequence input (batch_size=1, seq_len=0)
    input_ids = torch.empty((1, 0), dtype=torch.long)
    with pytest.raises(Exception):
        generate(
            model=model,
            input_ids=input_ids,
            max_tokens=2,
            temperature=1.0,
            top_k=None,
            eos_id=None
        )
