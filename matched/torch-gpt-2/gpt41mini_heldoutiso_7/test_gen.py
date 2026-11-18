import pytest
import torch
import torch.nn.functional as F
from torch_gpt_2 import BytePairEncoding, generate, gelu, GPT_2


def test_gelu_values():
    # Test basic gelu behavior on known input
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = gelu(x)
    # GELU is close to input * sigmoid approximation, check monotonicity and range
    assert y.shape == x.shape
    assert torch.all(y[1:] >= y[:-1])
    # At zero, gelu(0) ~ 0
    assert torch.isclose(y[2], torch.tensor(0.), atol=1e-5)
    # Values roughly smaller magnitude near 0 than input
    assert abs(y[1].item()) < abs(x[1].item())
    assert abs(y[3].item()) < abs(x[3].item())


@pytest.mark.parametrize("text", ["hello world", "", " ", "123", "The quick brown fox"])
def test_bytepairencoding_encode_decode_roundtrip(text):
    bpe = BytePairEncoding()
    # encode a text string to tokens and decode back to string
    tokens = bpe.encode(text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    decoded = bpe.decode(tokens)
    # Decoded string should be str and contain the original text tokens (possibly different spacing)
    assert isinstance(decoded, str)
    # Because of subword tokenization decoded can differ in spacing, so test encode->decode->encode consistent tokens
    tokens2 = bpe.encode(decoded)
    assert tokens == tokens2


def test_bytepairencoding_encode_empty_string():
    bpe = BytePairEncoding()
    tokens = bpe.encode("")
    assert tokens == []


def test_bytepairencoding_decode_empty_list():
    bpe = BytePairEncoding()
    decoded = bpe.decode([])
    assert decoded == ""


def test_generate_returns_string():
    # generate requires a model and some input tokens
    # Setup minimal GPT_2 config and model
    model = GPT_2()
    prompt = "hello"
    bpe = BytePairEncoding()
    tokens = bpe.encode(prompt)
    assert len(tokens) > 0
    # Generate some text with default parameters (max_length=20)
    output_text = generate(model, tokens, max_length=20)
    assert isinstance(output_text, str)
    # Output text should contain some generated content
    assert len(output_text) > 0


def test_generate_with_custom_max_length():
    model = GPT_2()
    bpe = BytePairEncoding()
    tokens = bpe.encode("test")
    output = generate(model, tokens, max_length=5)
    # Output string length should be >= input length (since input tokens decoded usually shorter than tokens count)
    assert isinstance(output, str)
    assert len(output) > 0


def test_gpt2_forward_output_shape():
    model = GPT_2()
    batch_size = 2
    seq_len = 5
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    # Output shape should be batch_size x seq_len x vocab_size
    assert output.shape == (batch_size, seq_len, model.config.vocab_size)
    # Output should be a torch Tensor
    assert torch.is_tensor(output)
    # Output values are logits, so can be positive or negative
    assert output.min() <= output.max()


def test_gpt2_generate_returns_output_with_no_errors():
    model = GPT_2()
    bpe = BytePairEncoding()
    input_text = "Hello GPT2"
    tokens = bpe.encode(input_text)
    # generate usually returns a string, call and check no exceptions
    output = generate(model, tokens, max_length=10)
    assert isinstance(output, str)
    assert len(output) > 0


def test_bytepairencoding_encode_invalid_type():
    bpe = BytePairEncoding()
    with pytest.raises(TypeError):
        # Passing non-str should raise TypeError (assuming implementation)
        bpe.encode(123)


def test_bytepairencoding_decode_invalid_type():
    bpe = BytePairEncoding()
    with pytest.raises(TypeError):
        # Passing non-list should raise TypeError (assuming implementation)
        bpe.decode("not a list")


def test_generate_with_invalid_model_type():
    bpe = BytePairEncoding()
    tokens = bpe.encode("hi")
    with pytest.raises(AttributeError):
        # Passing an invalid model should raise an exception (e.g., missing forward method)
        generate("not a model", tokens)


def test_generate_with_empty_tokens():
    model = GPT_2()
    output = generate(model, [], max_length=5)
    # Even with empty tokens, generate returns a string (maybe empty or short)
    assert isinstance(output, str)


def test_gpt2_config_default_values():
    model = GPT_2()
    config = model.config
    # Test config key presence and reasonable defaults
    assert hasattr(config, "vocab_size")
    assert hasattr(config, "n_ctx")
    assert hasattr(config, "n_embd")
    assert hasattr(config, "n_layer")
    assert hasattr(config, "n_head")
    assert isinstance(config.vocab_size, int)
    assert isinstance(config.n_ctx, int)
    assert 0 < config.n_embd < 10000
    assert 0 < config.n_layer < 100
    assert 0 < config.n_head < 100


def test_gelus_gradient():
    x = torch.randn(5, requires_grad=True)
    y = gelu(x)
    assert y.requires_grad
    y.sum().backward()
    # gradients should be populated
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_generate_produces_different_outputs_on_different_seeds():
    # Some minimal stochasticity test if random seed changed, output differs
    model = GPT_2()
    bpe = BytePairEncoding()
    tokens = bpe.encode("seed test")
    torch.manual_seed(42)
    output1 = generate(model, tokens, max_length=8)
    torch.manual_seed(43)
    output2 = generate(model, tokens, max_length=8)
    # Outputs likely differ
    assert output1 != output2


def test_generate_with_zero_max_length_raises():
    model = GPT_2()
    bpe = BytePairEncoding()
    tokens = bpe.encode("test")
    with pytest.raises(ValueError):
        generate(model, tokens, max_length=0)


def test_gpt2_forward_requires_long_tensor():
    model = GPT_2()
    with pytest.raises(RuntimeError):
        # Passing float tensor instead of int should raise runtime (index) error
        model(torch.randn(2, 2))


def test_gpt2_can_process_single_token_batch():
    model = GPT_2()
    input_ids = torch.tensor([[0]])
    output = model(input_ids)
    assert output.shape == (1, 1, model.config.vocab_size)


def test_bytepairencoding_decode_invalid_token_value():
    bpe = BytePairEncoding()
    # Decoding token ids not in vocab should not crash and produce a string
    invalid_tokens = [999999, 123456]
    decoded = bpe.decode(invalid_tokens)
    assert isinstance(decoded, str)


def test_generate_with_custom_temperature():
    model = GPT_2()
    bpe = BytePairEncoding()
    tokens = bpe.encode("temperature test")
    # generate supports temperature param (assuming)
    output = generate(model, tokens, max_length=10, temperature=0.5)
    assert isinstance(output, str)
    assert len(output) > 0
