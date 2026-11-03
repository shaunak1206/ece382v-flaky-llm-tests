import pytest
import torch
import torch.nn.functional as F
from torch_gpt_2 import BytePairEncoding, GPT_2, gelu, generate


def test_gelu_basic():
    # gelu(x) should be close to official gelu implementation
    x = torch.linspace(-3, 3, 100)
    expected = 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * x ** 3)))
    actual = gelu(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_gelu_scalar_value():
    # Test gelu with a scalar tensor
    x = torch.tensor(0.5)
    out = gelu(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape


@pytest.fixture
def sample_bpe_data(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"

    vocab_content = '{"hello": 0, "world": 1, "!": 2}'
    merges_content = "h e\nhe l\nhel l o\nw o\nwo r\nwor l\ndl !\n"

    vocab_path.write_text(vocab_content)
    merges_path.write_text(merges_content)
    return str(vocab_path), str(merges_path)


def test_bytepairencoding_init_and_encode_decode(sample_bpe_data):
    vocab_path, merges_path = sample_bpe_data

    bpe = BytePairEncoding.from_files(vocab_path, merges_path)
    text = "hello world!"
    ids = bpe.encode(text)
    decoded = bpe.decode(ids)
    # decoded text might differ on spaces etc, but decoding then encoding should be stable
    ids_2 = bpe.encode(decoded)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert ids == ids_2


def test_bpe_encode_unknown_tokens(sample_bpe_data):
    vocab_path, merges_path = sample_bpe_data
    bpe = BytePairEncoding.from_files(vocab_path, merges_path)
    unknown_text = "@@@unknown@@@"
    ids = bpe.encode(unknown_text)
    assert isinstance(ids, list)
    # Should return some token ids, not empty
    assert len(ids) > 0


def test_bpe_decode_empty_list(sample_bpe_data):
    vocab_path, merges_path = sample_bpe_data
    bpe = BytePairEncoding.from_files(vocab_path, merges_path)
    decoded = bpe.decode([])
    # Should return an empty string on empty input list
    assert decoded == ""


def test_gpt2_forward_shapes():
    device = "cpu"
    model = GPT_2(n_vocab=1000, n_ctx=32, n_embd=128, n_head=4, n_layer=2).to(device)
    x = torch.randint(0, 1000, (2, 10), device=device)
    logits = model(x)
    assert logits.shape == (2, 10, 1000)


def test_gpt2_forward_different_length():
    device = "cpu"
    model = GPT_2(n_vocab=500, n_ctx=16, n_embd=64, n_head=4, n_layer=1).to(device)
    for length in (1, 5, 16):
        x = torch.randint(0, 500, (1, length), device=device)
        logits = model(x)
        assert logits.shape == (1, length, 500)


def test_gpt2_generate_output_length():
    device = "cpu"
    model = GPT_2(n_vocab=100, n_ctx=20, n_embd=64, n_head=4, n_layer=1).to(device)
    bpe = BytePairEncoding()
    # mock a minimal vocab to pass in generate example
    bpe.encode = lambda txt: [1, 2, 3]
    bpe.decode = lambda ids: "abc"
    bpe.n_vocab = 100

    # monkeypatch model generation to do controlled output
    def fake_forward(x):
        b, t = x.shape
        logits = torch.full((b, t, 100), -float('inf'))
        logits[:, -1, 10] = 10.0  # make token id=10 highest prob always
        return logits

    model.forward = fake_forward

    prompt = "abc"
    output = generate(model, bpe, prompt, n=5, temperature=1.0, device=device)
    # output should be prompt + generated tokens
    assert output.startswith(prompt)
    assert len(output) > len(prompt)


def test_generate_greedy_sampling(monkeypatch):
    device = "cpu"
    vocab_size = 10
    model = GPT_2(n_vocab=vocab_size, n_ctx=10, n_embd=32, n_head=4, n_layer=1).to(device)
    bpe = BytePairEncoding()
    bpe.n_vocab = vocab_size
    # encode "hi" as tokens [1,2]
    bpe.encode = lambda _: [1, 2]
    # decode token lists by joining numbers as string (for test)
    bpe.decode = lambda ids: "".join(str(i) for i in ids)

    # model emits logits that always predict token 3 next
    def fake_forward(x):
        b, t = x.shape
        logits = torch.full((b, t, vocab_size), float('-inf'))
        logits[:, -1, 3] = 10  # token 3 highest logit
        return logits

    model.forward = fake_forward
    output = generate(model, bpe, "hi", n=3, temperature=0, device=device)
    # Since temperature=0, should pick argmax token 3 repeatedly
    assert output == "12" + "3" * 3


def test_generate_with_temperature(monkeypatch):
    device = "cpu"
    vocab_size = 5
    model = GPT_2(n_vocab=vocab_size, n_ctx=10, n_embd=32, n_head=2, n_layer=1).to(device)
    bpe = BytePairEncoding()
    bpe.n_vocab = vocab_size
    bpe.encode = lambda _: [0]
    bpe.decode = lambda ids: "".join(str(i) for i in ids)

    logits = torch.ones((1, 1, vocab_size))
    logits[0, 0, :] = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])

    def fake_forward(x):
        return logits

    model.forward = fake_forward

    # temperature > 0 should allow sampling not just argmax
    output = generate(model, bpe, "", n=1, temperature=1.0, device=device)
    assert len(output) == 1 or len(output) == 2  # Either empty prompt + 1 token


def test_bpe_from_files_invalid_path(tmp_path):
    # Using get_bpe_from_files from keras_gpt_2 should fail with invalid file
    with pytest.raises(Exception):
        from torch_gpt_2 import get_bpe_from_files
        get_bpe_from_files("nonexistent_vocab.json", "nonexistent_merges.txt")
