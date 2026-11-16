import torch

from torch_layer_normalization import LayerNormalization


def test_default_parameters_initialized():
    torch.manual_seed(0)
    layer = LayerNormalization(5)
    assert layer.gamma is not None
    assert layer.beta is not None
    assert torch.allclose(layer.gamma, torch.ones(5))
    assert torch.allclose(layer.beta, torch.zeros(5))


def test_int_normal_shape():
    torch.manual_seed(0)
    layer = LayerNormalization(7)
    assert tuple(layer.normal_shape) == (7,)
    assert tuple(layer.gamma.shape) == (7,)
    assert tuple(layer.beta.shape) == (7,)


def test_tuple_normal_shape_uses_last_dim():
    torch.manual_seed(0)
    layer = LayerNormalization((3, 4, 6))
    assert tuple(layer.normal_shape) == (6,)
    assert tuple(layer.gamma.shape) == (6,)
    assert tuple(layer.beta.shape) == (6,)


def test_no_gamma():
    torch.manual_seed(0)
    layer = LayerNormalization(4, gamma=False)
    assert layer.gamma is None
    assert layer.beta is not None
    names = [n for n, _ in layer.named_parameters()]
    assert names == ["beta"]


def test_no_beta():
    torch.manual_seed(0)
    layer = LayerNormalization(4, beta=False)
    assert layer.beta is None
    assert layer.gamma is not None
    names = [n for n, _ in layer.named_parameters()]
    assert names == ["gamma"]


def test_no_gamma_no_beta_has_no_parameters():
    torch.manual_seed(0)
    layer = LayerNormalization(4, gamma=False, beta=False)
    assert len(list(layer.parameters())) == 0
    x = torch.randn(3, 4)
    y = layer(x)
    assert y.shape == x.shape


def test_epsilon_stored():
    torch.manual_seed(0)
    assert LayerNormalization(3).epsilon == 1e-10
    assert LayerNormalization(3, epsilon=1e-3).epsilon == 1e-3


def test_output_shape_preserved_2d_and_3d():
    torch.manual_seed(0)
    layer = LayerNormalization(8)
    assert layer(torch.randn(4, 8)).shape == (4, 8)
    assert layer(torch.randn(2, 3, 8)).shape == (2, 3, 8)


def test_output_zero_mean_unit_variance_with_defaults():
    torch.manual_seed(0)
    layer = LayerNormalization(16)
    x = torch.randn(5, 16) * 3.0 + 2.0
    y = layer(x)
    mean = y.mean(dim=-1)
    var = ((y - mean.unsqueeze(-1)) ** 2).mean(dim=-1)
    assert torch.allclose(mean, torch.zeros(5), atol=1e-5)
    assert torch.allclose(var, torch.ones(5), atol=1e-4)


def test_matches_torch_functional_layer_norm():
    torch.manual_seed(0)
    layer = LayerNormalization(10, epsilon=1e-5)
    with torch.no_grad():
        layer.gamma.copy_(torch.randn(10))
        layer.beta.copy_(torch.randn(10))
    x = torch.randn(3, 4, 10)
    expected = torch.nn.functional.layer_norm(
        x, (10,), layer.gamma.detach(), layer.beta.detach(), eps=1e-5)
    out = layer(x)
    assert torch.allclose(out, expected, atol=1e-5)


def test_gamma_and_beta_applied():
    torch.manual_seed(0)
    x = torch.randn(4, 6)
    base = LayerNormalization(6, gamma=False, beta=False)(x)
    layer = LayerNormalization(6)
    with torch.no_grad():
        layer.gamma.fill_(2.0)
        layer.beta.fill_(0.5)
    out = layer(x)
    assert torch.allclose(out, base * 2.0 + 0.5, atol=1e-5)


def test_constant_input_gives_beta():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.beta.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.full((2, 4), 2.0)
    out = layer(x)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(2, 4)
    assert torch.allclose(out, expected, atol=1e-4)


def test_rows_normalized_independently():
    torch.manual_seed(0)
    layer = LayerNormalization(5)
    x = torch.randn(3, 5)
    full = layer(x)
    for i in range(3):
        row = layer(x[i:i + 1])
        assert torch.allclose(full[i:i + 1], row, atol=1e-6)


def test_invariant_to_shift_and_positive_scale():
    torch.manual_seed(0)
    layer = LayerNormalization(8, epsilon=1e-10)
    x = torch.randn(4, 8)
    y1 = layer(x)
    y2 = layer(x * 5.0 + 3.0)
    assert torch.allclose(y1, y2, atol=1e-3)


def test_gradients_flow_to_parameters_and_input():
    torch.manual_seed(0)
    layer = LayerNormalization(6)
    x = torch.randn(3, 6, requires_grad=True)
    layer(x).pow(2).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None
    assert layer.gamma.grad.shape == (6,)
    assert layer.beta.grad.shape == (6,)


def test_reset_parameters_restores_defaults():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.gamma.fill_(7.0)
        layer.beta.fill_(-3.0)
    layer.reset_parameters()
    assert torch.allclose(layer.gamma, torch.ones(4))
    assert torch.allclose(layer.beta, torch.zeros(4))


def test_reset_parameters_without_params_does_not_fail():
    torch.manual_seed(0)
    layer = LayerNormalization(4, gamma=False, beta=False)
    layer.reset_parameters()
    assert layer.gamma is None and layer.beta is None


def test_extra_repr_contains_configuration():
    torch.manual_seed(0)
    layer = LayerNormalization(4, gamma=True, beta=False, epsilon=0.5)
    text = layer.extra_repr()
    assert "normal_shape" in text
    assert "gamma=True" in text
    assert "beta=False" in text
    assert "0.5" in text


def test_state_dict_keys():
    torch.manual_seed(0)
    layer = LayerNormalization(3)
    assert sorted(layer.state_dict().keys()) == ["beta", "gamma"]
