import torch

from torch_layer_normalization import LayerNormalization


def test_int_normal_shape():
    layer = LayerNormalization(5)
    assert layer.normal_shape == torch.Size([5])


def test_tuple_normal_shape_uses_last_dim():
    layer = LayerNormalization((3, 4, 6))
    assert layer.normal_shape == torch.Size([6])
    assert layer.gamma.shape == (6,)
    assert layer.beta.shape == (6,)


def test_list_normal_shape_uses_last_dim():
    layer = LayerNormalization([2, 7])
    assert layer.normal_shape == torch.Size([7])


def test_default_parameter_initialization():
    layer = LayerNormalization(4)
    assert torch.allclose(layer.gamma, torch.ones(4))
    assert torch.allclose(layer.beta, torch.zeros(4))


def test_parameters_registered():
    layer = LayerNormalization(4)
    names = sorted(name for name, _ in layer.named_parameters())
    assert names == ['beta', 'gamma']
    assert all(p.requires_grad for p in layer.parameters())


def test_no_gamma():
    layer = LayerNormalization(4, gamma=False)
    assert layer.gamma is None
    names = [name for name, _ in layer.named_parameters()]
    assert names == ['beta']


def test_no_beta():
    layer = LayerNormalization(4, beta=False)
    assert layer.beta is None
    names = [name for name, _ in layer.named_parameters()]
    assert names == ['gamma']


def test_no_gamma_no_beta():
    layer = LayerNormalization(4, gamma=False, beta=False)
    assert layer.gamma is None
    assert layer.beta is None
    assert list(layer.parameters()) == []


def test_epsilon_stored():
    layer = LayerNormalization(3, epsilon=1e-5)
    assert layer.epsilon == 1e-5
    assert LayerNormalization(3).epsilon == 1e-10


def test_forward_output_shape():
    layer = LayerNormalization(8)
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    y = layer(x)
    assert y.shape == x.shape


def test_forward_3d_input_shape():
    layer = LayerNormalization(6)
    x = torch.arange(48, dtype=torch.float32).reshape(2, 4, 6)
    y = layer(x)
    assert y.shape == (2, 4, 6)


def test_forward_zero_mean_unit_variance():
    layer = LayerNormalization(5, epsilon=1e-12)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                      [10.0, -3.0, 7.0, 0.5, 2.0]], dtype=torch.float64)
    layer = layer.double()
    y = layer(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(2, dtype=torch.float64), atol=1e-8)
    var = ((y - y.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
    assert torch.allclose(var, torch.ones(2, dtype=torch.float64), atol=1e-6)


def test_forward_known_values():
    layer = LayerNormalization(2, epsilon=0.0)
    x = torch.tensor([[1.0, 3.0]])
    y = layer(x)
    assert torch.allclose(y, torch.tensor([[-1.0, 1.0]]))


def test_forward_gamma_and_beta_applied():
    layer = LayerNormalization(2, epsilon=0.0)
    with torch.no_grad():
        layer.gamma.copy_(torch.tensor([2.0, 3.0]))
        layer.beta.copy_(torch.tensor([10.0, -10.0]))
    x = torch.tensor([[1.0, 3.0]])
    y = layer(x)
    assert torch.allclose(y, torch.tensor([[8.0, -7.0]]))


def test_forward_constant_input_gives_beta():
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.beta.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.full((2, 4), 7.0)
    y = layer(x)
    assert torch.allclose(y, torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(2, 4))


def test_forward_does_not_modify_input():
    layer = LayerNormalization(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    original = x.clone()
    layer(x)
    assert torch.equal(x, original)


def test_forward_matches_torch_layer_norm():
    layer = LayerNormalization(6, epsilon=1e-5)
    x = torch.tensor([[0.5, -1.0, 2.0, 3.5, -0.25, 1.0],
                      [4.0, 3.0, 2.0, 1.0, 0.0, -1.0]])
    expected = torch.nn.functional.layer_norm(x, (6,), eps=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_reset_parameters_restores_defaults():
    layer = LayerNormalization(3)
    with torch.no_grad():
        layer.gamma.fill_(5.0)
        layer.beta.fill_(-2.0)
    layer.reset_parameters()
    assert torch.allclose(layer.gamma, torch.ones(3))
    assert torch.allclose(layer.beta, torch.zeros(3))


def test_backward_produces_gradients():
    layer = LayerNormalization(4)
    x = torch.tensor([[1.0, 2.0, 4.0, 8.0]], requires_grad=True)
    layer(x).sum().backward()
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None
    assert x.grad is not None


def test_extra_repr():
    layer = LayerNormalization(4, gamma=False, epsilon=1e-5)
    text = repr(layer)
    assert 'normal_shape=torch.Size([4])' in text
    assert 'gamma=False' in text
    assert 'beta=True' in text
    assert 'epsilon=1e-05' in text
