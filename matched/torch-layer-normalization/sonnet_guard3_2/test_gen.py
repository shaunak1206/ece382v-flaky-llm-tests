import torch
import torch.nn.functional as F

import torch_layer_normalization
from torch_layer_normalization import LayerNormalization


def test_version_defined():
    torch.manual_seed(0)
    assert isinstance(torch_layer_normalization.__version__, str)
    assert torch_layer_normalization.__version__ != ''


def test_default_parameters_initialized():
    torch.manual_seed(0)
    layer = LayerNormalization(5)
    assert layer.gamma is not None
    assert layer.beta is not None
    assert tuple(layer.gamma.shape) == (5,)
    assert tuple(layer.beta.shape) == (5,)
    assert torch.allclose(layer.gamma, torch.ones(5))
    assert torch.allclose(layer.beta, torch.zeros(5))


def test_parameters_are_trainable_parameters():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    params = list(layer.parameters())
    assert len(params) == 2
    for p in params:
        assert isinstance(p, torch.nn.Parameter)
        assert p.requires_grad


def test_normal_shape_int():
    torch.manual_seed(0)
    layer = LayerNormalization(7)
    assert tuple(layer.normal_shape) == (7,)


def test_normal_shape_tuple_uses_last_dimension():
    torch.manual_seed(0)
    layer = LayerNormalization((3, 6))
    assert tuple(layer.normal_shape) == (6,)
    assert tuple(layer.gamma.shape) == (6,)
    assert tuple(layer.beta.shape) == (6,)


def test_no_gamma_no_beta():
    torch.manual_seed(0)
    layer = LayerNormalization(4, gamma=False, beta=False)
    assert layer.gamma is None
    assert layer.beta is None
    assert len(list(layer.parameters())) == 0
    x = torch.randn(3, 4)
    y = layer(x)
    assert tuple(y.shape) == (3, 4)


def test_only_gamma_or_only_beta():
    torch.manual_seed(0)
    only_gamma = LayerNormalization(4, beta=False)
    assert only_gamma.gamma is not None and only_gamma.beta is None
    only_beta = LayerNormalization(4, gamma=False)
    assert only_beta.gamma is None and only_beta.beta is not None
    assert len(list(only_gamma.parameters())) == 1
    assert len(list(only_beta.parameters())) == 1


def test_epsilon_stored():
    torch.manual_seed(0)
    layer = LayerNormalization(4, epsilon=1e-3)
    assert layer.epsilon == 1e-3


def test_output_shape_preserved_2d_and_3d():
    torch.manual_seed(0)
    layer = LayerNormalization(8)
    assert tuple(layer(torch.randn(5, 8)).shape) == (5, 8)
    assert tuple(layer(torch.randn(2, 3, 8)).shape) == (2, 3, 8)


def test_output_zero_mean_unit_variance():
    torch.manual_seed(0)
    layer = LayerNormalization(16)
    x = torch.randn(4, 5, 16) * 3.0 + 2.0
    y = layer(x)
    mean = y.mean(dim=-1)
    var = ((y - y.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_matches_torch_functional_layer_norm():
    torch.manual_seed(0)
    layer = LayerNormalization(10, epsilon=1e-5)
    x = torch.randn(6, 10)
    expected = F.layer_norm(x, (10,), eps=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-4)


def test_matches_functional_with_gamma_beta():
    torch.manual_seed(0)
    layer = LayerNormalization(6, epsilon=1e-5)
    with torch.no_grad():
        layer.gamma.copy_(torch.randn(6))
        layer.beta.copy_(torch.randn(6))
    x = torch.randn(3, 2, 6)
    expected = F.layer_norm(x, (6,), weight=layer.gamma.detach(),
                            bias=layer.beta.detach(), eps=1e-5)
    assert torch.allclose(layer(x), expected, atol=1e-4)


def test_gamma_and_beta_applied():
    torch.manual_seed(0)
    layer = LayerNormalization(5)
    plain = LayerNormalization(5, gamma=False, beta=False)
    with torch.no_grad():
        layer.gamma.fill_(2.0)
        layer.beta.fill_(3.0)
    x = torch.randn(4, 5)
    expected = plain(x) * 2.0 + 3.0
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_constant_input_gives_beta():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.beta.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.full((2, 4), 5.0)
    y = layer(x)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(2, 4)
    assert torch.allclose(y, expected, atol=1e-5)


def test_shift_and_scale_invariance():
    torch.manual_seed(0)
    layer = LayerNormalization(12)
    x = torch.randn(3, 12)
    y1 = layer(x)
    y2 = layer(x * 10.0 + 5.0)
    assert torch.allclose(y1, y2, atol=1e-3)


def test_rows_are_normalized_independently():
    torch.manual_seed(0)
    layer = LayerNormalization(6)
    x = torch.randn(4, 6)
    full = layer(x)
    for i in range(4):
        single = layer(x[i:i + 1])
        assert torch.allclose(full[i:i + 1], single, atol=1e-5)


def test_backward_produces_gradients():
    torch.manual_seed(0)
    layer = LayerNormalization(5)
    x = torch.randn(3, 5, requires_grad=True)
    w = torch.randn(3, 5)
    loss = (layer(x) * w).sum()
    loss.backward()
    assert x.grad is not None
    assert tuple(x.grad.shape) == (3, 5)
    assert layer.gamma.grad is not None
    assert layer.beta.grad is not None
    assert torch.allclose(layer.beta.grad, w.sum(dim=0), atol=1e-5)


def test_reset_parameters_restores_defaults():
    torch.manual_seed(0)
    layer = LayerNormalization(4)
    with torch.no_grad():
        layer.gamma.fill_(7.0)
        layer.beta.fill_(-2.0)
    layer.reset_parameters()
    assert torch.allclose(layer.gamma, torch.ones(4))
    assert torch.allclose(layer.beta, torch.zeros(4))


def test_extra_repr_and_state_dict():
    torch.manual_seed(0)
    layer = LayerNormalization(3, epsilon=1e-4)
    text = layer.extra_repr()
    assert 'epsilon' in text
    assert 'normal_shape' in text
    assert set(layer.state_dict().keys()) == {'gamma', 'beta'}
