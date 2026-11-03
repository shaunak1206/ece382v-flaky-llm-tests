import pytest
import torch
import torch.nn.functional as F

from torch_layer_normalization import LayerNormalization


def _module(normal_shape, **kwargs):
    return LayerNormalization(normal_shape, **kwargs).double()


def _rand(shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float64)


def test_int_normal_shape_sets_feature_dimension():
    ln = LayerNormalization(5)
    assert ln.normal_shape == torch.Size([5])
    assert tuple(ln.gamma.shape) == (5,)
    assert tuple(ln.beta.shape) == (5,)


def test_tuple_normal_shape_uses_only_last_dimension():
    ln = LayerNormalization((2, 3, 7))
    assert ln.normal_shape == torch.Size([7])
    assert tuple(ln.gamma.shape) == (7,)
    assert tuple(ln.beta.shape) == (7,)


def test_list_normal_shape_uses_only_last_dimension():
    ln = LayerNormalization([4, 6])
    assert ln.normal_shape == torch.Size([6])
    assert tuple(ln.gamma.shape) == (6,)


def test_default_parameters_initialised_to_identity_affine():
    ln = LayerNormalization(6)
    assert torch.all(ln.gamma.data == 1)
    assert torch.all(ln.beta.data == 0)


def test_default_epsilon_is_1e_minus_10():
    assert LayerNormalization(3).epsilon == 1e-10


def test_gamma_false_removes_scale_parameter():
    ln = LayerNormalization(4, gamma=False)
    assert ln.gamma is None
    assert [name for name, _ in ln.named_parameters()] == ['beta']


def test_beta_false_removes_shift_parameter():
    ln = LayerNormalization(4, beta=False)
    assert ln.beta is None
    assert [name for name, _ in ln.named_parameters()] == ['gamma']


def test_no_affine_parameters_when_gamma_and_beta_disabled():
    ln = _module(4, gamma=False, beta=False)
    assert list(ln.parameters()) == []
    x = _rand((2, 4), seed=0)
    assert ln(x).shape == x.shape


def test_forward_preserves_input_shape():
    ln = _module(8)
    x = _rand((2, 3, 8), seed=0)
    assert ln(x).shape == x.shape


def test_forward_does_not_modify_input():
    ln = _module(8)
    x = _rand((4, 8), seed=0)
    snapshot = x.clone()
    ln(x)
    assert torch.equal(x, snapshot)


def test_default_output_has_zero_mean_and_unit_variance_per_row():
    ln = _module(16)
    x = _rand((5, 16), seed=1) * 3.0 + 2.0
    y = ln(x)
    mean = y.mean(dim=-1)
    var = ((y - y.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1)
    assert torch.allclose(mean, torch.zeros(5, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(var, torch.ones(5, dtype=torch.float64), atol=1e-6)


def test_matches_hand_written_reference_with_affine_parameters():
    ln = _module(6)
    with torch.no_grad():
        ln.gamma.copy_(_rand((6,), seed=2))
        ln.beta.copy_(_rand((6,), seed=3))
    x = _rand((3, 4, 6), seed=4)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    expected = (x - mean) / torch.sqrt(var + ln.epsilon) * ln.gamma + ln.beta
    assert torch.allclose(ln(x), expected, atol=1e-10)


def test_matches_torch_functional_layer_norm():
    ln = _module(10, epsilon=1e-5)
    with torch.no_grad():
        ln.gamma.copy_(_rand((10,), seed=5))
        ln.beta.copy_(_rand((10,), seed=6))
    x = _rand((7, 10), seed=7)
    expected = F.layer_norm(x, (10,), weight=ln.gamma, bias=ln.beta, eps=1e-5)
    assert torch.allclose(ln(x), expected, atol=1e-10)


def test_affine_parameters_scale_and_shift_normalized_output():
    base = _module(5)
    scaled = _module(5)
    with torch.no_grad():
        scaled.gamma.fill_(2.0)
        scaled.beta.fill_(3.0)
    x = _rand((4, 5), seed=8)
    assert torch.allclose(scaled(x), 2.0 * base(x) + 3.0, atol=1e-10)


def test_constant_row_maps_to_beta():
    ln = _module(4)
    with torch.no_grad():
        ln.beta.copy_(torch.tensor([0.5, -1.0, 2.0, 0.0], dtype=torch.float64))
    x = torch.full((3, 4), 3.0, dtype=torch.float64)
    expected = ln.beta.detach().expand(3, 4)
    assert torch.allclose(ln(x), expected, atol=1e-6)


def test_single_feature_output_equals_beta():
    ln = _module(1)
    with torch.no_grad():
        ln.gamma.fill_(5.0)
        ln.beta.fill_(-2.0)
    x = _rand((6, 1), seed=9) * 10.0
    expected = torch.full((6, 1), -2.0, dtype=torch.float64)
    assert torch.allclose(ln(x), expected, atol=1e-6)


def test_shift_invariance_along_feature_axis():
    ln = _module(9)
    x = _rand((4, 9), seed=10)
    assert torch.allclose(ln(x + 7.5), ln(x), atol=1e-8)


def test_scale_invariance_of_normalized_output():
    ln = _module(9)
    x = _rand((4, 9), seed=11)
    assert torch.allclose(ln(x * 4.0), ln(x), atol=1e-6)


def test_rows_are_normalized_independently():
    ln = _module(6)
    x = _rand((3, 6), seed=12)
    y_before = ln(x)
    x_changed = x.clone()
    x_changed[0] = x_changed[0] ** 3
    y_after = ln(x_changed)
    assert torch.allclose(y_after[1:], y_before[1:], atol=1e-12)
    assert not torch.allclose(y_after[0], y_before[0])


def test_small_epsilon_amplifies_low_variance_rows_more_than_large_epsilon():
    x = torch.tensor([[0.0, 0.0, 0.0, 1e-6]], dtype=torch.float64)
    tiny = _module(4, epsilon=1e-10)
    large = _module(4, epsilon=1.0)
    assert tiny.epsilon == 1e-10
    assert tiny(x).abs().max() > 100 * large(x).abs().max()


def test_gradcheck_passes_for_input_and_parameters():
    ln = _module(4)
    with torch.no_grad():
        ln.gamma.copy_(_rand((4,), seed=13))
        ln.beta.copy_(_rand((4,), seed=14))
    x = _rand((3, 4), seed=15).requires_grad_(True)
    assert torch.autograd.gradcheck(ln, (x,), eps=1e-6, atol=1e-5)


def test_backward_populates_parameter_gradients_with_expected_values():
    ln = _module(4)
    x = _rand((5, 4), seed=16)
    ln(x).sum().backward()
    assert tuple(ln.gamma.grad.shape) == (4,)
    assert tuple(ln.beta.grad.shape) == (4,)
    # d(sum y)/d(beta) is the number of rows for every feature.
    assert torch.allclose(ln.beta.grad, torch.full((4,), 5.0, dtype=torch.float64))
    # d(sum y)/d(gamma) is the column sum of the normalized input.
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + ln.epsilon)
    assert torch.allclose(ln.gamma.grad, normalized.sum(dim=0), atol=1e-8)


def test_input_gradient_of_feature_sum_vanishes():
    ln = _module(8)
    x = _rand((2, 8), seed=17).requires_grad_(True)
    ln(x).sum().backward()
    assert torch.allclose(x.grad, torch.zeros_like(x), atol=1e-6)


def test_extra_repr_reports_configuration():
    ln = LayerNormalization((3, 4), gamma=True, beta=False, epsilon=1e-5)
    text = ln.extra_repr()
    assert 'normal_shape=torch.Size([4])' in text
    assert 'gamma=True' in text
    assert 'beta=False' in text
    assert 'epsilon=1e-05' in text
