import pytest
import torch
import torch_layer_normalization as tln


def test_layer_norm_basic_forward():
    ln = tln.LayerNormalization(10)
    x = torch.randn(2, 10)
    y = ln(x)
    assert y.shape == x.shape


def test_layer_norm_forward_different_last_dim():
    last_dim = 5
    ln = tln.LayerNormalization(last_dim)
    x = torch.randn(4, 3, last_dim)
    y = ln(x)
    assert y.shape == x.shape


def test_layer_norm_forward_with_bias_and_scale():
    ln = tln.LayerNormalization(7, elementwise_affine=True)
    x = torch.randn(2, 7)
    y = ln(x)
    assert y.shape == x.shape
    # Check that bias and weight params are present
    assert ln.weight is not None
    assert ln.bias is not None


def test_layer_norm_forward_without_affine():
    ln = tln.LayerNormalization(8, elementwise_affine=False)
    x = torch.randn(3, 8)
    y = ln(x)
    assert y.shape == x.shape
    # weight and bias should not be present when affine is False
    assert not hasattr(ln, 'weight')
    assert not hasattr(ln, 'bias')


def test_layer_norm_forward_with_different_epsilon():
    ln = tln.LayerNormalization(6, eps=1e-3)
    x = torch.randn(1, 6)
    y = ln(x)
    assert y.shape == x.shape


def test_layer_norm_change_epsilon_after_init():
    ln = tln.LayerNormalization(4)
    old_eps = ln.eps
    ln.eps = 1e-4
    assert ln.eps != old_eps


def test_layer_norm_forward_device_cpu_and_cuda():
    if torch.cuda.is_available():
        ln = tln.LayerNormalization(5).cuda()
        x = torch.randn(2, 5).cuda()
        y = ln(x)
        assert y.is_cuda
        assert y.shape == x.shape
    else:
        pytest.skip("CUDA not available")


def test_layer_norm_forward_grad_enabled():
    ln = tln.LayerNormalization(3)
    x = torch.randn(2, 3, requires_grad=True)
    y = ln(x)
    y.sum().backward()
    assert x.grad is not None


def test_layer_norm_forward_grad_disabled_inference():
    ln = tln.LayerNormalization(3)
    x = torch.randn(2, 3)
    with torch.no_grad():
        y = ln(x)
    assert y.requires_grad is False


def test_layer_norm_double_forward_same_output_for_same_input():
    ln = tln.LayerNormalization(10)
    x = torch.randn(4, 10)
    y1 = ln(x)
    y2 = ln(x)
    assert torch.allclose(y1, y2)


def test_layer_norm_output_mean_and_var_close_to_expected():
    ln = tln.LayerNormalization(5)
    x = torch.randn(8, 5)
    y = ln(x)
    # Layer norm normalizes along last dimension to zero mean and unit variance.
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, unbiased=False)
    # Means near zero
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    # Vars near one
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_layer_norm_backwards_affine_enabled():
    ln = tln.LayerNormalization(7, elementwise_affine=True)
    x = torch.randn(2, 7, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_layer_norm_backwards_affine_disabled():
    ln = tln.LayerNormalization(4, elementwise_affine=False)
    x = torch.randn(3, 4, requires_grad=True)
    y = ln(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_layer_norm_can_handle_high_dimensional_input():
    ln = tln.LayerNormalization(2)
    x = torch.randn(3, 4, 5, 2)
    y = ln(x)
    assert y.shape == x.shape


def test_layer_norm_weight_and_bias_initial_values():
    ln = tln.LayerNormalization(3)
    # Weight should be ones by default
    torch.testing.assert_close(ln.weight.data, torch.ones_like(ln.weight))
    # Bias should be zeros by default
    torch.testing.assert_close(ln.bias.data, torch.zeros_like(ln.bias))


def test_layer_norm_repr_output_contains_class_name_and_params():
    ln = tln.LayerNormalization(6, eps=1e-3, elementwise_affine=False)
    r = repr(ln)
    assert "LayerNormalization" in r
    assert "eps=0.001" in r or "eps=0.001000" in r
    assert "elementwise_affine=False" in r


def test_layer_norm_invalid_input_raises():
    ln = tln.LayerNormalization(3)
    with pytest.raises(RuntimeError):
        # Input without the expected last dimension size
        ln(torch.randn(2, 2))


def test_layer_norm_input_on_different_device_than_module():
    if torch.cuda.is_available():
        ln = tln.LayerNormalization(4).cuda()
        x = torch.randn(2, 4)
        with pytest.raises(RuntimeError):
            # Passing CPU tensor to CUDA module should error
            ln(x)
    else:
        pytest.skip("CUDA not available")


def test_layer_norm_with_single_dimension_input():
    ln = tln.LayerNormalization(1)
    x = torch.randn(10, 1)
    y = ln(x)
    assert y.shape == x.shape


def test_layer_norm_with_input_requires_grad_false_then_forward():
    ln = tln.LayerNormalization(3)
    x = torch.randn(5,3)
    x.requires_grad_(False)
    y = ln(x)
    assert y.requires_grad is False


def test_layer_norm_elementwise_affine_set_false_removes_params():
    ln = tln.LayerNormalization(5, elementwise_affine=True)
    x = torch.randn(3,5)
    # Change affine param afterwards is not supported, but test attribute removal manually
    ln.elementwise_affine = False
    # Manually delete params to mimic effect
    ln._parameters.pop('weight', None)
    ln._parameters.pop('bias', None)
    # Forward works without weight and bias
    y = ln(x)
    assert y.shape == x.shape


def test_layer_norm_epsilon_effect_on_output_variance():
    ln1 = tln.LayerNormalization(4, eps=1e-5)
    ln2 = tln.LayerNormalization(4, eps=1e-1)
    x = torch.randn(20, 4)
    y1 = ln1(x)
    y2 = ln2(x)
    # Both outputs should have similar mean ~0 but var may differ slightly
    assert torch.allclose(y1.mean(dim=-1), torch.zeros_like(y1.mean(dim=-1)), atol=1e-4)
    assert torch.allclose(y2.mean(dim=-1), torch.zeros_like(y2.mean(dim=-1)), atol=1e-4)
    # Var difference due to epsilon differences
    var_diff = torch.abs(y1.var(dim=-1, unbiased=False) - y2.var(dim=-1, unbiased=False))
    assert (var_diff > 0).any()


def test_layer_norm_attention_like_shapes():
    # Simulate input tensor shape typical for attention: batch, seq_len, embed_dim.
    batch, seq_len, embed_dim = 2, 5, 16
    ln = tln.LayerNormalization(embed_dim)
    x = torch.randn(batch, seq_len, embed_dim)
    y = ln(x)
    assert y.shape == x.shape
    # Check normalized mean near zero for all batch and seq_len positions
    mean = y.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
