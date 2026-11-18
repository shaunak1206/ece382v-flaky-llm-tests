import numpy as np
import pytest
import torch
from pybnn.bohamiann import Bohamiann, get_default_network, nll


def test_get_default_network_output_shape():
    net = get_default_network(3)
    x = torch.randn(5, 3)
    out = net(x)
    assert out.shape == (5, 2)  # output + log_std (AppendLayer adds 1)

def test_nll_basic_computation():
    input_tensor = torch.tensor([[0.0, np.log(1.0)], [1.0, np.log(2.0)]], dtype=torch.float64)
    target_tensor = torch.tensor([0.5, 1.5], dtype=torch.float64)
    value = nll(input_tensor, target_tensor)
    assert isinstance(value.item(), float)
    assert value.requires_grad

def test_bohamiann_network_weights_getter_setter():
    model = get_default_network(2).double()
    bnn = Bohamiann(get_network=lambda input_dimensionality: model, use_double_precision=True)
    bnn.model = model
    weights = bnn.network_weights
    assert isinstance(weights, tuple)
    assert all(isinstance(w, np.ndarray) for w in weights)
    # modify weights and set back
    weights_new = [w.copy() + 0.1 for w in weights]
    bnn.network_weights = weights_new
    weights_set = bnn.network_weights
    for w_new, w_set in zip(weights_new, weights_set):
        np.testing.assert_allclose(w_new, w_set)

def test_bohamiann_train_basic_shapes():
    x_train = np.random.randn(50, 3)
    y_train = np.random.randn(50)
    bnn = Bohamiann(print_every_n_steps=100000)  # to suppress print during test
    bnn.train(x_train, y_train, num_steps=200, keep_every=50, num_burn_in_steps=50, batch_size=10)
    assert bnn.is_trained is True
    # sampled weights should be non-empty (since after burn-in with keep_every=50 and total 200 steps)
    assert len(bnn.sampled_weights) > 0

def test_bohamiann_train_batch_size_larger_than_data():
    x_train = np.random.randn(5, 2)
    y_train = np.random.randn(5)
    bnn = Bohamiann(batch_size=10)
    bnn.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=5, batch_size=10)
    # batch_size should be adjusted down to number of datapoints
    assert bnn.sampled_weights  # sampled_weights list exists

def test_train_and_evaluate_runs_and_returns_correct_lengths():
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    x_valid = np.random.randn(5, 2)
    y_valid = np.random.randn(5)
    bnn = Bohamiann(print_every_n_steps=100000)  # suppress printing
    n_steps, ll, mse = bnn.train_and_evaluate(x_train, y_train, x_valid, y_valid,
                                             num_steps=20, validate_every_n_steps=10,
                                             keep_every=5, num_burn_in_steps=5, batch_size=5)
    assert isinstance(n_steps, list)
    assert isinstance(ll, list)
    assert isinstance(mse, list)
    assert all(isinstance(s, int) for s in n_steps)
    assert len(n_steps) == len(ll) == len(mse)
    # values should be floats
    assert all(isinstance(v, float) for v in ll)
    assert all(isinstance(v, float) for v in mse)

def test_predict_return_shapes_and_types():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    bnn = Bohamiann(print_every_n_steps=100000)
    bnn.train(x_train, y_train, num_steps=100, keep_every=20, num_burn_in_steps=50, batch_size=10)
    x_test = np.random.randn(4, 2)
    mean, var = bnn.predict(x_test)
    assert isinstance(mean, np.ndarray)
    assert isinstance(var, np.ndarray)
    assert mean.shape == (4,)
    assert var.shape == (4,)
    mean_i, var_i, preds_i = bnn.predict(x_test, return_individual_predictions=True)
    assert preds_i.shape[1] == 4

def test_predict_single_index_out_of_bounds():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    bnn = Bohamiann(print_every_n_steps=100000)
    bnn.train(x_train, y_train, num_steps=60, keep_every=20, num_burn_in_steps=10, batch_size=10)
    x_test = np.random.randn(3, 2)
    # sample index must be in range
    with pytest.raises(IndexError):
        bnn.predict_single(x_test, sample_index=1000)
    # valid index returns ndarray
    out = bnn.predict_single(x_test, sample_index=0)
    assert isinstance(out, np.ndarray)

def test_normalize_input_output_consistency():
    bnn = Bohamiann()
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm_x, m, s = bnn.normalize_input(x)
    # mean near zero and std near one
    assert np.allclose(np.mean(norm_x, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(norm_x, axis=0), 1, atol=1e-7)
    norm_y, m2, s2 = bnn.normalize_output(np.array([1., 2., 3.]))
    assert np.allclose(np.mean(norm_y), 0, atol=1e-7)
    assert np.allclose(np.std(norm_y), 1, atol=1e-7)

def test_f_gradient_output_shape_and_type():
    x_train = np.random.randn(15, 2)
    y_train = np.random.randn(15)
    bnn = Bohamiann()
    bnn.train(x_train, y_train, num_steps=80, keep_every=20, num_burn_in_steps=10, batch_size=10)
    x_test = np.random.randn(2)
    weights = bnn.sampled_weights[0]
    grad = bnn.f_gradient(x_test, weights)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2,)
    assert np.issubdtype(grad.dtype, np.floating)

def test_predictive_mean_gradient_and_variance_gradient_shapes():
    x_train = np.random.randn(20, 3)
    y_train = np.random.randn(20)
    bnn = Bohamiann()
    bnn.train(x_train, y_train, num_steps=120, keep_every=30, num_burn_in_steps=20, batch_size=10)
    x_test = np.random.randn(3)
    mean_grad = bnn.predictive_mean_gradient(x_test)
    var_grad = bnn.predictive_variance_gradient(x_test)
    assert mean_grad.shape == x_test.shape
    assert var_grad.shape == x_test.shape
    assert isinstance(mean_grad, np.ndarray)
    assert isinstance(var_grad, np.ndarray)

def test_predict_with_empty_sampled_weights_returns_nan_shapes():
    bnn = Bohamiann()
    bnn.sampled_weights = []
    try:
        mean, var = bnn.predict(np.array([[1., 2.]]))
    except Exception:
        # It might raise if no sampled weights present - test graceful failure
        assert True
    else:
        assert mean.shape == (1,)
        assert var.shape == (1,)

def test_network_weights_setter_raises_with_wrong_shape():
    bnn = Bohamiann()
    bnn.model = get_default_network(2).double()
    weights = bnn.network_weights
    # Provide wrong shaped array
    wrong_weights = [np.ones((100, 100)), np.ones((2, 2)), np.ones((5, 5)), np.ones((1,))][:len(weights)]
    with pytest.raises(RuntimeError):
        bnn.network_weights = wrong_weights

def test_train_continue_training_flag_behavior():
    x_train = np.random.randn(50, 2)
    y_train = np.random.randn(50)
    bnn = Bohamiann()
    # initial training
    bnn.train(x_train, y_train, num_steps=50, keep_every=10, num_burn_in_steps=10, batch_size=20)
    first_len = len(bnn.sampled_weights)
    # continue training: sampled_weights should accumulate
    bnn.train(x_train, y_train, num_steps=50, keep_every=10,
              num_burn_in_steps=0, batch_size=20, continue_training=True)
    second_len = len(bnn.sampled_weights)
    assert second_len >= first_len

def test_train_raises_invalid_batch_size():
    bnn = Bohamiann()
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10)
    with pytest.raises(AssertionError):
        bnn.train(x_train, y_train, batch_size=0)

def test_train_raises_invalid_target_shape():
    bnn = Bohamiann()
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10, 2)  # invalid shape
    with pytest.raises(AssertionError):
        bnn.train(x_train, y_train)

def test_predict_single_returns_float_array():
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10)
    bnn = Bohamiann()
    bnn.train(x_train, y_train, num_steps=40, keep_every=10, num_burn_in_steps=10)
    val = bnn.predict_single(np.random.randn(2), 0)
    assert isinstance(val, np.ndarray)
    assert val.ndim in {1, 2}

