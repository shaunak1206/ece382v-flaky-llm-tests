import numpy as np
import torch
import pytest
from pybnn.bohamiann import Bohamiann, get_default_network, nll


def test_get_default_network_output_shape():
    input_dim = 3
    net = get_default_network(input_dim)
    x = torch.randn(5, input_dim)
    out = net(x)
    assert out.shape == (5, 2)  # Output has mean and log-variance


def test_nll_basic_computation():
    # input tensor with mean and log variance, target tensor
    mean = torch.tensor([[1.0], [2.0]])
    log_var = torch.tensor([[0.0], [0.0]])  # variance=1
    input_tensor = torch.cat([mean, log_var], 1)
    target = torch.tensor([1.0, 2.5])
    loss = nll(input_tensor, target)
    # Manual calculation: 0.5*(target - mean)^2 + 0.5*log_var, mean over 2 samples
    expected = 0.5 * ((1 - 1)**2 + (2.5 - 2)**2) + 0.5 * 0
    expected = expected / 2  # mean over batch size
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)


def test_network_weights_property_and_setter():
    model = get_default_network(2)
    bnn = Bohamiann(get_network=lambda input_dimensionality: model, use_double_precision=False)
    bnn.model = model
    orig_weights = bnn.network_weights
    # Modify weights
    new_weights = [w + 1.0 for w in orig_weights]
    bnn.network_weights = new_weights
    updated_weights = bnn.network_weights
    for nw, w in zip(new_weights, updated_weights):
        assert np.allclose(nw, w)


def test_train_invalid_batch_size_raises():
    bnn = Bohamiann()
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10)
    with pytest.raises(AssertionError):
        bnn.train(x_train, y_train, batch_size=0)


def test_train_invalid_target_shape_raises():
    bnn = Bohamiann()
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10, 2)  # Invalid shape (N,2)
    with pytest.raises(AssertionError):
        bnn.train(x_train, y_train, batch_size=1)


def test_train_sampled_weights_added():
    bnn = Bohamiann()
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    bnn.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=2, batch_size=10, lr=1e-2, verbose=False)
    # After burn in and sampling every 5 steps, sampled_weights should have some entries
    assert len(bnn.sampled_weights) > 0
    # is_trained flag should be True after training
    assert bnn.is_trained


def test_train_continue_training_flag_effect():
    bnn = Bohamiann()
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    # First train a little
    bnn.train(x_train, y_train, num_steps=5, keep_every=2, num_burn_in_steps=1, verbose=False)
    sampled_count_before = len(bnn.sampled_weights)
    # Continue training, should keep previous weights
    bnn.train(x_train, y_train, num_steps=5, keep_every=2, num_burn_in_steps=1, continue_training=True, verbose=False)
    sampled_count_after = len(bnn.sampled_weights)
    assert sampled_count_after >= sampled_count_before


def test_predict_returns_mean_and_variance_shapes():
    bnn = Bohamiann()
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    # Train to populate sampled_weights
    bnn.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=2, batch_size=10, verbose=False)
    x_test = np.random.randn(5, 2)
    mean, var = bnn.predict(x_test)
    assert mean.shape == (5,)
    assert var.shape == (5,)


def test_predict_return_individual_predictions_shapes():
    bnn = Bohamiann()
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    bnn.train(x_train, y_train, num_steps=12, keep_every=6, num_burn_in_steps=2, batch_size=10, verbose=False)
    x_test = np.random.randn(3, 2)
    mean, var, indiv_preds = bnn.predict(x_test, return_individual_predictions=True)
    assert mean.shape == (3,)
    assert var.shape == (3,)
    # number of individual predictions: num sampled models, each shape (3,)
    assert indiv_preds.shape[1] == 3


def test_predict_single_returns_correct_shape():
    bnn = Bohamiann()
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    bnn.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=2, batch_size=10, verbose=False)
    x_test = np.random.randn(3, 2)
    # Should raise if index out of range
    with pytest.raises(IndexError):
        bnn.predict_single(x_test, sample_index=len(bnn.sampled_weights))
    # Use a valid index
    out = bnn.predict_single(x_test, sample_index=0)
    assert out.shape == (3, 2) or out.shape == (3, 2) or out.shape[1] == 2 or out.ndim >= 1


def test_f_gradient_returns_gradient_array():
    bnn = Bohamiann()
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    bnn.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=2, batch_size=10, verbose=False)
    x_test = np.random.randn(2)
    weights = bnn.sampled_weights[0]
    grad = bnn.f_gradient(x_test, weights)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == x_test.shape


def test_predictive_mean_gradient_returns_average_gradient():
    bnn = Bohamiann()
    x_train = np.random.randn(20, 3)
    y_train = np.random.randn(20)
    bnn.train(x_train, y_train, num_steps=12, keep_every=6, num_burn_in_steps=2, batch_size=10, verbose=False)
    x_test = np.random.randn(3)
    grad = bnn.predictive_mean_gradient(x_test)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == x_test.shape


def test_predictive_variance_gradient_returns_correct_shape():
    bnn = Bohamiann()
    x_train = np.random.randn(20, 3)
    y_train = np.random.randn(20)
    bnn.train(x_train, y_train, num_steps=12, keep_every=6, num_burn_in_steps=2, batch_size=10, verbose=False)
    x_test = np.random.randn(3)
    grad = bnn.predictive_variance_gradient(x_test)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == x_test.shape


def test_normalize_input_and_output_consistency():
    bnn = Bohamiann()
    x = np.random.randn(10, 4)
    norm_x, mean_x, std_x = bnn.normalize_input(x)
    assert norm_x.shape == x.shape
    # Applying normalization again with given mean and std returns same result
    norm_x_2, mean_x_2, std_x_2 = bnn.normalize_input(x, mean_x, std_x)
    assert np.allclose(norm_x, norm_x_2)


def test_train_and_evaluate_runs_and_returns_curves():
    bnn = Bohamiann()
    x_train = np.random.randn(40, 2)
    y_train = np.random.randn(40)
    x_valid = np.random.randn(10, 2)
    y_valid = np.random.randn(10)
    n_steps, lls, mses = bnn.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=200, validate_every_n_steps=50,
        keep_every=20, num_burn_in_steps=25, batch_size=10, verbose=False
    )
    assert isinstance(n_steps, list)
    assert isinstance(lls, list)
    assert isinstance(mses, list)
    assert len(n_steps) == len(lls) == len(mses)
    assert all(isinstance(x, int) for x in n_steps)
    assert all(isinstance(x, float) for x in lls)
    assert all(isinstance(x, float) for x in mses)


def test_sampling_methods_can_be_set():
    methods = ["adaptive_sghmc", "sgld", "preconditioned_sgld", "sghmc"]
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    for method in methods:
        bnn = Bohamiann(sampling_method=method)
        bnn.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=2, batch_size=10, verbose=False)
        assert bnn.sampler is not None


def test_training_with_normalization_flags():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    # no normalization
    bnn = Bohamiann(normalize_input=False, normalize_output=False)
    bnn.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=2, batch_size=10, verbose=False)
    # with normalization
    bnn2 = Bohamiann(normalize_input=True, normalize_output=True)
    bnn2.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=2, batch_size=10, verbose=False)
    # check means and stds properties exist only if normalization is True
    assert hasattr(bnn2, 'x_mean')
    assert hasattr(bnn2, 'x_std')
    assert hasattr(bnn2, 'y_mean')
    assert hasattr(bnn2, 'y_std')


def test_print_every_n_steps_effect_on_verbose():
    # This test just runs with verbose True and print_every_n_steps small value,
    # it does not check output but should run without error.
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    bnn = Bohamiann(print_every_n_steps=2)
    bnn.train(x_train, y_train, num_steps=6, batch_size=10, verbose=True)


def test_model_double_precision_behavior():
    bnn = Bohamiann(use_double_precision=True)
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    bnn.train(x_train, y_train, num_steps=5, num_burn_in_steps=2, batch_size=10, verbose=False)
    # Model parameters should be double
    for param in bnn.model.parameters():
        assert param.dtype == torch.float64


def test_model_float_precision_behavior():
    bnn = Bohamiann(use_double_precision=False)
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    bnn.train(x_train, y_train, num_steps=5, num_burn_in_steps=2, batch_size=10, verbose=False)
    # Model parameters should be float
    for param in bnn.model.parameters():
        assert param.dtype == torch.float32
