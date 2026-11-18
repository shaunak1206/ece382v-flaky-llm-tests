import numpy as np
import torch
import pytest
from pybnn.bohamiann import Bohamiann, nll, get_default_network
import torch.nn as nn


def test_get_default_network_structure():
    net = get_default_network(3)
    # Check if net has expected attributes
    assert hasattr(net, "fc1")
    assert hasattr(net, "fc2")
    assert hasattr(net, "fc3")
    assert hasattr(net, "log_std")

    # Check input and output sizes
    x = torch.randn(5, 3)
    y = net(x)
    assert y.shape == (5, 2)  # The AppendLayer appends noise, so output size should be 2


def test_nll_basic_computation():
    # input tensor with mean and log variance
    mean = torch.tensor([[0.0], [1.0]])
    log_var = torch.tensor([[np.log(1.0)], [np.log(0.5)]])
    input_tensor = torch.cat([mean, log_var], dim=1)
    target = torch.tensor([0.0, 1.0])

    computed_nll = nll(input_tensor, target)
    # Manual calculation
    var = torch.exp(log_var)
    expected = 0.5 * (torch.log(var) + (target.view(-1, 1) - mean) ** 2 / var)
    expected = expected.sum() / input_tensor.size(0)
    expected = expected.item()
    assert np.isclose(computed_nll.item(), expected, atol=1e-6)


def test_bohamiann_initialization_defaults():
    bnn = Bohamiann()
    assert bnn.do_normalize_input is True
    assert bnn.do_normalize_output is True
    assert bnn.sampling_method == "adaptive_sghmc"
    assert bnn.use_double_precision is True
    assert bnn.metrics == (nn.MSELoss,)
    assert bnn.likelihood_function == nll
    assert bnn.is_trained is False
    assert bnn.sampled_weights == []


def test_network_weights_property_and_setter():
    bnn = Bohamiann()
    bnn.model = get_default_network(2)().double()
    initial_weights = bnn.network_weights

    # Modify weights (set to zero)
    zeros = [np.zeros_like(w) for w in initial_weights]
    bnn.network_weights = zeros
    updated_weights = bnn.network_weights

    for w in updated_weights:
        assert np.allclose(w, 0)


def test_train_and_sampling_weights_shape():
    # Minimal synthetic data
    np.random.seed(0)
    x_train = np.random.randn(10, 2).astype(np.float64)
    y_train = np.random.randn(10).astype(np.float64)

    bnn = Bohamiann(use_double_precision=True, sampling_method="sgld")
    bnn.train(x_train, y_train, num_steps=10, keep_every=2, num_burn_in_steps=3, batch_size=5)
    assert bnn.is_trained

    # Check sampled_weights not empty and consistent length
    assert len(bnn.sampled_weights) > 0
    for weights in bnn.sampled_weights:
        for w in weights:
            assert isinstance(w, np.ndarray)


def test_train_continue_training_flag_effect():
    np.random.seed(0)
    x_train = np.random.randn(15, 3).astype(np.float64)
    y_train = np.random.randn(15).astype(np.float64)

    bnn = Bohamiann(use_double_precision=False, sampling_method="sghmc")

    bnn.train(x_train, y_train, num_steps=5, keep_every=2, num_burn_in_steps=2, batch_size=5)
    initial_sampled_weights = bnn.sampled_weights.copy()
    model_id_1 = id(bnn.model)

    bnn.train(x_train, y_train, num_steps=5, keep_every=2, num_burn_in_steps=2, batch_size=5, continue_training=True)
    model_id_2 = id(bnn.model)

    # When continue_training=True, model object (and weights list) is preserved
    assert model_id_1 == model_id_2
    # sampled_weights should accumulate
    assert len(bnn.sampled_weights) >= len(initial_sampled_weights)


def test_predict_output_shapes_and_values():
    np.random.seed(0)
    x_train = np.random.randn(20, 2).astype(np.float64)
    y_train = np.random.randn(20).astype(np.float64)
    x_test = np.random.randn(5, 2).astype(np.float64)

    bnn = Bohamiann()
    bnn.train(x_train, y_train, num_steps=50, keep_every=5, num_burn_in_steps=10, batch_size=10)

    mean, var = bnn.predict(x_test)
    assert mean.shape == (5,)
    assert var.shape == (5,)
    assert np.all(var >= 0)

    mean, var, indiv_pred = bnn.predict(x_test, return_individual_predictions=True)
    assert indiv_pred.shape[0] == len(bnn.sampled_weights)
    assert indiv_pred.shape[1] == 5


def test_predict_single_consistency():
    # Simple data
    np.random.seed(1)
    x_train = np.random.randn(30, 2).astype(np.float64)
    y_train = np.random.randn(30).astype(np.float64)
    x_test = np.random.randn(3, 2).astype(np.float64)

    bnn = Bohamiann()
    bnn.train(x_train, y_train, num_steps=40, keep_every=4, num_burn_in_steps=10, batch_size=10)
    assert len(bnn.sampled_weights) > 0

    # Get prediction from a single sampled weight
    pred_single = bnn.predict_single(x_test, sample_index=0)
    assert pred_single.shape[0] == x_test.shape[0]


def test_f_gradient_and_predictive_gradients():
    np.random.seed(2)
    x_train = np.random.randn(20, 1).astype(np.float64)
    y_train = np.random.randn(20).astype(np.float64)
    x_test = np.array([0.5])

    bnn = Bohamiann()
    bnn.train(x_train, y_train, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=5)

    assert len(bnn.sampled_weights) > 0
    # Test f_gradient returns a numpy array with the right size
    grad = bnn.f_gradient(x_test, bnn.sampled_weights[0])
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (x_test.shape[0],)

    # Test predictive mean gradient returns correct shape
    pmg = bnn.predictive_mean_gradient(x_test)
    assert isinstance(pmg, np.ndarray)
    assert pmg.shape == (x_test.shape[0],)

    # Test predictive variance gradient returns correct shape
    pvg = bnn.predictive_variance_gradient(x_test)
    assert isinstance(pvg, np.ndarray)
    assert pvg.shape == (x_test.shape[0],)


def test_normalization_methods_consistent():
    bnn = Bohamiann()
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    normed, m, s = bnn.normalize_input(x)
    recon = (normed * s) + m
    assert np.allclose(x, recon)


def test_train_batch_size_adjustment_warning(caplog):
    np.random.seed(0)
    x_train = np.random.randn(3, 2).astype(np.float64)
    y_train = np.random.randn(3).astype(np.float64)

    bnn = Bohamiann()
    with caplog.at_level("WARNING"):
        bnn.train(x_train, y_train, batch_size=20, num_steps=5)
    assert "Not enough datapoints to form a batch" in caplog.text


def test_train_invalid_batch_size_assertion():
    np.random.seed(0)
    x_train = np.random.randn(10, 2).astype(np.float64)
    y_train = np.random.randn(10).astype(np.float64)

    bnn = Bohamiann()
    with pytest.raises(AssertionError):
        bnn.train(x_train, y_train, batch_size=0)


def test_train_y_shape_assertion():
    np.random.seed(0)
    x_train = np.random.randn(10, 2).astype(np.float64)
    y_train = np.random.randn(10, 2).astype(np.float64)  # Invalid shape, second dim != 1

    bnn = Bohamiann()
    with pytest.raises(AssertionError):
        bnn.train(x_train, y_train, batch_size=5)

def test_train_verbose_print(capsys):
    np.random.seed(0)
    x_train = np.random.randn(10, 2).astype(np.float64)
    y_train = np.random.randn(10).astype(np.float64)

    bnn = Bohamiann(print_every_n_steps=2)
    bnn.train(x_train, y_train, num_steps=5, keep_every=1, num_burn_in_steps=1, batch_size=5, verbose=True)
    captured = capsys.readouterr()
    # Should print at least once (step 2 or 4 depending on print_every_n_steps)
    assert "Step" in captured.out


def test_train_and_evaluate_functionality():
    np.random.seed(0)
    x_train = np.random.randn(30, 2).astype(np.float64)
    y_train = np.random.randn(30).astype(np.float64)
    x_valid = np.random.randn(10, 2).astype(np.float64)
    y_valid = np.random.randn(10).astype(np.float64)

    bnn = Bohamiann()
    n_steps, lls, mses = bnn.train_and_evaluate(x_train, y_train, x_valid, y_valid,
                                               num_steps=100,
                                               validate_every_n_steps=50,
                                               keep_every=10,
                                               num_burn_in_steps=20,
                                               batch_size=10)
    assert isinstance(n_steps, list)
    assert isinstance(lls, list)
    assert isinstance(mses, list)
    # Should have length 2 because 100/50 = 2 validation points
    assert len(n_steps) == 2
    assert len(lls) == 2
    assert len(mses) == 2


def test_predict_with_no_samples_returns_nan():
    # Predict without any sampled weights should logically fail or return nan
    bnn = Bohamiann()
    x_test = np.array([[1.0, 2.0]])

    with pytest.raises(IndexError):
        # No weights sampled, sample list empty -> indexing fails inside predict
        bnn.predict(x_test)


def test_predict_single_invalid_sample_index():
    np.random.seed(0)
    x_train = np.random.randn(10, 2).astype(np.float64)
    y_train = np.random.randn(10).astype(np.float64)
    bnn = Bohamiann()
    bnn.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=5, batch_size=5)

    with pytest.raises(IndexError):
        bnn.predict_single(np.array([[1.0, 2.0]]), sample_index=1000)
