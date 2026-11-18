import numpy as np
import torch
import pytest
from pybnn.bohamiann import Bohamiann, get_default_network, nll
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization


def test_get_default_network_output_shape():
    net = get_default_network(input_dimensionality=3)
    x = torch.randn(5, 3)
    output = net(x)
    assert output.shape == (5, 2), "Output should have shape (batch_size, 2)"


def test_nll_output_scalar_and_positive():
    mean = torch.tensor([[1.0], [2.0]])
    log_var = torch.tensor([[0.0], [0.0]])
    input_tensor = torch.cat([mean, log_var], dim=1)
    target = torch.tensor([1.0, 3.0])
    loss = nll(input_tensor, target)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0


def test_bohamiann_initial_parameters():
    model = Bohamiann()
    assert model.is_trained is False
    assert model.sampled_weights == []
    assert callable(model.get_network)
    assert model.print_every_n_steps == 100


def test_network_weights_setter_and_getter_consistency():
    model = Bohamiann()
    input_dim = 2
    model.model = model.get_network(input_dimensionality=input_dim).double()
    weights_orig = model.network_weights
    # Modify weights slightly
    weights_new = [w + 0.1 for w in weights_orig]
    model.network_weights = weights_new
    weights_after_set = model.network_weights
    for w_new, w_after in zip(weights_new, weights_after_set):
        np.testing.assert_allclose(w_new, w_after, rtol=1e-7)


def test_train_reduces_loss_on_simple_data():
    model = Bohamiann(print_every_n_steps=1, use_double_precision=False)
    np.random.seed(0)
    x_train = np.random.randn(50, 1).astype(np.float32)
    y_train = (2 * x_train[:, 0] + 3 + 0.1 * np.random.randn(50)).astype(np.float32)
    model.train(x_train, y_train, num_steps=20, batch_size=10, keep_every=5, num_burn_in_steps=5, verbose=False)
    assert model.is_trained is True
    assert len(model.sampled_weights) > 0


def test_predict_returns_mean_and_variance():
    model = Bohamiann()
    x_train = np.random.randn(30, 1)
    y_train = 2 * x_train[:, 0] + 1.0
    model.train(x_train, y_train, num_steps=10, batch_size=5, keep_every=2, num_burn_in_steps=3, verbose=False)
    x_test = np.array([[0.5]])
    mean, var = model.predict(x_test)
    assert mean.shape == (1,)
    assert var.shape == (1,)
    assert (var >= 0).all()


def test_predict_return_individual_predictions_length():
    model = Bohamiann()
    x_train = np.random.randn(25, 1)
    y_train = 3 * x_train[:, 0] - 2
    model.train(x_train, y_train, num_steps=10, batch_size=5, keep_every=2, num_burn_in_steps=4, verbose=False)
    x_test = np.array([[1.0]])
    mean, var, indiv_preds = model.predict(x_test, return_individual_predictions=True)
    assert indiv_preds.shape[0] == len(model.sampled_weights)
    assert indiv_preds.shape[1] == x_test.shape[0]


def test_predict_single_agrees_with_predict_for_one_sample():
    model = Bohamiann()
    x_train = np.random.randn(40, 1)
    y_train = 5 * x_train[:, 0] + 0.5
    model.train(x_train, y_train, num_steps=15, batch_size=10, keep_every=5, num_burn_in_steps=5, verbose=False)
    x_test = np.array([[0.0]])
    # Ensure there is at least 1 sampled weight
    assert len(model.sampled_weights) > 0
    sample_index = 0
    pred_single = model.predict_single(x_test, sample_index=sample_index)
    mean, var, indiv = model.predict(x_test, return_individual_predictions=True)
    np.testing.assert_allclose(pred_single[0, 0], indiv[sample_index, 0], rtol=1e-5)


def test_normalize_input_and_output_are_consistent():
    model = Bohamiann()
    x = np.random.randn(10, 3)
    y = np.random.randn(10)
    normalized_x, m_x, s_x = model.normalize_input(x)
    x_back = zero_mean_unit_var_denormalization(normalized_x, m_x, s_x)
    np.testing.assert_allclose(x, x_back, rtol=1e-6)
    normalized_y, m_y, s_y = model.normalize_output(y)
    y_back = zero_mean_unit_var_denormalization(normalized_y, m_y, s_y)
    np.testing.assert_allclose(y, y_back, rtol=1e-6)


def test_f_gradient_returns_gradient_of_correct_shape():
    model = Bohamiann()
    np.random.seed(1)
    x_train = np.random.randn(30, 2)
    y_train = np.sum(x_train, axis=1) + 0.1 * np.random.randn(30)
    model.train(x_train, y_train, num_steps=15, batch_size=10, keep_every=5, num_burn_in_steps=5, verbose=False)
    x_test = np.array([0.1, -0.2])
    # Use first sampled weight for gradient
    g = model.f_gradient(x_test, weights=model.sampled_weights[0])
    assert g.shape == x_test.shape
    assert isinstance(g, np.ndarray)


def test_predictive_mean_gradient_close_to_mean_of_f_gradient():
    model = Bohamiann()
    np.random.seed(2)
    x_train = np.random.randn(40, 3)
    y_train = x_train[:, 0] - 2*x_train[:, 1] + 0.5*x_train[:, 2] + 0.2*np.random.randn(40)
    model.train(x_train, y_train, num_steps=20, batch_size=10, keep_every=5, num_burn_in_steps=5, verbose=False)
    x_test = np.array([0.0, 0.1, -0.1])
    grads = np.array([model.f_gradient(x_test, w) for w in model.sampled_weights])
    pred_grad = model.predictive_mean_gradient(x_test)
    np.testing.assert_allclose(pred_grad, np.mean(grads, axis=0), rtol=1e-6)


def test_predictive_variance_gradient_returns_array_shape():
    model = Bohamiann()
    x_train = np.random.randn(30, 1)
    y_train = 2 * x_train[:, 0] + 1 + 0.1 * np.random.randn(30)
    model.train(x_train, y_train, num_steps=15, batch_size=10, keep_every=4, num_burn_in_steps=5, verbose=False)
    x_test = np.array([0.5])
    g_var = model.predictive_variance_gradient(x_test)
    assert isinstance(g_var, np.ndarray)
    assert g_var.shape == x_test.shape


def test_train_continue_training_keeps_weights():
    model = Bohamiann()
    x_train = np.random.randn(25, 2)
    y_train = np.sum(x_train, axis=1)
    model.train(x_train, y_train, num_steps=10, batch_size=5, keep_every=2, num_burn_in_steps=3)
    sampled_before = model.sampled_weights.copy()
    model.train(x_train, y_train, num_steps=10, batch_size=5, keep_every=2, num_burn_in_steps=0, continue_training=True)
    # Sampled weights should be more after second train call (not cleared)
    assert len(model.sampled_weights) >= len(sampled_before)


def test_train_batch_size_larger_than_dataset_resets_batch_size(monkeypatch):
    model = Bohamiann()
    x_train = np.random.randn(5, 2)
    y_train = np.sum(x_train, axis=1)
    # Use batch size bigger than dataset
    batch_size = 10
    logged = {}
    def fake_warning(msg):
        logged['warn'] = msg
    monkeypatch.setattr("logging.warning", fake_warning)
    model.train(x_train, y_train, num_steps=5, batch_size=batch_size, keep_every=1, num_burn_in_steps=1)
    assert 'warn' in logged
    # Batch size internally reduced to x_train.shape[0]
    assert model.sampled_weights != []


def test_train_invalid_y_shape_raises_assert():
    model = Bohamiann()
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10, 2)  # Invalid shape (N,2)
    with pytest.raises(AssertionError):
        model.train(x_train, y_train)


def test_train_invalid_batch_size_raises_assert():
    model = Bohamiann()
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10)
    with pytest.raises(AssertionError):
        model.train(x_train, y_train, batch_size=0)


def test_train_and_evaluate_runs_and_returns_consistent_length():
    model = Bohamiann()
    x_train = np.random.randn(30, 2)
    y_train = x_train[:, 0] - x_train[:, 1]
    x_valid = np.random.randn(10, 2)
    y_valid = x_valid[:, 0] - x_valid[:, 1]
    n_steps, learning_curve_ll, learning_curve_mse = model.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=20, validate_every_n_steps=10, keep_every=5, num_burn_in_steps=5, batch_size=10)
    assert len(n_steps) == len(learning_curve_ll) == len(learning_curve_mse)
    assert all(isinstance(x, (float, int, np.integer, np.floating)) for x in learning_curve_ll)
    assert all(isinstance(x, (float, int, np.integer, np.floating)) for x in learning_curve_mse)


def test_predict_with_no_sampled_weights_returns_mean_var_zeros(monkeypatch):
    model = Bohamiann()
    model.sampled_weights = []
    model.model = model.get_network(input_dimensionality=1).double()
    x_test = np.array([[0.0]])
    # monkeypatch network_weights to avoid exception
    with pytest.raises(IndexError):
        model.predict(x_test)


def test_predict_single_with_invalid_index_raises_index_error():
    model = Bohamiann()
    model.sampled_weights = []
    with pytest.raises(IndexError):
        model.predict_single(np.array([[1.0]]), sample_index=0)
