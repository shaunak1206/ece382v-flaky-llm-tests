import numpy as np
import torch
import pytest

from pybnn.bohamiann import Bohamiann, nll, get_default_network


def seed_all():
    torch.manual_seed(0)
    np.random.seed(0)


def create_dummy_data(n=100, d=3):
    np.random.seed(0)
    x = np.random.randn(n, d)
    y = np.random.randn(n)
    return x, y


def test_get_default_network_output_shape_and_type():
    seed_all()
    model = get_default_network(5)
    inp = torch.randn(2, 5)
    out = model(inp)
    assert out.shape == (2, 2)  # output is mean and log variance per sample
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32 or out.dtype == torch.float64


def test_nll_returns_correct_shape_and_scalar_value():
    seed_all()
    input_ = torch.randn(10, 2)
    target = torch.randn(10)
    loss = nll(input_, target)
    assert loss.shape == torch.Size([])
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss).all()
    # The loss should be a float scalar tensor


def test_bohamiann_init_defaults_and_attributes():
    seed_all()
    model = Bohamiann()
    assert model.do_normalize_input
    assert model.do_normalize_output
    assert model.sampling_method == "adaptive_sghmc"
    assert model.is_trained is False
    assert model.likelihood_function == nll
    assert model.sampled_weights == []
    assert callable(model.get_network)
    assert model.metrics == (torch.nn.MSELoss,)
    assert isinstance(model.print_every_n_steps, int)


def test_network_weights_properties_and_setter():
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(20, 3)
    model.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=0, verbose=False)
    weights_before = model.network_weights
    # Modify weights manually with numpy arrays of correct shapes
    new_weights = [w + 0.1 for w in weights_before]
    model.network_weights = new_weights
    new_weights_after = model.network_weights
    for nw, expected in zip(new_weights_after, new_weights):
        # Use almost equal comparison
        assert np.allclose(nw, expected, rtol=1e-6, atol=1e-8)


def test_train_runs_and_updates_is_trained(monkeypatch):
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(30, 2)
    # patch print_every_n_steps to avoid printing to console
    monkeypatch.setattr(model, "print_every_n_steps", 1000)
    model.train(x_train, y_train, num_steps=20, num_burn_in_steps=5, keep_every=5, verbose=False)
    assert model.is_trained
    assert len(model.sampled_weights) > 0
    # Sampled weights should be tuples of numpy arrays
    for w in model.sampled_weights:
        assert isinstance(w, tuple)
        assert all(isinstance(arr, np.ndarray) for arr in w)


def test_train_with_batch_size_larger_than_data(monkeypatch):
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(5, 2)
    monkeypatch.setattr(model, "print_every_n_steps", 1000)
    # batch_size > number of datapoints triggers warning and adjusts batch_size automatically
    model.train(x_train, y_train, batch_size=10, num_steps=15, num_burn_in_steps=5, keep_every=5, verbose=False)


def test_train_continue_training_respects_continue_training_flag(monkeypatch):
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(20, 2)
    monkeypatch.setattr(model, "print_every_n_steps", 1000)
    model.train(x_train, y_train, num_steps=10, num_burn_in_steps=0, keep_every=5, verbose=False)
    first_samples = list(model.sampled_weights)
    # Train again with continue_training = True
    model.train(x_train, y_train, num_steps=10, num_burn_in_steps=0, keep_every=5, verbose=False, continue_training=True)
    # sampled_weights should be added to, not cleared
    assert len(model.sampled_weights) >= len(first_samples)


def test_predict_and_predict_single_consistency():
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(30, 4)
    model.train(x_train, y_train, num_steps=20, num_burn_in_steps=0, keep_every=10, verbose=False)
    x_test = np.random.randn(2, 4)
    mu, var = model.predict(x_test)
    assert mu.shape == (2,)
    assert var.shape == (2,)
    assert np.all(var >= 0)
    # predict_single for first sampled weight returns shape (2,2) predicted mean+logvar per point
    val = model.predict_single(x_test, sample_index=0)
    assert val.shape == (2, 2)
    # Values are finite floats
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(var))
    assert np.all(np.isfinite(val))


def test_predict_individual_predictions_return_individual_predictions_shapes():
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(30, 3)
    model.train(x_train, y_train, num_steps=20, num_burn_in_steps=0, keep_every=10, verbose=False)
    x_test = np.random.randn(4, 3)
    mu, var, indiv_preds = model.predict(x_test, return_individual_predictions=True)
    # mu and var shape: (number of points,)
    # indiv_preds shape: (num_samples, number of test points)
    assert mu.shape == (4,)
    assert var.shape == (4,)
    assert indiv_preds.shape[1] == 4
    assert indiv_preds.shape[0] == len(model.sampled_weights)


def test_f_gradient_output_shape_and_values():
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(30, 2)
    model.train(x_train, y_train, num_steps=20, num_burn_in_steps=0, keep_every=10, verbose=False)
    x_test = np.random.randn(2)
    weights = model.sampled_weights[0]
    grad = model.f_gradient(x_test, weights)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2,)
    # gradients can be positive or negative; ensure finite
    assert np.all(np.isfinite(grad))


def test_predictive_mean_gradient_output_shape_and_values():
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(50, 3)
    model.train(x_train, y_train, num_steps=20, num_burn_in_steps=0, keep_every=5, verbose=False)
    x_test = np.random.randn(3)
    grad = model.predictive_mean_gradient(x_test)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (3,)
    assert np.all(np.isfinite(grad))


def test_predictive_variance_gradient_output_shape_and_values():
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(60, 3)
    model.train(x_train, y_train, num_steps=20, num_burn_in_steps=0, keep_every=5, verbose=False)
    x_test = np.random.randn(3)
    grad = model.predictive_variance_gradient(x_test)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (3,)
    assert np.all(np.isfinite(grad))


def test_train_and_evaluate_runs_without_errors(monkeypatch):
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(40, 2)
    x_valid, y_valid = create_dummy_data(20, 2)
    monkeypatch.setattr(model, "print_every_n_steps", 1000)
    n_steps, ll_curve, mse_curve = model.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=20,
        validate_every_n_steps=10,
        keep_every=5,
        num_burn_in_steps=5,
        verbose=False
    )
    assert isinstance(n_steps, list)
    assert isinstance(ll_curve, list)
    assert isinstance(mse_curve, list)
    assert all(isinstance(x, (int, np.integer)) for x in n_steps)
    assert all(isinstance(x, float) for x in ll_curve)
    assert all(isinstance(x, float) for x in mse_curve)
    # The length of curves should match expected number of validations
    assert len(n_steps) == len(ll_curve) == len(mse_curve)
    # Values should be finite
    assert all(np.isfinite(ll_curve))
    assert all(np.isfinite(mse_curve))


def test_normalize_input_and_output_consistency():
    seed_all()
    model = Bohamiann()
    x = np.random.randn(100, 2)
    # Normalize input without provided mean/std
    norm_x, mean, std = model.normalize_input(x)
    assert norm_x.shape == x.shape
    assert np.allclose(np.mean(norm_x, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(norm_x, axis=0), 1, atol=1e-8)
    # Normalize output similarly
    y = np.random.randn(100)
    norm_y, mean_y, std_y = model.normalize_output(y)
    assert norm_y.shape == y.shape
    assert np.allclose(np.mean(norm_y), 0, atol=1e-10)
    assert np.allclose(np.std(norm_y), 1, atol=1e-8)
    # Normalize input with given mean and std returns consistent results
    norm_x2, mean2, std2 = model.normalize_input(x, m=mean, s=std)
    assert np.allclose(norm_x, norm_x2)
    # Normalize output with given mean and std returns consistent results
    norm_y2, mean_y2, std_y2 = model.normalize_output(y, m=mean_y, s=std_y)
    assert np.allclose(norm_y, norm_y2)


def test_train_invalid_batch_size_raises():
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(10, 2)
    with pytest.raises(AssertionError):
        model.train(x_train, y_train, batch_size=0, num_steps=1)


def test_train_invalid_target_shape_raises():
    seed_all()
    model = Bohamiann()
    x_train, _ = create_dummy_data(10, 2)
    y_bad = np.random.randn(10, 2)  # invalid shape (N,2)
    with pytest.raises(AssertionError):
        model.train(x_train, y_bad, num_steps=1)


def test_predict_with_no_sampled_weights_raises(monkeypatch):
    seed_all()
    model = Bohamiann()
    with pytest.raises(IndexError):
        model.predict(np.array([[0.0]]))
    # add a sample so predict no longer raises
    x_train, y_train = create_dummy_data(20, 1)
    monkeypatch.setattr(model, "print_every_n_steps", 1000)
    model.train(x_train, y_train, num_steps=10, num_burn_in_steps=0, keep_every=5, verbose=False)
    mu, var = model.predict(np.array([[0.0]]))
    assert mu.shape == (1,)
    assert var.shape == (1,)


def test_predict_single_with_invalid_sample_index_raises():
    seed_all()
    model = Bohamiann()
    x_train, y_train = create_dummy_data(20, 2)
    model.train(x_train, y_train, num_steps=10, num_burn_in_steps=0, keep_every=10, verbose=False)
    invalid_index = len(model.sampled_weights) + 1
    with pytest.raises(IndexError):
        model.predict_single(np.array([0.0, 0.0]), invalid_index)
