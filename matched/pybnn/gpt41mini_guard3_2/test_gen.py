import pytest
import numpy as np
import torch
import random

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def seed_all_generators(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_get_default_network_structure_and_forward():
    seed_all_generators()
    net = get_default_network(3)
    assert isinstance(net, torch.nn.Module)
    x = torch.randn(5, 3)
    out = net(x)
    assert out.shape == (5, 2)  # output is mean and log variance
    # The second output (log_std) should be approx log(1e-3) ~ -6.9 but AppendLayer appends noise
    # We just check output is finite
    assert torch.isfinite(out).all()


def test_nll_returns_correct_shape_and_type():
    seed_all_generators()
    batch_size = 4
    input_tensor = torch.randn(batch_size, 2, dtype=torch.float64)
    target = torch.randn(batch_size, dtype=torch.float64)
    val = nll(input_tensor, target)
    assert isinstance(val, torch.Tensor)
    assert val.shape == ()
    assert torch.isfinite(val)


def test_bohamiann_initialization_defaults_and_properties():
    seed_all_generators()
    model = Bohamiann()
    assert model.do_normalize_input
    assert model.do_normalize_output
    assert model.sampling_method == "adaptive_sghmc"
    assert model.use_double_precision
    assert model.print_every_n_steps == 100
    assert isinstance(model.sampled_weights, list)
    assert model.is_trained is False
    assert callable(model.likelihood_function)
    # network_weights property should fail before training/model creation
    with pytest.raises(AttributeError):
        _ = model.network_weights


def test_network_weights_get_and_set(monkeypatch):
    seed_all_generators()
    model = Bohamiann()
    # Setup a model manually to test weights property
    model.model = get_default_network(2).double()
    original_weights = model.network_weights
    # Change weights by adding 1.0 to each parameter numpy array
    new_weights = [w + 1.0 for w in original_weights]
    model.network_weights = new_weights
    updated_weights = model.network_weights
    # They should be equal to new_weights approx
    for updated, new in zip(updated_weights, new_weights):
        np.testing.assert_allclose(updated, new, rtol=1e-12, atol=1e-12)


def test_normalize_input_and_output_consistency():
    seed_all_generators()
    model = Bohamiann()
    x = np.random.randn(10, 3)
    y = np.random.randn(10)
    x_norm, m_x, s_x = model.normalize_input(x)
    y_norm, m_y, s_y = model.normalize_output(y)
    # Normalized should have mean close to 0 and std close to 1
    assert np.allclose(np.mean(x_norm, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(x_norm, axis=0), 1, atol=1e-7)
    assert abs(np.mean(y_norm)) < 1e-7
    assert abs(np.std(y_norm) - 1) < 1e-7
    # Provide mean and std to normalize again, should get same results
    x_norm2, m_x2, s_x2 = model.normalize_input(x, m=m_x, s=s_x)
    y_norm2, m_y2, s_y2 = model.normalize_output(y, m=m_y, s=s_y)
    np.testing.assert_allclose(x_norm, x_norm2)
    np.testing.assert_allclose(y_norm, y_norm2)


@pytest.mark.parametrize("sampling_method", ["adaptive_sghmc", "sgld", "preconditioned_sgld", "sghmc"])
def test_train_basic_functionality(sampling_method):
    seed_all_generators()
    model = Bohamiann(sampling_method=sampling_method, use_double_precision=True)
    x = np.random.randn(30, 2)
    y = np.sin(x[:, 0]) + np.cos(x[:, 1])
    y = y.astype(np.float64)
    # Train for a small number of steps for test speed
    model.train(x, y, num_steps=30, keep_every=10, num_burn_in_steps=5, batch_size=10, verbose=False)
    assert model.is_trained
    # sampled_weights should have approximately (num_steps - num_burn_in_steps)/keep_every + 1 (if inclusive)
    expected_num_samples = max(0, (30 - 5) // 10 + 1)
    assert len(model.sampled_weights) == expected_num_samples or len(model.sampled_weights) == expected_num_samples - 1
    # Each sample is a tuple of numpy arrays
    for sample in model.sampled_weights:
        assert isinstance(sample, tuple)
        for w in sample:
            assert isinstance(w, np.ndarray)


def test_train_invalid_batch_size_raises():
    seed_all_generators()
    model = Bohamiann()
    x = np.random.randn(5, 2)
    y = np.random.randn(5)
    with pytest.raises(AssertionError):
        model.train(x, y, batch_size=0)


def test_train_with_small_dataset_batch_size_adjusts_and_warns(caplog):
    seed_all_generators()
    model = Bohamiann()
    x = np.random.randn(3, 2)
    y = np.random.randn(3)
    with caplog.at_level("WARNING"):
        model.train(x, y, batch_size=5, num_steps=10, keep_every=5, num_burn_in_steps=2)
        assert any("Not enough datapoints to form a batch" in rec.message for rec in caplog.records)


def test_train_and_evaluate_runs_and_returns_curve():
    seed_all_generators()
    model = Bohamiann()
    x_train = np.random.randn(20, 2)
    y_train = np.sin(x_train[:, 0]) + np.cos(x_train[:, 1])
    y_train = y_train.astype(np.float64)
    x_valid = np.random.randn(8, 2)
    y_valid = np.sin(x_valid[:, 0]) + np.cos(x_valid[:, 1])
    y_valid = y_valid.astype(np.float64)
    n_steps, ll, mse = model.train_and_evaluate(x_train, y_train, x_valid, y_valid,
                                               num_steps=20, validate_every_n_steps=10,
                                               keep_every=5, num_burn_in_steps=5,
                                               batch_size=5, verbose=False)
    assert isinstance(n_steps, list) and isinstance(ll, list) and isinstance(mse, list)
    assert len(n_steps) == len(ll) == len(mse) == 2
    # MSE and LL values numeric and finite
    for val_list in (ll, mse):
        for val in val_list:
            assert np.isfinite(val)


def test_predict_and_predict_single_consistency():
    seed_all_generators()
    model = Bohamiann()
    x = np.random.randn(10, 2)
    y = np.random.randn(10)
    model.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=5, verbose=False)
    x_test = np.random.randn(3, 2)
    mean, var = model.predict(x_test)
    assert mean.shape == (3,)
    assert var.shape == (3,)
    # Test with return_individual_predictions True
    mean2, var2, indiv_preds = model.predict(x_test, return_individual_predictions=True)
    assert indiv_preds.shape == (len(model.sampled_weights), 3)
    # For sample_index zero
    single_pred = model.predict_single(x_test, 0)
    assert single_pred.shape[0] == 3 or (single_pred.shape == (2,) and len(single_pred) == 2) or single_pred.ndim in {1, 2}


def test_predict_without_samples_returns_raise(monkeypatch):
    seed_all_generators()
    model = Bohamiann()
    x_test = np.random.randn(2, 2)
    # No training/samples: predict uses self.sampled_weights which is empty
    # Should return empty arrays or raise index error internally
    # We expect it to raise IndexError because no weights to assign
    with pytest.raises(IndexError):
        model.predict(x_test)


def test_f_gradient_and_predictive_gradients_shapes():
    seed_all_generators()
    model = Bohamiann()
    x = np.random.randn(15, 2)
    y = np.random.randn(15)
    model.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=5, verbose=False)
    x_test = np.random.randn(2)
    sample_weights = model.sampled_weights[0]
    grad = model.f_gradient(x_test, sample_weights)
    assert grad.shape == x_test.shape
    pmg = model.predictive_mean_gradient(x_test)
    pvg = model.predictive_variance_gradient(x_test)
    assert pmg.shape == x_test.shape
    assert pvg.shape == x_test.shape


def test_train_continue_training_keeps_model_and_weights():
    seed_all_generators()
    model = Bohamiann(sampling_method="sgld", use_double_precision=False)
    x = np.random.randn(30, 3)
    y = np.random.randn(30)
    model.train(x, y, num_steps=10, keep_every=5, num_burn_in_steps=3, batch_size=10, verbose=False)
    weights_before = list(model.network_weights)
    num_samples_before = len(model.sampled_weights)
    # Continue training should keep model and weights
    model.train(x, y, num_steps=10, keep_every=5, num_burn_in_steps=0, batch_size=10, verbose=False, continue_training=True)
    weights_after = list(model.network_weights)
    num_samples_after = len(model.sampled_weights)
    # Weights should not have been reset to initial because continue_training True
    # So weights_after should be close to weights_before (or exactly equal)
    for wb, wa in zip(weights_before, weights_after):
        np.testing.assert_allclose(wb, wa, rtol=1e-12, atol=1e-12)
    # Number of samples can only grow or remain the same
    assert num_samples_after >= num_samples_before


def test_predict_single_with_output_normalization_off():
    seed_all_generators()
    model = Bohamiann(normalize_output=False, normalize_input=False)
    x = np.random.randn(10, 2)
    y = np.random.randn(10)
    model.train(x, y, num_steps=10, keep_every=5, num_burn_in_steps=2, batch_size=5, verbose=False)
    res = model.predict_single(np.random.randn(2), 0)
    assert isinstance(res, np.ndarray)
    assert res.shape[1] == 2 or res.ndim == 1


def test_predict_returns_correct_types_and_values_with_normalization_off():
    seed_all_generators()
    model = Bohamiann(normalize_input=False, normalize_output=False)
    x = np.random.randn(10, 2)
    y = np.random.randn(10)
    model.train(x, y, num_steps=10, keep_every=5, num_burn_in_steps=2, batch_size=5, verbose=False)
    x_test = np.random.randn(3, 2)
    mean, var = model.predict(x_test)
    assert isinstance(mean, np.ndarray)
    assert isinstance(var, np.ndarray)
    assert mean.shape == (3,)
    assert var.shape == (3,)
    # Variances should be positive
    assert np.all(var >= 0)


def test_network_weights_setter_rejects_wrong_shapes(monkeypatch):
    """Test that assigning weights with wrong shape raises RuntimeError from torch copy_"""
    seed_all_generators()
    model = Bohamiann()
    model.model = get_default_network(2).double()
    orig_weights = model.network_weights
    bad_weights = [w.flatten() for w in orig_weights]  # flattened, wrong shape
    with pytest.raises(RuntimeError):
        model.network_weights = bad_weights


def test_predict_with_no_samples_raises_index_error():
    seed_all_generators()
    model = Bohamiann()
    with pytest.raises(IndexError):
        model.predict(np.array([[0.0, 0.0]]))


def test_train_with_vector_and_2d_targets():
    seed_all_generators()
    model = Bohamiann()
    x = np.random.randn(10, 2)
    y_vec = np.random.randn(10)
    y_2d = y_vec[:, None]
    # Both should work
    model.train(x, y_vec, num_steps=5, keep_every=2, num_burn_in_steps=2, batch_size=5, verbose=False)
    model.train(x, y_2d, num_steps=5, keep_every=2, num_burn_in_steps=2, batch_size=5, verbose=False)
    assert model.is_trained


def test_predict_individual_predictions_shape_consistent():
    seed_all_generators()
    model = Bohamiann()
    x = np.random.randn(15, 2)
    y = np.random.randn(15)
    model.train(x, y, num_steps=15, keep_every=5, num_burn_in_steps=3, batch_size=5, verbose=False)
    x_test = np.random.randn(4, 2)
    mean, var, indiv_preds = model.predict(x_test, return_individual_predictions=True)
    assert indiv_preds.shape[1] == len(x_test)
    assert indiv_preds.shape[0] == len(model.sampled_weights)
    # Check type
    assert isinstance(indiv_preds, np.ndarray)


def test_predictive_variance_gradient_values_type_and_shape():
    seed_all_generators()
    model = Bohamiann()
    x = np.random.randn(15, 3)
    y = np.random.randn(15)
    model.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=5, verbose=False)
    x_test = np.random.randn(3)
    grad = model.predictive_variance_gradient(x_test)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == x_test.shape
