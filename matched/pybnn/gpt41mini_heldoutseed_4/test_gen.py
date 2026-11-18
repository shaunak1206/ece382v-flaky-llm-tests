import numpy as np
import pytest
import torch
from pybnn.bohamiann import Bohamiann, get_default_network, nll
from scipy.stats import norm


def test_get_default_network_output_shape():
    net = get_default_network(3)
    x = torch.randn(10, 3)
    y = net(x)
    assert y.shape == (10, 2)


def test_nll_basic_values():
    input_tensor = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    target_tensor = torch.tensor([1.0, 0.0])
    val = nll(input_tensor, target_tensor)
    assert isinstance(val, torch.Tensor)
    assert val.shape == ()


def test_nll_positive_output_for_nonzero_error():
    input_tensor = torch.tensor([[0.0, 0.0]])
    target_tensor = torch.tensor([1.0])
    val = nll(input_tensor, target_tensor)
    assert val.item() > 0


def test_bohamiann_network_weights_set_and_get():
    model = Bohamiann(get_network=get_default_network)
    model.model = model.get_network(2)
    initial_weights = model.network_weights
    new_weights = [w + 1 for w in initial_weights]
    model.network_weights = new_weights
    updated_weights = model.network_weights
    for nw, uw in zip(new_weights, updated_weights):
        assert np.allclose(nw, uw)


def test_bohamiann_train_runs_and_sets_trained_flag():
    np.random.seed(42)
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann(get_network=get_default_network, use_double_precision=False, sampling_method="sgld")
    model.train(x_train, y_train, num_steps=10, keep_every=2, num_burn_in_steps=5, batch_size=10)
    assert model.is_trained
    assert len(model.sampled_weights) > 0


def test_train_with_batch_size_larger_than_data_warns(caplog):
    x_train = np.random.randn(5, 2)
    y_train = np.random.randn(5)
    model = Bohamiann(get_network=get_default_network)
    caplog.set_level("WARNING")
    model.train(x_train, y_train, num_steps=5, batch_size=10)
    assert any("Not enough datapoints to form a batch" in rec.message for rec in caplog.records)


def test_train_and_evaluate_returns_correct_lengths():
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    x_valid = np.random.randn(10, 2)
    y_valid = np.random.randn(10)
    model = Bohamiann(get_network=get_default_network, use_double_precision=False, sampling_method="sghmc")
    n_steps, lls, mses = model.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=20, validate_every_n_steps=10, batch_size=10, verbose=False)
    assert len(n_steps) == 2
    assert len(lls) == 2
    assert len(mses) == 2


def test_normalize_input_and_output_are_inverses():
    x = np.random.randn(10, 2)
    model = Bohamiann()
    norm_x, mean_x, std_x = model.normalize_input(x)
    assert np.allclose(np.mean(norm_x, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(norm_x, axis=0), 1, atol=1e-7)

    y = np.random.randn(10)
    norm_y, mean_y, std_y = model.normalize_output(y)
    assert np.allclose(np.mean(norm_y), 0, atol=1e-7)
    assert np.allclose(np.std(norm_y), 1, atol=1e-7)


def test_predict_returns_mean_and_variance():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann(get_network=get_default_network, use_double_precision=False, sampling_method="sgld")
    model.train(x_train, y_train, num_steps=10, keep_every=2, num_burn_in_steps=5, batch_size=10)

    x_test = np.random.randn(5, 2)
    m, v = model.predict(x_test)
    assert m.shape == (5,)
    assert v.shape == (5,)
    assert np.all(v >= 0)


def test_predict_return_individual_predictions_shape():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann(get_network=get_default_network, use_double_precision=False, sampling_method="sgld")
    model.train(x_train, y_train, num_steps=10, keep_every=2, num_burn_in_steps=5, batch_size=10)

    x_test = np.random.randn(5, 2)
    m, v, preds = model.predict(x_test, return_individual_predictions=True)
    assert preds.shape == (len(model.sampled_weights), 5)
    assert m.shape == (5,)
    assert v.shape == (5,)


def test_predict_single_returns_proper_shape():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann(get_network=get_default_network, use_double_precision=False, sampling_method="sgld")
    model.train(x_train, y_train, num_steps=10, keep_every=2, num_burn_in_steps=5, batch_size=10)

    x_test = np.random.randn(5, 2)
    res = model.predict_single(x_test, sample_index=0)
    assert res.shape == (5, 2) or res.shape == (2,)  # single batch or single point


def test_f_gradient_returns_gradient_of_correct_shape():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann(get_network=get_default_network, use_double_precision=True, sampling_method="adaptive_sghmc")
    model.train(x_train, y_train, num_steps=10, keep_every=2, num_burn_in_steps=5, batch_size=10)

    x = np.random.randn(2)
    grad = model.f_gradient(x, weights=model.sampled_weights[0])
    assert grad.shape == (2,)
    assert np.all(np.isfinite(grad))


def test_predictive_mean_gradient_returns_correct_shape():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann(get_network=get_default_network, use_double_precision=True, sampling_method="adaptive_sghmc")
    model.train(x_train, y_train, num_steps=10, keep_every=2, num_burn_in_steps=5, batch_size=10)

    x = np.random.randn(2)
    g = model.predictive_mean_gradient(x)
    assert g.shape == (2,)
    assert np.all(np.isfinite(g))


def test_predictive_variance_gradient_returns_correct_shape():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann(get_network=get_default_network, use_double_precision=True, sampling_method="adaptive_sghmc")
    model.train(x_train, y_train, num_steps=10, keep_every=2, num_burn_in_steps=5, batch_size=10)

    x = np.random.randn(2)
    g = model.predictive_variance_gradient(x)
    assert g.shape == (2,)
    assert np.all(np.isfinite(g))


def test_network_weights_setter_works_with_different_dtype():
    model = Bohamiann(get_network=get_default_network, use_double_precision=False)
    model.model = model.get_network(2).float()
    weights = model.network_weights
    add_ones = [w + 1 for w in weights]
    model.network_weights = add_ones
    updated = model.network_weights
    for w_new, w_old in zip(add_ones, updated):
        assert np.allclose(w_new, w_old)


def test_train_continue_training_reuses_model():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann(get_network=get_default_network)
    model.train(x_train, y_train, num_steps=5, num_burn_in_steps=2, batch_size=10)
    old_model_params = [p.detach().clone() for p in model.model.parameters()]
    model.train(x_train, y_train, num_steps=5, continue_training=True, batch_size=10)
    new_model_params = [p.detach().clone() for p in model.model.parameters()]
    # some parameters should be updated after training again
    differing = any(not torch.allclose(o, n) for o, n in zip(old_model_params, new_model_params))
    assert differing


def test_predict_works_without_normalization():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann(get_network=get_default_network, do_normalize_input=False, do_normalize_output=False, use_double_precision=False)
    model.train(x_train, y_train, num_steps=10, keep_every=2, num_burn_in_steps=5, batch_size=10)

    x_test = np.random.randn(5, 2)
    m, v = model.predict(x_test)
    assert m.shape == (5,)
    assert v.shape == (5,)


def test_predict_single_index_out_of_range_raises():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann(get_network=get_default_network)
    model.train(x_train, y_train, num_steps=10, keep_every=2, num_burn_in_steps=5, batch_size=10)
    x_test = np.random.randn(5, 2)
    with pytest.raises(IndexError):
        model.predict_single(x_test, sample_index=len(model.sampled_weights))


def test_train_invalid_batch_size_raises():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x_train, y_train, batch_size=0)


def test_train_invalid_target_shape_raises():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30, 2)  # Invalid shape
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x_train, y_train)


def test_train_and_evaluate_invalid_batch_size_raises():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    x_valid = np.random.randn(10, 2)
    y_valid = np.random.randn(10)
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train_and_evaluate(x_train, y_train, x_valid, y_valid, batch_size=0)
