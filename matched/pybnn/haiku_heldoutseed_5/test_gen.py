import logging

import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def _make_data(n_points=30, n_dims=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(n_points, n_dims)
    y = x.sum(axis=1) + 0.01 * rng.randn(n_points)
    return x, y


def _train_model(model_kwargs=None, **train_kwargs):
    """Train a small Bohamiann model; with the settings below, 3 weight samples are kept."""
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = _make_data()
    model = Bohamiann(**(model_kwargs or {}))
    options = dict(num_steps=12, keep_every=2, num_burn_in_steps=4, batch_size=8)
    options.update(train_kwargs)
    model.train(x, y, **options)
    return model


def test_init_defaults():
    model = Bohamiann()

    assert model.print_every_n_steps == 100
    assert model.do_normalize_input is True
    assert model.do_normalize_output is True
    assert model.sampling_method == "adaptive_sghmc"
    assert model.use_double_precision is True
    assert model.get_network is get_default_network
    assert model.likelihood_function is nll
    assert model.sampled_weights == []
    assert model.is_trained is False
    assert model.sampler is None


def test_default_network_shapes_and_output():
    net = get_default_network(input_dimensionality=3)
    assert isinstance(net, torch.nn.Module)
    assert net.fc1.in_features == 3

    out = net(torch.zeros(7, 3))
    assert tuple(out.shape) == (7, 2)


def test_nll_is_zero_for_perfect_prediction_with_unit_variance():
    inputs = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)

    value = nll(input=inputs, target=target)

    assert float(value) == pytest.approx(0.0, abs=1e-12)


def test_nll_known_value_with_nonunit_variance():
    # mean = 1, target = 0, variance = 2 -> nll = 0.25 + 0.5 * log(2)
    inputs = torch.tensor([[1.0, np.log(2.0)]], dtype=torch.float64)
    target = torch.tensor([0.0], dtype=torch.float64)

    value = nll(input=inputs, target=target)

    assert float(value) == pytest.approx(0.25 + 0.5 * np.log(2.0), rel=1e-6)


def test_nll_averages_over_batch():
    inputs = torch.tensor([[1.0, 0.0], [1.0, np.log(2.0)]], dtype=torch.float64)
    target = torch.tensor([0.0, 0.0], dtype=torch.float64)

    value = nll(input=inputs, target=target)

    expected = (0.75 + 0.5 * np.log(2.0)) / 2.0
    assert float(value) == pytest.approx(expected, rel=1e-6)


def test_network_weights_getter_returns_numpy_arrays_matching_parameters():
    model = _train_model()

    weights = model.network_weights

    assert isinstance(weights, tuple)
    params = list(model.model.parameters())
    assert len(weights) == len(params)
    for w, p in zip(weights, params):
        assert isinstance(w, np.ndarray)
        assert w.shape == tuple(p.shape)
        np.testing.assert_allclose(w, p.detach().numpy())


def test_network_weights_setter_overwrites_parameters():
    model = Bohamiann()
    model.model = get_default_network(input_dimensionality=2).double()

    zeros = [np.zeros_like(w) for w in model.network_weights]
    model.network_weights = zeros
    assert all(np.all(w == 0) for w in model.network_weights)

    ones = [np.ones_like(w) for w in model.network_weights]
    model.network_weights = ones
    assert all(np.all(w == 1) for w in model.network_weights)


def test_train_sets_is_trained_and_collects_expected_number_of_samples():
    model = _train_model()

    assert model.is_trained is True
    # steps 0..11, keep steps s > 4 with (s - 4) % 2 == 0 -> s in {6, 8, 10}
    assert len(model.sampled_weights) == 3
    assert all(isinstance(w, tuple) for w in model.sampled_weights)


def test_train_without_continue_resets_sampled_weights():
    model = _train_model()
    assert len(model.sampled_weights) == 3

    x, y = _make_data()
    model.train(x, y, num_steps=12, keep_every=2, num_burn_in_steps=4, batch_size=8,
                continue_training=False)

    assert len(model.sampled_weights) == 3


def test_train_with_continue_training_accumulates_samples():
    model = _train_model()
    assert len(model.sampled_weights) == 3

    x, y = _make_data()
    model.train(x, y, num_steps=12, keep_every=2, num_burn_in_steps=4, batch_size=8,
                continue_training=True)

    assert len(model.sampled_weights) == 6


def test_train_rejects_zero_batch_size():
    model = Bohamiann()
    x, y = _make_data()

    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=2, batch_size=0)


def test_train_rejects_multi_column_targets():
    model = Bohamiann()
    x, _ = _make_data()
    bad_targets = np.random.rand(x.shape[0], 2)

    with pytest.raises(AssertionError):
        model.train(x, bad_targets, num_steps=2, batch_size=4)


def test_train_warns_when_fewer_points_than_batch_size(caplog):
    model = Bohamiann()
    x, y = _make_data(n_points=5)

    with caplog.at_level(logging.WARNING):
        model.train(x, y, num_steps=6, keep_every=1, num_burn_in_steps=2, batch_size=20)

    assert "Not enough datapoints" in caplog.text
    assert model.is_trained is True


def test_train_verbose_prints_progress(capsys):
    model = Bohamiann(normalize_output=False, print_every_n_steps=1)
    x, y = _make_data()

    model.train(x, y, num_steps=6, keep_every=2, num_burn_in_steps=2, batch_size=8, verbose=True)

    captured = capsys.readouterr().out
    assert "Step" in captured
    assert "NLL" in captured
    assert "Samples=" in captured


@pytest.mark.parametrize("sampling_method", ["adaptive_sghmc", "sgld", "sghmc", "preconditioned_sgld"])
def test_train_supports_all_sampling_methods(sampling_method):
    model = _train_model(model_kwargs={"sampling_method": sampling_method})

    assert model.is_trained is True
    assert len(model.sampled_weights) == 3


def test_train_and_predict_in_single_precision():
    model = _train_model(model_kwargs={"use_double_precision": False})
    x, _ = _make_data()

    assert next(model.model.parameters()).dtype == torch.float32

    mean, var = model.predict(x[:5])
    assert mean.shape == (5,)
    assert var.shape == (5,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_predict_returns_mean_and_positive_variance():
    model = _train_model()
    x, _ = _make_data()

    mean, var = model.predict(x[:5])

    assert mean.shape == (5,)
    assert var.shape == (5,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_predict_individual_predictions_shape():
    model = _train_model()
    x, _ = _make_data()

    mean, var, individual = model.predict(x[:5], return_individual_predictions=True)

    assert individual.shape == (len(model.sampled_weights), 5)
    assert mean.shape == (5,)
    assert var.shape == (5,)


def test_predict_single_returns_mean_and_log_variance_per_point():
    model = _train_model()
    x, _ = _make_data()

    output = model.predict_single(x[:5], sample_index=0)

    assert output.shape == (5, 2)
    assert np.all(np.isfinite(output))


def test_predict_single_raises_for_out_of_range_index():
    model = _train_model()
    x, _ = _make_data()

    with pytest.raises(IndexError):
        model.predict_single(x[:5], sample_index=len(model.sampled_weights) + 10)


def test_f_gradient_matches_finite_differences():
    model = _train_model(model_kwargs={"normalize_input": False, "normalize_output": False})
    x, _ = _make_data()
    x0 = x[0].copy()
    weights = model.network_weights

    grad = model.f_gradient(x0, weights)

    def f(point):
        with torch.no_grad():
            out = model.model(torch.from_numpy(point[None, :]).double()).numpy()
        return float(out[0, 0])

    eps = 1e-6
    numeric = np.zeros_like(x0)
    for i in range(len(x0)):
        step = np.zeros_like(x0)
        step[i] = eps
        numeric[i] = (f(x0 + step) - f(x0 - step)) / (2 * eps)

    assert grad.shape == (x0.shape[0],)
    np.testing.assert_allclose(grad, numeric, rtol=1e-4, atol=1e-5)


def test_predictive_mean_gradient_shape():
    model = _train_model()
    x, _ = _make_data()

    g = model.predictive_mean_gradient(x[0])

    assert g.shape == (x.shape[1],)
    assert np.all(np.isfinite(g))


def test_predictive_variance_gradient_shape():
    model = _train_model()
    x, _ = _make_data()

    g = model.predictive_variance_gradient(x[0])

    assert g.shape == (x.shape[1],)
    assert np.all(np.isfinite(g))


def test_train_and_evaluate_returns_learning_curve():
    model = Bohamiann()
    x_train, y_train = _make_data(n_points=30)
    x_valid, y_valid = _make_data(n_points=10, seed=1)

    n_steps, ll, mse = model.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=6,
        validate_every_n_steps=2,
        keep_every=1,
        num_burn_in_steps=2,
        batch_size=8,
    )

    assert n_steps == [4, 6, 8]
    assert len(ll) == 3
    assert len(mse) == 3
    assert all(np.isfinite(v) for v in ll)
    assert all(v >= 0 for v in mse)
