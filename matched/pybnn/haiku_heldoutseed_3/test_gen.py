import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from pybnn.bohamiann import Bohamiann, get_default_network, nll
from pybnn.sampler import AdaptiveSGHMC, PreconditionedSGLD, SGHMC, SGLD


@pytest.fixture(autouse=True)
def seed_everything():
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    x = rng.uniform(-1.0, 1.0, size=(30, 2))
    y = x[:, 0] - 2.0 * x[:, 1] + 0.1 * rng.randn(30)
    return x, y


def _train_small(model, x, y, **overrides):
    kwargs = dict(num_steps=10, num_burn_in_steps=2, keep_every=2, batch_size=10)
    kwargs.update(overrides)
    model.train(x, y, **kwargs)
    return model


def _manual_model(normalize_output=False, y_mean=0.0, y_std=1.0):
    """Model with hand-picked weight samples and no training."""
    model = Bohamiann(normalize_input=False, normalize_output=normalize_output)
    model.model = get_default_network(input_dimensionality=2).double()
    base = model.network_weights
    model.sampled_weights = [
        base,
        tuple(w + 0.25 for w in base),
        tuple(w * 1.5 for w in base),
    ]
    model.y_mean = y_mean
    model.y_std = y_std
    return model


def _raw_outputs(model, x):
    """Raw network outputs (no denormalization), shape (n_samples, n_points, 2)."""
    outputs = []
    for weights in model.sampled_weights:
        model.network_weights = weights
        with torch.no_grad():
            outputs.append(model.model(torch.from_numpy(x).double()).numpy())
    return np.array(outputs)


def _closed_form(model, x):
    raw = _raw_outputs(model, x)
    mean_raw = raw[:, :, 0].mean(axis=0)
    var_raw = np.mean((raw[:, :, 0] - mean_raw) ** 2 + np.exp(raw[:, :, 1]), axis=0)
    return mean_raw, var_raw


def _scalar_output(model, x, weights):
    model.network_weights = weights
    with torch.no_grad():
        inp = torch.from_numpy(np.asarray(x, dtype=np.float64)[None, :]).double()
        return float(model.model(inp)[0, 0])


def test_init_defaults():
    model = Bohamiann()
    assert model.is_trained is False
    assert model.sampled_weights == []
    assert model.sampler is None
    assert model.sampling_method == "adaptive_sghmc"
    assert model.use_double_precision is True
    assert model.do_normalize_input is True
    assert model.do_normalize_output is True
    assert model.print_every_n_steps == 100
    assert model.get_network is get_default_network
    assert model.likelihood_function is nll
    assert model.metrics == (nn.MSELoss,)


def test_init_stores_custom_arguments():
    def custom_network(input_dimensionality):
        return get_default_network(input_dimensionality)

    def custom_likelihood(input, target):
        return nll(input=input, target=target)

    model = Bohamiann(
        get_network=custom_network,
        normalize_input=False,
        normalize_output=False,
        sampling_method="sgld",
        use_double_precision=False,
        metrics=(nn.L1Loss,),
        likelihood_function=custom_likelihood,
        print_every_n_steps=7,
    )
    assert model.get_network is custom_network
    assert model.do_normalize_input is False
    assert model.do_normalize_output is False
    assert model.sampling_method == "sgld"
    assert model.use_double_precision is False
    assert model.metrics == (nn.L1Loss,)
    assert model.likelihood_function is custom_likelihood
    assert model.print_every_n_steps == 7


def test_default_network_output_shape():
    net = get_default_network(input_dimensionality=3)
    assert net.fc1.in_features == 3
    assert net.fc3.out_features == 1
    out = net(torch.zeros(5, 3))
    assert tuple(out.shape) == (5, 2)


def test_nll_matches_closed_form():
    # sample 1: mean 0, log-variance 0, target 1 -> ll = -0.5
    # sample 2: mean 1, log-variance ln(2), target 1 -> ll = -0.5 * ln(2)
    inp = torch.tensor([[0.0, 0.0], [1.0, float(np.log(2.0))]], dtype=torch.float64)
    target = torch.tensor([1.0, 1.0], dtype=torch.float64)
    value = nll(input=inp, target=target)
    expected = 0.25 + 0.25 * np.log(2.0)
    assert value.dim() == 0
    assert value.item() == pytest.approx(expected, rel=1e-8)


def test_network_weights_getter_returns_detached_numpy_copies():
    model = Bohamiann(use_double_precision=True)
    model.model = get_default_network(input_dimensionality=2).double()
    params = list(model.model.parameters())
    weights = model.network_weights

    assert isinstance(weights, tuple)
    assert len(weights) == len(params)
    for w, p in zip(weights, params):
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float64
        assert w.shape == tuple(p.shape)

    before = [p.detach().clone() for p in params]
    for w in weights:
        w[...] = 123.0
    for p, b in zip(params, before):
        assert torch.allclose(p.detach(), b)


def test_network_weights_setter_assigns_values():
    model = Bohamiann(use_double_precision=True)
    model.model = get_default_network(input_dimensionality=2).double()
    new_weights = [np.full(p.shape, 0.5) for p in model.model.parameters()]
    model.network_weights = new_weights
    for p in model.model.parameters():
        assert torch.allclose(p.detach(), torch.full_like(p, 0.5))
    for w in model.network_weights:
        assert np.allclose(w, 0.5)


def test_normalize_helpers_statistics(data):
    x, y = data
    model = Bohamiann()

    x_norm, x_mean, x_std = model.normalize_input(x)
    assert np.allclose(x_norm.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(x_norm.std(axis=0), 1.0, atol=0.05)
    assert np.allclose(x_mean, x.mean(axis=0))

    y_norm, y_mean, y_std = model.normalize_output(y)
    assert np.isclose(y_norm.mean(), 0.0, atol=1e-10)
    assert np.isclose(y_norm.std(), 1.0, atol=0.05)
    assert np.isclose(y_mean, y.mean())

    # explicitly given statistics are used
    given, _, _ = model.normalize_output(y, m=1.0, s=2.0)
    assert np.allclose(given, (y - 1.0) / 2.0)


def test_train_collects_expected_number_of_samples(data):
    x, y = data
    model = _train_small(Bohamiann(), x, y)
    assert model.is_trained is True
    assert len(model.sampled_weights) == 3
    n_params = len(list(model.model.parameters()))
    for sample in model.sampled_weights:
        assert len(sample) == n_params


@pytest.mark.parametrize("num_steps,burn_in,keep_every,expected", [
    (10, 2, 2, 3),   # kept at steps 4, 6, 8
    (5, 0, 1, 4),    # kept at steps 1, 2, 3, 4
    (3, 5, 1, 0),    # still in burn-in, nothing kept
    (12, 0, 3, 3),   # kept at steps 3, 6, 9
])
def test_train_sample_count_follows_burn_in_and_thinning(data, num_steps, burn_in, keep_every, expected):
    x, y = data
    model = _train_small(Bohamiann(), x, y, num_steps=num_steps,
                         num_burn_in_steps=burn_in, keep_every=keep_every)
    assert len(model.sampled_weights) == expected


@pytest.mark.parametrize("method,sampler_cls", [
    ("adaptive_sghmc", AdaptiveSGHMC),
    ("sgld", SGLD),
    ("sghmc", SGHMC),
    ("preconditioned_sgld", PreconditionedSGLD),
])
def test_train_creates_requested_sampler(data, method, sampler_cls):
    x, y = data
    model = _train_small(Bohamiann(sampling_method=method), x, y,
                         num_steps=3, num_burn_in_steps=1, keep_every=1)
    assert isinstance(model.sampler, sampler_cls)
    assert model.is_trained is True


def test_train_continue_training_appends_samples_and_keeps_network(data):
    x, y = data
    model = _train_small(Bohamiann(), x, y)
    network_before = model.model
    assert len(model.sampled_weights) == 3

    # burn-in 0, keep every step: steps 1 and 2 are kept
    _train_small(model, x, y, num_steps=3, num_burn_in_steps=0, keep_every=1,
                 continue_training=True)
    assert len(model.sampled_weights) == 5
    assert model.model is network_before


def test_train_without_continue_resets_samples_and_network(data):
    x, y = data
    model = _train_small(Bohamiann(), x, y)
    network_before = model.model
    _train_small(model, x, y)
    assert len(model.sampled_weights) == 3
    assert model.model is not network_before


def test_train_rejects_zero_batch_size(data):
    x, y = data
    with pytest.raises(AssertionError):
        Bohamiann().train(x, y, num_steps=3, num_burn_in_steps=1, batch_size=0)


def test_train_rejects_matrix_targets(data):
    x, y = data
    y_matrix = np.stack([y, y], axis=1)
    with pytest.raises(AssertionError):
        Bohamiann().train(x, y_matrix, num_steps=3, num_burn_in_steps=1)


def test_train_accepts_column_vector_targets(data):
    x, y = data
    model = _train_small(Bohamiann(), x, y.reshape(-1, 1))
    assert model.is_trained is True
    assert model.y.shape == (30,)


def test_train_warns_when_too_few_points_for_batch(data, caplog):
    x, y = data
    caplog.set_level(logging.WARNING)
    model = Bohamiann()
    _train_small(model, x[:5], y[:5], batch_size=20, num_steps=4,
                 num_burn_in_steps=1, keep_every=1)
    assert model.is_trained is True
    assert any("Not enough datapoints" in record.getMessage() for record in caplog.records)


def test_predict_output_shapes_and_positive_variance(data):
    x, y = data
    model = _train_small(Bohamiann(), x, y)
    x_test = np.random.RandomState(1).uniform(-1.0, 1.0, size=(7, 2))
    mean, var = model.predict(x_test)
    assert mean.shape == (7,)
    assert var.shape == (7,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_predict_return_individual_predictions(data):
    x, y = data
    model = _train_small(Bohamiann(), x, y)
    x_test = x[:4]
    mean, var, individual = model.predict(x_test, return_individual_predictions=True)
    assert mean.shape == (4,)
    assert var.shape == (4,)
    assert individual.shape == (3, 4)


def test_predict_single_matches_individual_predictions(data):
    x, y = data
    model = _train_small(Bohamiann(), x, y)
    x_test = x[:4]
    _, _, individual = model.predict(x_test, return_individual_predictions=True)
    for i in range(len(model.sampled_weights)):
        single = model.predict_single(x_test, sample_index=i)
        assert single.shape == (4, 2)
        assert np.allclose(single[:, 0], individual[i])


def test_predict_matches_closed_form_without_normalization():
    model = _manual_model()
    x_test = np.random.RandomState(2).uniform(-1.0, 1.0, size=(6, 2))
    mean_raw, var_raw = _closed_form(model, x_test)
    mean, var = model.predict(x_test)
    assert np.allclose(mean, mean_raw)
    assert np.allclose(var, var_raw)


def test_predict_denormalizes_mean_and_variance():
    model = _manual_model(normalize_output=True, y_mean=5.0, y_std=2.0)
    x_test = np.random.RandomState(3).uniform(-1.0, 1.0, size=(6, 2))
    mean_raw, var_raw = _closed_form(model, x_test)
    mean, var = model.predict(x_test)
    assert np.allclose(mean, 2.0 * mean_raw + 5.0)
    assert np.allclose(var, 4.0 * var_raw)


def test_f_gradient_matches_finite_differences():
    model = _manual_model()
    weights = model.sampled_weights[1]
    x0 = np.array([0.3, -0.7])
    grad = model.f_gradient(x0, weights)

    eps = 1e-6
    numeric = np.zeros(2)
    for i in range(2):
        step = np.zeros(2)
        step[i] = eps
        numeric[i] = (_scalar_output(model, x0 + step, weights)
                      - _scalar_output(model, x0 - step, weights)) / (2 * eps)

    assert grad.shape == (2,)
    assert np.allclose(grad, numeric, atol=1e-6)


def test_predictive_mean_gradient_matches_finite_differences():
    model = _manual_model()
    x0 = np.array([0.2, 0.1])
    grad = model.predictive_mean_gradient(x0)

    def mean_at(x):
        return float(model.predict(np.asarray(x, dtype=np.float64)[None, :])[0][0])

    eps = 1e-6
    numeric = np.zeros(2)
    for i in range(2):
        step = np.zeros(2)
        step[i] = eps
        numeric[i] = (mean_at(x0 + step) - mean_at(x0 - step)) / (2 * eps)

    assert grad.shape == (2,)
    assert np.allclose(grad, numeric, atol=1e-5)


def test_train_and_evaluate_returns_learning_curve(data):
    x, y = data
    model = Bohamiann()
    n_steps, learning_curve_ll, learning_curve_mse = model.train_and_evaluate(
        x[:20], y[:20], x[20:], y[20:],
        num_steps=6,
        validate_every_n_steps=2,
        keep_every=1,
        num_burn_in_steps=2,
        batch_size=10,
    )
    assert n_steps == [4, 6, 8]
    assert len(learning_curve_ll) == 3
    assert len(learning_curve_mse) == 3
    assert all(np.isfinite(v) for v in learning_curve_ll)
    assert all(v >= 0 for v in learning_curve_mse)
    assert model.is_trained is True
