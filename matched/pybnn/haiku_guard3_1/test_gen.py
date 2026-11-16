import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from pybnn.bohamiann import Bohamiann, get_default_network, nll


SMALL_TRAIN = dict(num_steps=10, keep_every=2, num_burn_in_steps=2, batch_size=10)


def _toy_data(n=30, d=2):
    """Deterministic toy regression data (no randomness involved)."""
    x = np.linspace(-1.0, 1.0, n * d).reshape(n, d)
    y = np.sin(3.0 * x[:, 0]) + 0.5 * x[:, 1]
    return x, y


def _fit(bnn_kwargs=None, **train_overrides):
    x, y = _toy_data()
    bnn = Bohamiann(**(bnn_kwargs or {}))
    train_kwargs = dict(SMALL_TRAIN)
    train_kwargs.update(train_overrides)
    bnn.train(x, y, **train_kwargs)
    return bnn, x, y


def test_init_stores_default_and_custom_arguments():
    torch.manual_seed(0)
    np.random.seed(0)

    bnn = Bohamiann()
    assert bnn.print_every_n_steps == 100
    assert bnn.metrics == (nn.MSELoss,)
    assert bnn.do_normalize_input is True
    assert bnn.do_normalize_output is True
    assert bnn.sampling_method == "adaptive_sghmc"
    assert bnn.use_double_precision is True
    assert bnn.get_network is get_default_network
    assert bnn.likelihood_function is nll
    assert bnn.is_trained is False
    assert bnn.sampled_weights == []
    assert bnn.sampler is None

    def custom_network(input_dimensionality):
        return get_default_network(input_dimensionality)

    def custom_likelihood(input, target):
        return nll(input=input, target=target)

    bnn = Bohamiann(get_network=custom_network,
                    normalize_input=False,
                    normalize_output=False,
                    sampling_method="sgld",
                    use_double_precision=False,
                    metrics=(),
                    likelihood_function=custom_likelihood,
                    print_every_n_steps=7)
    assert bnn.get_network is custom_network
    assert bnn.do_normalize_input is False
    assert bnn.do_normalize_output is False
    assert bnn.sampling_method == "sgld"
    assert bnn.use_double_precision is False
    assert bnn.metrics == ()
    assert bnn.likelihood_function is custom_likelihood
    assert bnn.print_every_n_steps == 7


def test_nll_matches_closed_form():
    torch.manual_seed(0)
    np.random.seed(0)

    mu = np.array([0.5, -1.0, 2.0])
    logvar = np.array([0.0, -0.5, 1.0])
    y = np.array([1.0, -0.5, 1.5])
    inp = torch.tensor(np.stack([mu, logvar], axis=1), dtype=torch.float64)
    tgt = torch.tensor(y, dtype=torch.float64)

    expected = np.mean(0.5 * (y - mu) ** 2 / np.exp(logvar) + 0.5 * logvar)
    result = nll(inp, tgt)

    assert result.dim() == 0
    assert abs(result.item() - expected) < 1e-8


def test_nll_accepts_vector_or_column_targets():
    torch.manual_seed(0)
    np.random.seed(0)

    mu = np.array([0.1, 0.4, -0.7, 1.2])
    logvar = np.array([-0.2, 0.3, 0.0, 0.5])
    y = np.array([0.0, 0.5, -1.0, 1.0])
    inp = torch.tensor(np.stack([mu, logvar], axis=1), dtype=torch.float64)

    vector_result = nll(inp, torch.tensor(y, dtype=torch.float64))
    column_result = nll(inp, torch.tensor(y, dtype=torch.float64).view(-1, 1))

    assert abs(vector_result.item() - column_result.item()) < 1e-12


def test_nll_is_lower_when_mean_matches_targets():
    torch.manual_seed(0)
    np.random.seed(0)

    y = np.array([0.3, -0.2, 1.0, 0.7])
    logvar = np.zeros_like(y)
    tgt = torch.tensor(y, dtype=torch.float64)

    good = torch.tensor(np.stack([y, logvar], axis=1), dtype=torch.float64)
    bad = torch.tensor(np.stack([y + 1.0, logvar], axis=1), dtype=torch.float64)

    assert nll(good, tgt).item() < nll(bad, tgt).item()


def test_default_network_layer_shapes_and_forward_output():
    torch.manual_seed(0)
    np.random.seed(0)

    net = get_default_network(input_dimensionality=3)

    assert net.fc1.in_features == 3
    assert net.fc1.out_features == 50
    assert net.fc2.in_features == 50
    assert net.fc2.out_features == 50
    assert net.fc3.in_features == 50
    assert net.fc3.out_features == 1

    out = net(torch.randn(5, 3))
    assert tuple(out.shape) == (5, 2)


def test_network_weights_setter_and_getter_roundtrip():
    torch.manual_seed(0)
    np.random.seed(0)

    bnn = Bohamiann()
    bnn.model = get_default_network(input_dimensionality=2).double()
    params = list(bnn.model.parameters())

    weights = bnn.network_weights
    assert len(weights) == len(params)
    for w, p in zip(weights, params):
        assert w.shape == tuple(p.shape)
        assert w.dtype == np.float64

    bnn.network_weights = [np.zeros_like(w) for w in weights]
    for p in bnn.model.parameters():
        assert torch.all(p == 0).item()

    bnn.network_weights = [np.full_like(w, 0.25) for w in weights]
    for w in bnn.network_weights:
        np.testing.assert_allclose(w, 0.25, atol=1e-12)


def test_network_weights_getter_returns_copies():
    torch.manual_seed(0)
    np.random.seed(0)

    bnn = Bohamiann()
    bnn.model = get_default_network(input_dimensionality=2).double()

    weights = bnn.network_weights
    before = [w.copy() for w in weights]
    weights[0][...] = 123.0

    after = bnn.network_weights
    for a, b in zip(after, before):
        np.testing.assert_array_equal(a, b)


def test_normalize_input_gives_zero_mean_unit_variance():
    torch.manual_seed(0)
    np.random.seed(0)

    x = np.random.randn(200, 3) * 5.0 + 2.0
    bnn = Bohamiann()
    x_norm, m, s = bnn.normalize_input(x)

    assert x_norm.shape == x.shape
    np.testing.assert_allclose(x_norm.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(x_norm.std(axis=0), 1.0, atol=1e-2)


def test_train_sets_flag_and_collects_expected_number_of_samples():
    torch.manual_seed(0)
    np.random.seed(0)

    # steps 0..9, burn-in 2, keep every 2: kept steps are 4, 6, 8
    bnn, _, _ = _fit()

    assert bnn.is_trained is True
    assert len(bnn.sampled_weights) == 3
    n_params = len(list(bnn.model.parameters()))
    for sample in bnn.sampled_weights:
        assert len(sample) == n_params


def test_train_rejects_invalid_batch_size():
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _toy_data()
    bnn = Bohamiann()
    with pytest.raises(AssertionError):
        bnn.train(x, y, num_steps=2, num_burn_in_steps=0, batch_size=0)


def test_train_rejects_targets_that_are_not_vectors():
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _toy_data()
    y_two_columns = np.stack([y, y], axis=1)
    bnn = Bohamiann()
    with pytest.raises(AssertionError):
        bnn.train(x, y_two_columns, num_steps=2, num_burn_in_steps=0, batch_size=10)


def test_train_clips_batch_size_to_dataset_size_and_logs_warning(caplog):
    torch.manual_seed(0)
    np.random.seed(0)

    seen_batch_sizes = []

    def recording_likelihood(input, target):
        seen_batch_sizes.append(input.shape[0])
        return nll(input=input, target=target)

    x = np.linspace(-1.0, 1.0, 10).reshape(5, 2)
    y = np.arange(5, dtype=np.float64)
    bnn = Bohamiann(likelihood_function=recording_likelihood)

    with caplog.at_level(logging.WARNING):
        bnn.train(x, y, num_steps=3, num_burn_in_steps=0, keep_every=1, batch_size=20)

    assert "Not enough datapoints" in caplog.text
    assert seen_batch_sizes == [5, 5, 5]
    assert bnn.is_trained is True


def test_train_calls_likelihood_function_once_per_step():
    torch.manual_seed(0)
    np.random.seed(0)

    calls = []

    def counting_likelihood(input, target):
        calls.append(1)
        return nll(input=input, target=target)

    x, y = _toy_data()
    bnn = Bohamiann(likelihood_function=counting_likelihood)
    bnn.train(x, y, num_steps=6, num_burn_in_steps=2, keep_every=2, batch_size=10)

    assert len(calls) == 6


def test_train_passes_input_dimensionality_to_get_network():
    torch.manual_seed(0)
    np.random.seed(0)

    dims = []

    def factory(input_dimensionality):
        dims.append(input_dimensionality)
        return get_default_network(input_dimensionality)

    x, y = _toy_data(d=3)
    bnn = Bohamiann(get_network=factory)
    bnn.train(x, y, **SMALL_TRAIN)

    assert dims == [3]


def test_train_single_precision_produces_float32_weights():
    torch.manual_seed(0)
    np.random.seed(0)

    bnn, x, _ = _fit(bnn_kwargs={"use_double_precision": False})

    assert len(bnn.sampled_weights) == 3
    for sample in bnn.sampled_weights:
        for w in sample:
            assert w.dtype == np.float32

    mu, var = bnn.predict(x[:4])
    assert np.all(np.isfinite(mu))
    assert np.all(var > 0)


def test_train_verbose_prints_progress(capsys):
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _toy_data()
    bnn = Bohamiann(print_every_n_steps=1)
    bnn.train(x, y, num_steps=4, keep_every=1, num_burn_in_steps=2,
              batch_size=10, verbose=True)

    out = capsys.readouterr().out
    assert "NLL" in out
    assert "Samples=" in out


def test_predict_returns_mean_and_positive_variance_of_correct_shape():
    torch.manual_seed(0)
    np.random.seed(0)

    bnn, x, _ = _fit()
    mu, var = bnn.predict(x[:4])

    assert mu.shape == (4,)
    assert var.shape == (4,)
    assert np.all(np.isfinite(mu))
    assert np.all(var > 0)


def test_predict_individual_predictions_average_to_mean():
    torch.manual_seed(0)
    np.random.seed(0)

    bnn, x, _ = _fit()
    mu, var, individual = bnn.predict(x[:4], return_individual_predictions=True)

    assert individual.shape == (3, 4)
    np.testing.assert_allclose(mu, individual.mean(axis=0), atol=1e-8)


def test_predict_single_matches_individual_predictions():
    torch.manual_seed(0)
    np.random.seed(0)

    bnn, x, _ = _fit()
    _, _, individual = bnn.predict(x[:4], return_individual_predictions=True)

    for i in range(len(bnn.sampled_weights)):
        single = bnn.predict_single(x[:4], i)
        assert single.shape == (4, 2)
        np.testing.assert_allclose(single[:, 0], individual[i], atol=1e-10)


def test_predict_works_without_normalization():
    torch.manual_seed(0)
    np.random.seed(0)

    bnn, x, _ = _fit(bnn_kwargs={"normalize_input": False, "normalize_output": False})

    assert not hasattr(bnn, "x_mean")
    mu, var = bnn.predict(x[:4])
    assert mu.shape == (4,)
    assert np.all(np.isfinite(mu))
    assert np.all(var > 0)


def test_f_gradient_matches_finite_differences():
    torch.manual_seed(0)
    np.random.seed(0)

    bnn, x, _ = _fit()
    weights = bnn.sampled_weights[0]
    x0 = x[7].copy()

    g = bnn.f_gradient(x0, weights=weights)
    assert g.shape == (2,)

    eps = 1e-6
    fd = np.zeros(2)
    for j in range(2):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        f_plus = bnn.predict_single(x_plus[None, :], 0)[0, 0]
        f_minus = bnn.predict_single(x_minus[None, :], 0)[0, 0]
        fd[j] = (f_plus - f_minus) / (2 * eps)

    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


def test_predictive_mean_gradient_is_mean_of_sample_gradients():
    torch.manual_seed(0)
    np.random.seed(0)

    bnn, x, _ = _fit()
    x0 = x[7]

    g = bnn.predictive_mean_gradient(x0)
    expected = np.mean([bnn.f_gradient(x0, weights=w) for w in bnn.sampled_weights], axis=0)

    assert g.shape == (2,)
    np.testing.assert_allclose(g, expected, atol=1e-12)


def test_predictive_variance_gradient_is_zero_for_identical_samples():
    torch.manual_seed(0)
    np.random.seed(0)

    bnn, x, _ = _fit()
    w0 = bnn.sampled_weights[0]
    bnn.sampled_weights = [w0, w0]

    g = bnn.predictive_variance_gradient(x[7])

    assert g.shape == (2,)
    np.testing.assert_allclose(g, 0.0, atol=1e-12)


def test_train_and_evaluate_returns_learning_curve():
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _toy_data()
    bnn = Bohamiann()
    n_steps, learning_curve_ll, learning_curve_mse = bnn.train_and_evaluate(
        x, y, x[:5], y[:5],
        num_steps=4,
        validate_every_n_steps=2,
        keep_every=1,
        num_burn_in_steps=2,
        batch_size=10,
    )

    assert n_steps == [4, 6]
    assert len(learning_curve_ll) == 2
    assert len(learning_curve_mse) == 2
    assert all(np.isfinite(v) for v in learning_curve_ll)
    assert all(v >= 0 for v in learning_curve_mse)
    assert len(bnn.sampled_weights) == 2
