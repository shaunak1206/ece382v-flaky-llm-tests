"""Unit tests for pybnn.bohamiann.Bohamiann and its module-level helpers."""

import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll
from pybnn.sampler import AdaptiveSGHMC, PreconditionedSGLD, SGHMC, SGLD

EXPECTED_WEIGHT_SHAPES = [(50, 3), (50,), (50, 50), (50,), (1, 50), (1,), (1, 1)]


@pytest.fixture(autouse=True)
def _seed_everything():
    np.random.seed(0)
    torch.manual_seed(0)
    yield


def _toy_data(n=12, d=3, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, d))
    y = np.sin(x).sum(axis=1) + 0.05 * rng.randn(n)
    return x, y


def _trained_model(n=12, d=3, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=4, **init_kwargs):
    x, y = _toy_data(n=n, d=d)
    model = Bohamiann(**init_kwargs)
    model.train(x, y, num_steps=num_steps, keep_every=keep_every,
                num_burn_in_steps=num_burn_in_steps, batch_size=batch_size)
    return model, x, y


def test_init_stores_defaults():
    model = Bohamiann()
    assert model.print_every_n_steps == 100
    assert model.do_normalize_input is True
    assert model.do_normalize_output is True
    assert model.sampling_method == "adaptive_sghmc"
    assert model.use_double_precision is True
    assert model.get_network is get_default_network
    assert model.likelihood_function is nll
    assert model.is_trained is False
    assert model.sampled_weights == []
    assert model.sampler is None


def test_nll_matches_closed_form_and_is_zero_for_perfect_fit():
    pred = np.array([[0.5, -0.3], [1.2, 0.4], [-0.7, 0.0]])
    target = np.array([0.0, 1.0, -1.0])
    inv_var = 1.0 / (np.exp(pred[:, 1]) + 1e-16)
    expected = np.mean(0.5 * (target - pred[:, 0]) ** 2 * inv_var + 0.5 * pred[:, 1])

    value = nll(input=torch.from_numpy(pred), target=torch.from_numpy(target))
    assert value.dim() == 0
    assert float(value) == pytest.approx(expected, rel=1e-9)

    perfect = nll(input=torch.tensor([[2.0, 0.0]], dtype=torch.float64),
                  target=torch.tensor([2.0], dtype=torch.float64))
    assert float(perfect) == pytest.approx(0.0, abs=1e-12)


def test_nll_prefers_predictions_closer_to_target():
    target = torch.tensor([1.0], dtype=torch.float64)
    close = nll(input=torch.tensor([[1.0, 0.0]], dtype=torch.float64), target=target)
    far = nll(input=torch.tensor([[3.0, 0.0]], dtype=torch.float64), target=target)
    assert float(close) < float(far)


def test_default_network_shapes_and_constant_log_variance_column():
    net = get_default_network(input_dimensionality=3).double()
    assert net.fc1.in_features == 3
    assert net.fc1.out_features == 50
    assert net.fc3.out_features == 1

    out = net(torch.randn(6, 3, dtype=torch.float64))
    assert out.shape == (6, 2)
    expected_log_var = torch.full((6,), float(np.log(1e-3)), dtype=torch.float64)
    assert torch.allclose(out[:, 1], expected_log_var)


def test_network_weights_match_architecture_shapes():
    model, _, _ = _trained_model(d=3)
    weights = model.network_weights
    assert isinstance(weights, tuple)
    assert [w.shape for w in weights] == EXPECTED_WEIGHT_SHAPES


def test_network_weights_setter_round_trips_values():
    model, _, _ = _trained_model(d=3)
    new_weights = [np.full_like(w, 0.25) for w in model.network_weights]
    model.network_weights = new_weights
    for got, expected in zip(model.network_weights, new_weights):
        np.testing.assert_allclose(got, expected)


def test_train_sets_state_and_keeps_expected_number_of_samples():
    x, y = _toy_data(n=12, d=3)
    model = Bohamiann()
    model.train(x, y.reshape(-1, 1), num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=4)
    assert model.is_trained is True
    np.testing.assert_allclose(model.X, x)
    assert model.y.shape == (12,)
    np.testing.assert_allclose(model.y, y)
    # steps 10 and 15 of 0..19 are kept (after burn-in, every 5 steps)
    assert len(model.sampled_weights) == 2
    assert all(len(w) == 7 for w in model.sampled_weights)


def test_train_rejects_zero_batch_size_and_matrix_targets():
    x, y = _toy_data(n=12, d=3)
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=2, batch_size=0)
    with pytest.raises(AssertionError):
        model.train(x, np.zeros((len(y), 2)), num_steps=2)


def test_train_resets_samples_unless_continue_training():
    model, x, y = _trained_model(num_steps=20, keep_every=5, num_burn_in_steps=5)
    assert len(model.sampled_weights) == 2

    model.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=4)
    assert len(model.sampled_weights) == 2

    # continuing appends: steps 5 of 0..9 are kept with no burn-in
    model.train(x, y, num_steps=10, keep_every=5, num_burn_in_steps=0, batch_size=4, continue_training=True)
    assert len(model.sampled_weights) == 3


def test_train_uses_all_points_when_fewer_than_batch_size():
    x, y = _toy_data(n=3, d=2)
    model = Bohamiann()
    model.train(x, y, num_steps=10, keep_every=2, num_burn_in_steps=2, batch_size=20)
    assert model.is_trained is True
    # steps 4, 6 and 8 are kept
    assert len(model.sampled_weights) == 3


def test_train_builds_sampler_for_each_sampling_method():
    x, y = _toy_data()
    expected = {
        "adaptive_sghmc": AdaptiveSGHMC,
        "sgld": SGLD,
        "sghmc": SGHMC,
        "preconditioned_sgld": PreconditionedSGLD,
    }
    for method, sampler_cls in expected.items():
        model = Bohamiann(sampling_method=method)
        model.train(x, y, num_steps=4, keep_every=2, num_burn_in_steps=1, batch_size=4)
        assert isinstance(model.sampler, sampler_cls), method


def test_float32_training_keeps_float32_parameters_and_finite_predictions():
    model, x, _ = _trained_model(use_double_precision=False)
    assert all(p.dtype == torch.float32 for p in model.model.parameters())
    mu, var = model.predict(x[:4])
    assert mu.shape == (4,)
    assert var.shape == (4,)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(var))
    assert np.all(var > 0)


def test_verbose_training_prints_progress_lines(capsys):
    x, y = _toy_data()
    model = Bohamiann(print_every_n_steps=2)
    model.train(x, y, num_steps=6, keep_every=1, num_burn_in_steps=3, batch_size=4, verbose=True)
    out = capsys.readouterr().out
    assert "NLL" in out
    assert "MSE" in out
    assert "Samples=" in out


def test_training_stores_input_and_output_normalization_statistics():
    x, y = _toy_data(n=12, d=3)
    model = Bohamiann()
    model.train(x, y, num_steps=4, keep_every=2, num_burn_in_steps=1, batch_size=4)
    np.testing.assert_allclose(model.x_mean, x.mean(axis=0))
    np.testing.assert_allclose(model.x_std, x.std(axis=0))
    np.testing.assert_allclose(model.y_mean, y.mean())
    np.testing.assert_allclose(model.y_std, y.std())


def test_normalize_helpers_standardize_and_accept_fixed_statistics():
    model = Bohamiann()
    x, _ = _toy_data(n=20, d=3)

    x_norm, mean, std = model.normalize_input(x)
    np.testing.assert_allclose(x_norm.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(x_norm.std(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(mean, x.mean(axis=0))
    np.testing.assert_allclose(std, x.std(axis=0))

    x_fixed, m2, s2 = model.normalize_output(x, m=np.zeros(3), s=np.full(3, 2.0))
    np.testing.assert_allclose(x_fixed, x / 2.0)
    np.testing.assert_allclose(m2, 0.0)
    np.testing.assert_allclose(s2, 2.0)


def test_predict_returns_mean_and_positive_variance_per_point():
    model, _, _ = _trained_model()
    x_test = np.linspace(-1.0, 1.0, 15).reshape(5, 3)
    mu, var = model.predict(x_test)
    assert mu.shape == (5,)
    assert var.shape == (5,)
    assert np.all(np.isfinite(mu))
    assert np.all(var > 0)


def test_predict_is_deterministic_for_fixed_samples():
    model, x, _ = _trained_model()
    mu1, var1 = model.predict(x[:4])
    mu2, var2 = model.predict(x[:4])
    np.testing.assert_allclose(mu1, mu2, rtol=1e-12)
    np.testing.assert_allclose(var1, var2, rtol=1e-12)


def test_predict_mean_is_average_of_individual_predictions():
    model, x, _ = _trained_model()
    mu, _, individual = model.predict(x[:4], return_individual_predictions=True)
    assert individual.shape == (len(model.sampled_weights), 4)
    np.testing.assert_allclose(mu, individual.mean(axis=0), rtol=1e-9)


def test_predict_variance_matches_mixture_formula_without_output_normalization():
    model, x, _ = _trained_model(normalize_output=False)
    x_test = x[:4]
    mu, var = model.predict(x_test)
    singles = np.array([model.predict_single(x_test, i) for i in range(len(model.sampled_weights))])
    f_mean = singles[:, :, 0]
    f_logvar = singles[:, :, 1]
    expected_mu = f_mean.mean(axis=0)
    expected_var = np.mean((f_mean - expected_mu) ** 2 + np.exp(f_logvar), axis=0)
    np.testing.assert_allclose(mu, expected_mu, rtol=1e-9)
    np.testing.assert_allclose(var, expected_var, rtol=1e-9)


def test_predict_single_matches_each_individual_prediction():
    model, x, _ = _trained_model()
    x_test = x[:4]
    _, _, individual = model.predict(x_test, return_individual_predictions=True)
    for i in range(len(model.sampled_weights)):
        single = model.predict_single(x_test, sample_index=i)
        assert single.shape == (4, 2)
        np.testing.assert_allclose(single[:, 0], individual[i], rtol=1e-9)


def test_f_gradient_matches_finite_differences_of_single_sample():
    model, x, _ = _trained_model()
    x0 = x[0]
    grad = model.f_gradient(x0, model.sampled_weights[0])
    assert grad.shape == x0.shape

    h = 1e-5
    fd = np.zeros_like(x0)
    for j in range(x0.size):
        delta = np.zeros_like(x0)
        delta[j] = h
        f_plus = model.predict_single((x0 + delta)[None, :], sample_index=0)[0, 0]
        f_minus = model.predict_single((x0 - delta)[None, :], sample_index=0)[0, 0]
        fd[j] = (f_plus - f_minus) / (2 * h)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-6)


def test_predictive_mean_gradient_matches_finite_differences():
    model, x, _ = _trained_model()
    x0 = x[1]
    grad = model.predictive_mean_gradient(x0)
    assert grad.shape == x0.shape

    h = 1e-5
    fd = np.zeros_like(x0)
    for j in range(x0.size):
        delta = np.zeros_like(x0)
        delta[j] = h
        mu_plus, _ = model.predict((x0 + delta)[None, :])
        mu_minus, _ = model.predict((x0 - delta)[None, :])
        fd[j] = (mu_plus[0] - mu_minus[0]) / (2 * h)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-6)


def test_predictive_variance_gradient_vanishes_with_a_single_sample():
    model, x, _ = _trained_model()
    del model.sampled_weights[1:]
    x0 = x[2]
    grad = model.predictive_variance_gradient(x0)
    assert grad.shape == x0.shape
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)


def test_train_and_evaluate_returns_one_entry_per_validation_round():
    x, y = _toy_data(n=12, d=3)
    model = Bohamiann()
    n_steps, ll, mse = model.train_and_evaluate(
        x[:8], y[:8], x[8:], y[8:],
        num_steps=20, validate_every_n_steps=5, keep_every=1,
        num_burn_in_steps=10, batch_size=4,
    )
    assert n_steps == [15, 20, 25, 30]
    assert len(ll) == 4
    assert len(mse) == 4
    assert np.all(np.isfinite(ll))
    assert np.all(np.isfinite(mse))
    assert np.all(np.asarray(mse) >= 0)
