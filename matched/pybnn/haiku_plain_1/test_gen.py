import logging

import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll
from pybnn.sampler import AdaptiveSGHMC, PreconditionedSGLD, SGHMC, SGLD

# With num_steps=30, burn-in=10 and keep_every=5 the kept steps are 15, 20, 25.
TRAIN_KW = dict(num_steps=30, num_burn_in_steps=10, keep_every=5, batch_size=5)
EXPECTED_SAMPLES = 3

# Parameter shapes of get_default_network(input_dimensionality=3), in registration order:
# fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias, log_std.log_var
ARCH_SHAPES_D3 = [(50, 3), (50,), (50, 50), (50,), (1, 50), (1,), (1, 1)]


def _seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _data(n=20, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, d))
    y = np.sin(3.0 * x[:, 0]) + 0.5 * x[:, 1] + 0.05 * rng.randn(n)
    return x, y


def _trained_model(sampling_method="adaptive_sghmc", seed=0):
    _seed(seed)
    x, y = _data()
    model = Bohamiann(sampling_method=sampling_method)
    model.train(x, y, **TRAIN_KW)
    return model


def test_constructor_defaults_and_overrides():
    default = Bohamiann()
    assert default.print_every_n_steps == 100
    assert default.do_normalize_input is True
    assert default.do_normalize_output is True
    assert default.use_double_precision is True
    assert default.sampling_method == "adaptive_sghmc"
    assert default.get_network is get_default_network
    assert default.likelihood_function is nll
    assert default.sampled_weights == []
    assert default.is_trained is False
    assert default.sampler is None

    custom = Bohamiann(normalize_input=False, normalize_output=False,
                       sampling_method="sgld", use_double_precision=False,
                       print_every_n_steps=7)
    assert custom.do_normalize_input is False
    assert custom.do_normalize_output is False
    assert custom.use_double_precision is False
    assert custom.sampling_method == "sgld"
    assert custom.print_every_n_steps == 7


def test_nll_matches_closed_form_gaussian_likelihood():
    # Perfect mean and unit variance gives zero NLL.
    prediction = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert float(nll(prediction, target)) == pytest.approx(0.0, abs=1e-9)

    # Squared error 1 per point with unit variance gives 0.5 * (1 + 0) / 1 = 0.5.
    prediction = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    assert float(nll(prediction, target)) == pytest.approx(0.5, abs=1e-9)

    # Zero error with variance 3 gives 0.5 * log(3).
    prediction = torch.tensor([[1.0, np.log(3.0)]], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)
    assert float(nll(prediction, target)) == pytest.approx(0.5 * np.log(3.0), rel=1e-9)


def test_default_network_output_shape_and_constant_log_variance():
    net = get_default_network(input_dimensionality=3).double()
    x = torch.randn(6, 3, dtype=torch.float64)
    out = net(x)
    assert out.shape == (6, 2)
    expected_log_var = torch.full((6,), float(np.log(1e-3)), dtype=torch.float64)
    assert torch.allclose(out[:, 1], expected_log_var)


def test_network_weights_match_architecture_shapes():
    model = Bohamiann()
    model.model = get_default_network(input_dimensionality=3).double()
    weights = model.network_weights
    assert isinstance(weights, tuple)
    assert [w.shape for w in weights] == ARCH_SHAPES_D3
    assert all(w.dtype == np.float64 for w in weights)


def test_network_weights_getter_returns_detached_copies():
    model = Bohamiann()
    model.model = get_default_network(input_dimensionality=2).double()
    snapshot = model.network_weights
    original = [w.copy() for w in snapshot]
    with torch.no_grad():
        for parameter in model.model.parameters():
            parameter.add_(1.0)
    for kept, orig in zip(snapshot, original):
        assert np.array_equal(kept, orig)


def test_network_weights_setter_round_trip():
    model = Bohamiann()
    model.model = get_default_network(input_dimensionality=2).double()
    new_weights = [np.full_like(w, 0.25) for w in model.network_weights]
    model.network_weights = new_weights
    for got, want in zip(model.network_weights, new_weights):
        np.testing.assert_allclose(got, want)


def test_train_collects_expected_number_of_weight_samples():
    _seed()
    x, y = _data()
    model = Bohamiann()
    model.train(x, y, **TRAIN_KW)
    assert model.is_trained is True
    assert len(model.sampled_weights) == EXPECTED_SAMPLES
    assert isinstance(model.sampler, AdaptiveSGHMC)
    for sample in model.sampled_weights:
        assert len(sample) == len(ARCH_SHAPES_D3)
        assert all(np.all(np.isfinite(w)) for w in sample)


def test_train_rejects_invalid_batch_size_and_target_shapes():
    x, y = _data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=5, num_burn_in_steps=1, keep_every=1, batch_size=0)
    with pytest.raises(AssertionError):
        model.train(x, np.stack([y, y], axis=1), num_steps=5, num_burn_in_steps=1, keep_every=1)


def test_train_accepts_column_targets_and_stores_normalization_stats():
    _seed()
    x, y = _data()
    model = Bohamiann()
    model.train(x, y[:, None], **TRAIN_KW)
    assert model.y.shape == (20,)
    np.testing.assert_allclose(model.y, y)
    np.testing.assert_allclose(model.x_mean, x.mean(axis=0))
    np.testing.assert_allclose(model.x_std, x.std(axis=0))
    np.testing.assert_allclose(model.y_mean, y.mean())
    np.testing.assert_allclose(model.y_std, y.std())


def test_train_clamps_batch_size_and_logs_warning(caplog):
    _seed()
    x, y = _data(n=4)
    model = Bohamiann()
    with caplog.at_level(logging.WARNING):
        model.train(x, y, num_steps=20, num_burn_in_steps=5, keep_every=5, batch_size=10)
    assert any("Not enough datapoints" in record.getMessage() for record in caplog.records)
    assert model.is_trained is True
    # Kept steps with burn-in 5 and keep_every 5 within 20 steps are 10 and 15.
    assert len(model.sampled_weights) == 2


def test_train_continue_training_appends_samples():
    _seed()
    x, y = _data()
    model = Bohamiann()
    model.train(x, y, **TRAIN_KW)
    first_network = model.model
    first_sampler = model.sampler
    model.train(x, y, continue_training=True, **TRAIN_KW)
    assert len(model.sampled_weights) == 2 * EXPECTED_SAMPLES
    assert model.model is first_network
    assert model.sampler is first_sampler


def test_train_without_continue_resets_model_and_samples():
    _seed()
    x, y = _data()
    model = Bohamiann()
    model.train(x, y, **TRAIN_KW)
    old_network = model.model
    old_sampler = model.sampler
    model.train(x, y, **TRAIN_KW)
    assert len(model.sampled_weights) == EXPECTED_SAMPLES
    assert model.model is not old_network
    assert model.sampler is not old_sampler


def test_train_supports_every_sampling_method():
    expected = {
        "adaptive_sghmc": AdaptiveSGHMC,
        "sgld": SGLD,
        "sghmc": SGHMC,
        "preconditioned_sgld": PreconditionedSGLD,
    }
    x, y = _data()
    for method, sampler_cls in expected.items():
        _seed()
        model = Bohamiann(sampling_method=method)
        model.train(x, y, **TRAIN_KW)
        assert isinstance(model.sampler, sampler_cls), method
        assert len(model.sampled_weights) == EXPECTED_SAMPLES, method


def test_train_uses_injected_network_factory_and_likelihood():
    _seed()
    x, y = _data()
    seen_dims = []
    likelihood_calls = []

    def factory(input_dimensionality):
        seen_dims.append(input_dimensionality)
        return get_default_network(input_dimensionality)

    def likelihood(input, target):
        likelihood_calls.append(1)
        return nll(input=input, target=target)

    model = Bohamiann(get_network=factory, likelihood_function=likelihood)
    model.train(x, y, **TRAIN_KW)
    assert seen_dims == [x.shape[1]]
    # One likelihood evaluation per sampling step.
    assert len(likelihood_calls) == TRAIN_KW["num_steps"]


def test_train_verbose_prints_progress_lines(capsys):
    _seed()
    x, y = _data()
    model = Bohamiann(print_every_n_steps=2)
    model.train(x, y, verbose=True, **TRAIN_KW)
    lines = capsys.readouterr().out.strip().splitlines()
    step_lines = [line for line in lines if line.startswith("Step")]
    # Even steps in 2..28 except step 10 (neither burn-in nor post-burn-in branch) are printed.
    assert len(step_lines) == 13
    assert all("NLL" in line and "MSE" in line for line in step_lines)


def test_predict_returns_mean_and_variance_per_point():
    model = _trained_model()
    x_test = np.random.RandomState(1).uniform(-1.0, 1.0, size=(7, 2))
    mean, var = model.predict(x_test)
    assert mean.shape == (7,)
    assert var.shape == (7,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_predict_mean_equals_average_of_individual_predictions():
    model = _trained_model()
    x_test = np.random.RandomState(2).uniform(-1.0, 1.0, size=(7, 2))
    mean, var, individual = model.predict(x_test, return_individual_predictions=True)
    assert individual.shape == (EXPECTED_SAMPLES, 7)
    np.testing.assert_allclose(mean, individual.mean(axis=0), rtol=1e-6, atol=1e-8)
    assert np.all(var >= 0)


def test_predict_single_matches_individual_prediction():
    model = _trained_model()
    x_test = np.random.RandomState(3).uniform(-1.0, 1.0, size=(7, 2))
    _, _, individual = model.predict(x_test, return_individual_predictions=True)
    for i in range(EXPECTED_SAMPLES):
        single = model.predict_single(x_test, sample_index=i)
        assert single.shape == (7, 2)
        np.testing.assert_allclose(single[:, 0], individual[i], rtol=1e-6, atol=1e-8)


def test_train_and_evaluate_reports_learning_curve_per_validation_round():
    _seed()
    x, y = _data(n=20)
    model = Bohamiann()
    n_steps, lls, mses = model.train_and_evaluate(
        x[:15], y[:15], x[15:], y[15:],
        num_steps=20, validate_every_n_steps=5, keep_every=1,
        num_burn_in_steps=5, batch_size=5)
    assert n_steps == [10, 15, 20, 25]
    assert len(lls) == len(mses) == 4
    assert np.all(np.isfinite(lls))
    assert np.all(np.asarray(mses) >= 0)


def test_train_and_evaluate_rejects_zero_batch_size():
    x, y = _data(n=20)
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train_and_evaluate(x[:15], y[:15], x[15:], y[15:],
                                 num_steps=10, validate_every_n_steps=5,
                                 num_burn_in_steps=5, batch_size=0)


def test_normalize_helpers_zero_mean_unit_variance():
    model = Bohamiann()
    x = np.random.RandomState(4).normal(5.0, 2.0, size=(50, 3))
    xn, m, s = model.normalize_input(x)
    np.testing.assert_allclose(xn.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(xn.std(axis=0), 1.0, atol=1e-10)

    # Supplied statistics are reused rather than recomputed.
    xn2, m2, s2 = model.normalize_output(x[:5], m, s)
    np.testing.assert_allclose(xn2, (x[:5] - m) / s)
    assert m2 is m and s2 is s


def test_predictive_mean_gradient_matches_finite_differences():
    model = _trained_model()
    x0 = np.array([0.3, -0.2])
    analytic = model.predictive_mean_gradient(x0)
    eps = 1e-6
    numeric = np.zeros(2)
    for d in range(2):
        up = x0.copy()
        up[d] += eps
        down = x0.copy()
        down[d] -= eps
        numeric[d] = (model.predict(up[None, :])[0][0] - model.predict(down[None, :])[0][0]) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


def test_f_gradient_and_predictive_variance_gradient_shapes():
    model = _trained_model()
    x0 = np.array([0.1, 0.4])
    g_single = model.f_gradient(x0, weights=model.sampled_weights[0])
    assert g_single.shape == (2,)
    assert np.all(np.isfinite(g_single))
    g_var = model.predictive_variance_gradient(x0)
    assert g_var.shape == (2,)
    assert np.all(np.isfinite(g_var))
