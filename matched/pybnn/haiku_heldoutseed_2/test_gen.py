import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def make_data(n=20, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, d))
    y = np.sin(3.0 * x[:, 0]) + 0.1 * rng.randn(n)
    return x, y


@pytest.fixture
def trained_model():
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_data()
    model = Bohamiann()
    model.train(x, y, num_steps=12, keep_every=2, num_burn_in_steps=4, batch_size=5)
    return model


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_init_default_attributes():
    model = Bohamiann()
    assert model.print_every_n_steps == 100
    assert model.metrics == (nn.MSELoss,)
    assert model.do_normalize_input is True
    assert model.do_normalize_output is True
    assert model.get_network is get_default_network
    assert model.is_trained is False
    assert model.use_double_precision is True
    assert model.sampling_method == "adaptive_sghmc"
    assert model.sampled_weights == []
    assert model.likelihood_function is nll
    assert model.sampler is None


def test_init_custom_arguments_are_stored():
    def my_network(input_dimensionality):
        return get_default_network(input_dimensionality)

    def my_likelihood(input, target):
        return nll(input, target)

    model = Bohamiann(
        get_network=my_network,
        normalize_input=False,
        normalize_output=False,
        sampling_method="sgld",
        use_double_precision=False,
        metrics=(nn.L1Loss,),
        likelihood_function=my_likelihood,
        print_every_n_steps=7,
    )
    assert model.get_network is my_network
    assert model.do_normalize_input is False
    assert model.do_normalize_output is False
    assert model.sampling_method == "sgld"
    assert model.use_double_precision is False
    assert model.metrics == (nn.L1Loss,)
    assert model.likelihood_function is my_likelihood
    assert model.print_every_n_steps == 7


def test_instances_do_not_share_sampled_weights():
    a = Bohamiann()
    b = Bohamiann()
    a.sampled_weights.append("sample")
    assert b.sampled_weights == []


# ---------------------------------------------------------------------------
# get_default_network
# ---------------------------------------------------------------------------

def test_default_network_input_dimension_and_hidden_size():
    net = get_default_network(input_dimensionality=3)
    assert isinstance(net, nn.Module)
    assert net.fc1.in_features == 3
    assert net.fc1.out_features == 50
    assert net.fc3.out_features == 1


def test_default_network_forward_output_shape():
    net = get_default_network(input_dimensionality=3)
    out = net(torch.randn(4, 3))
    assert out.shape == (4, 2)


# ---------------------------------------------------------------------------
# nll
# ---------------------------------------------------------------------------

def test_nll_is_zero_for_perfect_prediction_with_unit_variance():
    inp = torch.zeros(5, 2, dtype=torch.float64)  # mean 0, log variance 0
    target = torch.zeros(5, dtype=torch.float64)
    assert nll(inp, target).item() == pytest.approx(0.0, abs=1e-12)


def test_nll_known_value_for_unit_error_unit_variance():
    inp = torch.zeros(3, 2, dtype=torch.float64)
    target = torch.ones(3, dtype=torch.float64)
    # per-sample: 0.5 * 1^2 / 1 + 0.5 * log(1) = 0.5
    assert nll(inp, target).item() == pytest.approx(0.5, rel=1e-8)


def test_nll_matches_reference_formula():
    rng = np.random.RandomState(1)
    mean = rng.randn(6)
    log_var = 0.5 * rng.randn(6)
    target = rng.randn(6)

    inp = torch.tensor(np.stack([mean, log_var], axis=1), dtype=torch.float64)
    tgt = torch.tensor(target, dtype=torch.float64)

    var = np.exp(log_var) + 1e-16
    ll = -((target - mean) ** 2) / (2.0 * var) - 0.5 * log_var
    expected = -np.mean(ll)

    result = nll(inp, tgt)
    assert result.item() == pytest.approx(expected, rel=1e-8)


def test_nll_returns_scalar_tensor():
    inp = torch.randn(4, 2, dtype=torch.float64)
    tgt = torch.randn(4, dtype=torch.float64)
    result = nll(inp, tgt)
    assert result.dim() == 0


def test_nll_increases_with_larger_error():
    tgt = torch.zeros(4, dtype=torch.float64)
    small = torch.tensor([[0.1, 0.0]] * 4, dtype=torch.float64)
    large = torch.tensor([[2.0, 0.0]] * 4, dtype=torch.float64)
    assert nll(large, tgt).item() > nll(small, tgt).item()


# ---------------------------------------------------------------------------
# network_weights property
# ---------------------------------------------------------------------------

def test_network_weights_getter_returns_arrays_per_parameter(trained_model):
    weights = trained_model.network_weights
    params = list(trained_model.model.parameters())
    assert isinstance(weights, tuple)
    assert len(weights) == len(params)
    for w, p in zip(weights, params):
        assert isinstance(w, np.ndarray)
        assert w.shape == tuple(p.shape)


def test_network_weights_setter_roundtrip(trained_model):
    original = trained_model.network_weights
    new_weights = [np.full_like(w, 0.25) for w in original]
    trained_model.network_weights = new_weights

    for w in trained_model.network_weights:
        np.testing.assert_allclose(w, 0.25)

    trained_model.network_weights = original
    for w_new, w_old in zip(trained_model.network_weights, original):
        np.testing.assert_allclose(w_new, w_old)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def test_train_sets_is_trained_and_creates_model_and_sampler():
    torch.manual_seed(0)
    x, y = make_data()
    model = Bohamiann()
    model.train(x, y, num_steps=5, keep_every=2, num_burn_in_steps=2, batch_size=5)
    assert model.is_trained is True
    assert model.model is not None
    assert model.sampler is not None


def test_train_collects_expected_number_of_samples():
    torch.manual_seed(0)
    x, y = make_data()
    model = Bohamiann()
    # Steps 0..9 are executed; samples kept at steps 6 and 8.
    model.train(x, y, num_steps=10, keep_every=2, num_burn_in_steps=4, batch_size=5)
    assert len(model.sampled_weights) == 2
    n_params = len(list(model.model.parameters()))
    for sample in model.sampled_weights:
        assert len(sample) == n_params


@pytest.mark.parametrize("method", ["sgld", "sghmc", "adaptive_sghmc", "preconditioned_sgld"])
def test_train_supports_all_sampling_methods(method):
    torch.manual_seed(0)
    x, y = make_data()
    model = Bohamiann(sampling_method=method)
    model.train(x, y, num_steps=12, keep_every=2, num_burn_in_steps=4, batch_size=5)
    assert model.is_trained is True
    assert len(model.sampled_weights) == 3


def test_train_float32_path_uses_float32_parameters():
    torch.manual_seed(0)
    x, y = make_data()
    model = Bohamiann(use_double_precision=False)
    model.train(x, y, num_steps=8, keep_every=2, num_burn_in_steps=2, batch_size=5)
    assert next(model.model.parameters()).dtype == torch.float32


def test_train_rejects_zero_batch_size():
    x, y = make_data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=2, batch_size=0)


def test_train_rejects_multi_column_targets():
    x, _ = make_data()
    y_bad = np.zeros((x.shape[0], 2))
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y_bad, num_steps=2, batch_size=5)


def test_train_accepts_column_vector_targets():
    torch.manual_seed(0)
    x, y = make_data()
    model = Bohamiann()
    model.train(x, y.reshape(-1, 1), num_steps=4, keep_every=1, num_burn_in_steps=1, batch_size=5)
    assert model.is_trained is True
    np.testing.assert_allclose(model.y, y)


def test_train_warns_and_shrinks_batch_when_too_few_points(caplog):
    torch.manual_seed(0)
    x, y = make_data(n=4)
    model = Bohamiann()
    with caplog.at_level(logging.WARNING):
        model.train(x, y, num_steps=3, keep_every=1, num_burn_in_steps=1, batch_size=50)
    assert "Not enough datapoints" in caplog.text
    assert model.is_trained is True


# ---------------------------------------------------------------------------
# predict / predict_single
# ---------------------------------------------------------------------------

def test_predict_returns_mean_and_positive_variance(trained_model):
    rng = np.random.RandomState(3)
    x_test = rng.uniform(-1, 1, size=(7, 2))
    mean, var = trained_model.predict(x_test)
    assert mean.shape == (7,)
    assert var.shape == (7,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_predict_is_deterministic(trained_model):
    rng = np.random.RandomState(4)
    x_test = rng.uniform(-1, 1, size=(5, 2))
    m1, v1 = trained_model.predict(x_test)
    m2, v2 = trained_model.predict(x_test)
    np.testing.assert_allclose(m1, m2)
    np.testing.assert_allclose(v1, v2)


def test_predict_return_individual_predictions_shape(trained_model):
    rng = np.random.RandomState(5)
    x_test = rng.uniform(-1, 1, size=(6, 2))
    mean, var, individual = trained_model.predict(x_test, return_individual_predictions=True)
    assert individual.shape == (len(trained_model.sampled_weights), 6)
    assert mean.shape == (6,)
    assert var.shape == (6,)


def test_predict_single_returns_output_for_each_test_point(trained_model):
    rng = np.random.RandomState(6)
    x_test = rng.uniform(-1, 1, size=(7, 2))
    out = trained_model.predict_single(x_test, sample_index=0)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 7


def test_predict_single_invalid_index_raises(trained_model):
    x_test = np.zeros((3, 2))
    with pytest.raises(IndexError):
        trained_model.predict_single(x_test, sample_index=len(trained_model.sampled_weights) + 10)


# ---------------------------------------------------------------------------
# train_and_evaluate
# ---------------------------------------------------------------------------

def test_train_and_evaluate_returns_learning_curve():
    torch.manual_seed(0)
    x_train, y_train = make_data(n=20, seed=0)
    x_valid, y_valid = make_data(n=10, seed=1)
    model = Bohamiann()
    n_steps, ll, mse = model.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=6,
        validate_every_n_steps=2,
        keep_every=1,
        num_burn_in_steps=2,
        batch_size=5,
    )
    assert n_steps == [4, 6, 8]
    assert len(ll) == 3
    assert len(mse) == 3
    assert all(np.isfinite(v) for v in ll)
    assert all(v >= 0 for v in mse)


def test_train_and_evaluate_rejects_zero_batch_size():
    x_train, y_train = make_data()
    x_valid, y_valid = make_data(n=5, seed=2)
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train_and_evaluate(x_train, y_train, x_valid, y_valid,
                                 num_steps=4, validate_every_n_steps=2, batch_size=0)


# ---------------------------------------------------------------------------
# normalization helpers
# ---------------------------------------------------------------------------

def test_normalize_input_gives_zero_mean_unit_variance():
    model = Bohamiann()
    rng = np.random.RandomState(7)
    x = rng.uniform(-5, 10, size=(50, 3))
    x_norm, mean, std = model.normalize_input(x)
    np.testing.assert_allclose(x_norm.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(x_norm.std(axis=0), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def test_f_gradient_shape(trained_model):
    x = np.array([0.1, -0.3])
    g = trained_model.f_gradient(x, weights=trained_model.sampled_weights[0])
    assert g.shape == (2,)
    assert np.all(np.isfinite(g))


def test_predictive_mean_gradient_shape(trained_model):
    x = np.array([0.2, 0.4])
    g = trained_model.predictive_mean_gradient(x)
    assert g.shape == (2,)
    assert np.all(np.isfinite(g))


def test_predictive_variance_gradient_shape(trained_model):
    x = np.array([-0.5, 0.6])
    g = trained_model.predictive_variance_gradient(x)
    assert g.shape == (2,)
    assert np.all(np.isfinite(g))
