import logging
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def _regression_data(n=20):
    """Deterministic 1-D regression data (no randomness involved)."""
    x = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = np.sin(3.0 * x[:, 0])
    return x, y


def _forward(model, points):
    """Run the network on float64 points without tracking gradients."""
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(points, dtype=np.float64))).numpy()


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_init_defaults_and_independent_sample_lists():
    b1 = Bohamiann()
    b2 = Bohamiann()

    assert b1.get_network is get_default_network
    assert b1.likelihood_function is nll
    assert b1.metrics == (nn.MSELoss,)
    assert b1.do_normalize_input is True
    assert b1.do_normalize_output is True
    assert b1.use_double_precision is True
    assert b1.sampling_method == "adaptive_sghmc"
    assert b1.print_every_n_steps == 100
    assert b1.is_trained is False
    assert b1.sampler is None
    assert b1.sampled_weights == []
    # Each instance must own its own list of sampled weights.
    assert b1.sampled_weights is not b2.sampled_weights


# ---------------------------------------------------------------------------
# nll
# ---------------------------------------------------------------------------

def test_nll_zero_for_perfect_prediction():
    inp = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)

    result = nll(input=inp, target=target)

    assert result.item() == pytest.approx(0.0, abs=1e-10)


def test_nll_matches_closed_form_value():
    # mean 0, variance 2, target 1 -> squared error 1
    # per-point log-likelihood = -1 * 0.5 / 2 - 0.5 * log(2)
    # nll = 0.25 + 0.5 * log(2)
    inp = torch.tensor([[0.0, math.log(2.0)]], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)

    result = nll(input=inp, target=target)

    expected = 0.25 + 0.5 * math.log(2.0)
    assert result.item() == pytest.approx(expected, rel=1e-8)


def test_nll_averages_over_batch():
    # Row 1: squared error 1, log variance 0 -> log-likelihood -0.5
    # Row 2: perfect prediction, log variance 0 -> log-likelihood 0
    # Average log-likelihood -0.25 -> nll 0.25
    inp = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([0.0, 2.0], dtype=torch.float64)

    result = nll(input=inp, target=target)

    assert result.item() == pytest.approx(0.25, rel=1e-8)


# ---------------------------------------------------------------------------
# get_default_network
# ---------------------------------------------------------------------------

def test_default_network_architecture_and_output_shape():
    torch.manual_seed(0)
    np.random.seed(0)

    net = get_default_network(input_dimensionality=3)

    assert isinstance(net, torch.nn.Module)
    assert net.fc1.in_features == 3
    assert net.fc1.out_features == 50
    assert net.fc2.in_features == 50
    assert net.fc2.out_features == 50
    assert net.fc3.in_features == 50
    assert net.fc3.out_features == 1

    net = net.double()
    out = net(torch.zeros(5, 3, dtype=torch.float64))
    assert tuple(out.shape) == (5, 2)


# ---------------------------------------------------------------------------
# network_weights property
# ---------------------------------------------------------------------------

def test_network_weights_getter_returns_independent_copies():
    torch.manual_seed(0)
    np.random.seed(0)

    b = Bohamiann()
    b.model = get_default_network(input_dimensionality=2).double()
    originals = [p.detach().numpy().copy() for p in b.model.parameters()]

    weights = b.network_weights
    assert len(weights) == len(originals)
    for w, o in zip(weights, originals):
        assert isinstance(w, np.ndarray)
        assert w.shape == o.shape
        assert w.dtype == np.float64
        w[...] = 123.0  # must not write through to the network

    for p, o in zip(b.model.parameters(), originals):
        np.testing.assert_allclose(p.detach().numpy(), o, rtol=0, atol=0)


def test_network_weights_setter_roundtrip():
    torch.manual_seed(0)
    np.random.seed(0)

    b = Bohamiann()
    b.model = get_default_network(input_dimensionality=2).double()
    new_weights = [np.full(tuple(p.shape), 0.5, dtype=np.float64)
                   for p in b.model.parameters()]

    b.network_weights = new_weights

    got = b.network_weights
    assert len(got) == len(new_weights)
    for g, e in zip(got, new_weights):
        np.testing.assert_allclose(g, e, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def test_train_sets_state_and_normalization_statistics():
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _regression_data(20)
    b = Bohamiann()
    b.train(x, y, num_steps=4, num_burn_in_steps=1, keep_every=1, batch_size=5)

    assert b.is_trained is True
    np.testing.assert_allclose(b.X, x)
    assert b.y.shape == (20,)
    np.testing.assert_allclose(np.asarray(b.x_mean), x.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(np.asarray(b.y_mean), y.mean(), atol=1e-10)
    assert np.all(np.asarray(b.x_std) > 0)


def test_train_keeps_expected_number_of_samples():
    torch.manual_seed(0)
    np.random.seed(0)

    # Loop runs steps 0..9. Samples are kept when step > 2 and (step - 2) % 2 == 0,
    # i.e. at steps 4, 6 and 8.
    x, y = _regression_data(20)
    b = Bohamiann()
    b.train(x, y, num_steps=10, num_burn_in_steps=2, keep_every=2, batch_size=5)

    assert len(b.sampled_weights) == 3
    n_params = len(list(b.model.parameters()))
    for sample in b.sampled_weights:
        assert len(sample) == n_params


def test_train_without_continue_resets_samples():
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _regression_data(20)
    b = Bohamiann()
    # Steps 1..5 are kept (5 samples).
    b.train(x, y, num_steps=6, num_burn_in_steps=0, keep_every=1, batch_size=5)
    assert len(b.sampled_weights) == 5

    # Steps 1..2 are kept (2 samples); the old list must be cleared first.
    b.train(x, y, num_steps=3, num_burn_in_steps=0, keep_every=1, batch_size=5)
    assert len(b.sampled_weights) == 2


def test_train_continue_training_appends_and_keeps_model():
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _regression_data(20)
    b = Bohamiann()
    b.train(x, y, num_steps=4, num_burn_in_steps=0, keep_every=2, batch_size=5)
    assert len(b.sampled_weights) == 1

    first_sample = [w.copy() for w in b.sampled_weights[0]]
    model_before = b.model

    b.train(x, y, num_steps=4, num_burn_in_steps=0, keep_every=2, batch_size=5,
            continue_training=True)

    assert len(b.sampled_weights) == 2
    assert b.model is model_before
    for w_new, w_old in zip(b.sampled_weights[0], first_sample):
        np.testing.assert_allclose(w_new, w_old, rtol=0, atol=0)


def test_train_rejects_zero_batch_size():
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _regression_data(20)
    b = Bohamiann()
    with pytest.raises(AssertionError, match="Invalid batch size"):
        b.train(x, y, num_steps=1, batch_size=0)


def test_train_rejects_two_column_targets():
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _regression_data(20)
    y_bad = np.column_stack([y, y])  # shape (N, 2)
    b = Bohamiann()
    with pytest.raises(AssertionError, match="Targets need to be in vector format"):
        b.train(x, y_bad, num_steps=1, batch_size=5)


def test_train_small_dataset_warns_and_uses_all_points(caplog):
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _regression_data(3)
    b = Bohamiann()
    caplog.set_level(logging.WARNING)
    b.train(x, y, num_steps=2, num_burn_in_steps=0, keep_every=1, batch_size=20)

    assert b.is_trained is True
    assert "Not enough datapoints" in caplog.text
    assert b.X.shape == (3, 1)


def test_train_single_precision_produces_float32_weights():
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _regression_data(20)
    b = Bohamiann(use_double_precision=False)
    b.train(x, y, num_steps=3, num_burn_in_steps=0, keep_every=1, batch_size=5)

    assert next(b.model.parameters()).dtype == torch.float32
    assert len(b.sampled_weights) > 0
    for w in b.sampled_weights[0]:
        assert w.dtype == np.float32


def test_train_verbose_prints_progress(capsys):
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _regression_data(20)
    b = Bohamiann(print_every_n_steps=1)
    b.train(x, y, num_steps=4, num_burn_in_steps=1, keep_every=1, batch_size=5,
            verbose=True)

    out = capsys.readouterr().out
    assert "Step" in out
    # At step 3 exactly one sample has been kept, so the post-burn-in line appears.
    assert "Samples=" in out


# ---------------------------------------------------------------------------
# predict / predict_single
# ---------------------------------------------------------------------------

def test_predict_single_sample_matches_direct_forward_pass():
    torch.manual_seed(0)
    np.random.seed(0)

    b = Bohamiann(normalize_input=False, normalize_output=False)
    b.model = get_default_network(input_dimensionality=2).double()
    b.sampled_weights = [b.network_weights]

    x = np.array([[0.1, -0.2], [0.5, 0.3], [-1.0, 0.7]])
    mu, var = b.predict(x)
    out = _forward(b.model, x)

    assert mu.shape == (3,)
    assert var.shape == (3,)
    np.testing.assert_allclose(mu, out[:, 0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(var, np.exp(out[:, 1]), rtol=1e-10, atol=1e-10)


def test_predict_identical_samples_have_no_disagreement_variance():
    torch.manual_seed(0)
    np.random.seed(0)

    b = Bohamiann(normalize_input=False, normalize_output=False)
    b.model = get_default_network(input_dimensionality=2).double()
    w = b.network_weights
    b.sampled_weights = [w, w]

    x = np.array([[0.2, 0.4], [-0.3, 0.9]])
    mu, var = b.predict(x)
    out = _forward(b.model, x)

    # With identical samples the between-sample term vanishes, leaving exp(log_var).
    np.testing.assert_allclose(mu, out[:, 0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(var, np.exp(out[:, 1]), rtol=1e-10, atol=1e-10)


def test_predict_individual_predictions_are_consistent():
    torch.manual_seed(0)
    np.random.seed(0)

    b = Bohamiann(normalize_input=False, normalize_output=False)
    weight_sets = []
    for _ in range(4):
        b.model = get_default_network(input_dimensionality=1).double()
        weight_sets.append(b.network_weights)
    b.sampled_weights = weight_sets

    x = np.array([[0.0], [0.5], [-0.5]])
    mu, var, individual = b.predict(x, return_individual_predictions=True)

    assert individual.shape == (4, 3)
    np.testing.assert_allclose(mu, individual.mean(axis=0), rtol=1e-10, atol=1e-10)
    # Predictive variance is a sum of a non-negative term and exp(...) > 0.
    assert np.all(var > 0)

    single = b.predict_single(x, sample_index=1)
    np.testing.assert_allclose(single[:, 0], individual[1], rtol=1e-10, atol=1e-10)


def test_predict_denormalizes_outputs():
    torch.manual_seed(0)
    np.random.seed(0)

    y_mean, y_std = 3.0, 2.0
    b = Bohamiann(normalize_input=False, normalize_output=True)
    b.y_mean = y_mean
    b.y_std = y_std
    b.model = get_default_network(input_dimensionality=1).double()
    b.sampled_weights = [b.network_weights]

    x = np.array([[0.2], [-0.5]])
    mu, var = b.predict(x)
    out = _forward(b.model, x)

    np.testing.assert_allclose(mu, out[:, 0] * y_std + y_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(var, np.exp(out[:, 1]) * y_std ** 2, rtol=1e-10, atol=1e-10)

    single = b.predict_single(x, sample_index=0)
    np.testing.assert_allclose(single[:, 0], out[:, 0] * y_std + y_mean,
                               rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def test_f_gradient_matches_finite_differences():
    torch.manual_seed(0)
    np.random.seed(0)

    b = Bohamiann(normalize_input=False, normalize_output=False)
    b.model = get_default_network(input_dimensionality=2).double()
    w = b.network_weights

    x = np.array([0.3, -0.4])
    g = b.f_gradient(x, weights=w)

    eps = 1e-6
    fd = np.zeros(2)
    for i in range(2):
        e = np.zeros(2)
        e[i] = eps
        f_plus = _forward(b.model, (x + e)[None, :])[0, 0]
        f_minus = _forward(b.model, (x - e)[None, :])[0, 0]
        fd[i] = (f_plus - f_minus) / (2 * eps)

    assert g.shape == (2,)
    np.testing.assert_allclose(g, fd, rtol=0, atol=1e-6)


def test_predictive_variance_gradient_is_zero_for_single_sample():
    torch.manual_seed(0)
    np.random.seed(0)

    b = Bohamiann(normalize_input=False, normalize_output=False)
    b.model = get_default_network(input_dimensionality=2).double()
    b.sampled_weights = [b.network_weights]

    x = np.array([0.1, 0.2])
    g = b.predictive_variance_gradient(x)

    assert g.shape == (2,)
    # With one sample the ensemble has no disagreement, so the gradient vanishes.
    np.testing.assert_allclose(g, 0.0, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# train_and_evaluate
# ---------------------------------------------------------------------------

def test_train_and_evaluate_returns_learning_curve():
    torch.manual_seed(0)
    np.random.seed(0)

    x, y = _regression_data(20)
    x_valid = np.linspace(-0.9, 0.9, 5).reshape(-1, 1)
    y_valid = np.sin(3.0 * x_valid[:, 0])

    b = Bohamiann()
    n_steps, lls, mses = b.train_and_evaluate(
        x, y, x_valid, y_valid,
        num_steps=4,
        validate_every_n_steps=2,
        keep_every=1,
        num_burn_in_steps=2,
        batch_size=5,
    )

    # Burn-in of 2 steps, then 4 // 2 = 2 validation rounds of 2 steps each.
    assert n_steps == [4, 6]
    assert len(lls) == 2
    assert len(mses) == 2
    assert all(np.isfinite(v) for v in lls)
    assert all(v >= 0 for v in mses)
    assert b.is_trained is True
