import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from pybnn.bohamiann import Bohamiann, get_default_network, nll
from pybnn.sampler import AdaptiveSGHMC, PreconditionedSGLD, SGHMC, SGLD


@pytest.fixture(autouse=True)
def seed_everything():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    x = rng.uniform(-1, 1, size=(20, 2))
    y = np.sin(3 * x[:, 0]) + 0.1 * rng.randn(20)
    return x, y


def small_train(model, x, y, **kwargs):
    """Run a short training run: 5 steps, burn-in 2, keep every step -> 2 samples."""
    params = dict(num_steps=5, keep_every=1, num_burn_in_steps=2, batch_size=10)
    params.update(kwargs)
    model.train(x, y, **params)
    return model


def test_default_constructor_attributes():
    b = Bohamiann()
    assert b.print_every_n_steps == 100
    assert b.metrics == (nn.MSELoss,)
    assert b.do_normalize_input is True
    assert b.do_normalize_output is True
    assert b.get_network is get_default_network
    assert b.is_trained is False
    assert b.use_double_precision is True
    assert b.sampling_method == "adaptive_sghmc"
    assert b.sampled_weights == []
    assert b.likelihood_function is nll
    assert b.sampler is None


def test_custom_constructor_arguments_are_stored():
    def custom_likelihood(input, target):
        return nll(input=input, target=target)

    b = Bohamiann(
        get_network=get_default_network,
        normalize_input=False,
        normalize_output=False,
        sampling_method="sgld",
        use_double_precision=False,
        metrics=(),
        likelihood_function=custom_likelihood,
        print_every_n_steps=7,
    )
    assert b.do_normalize_input is False
    assert b.do_normalize_output is False
    assert b.sampling_method == "sgld"
    assert b.use_double_precision is False
    assert b.metrics == ()
    assert b.likelihood_function is custom_likelihood
    assert b.print_every_n_steps == 7


def test_get_default_network_output_shape():
    net = get_default_network(3)
    assert isinstance(net, nn.Module)
    out = net(torch.zeros(4, 3))
    assert tuple(out.shape) == (4, 2)


def test_nll_closed_form_single_point():
    # mean = 1.0, log variance = log(4), target = 2.0
    # NLL = 0.5 * (diff^2) / var + 0.5 * log(var) = 0.5 * 1 / 4 + 0.5 * log(4)
    inp = torch.tensor([[1.0, float(np.log(4.0))]], dtype=torch.float64)
    target = torch.tensor([2.0], dtype=torch.float64)
    value = nll(input=inp, target=target)
    expected = 0.5 * 1.0 / 4.0 + 0.5 * np.log(4.0)
    assert float(value) == pytest.approx(expected, rel=1e-9)


def test_nll_is_average_over_batch():
    inp = torch.tensor([[0.5, float(np.log(2.0))],
                        [1.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 0.0], dtype=torch.float64)
    value = nll(input=inp, target=target)
    row1 = 0.5 * 0.25 / 2.0 + 0.5 * np.log(2.0)
    row2 = 0.5 * 1.0 / 1.0 + 0.5 * 0.0
    assert float(value) == pytest.approx(np.mean([row1, row2]), rel=1e-9)


def test_network_weights_getter_requires_model():
    b = Bohamiann()
    with pytest.raises(AttributeError):
        _ = b.network_weights


def test_network_weights_setter_roundtrip():
    b = Bohamiann()
    b.model = get_default_network(2).double()
    weights = [np.full(p.detach().numpy().shape, 0.25) for p in b.model.parameters()]
    b.network_weights = weights
    got = b.network_weights
    assert len(got) == len(weights)
    for g, w in zip(got, weights):
        np.testing.assert_allclose(g, w)


def test_train_collects_expected_number_of_samples(data):
    x, y = data
    b = small_train(Bohamiann(), x, y)
    assert b.is_trained is True
    # steps 0..4, burn-in 2, keep_every 1 -> steps 3 and 4 are kept
    assert len(b.sampled_weights) == 2
    assert len(b.sampled_weights[0]) == len(list(b.model.parameters()))


def test_train_resets_samples_without_continue(data):
    x, y = data
    b = small_train(Bohamiann(), x, y)
    assert len(b.sampled_weights) == 2
    small_train(b, x, y, continue_training=False)
    assert len(b.sampled_weights) == 2


def test_train_continue_training_appends_samples(data):
    x, y = data
    b = small_train(Bohamiann(), x, y)
    assert len(b.sampled_weights) == 2
    b.train(x, y, num_steps=3, keep_every=1, num_burn_in_steps=0,
            batch_size=10, continue_training=True)
    # steps 0..2 with burn-in 0 -> steps 1 and 2 kept
    assert len(b.sampled_weights) == 4


def test_train_invalid_batch_size_raises(data):
    x, y = data
    b = Bohamiann()
    with pytest.raises(AssertionError):
        b.train(x, y, num_steps=2, batch_size=0)


def test_train_rejects_multi_column_targets(data):
    x, y = data
    y_two_cols = np.stack([y, y], axis=1)
    b = Bohamiann()
    with pytest.raises(AssertionError):
        b.train(x, y_two_cols, num_steps=2)


def test_train_accepts_column_vector_targets(data):
    x, y = data
    b = small_train(Bohamiann(), x, y[:, None])
    assert b.is_trained is True
    assert len(b.sampled_weights) == 2


def test_train_warns_when_batch_exceeds_dataset(data, caplog):
    x, y = data
    b = Bohamiann()
    with caplog.at_level(logging.WARNING):
        b.train(x[:5], y[:5], num_steps=3, keep_every=1,
                num_burn_in_steps=0, batch_size=20)
    messages = [r.getMessage() for r in caplog.records]
    assert any("Not enough datapoints" in m for m in messages)
    assert b.is_trained is True


@pytest.mark.parametrize(
    "method, expected_cls",
    [
        ("adaptive_sghmc", AdaptiveSGHMC),
        ("sgld", SGLD),
        ("sghmc", SGHMC),
        ("preconditioned_sgld", PreconditionedSGLD),
    ],
)
def test_train_selects_sampler(method, expected_cls, data):
    x, y = data
    b = small_train(Bohamiann(sampling_method=method), x, y)
    assert isinstance(b.sampler, expected_cls)
    assert len(b.sampled_weights) == 2


def test_train_float32_mode(data):
    x, y = data
    b = small_train(Bohamiann(use_double_precision=False), x, y)
    assert next(b.model.parameters()).dtype == torch.float32
    mean, var = b.predict(x)
    assert mean.shape == (20,)
    assert var.shape == (20,)
    assert np.all(np.isfinite(mean))


def test_train_uses_custom_likelihood_function(data):
    x, y = data
    calls = []

    def likelihood(input, target):
        calls.append(1)
        return nll(input=input, target=target)

    b = Bohamiann(likelihood_function=likelihood)
    small_train(b, x, y, num_steps=4, num_burn_in_steps=1)
    assert len(calls) == 4


def test_train_verbose_prints_progress(data, capsys):
    x, y = data
    b = Bohamiann(print_every_n_steps=1)
    small_train(b, x, y, num_steps=3, num_burn_in_steps=0, verbose=True)
    out = capsys.readouterr().out
    nll_lines = [line for line in out.splitlines() if "NLL" in line]
    assert len(nll_lines) == 2  # steps 1 and 2
    assert "Samples=" in out


def test_normalization_statistics_are_set(data):
    x, y = data
    b = small_train(Bohamiann(), x, y)
    assert b.x_mean.shape == (2,)
    assert b.x_std.shape == (2,)
    np.testing.assert_allclose(b.x_mean, x.mean(axis=0), rtol=1e-6, atol=1e-8)
    assert hasattr(b, "y_mean") and hasattr(b, "y_std")

    b_raw = small_train(Bohamiann(normalize_input=False, normalize_output=False), x, y)
    assert not hasattr(b_raw, "x_mean")
    assert not hasattr(b_raw, "y_mean")


def test_predict_shapes_and_positive_variance(data):
    x, y = data
    b = small_train(Bohamiann(), x, y)
    x_test = np.random.RandomState(1).uniform(-1, 1, size=(7, 2))
    mean, var = b.predict(x_test)
    assert mean.shape == (7,)
    assert var.shape == (7,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_predict_return_individual_predictions(data):
    x, y = data
    b = small_train(Bohamiann(), x, y)
    x_test = np.random.RandomState(1).uniform(-1, 1, size=(7, 2))
    mean, var, individual = b.predict(x_test, return_individual_predictions=True)
    assert individual.shape == (2, 7)  # two sampled networks, seven test points
    np.testing.assert_allclose(individual.mean(axis=0), mean, rtol=1e-6, atol=1e-8)


def test_predict_single_mean_consistent_with_predict(data):
    x, y = data
    # exactly one sample: num_steps=3, burn-in=2, keep_every=1 -> step 2 kept
    b = small_train(Bohamiann(), x, y, num_steps=3)
    assert len(b.sampled_weights) == 1
    x_test = np.random.RandomState(1).uniform(-1, 1, size=(7, 2))
    mean, _ = b.predict(x_test)
    single = b.predict_single(x_test, sample_index=0)
    assert single.shape == (7, 2)
    np.testing.assert_allclose(single[:, 0], mean, rtol=1e-5, atol=1e-6)


def test_f_gradient_matches_finite_difference(data):
    x, y = data
    b = small_train(Bohamiann(normalize_input=False, normalize_output=False), x, y, num_steps=3)
    weights = b.sampled_weights[0]
    x0 = np.array([0.3, -0.2])

    grad = b.f_gradient(x0, weights)
    assert grad.shape == (2,)

    eps = 1e-6
    fd = np.zeros(2)
    for i in range(2):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = b.predict_single(x_plus[None, :], sample_index=0)[0, 0]
        f_minus = b.predict_single(x_minus[None, :], sample_index=0)[0, 0]
        fd[i] = (f_plus - f_minus) / (2 * eps)

    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-6)


def test_predictive_gradients_shapes(data):
    x, y = data
    b = small_train(Bohamiann(), x, y)
    x0 = np.array([0.1, 0.4])
    mean_grad = b.predictive_mean_gradient(x0)
    var_grad = b.predictive_variance_gradient(x0)
    assert mean_grad.shape == (2,)
    assert var_grad.shape == (2,)
    assert np.all(np.isfinite(mean_grad))
    assert np.all(np.isfinite(var_grad))


def test_train_and_evaluate_learning_curve(data):
    x, y = data
    b = Bohamiann()
    x_valid, y_valid = x[:8], y[:8]
    n_steps, ll, mse = b.train_and_evaluate(
        x, y, x_valid, y_valid,
        num_steps=4,
        validate_every_n_steps=2,
        keep_every=1,
        num_burn_in_steps=2,
        batch_size=10,
    )
    assert n_steps == [4, 6]
    assert len(ll) == 2
    assert len(mse) == 2
    assert np.all(np.isfinite(ll))
    assert np.all(np.asarray(mse) >= 0)
