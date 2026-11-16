import numpy as np
import pytest
import torch
import torch.nn as nn

from pybnn.bohamiann import Bohamiann, get_default_network, nll


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    x = rng.uniform(-1.0, 1.0, size=(30, 2))
    y = np.sin(3.0 * x[:, 0]) + 0.5 * x[:, 1] + 0.05 * rng.randn(30)
    return x, y


@pytest.fixture
def trained(data):
    x, y = data
    model = Bohamiann()
    model.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=8)
    return model


def test_default_constructor_attributes():
    bnn = Bohamiann()
    assert bnn.get_network is get_default_network
    assert bnn.metrics == (nn.MSELoss,)
    assert bnn.likelihood_function is nll
    assert bnn.do_normalize_input is True
    assert bnn.do_normalize_output is True
    assert bnn.sampling_method == "adaptive_sghmc"
    assert bnn.use_double_precision is True
    assert bnn.print_every_n_steps == 100
    assert bnn.is_trained is False
    assert bnn.sampled_weights == []
    assert bnn.sampler is None


def test_constructor_stores_custom_arguments():
    def my_network(input_dimensionality):
        return get_default_network(input_dimensionality)

    def my_likelihood(input, target):
        return nll(input, target)

    bnn = Bohamiann(
        get_network=my_network,
        normalize_input=False,
        normalize_output=False,
        sampling_method="sgld",
        use_double_precision=False,
        metrics=(),
        likelihood_function=my_likelihood,
        print_every_n_steps=7,
    )
    assert bnn.get_network is my_network
    assert bnn.do_normalize_input is False
    assert bnn.do_normalize_output is False
    assert bnn.sampling_method == "sgld"
    assert bnn.use_double_precision is False
    assert bnn.metrics == ()
    assert bnn.likelihood_function is my_likelihood
    assert bnn.print_every_n_steps == 7


def test_nll_matches_manual_computation():
    inp = torch.tensor([[0.5, 0.0], [-1.0, 1.0]], dtype=torch.float64, requires_grad=True)
    tgt = torch.tensor([1.0, 0.0], dtype=torch.float64)

    result = nll(inp, tgt)

    mu = inp[:, 0].detach().numpy()
    log_var = inp[:, 1].detach().numpy()
    t = tgt.numpy()
    var_inv = 1.0 / (np.exp(log_var) + 1e-16)
    per_sample_ll = -((t - mu) ** 2) * 0.5 * var_inv - 0.5 * log_var
    expected = -per_sample_ll.mean()

    assert result.dim() == 0
    assert result.requires_grad
    assert result.item() == pytest.approx(expected)


def test_nll_batch_equals_mean_of_single_sample_losses():
    inp = torch.tensor([[0.2, -0.5], [1.5, 0.3], [-0.7, 1.0]], dtype=torch.float64)
    tgt = torch.tensor([0.1, 1.4, -0.6], dtype=torch.float64)

    batch_value = nll(inp, tgt).item()
    single_values = [nll(inp[i:i + 1], tgt[i:i + 1]).item() for i in range(3)]

    assert batch_value == pytest.approx(np.mean(single_values))


def test_default_network_output_and_first_layer_shapes():
    net = get_default_network(3)
    out = net(torch.zeros(5, 3, dtype=torch.float32))
    assert out.shape == (5, 2)

    first_param = list(net.parameters())[0]
    assert first_param.shape == (50, 3)


def test_network_weights_getter_returns_arrays_matching_parameters():
    bnn = Bohamiann()
    model = get_default_network(2).double()
    bnn.model = model

    weights = bnn.network_weights
    params = list(model.parameters())

    assert len(weights) == len(params)
    for w, p in zip(weights, params):
        assert isinstance(w, np.ndarray)
        assert w.shape == tuple(p.shape)


def test_network_weights_getter_returns_copies():
    bnn = Bohamiann()
    model = get_default_network(2).double()
    bnn.model = model

    weights = bnn.network_weights
    for arr in weights:
        arr[...] = 123.0

    for p in model.parameters():
        assert not np.any(p.detach().numpy() == 123.0)


def test_network_weights_setter_roundtrip():
    bnn = Bohamiann()
    model = get_default_network(2).double()
    bnn.model = model

    new_weights = [np.full(p.shape, 0.25, dtype=np.float64) for p in model.parameters()]
    bnn.network_weights = new_weights

    got = bnn.network_weights
    assert len(got) == len(new_weights)
    for g, n in zip(got, new_weights):
        np.testing.assert_allclose(g, n)


def test_normalize_input_zero_mean_and_consistent_statistics():
    x = np.random.RandomState(1).normal(loc=5.0, scale=3.0, size=(50, 2))
    bnn = Bohamiann()

    out, m, s = bnn.normalize_input(x)

    np.testing.assert_allclose(m, x.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(out, (x - m) / s, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-8)


def test_train_sets_flag_and_collects_expected_samples(data):
    x, y = data
    bnn = Bohamiann()
    bnn.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=8)

    assert bnn.is_trained is True
    assert bnn.sampler is not None
    # Steps 0..19 run; samples kept at steps 10 and 15 (step > burn-in and divisible by keep_every)
    assert len(bnn.sampled_weights) == 2

    n_params = len(list(bnn.model.parameters()))
    for sample in bnn.sampled_weights:
        assert len(sample) == n_params


def test_train_rejects_zero_batch_size(data):
    x, y = data
    bnn = Bohamiann()
    with pytest.raises(AssertionError):
        bnn.train(x, y, num_steps=5, num_burn_in_steps=0, batch_size=0)


def test_train_rejects_invalid_target_shape(data):
    x, y = data
    bnn = Bohamiann()
    bad_targets = np.stack([y, y], axis=1)  # shape (N, 2)
    with pytest.raises(AssertionError):
        bnn.train(x, bad_targets, num_steps=5, num_burn_in_steps=0, batch_size=8)


def test_train_restarts_sample_list_without_continue(data):
    x, y = data
    bnn = Bohamiann()
    kwargs = dict(num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=8)

    bnn.train(x, y, **kwargs)
    first_count = len(bnn.sampled_weights)
    bnn.train(x, y, **kwargs)

    assert first_count == 2
    assert len(bnn.sampled_weights) == first_count


def test_train_continue_training_accumulates_samples(data):
    x, y = data
    bnn = Bohamiann()
    bnn.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=8)
    assert len(bnn.sampled_weights) == 2

    # Steps 0..19 without burn-in: samples kept at steps 5, 10, 15
    bnn.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=0, batch_size=8,
              continue_training=True)
    assert len(bnn.sampled_weights) == 5


@pytest.mark.parametrize("sampling_method",
                         ["adaptive_sghmc", "sgld", "sghmc", "preconditioned_sgld"])
def test_train_supports_all_sampling_methods(data, sampling_method):
    x, y = data
    bnn = Bohamiann(sampling_method=sampling_method)
    bnn.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=8)

    assert bnn.is_trained is True
    assert bnn.sampler is not None
    assert len(bnn.sampled_weights) == 2


def test_train_single_precision_produces_float32_weights(data):
    x, y = data
    bnn = Bohamiann(use_double_precision=False)
    bnn.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=8)

    for w in bnn.network_weights:
        assert w.dtype == np.float32


def test_train_verbose_prints_progress(data, capsys):
    x, y = data
    bnn = Bohamiann(print_every_n_steps=2)
    bnn.train(x, y, num_steps=10, keep_every=2, num_burn_in_steps=2, batch_size=8, verbose=True)

    captured = capsys.readouterr()
    assert "NLL" in captured.out
    assert "MSE" in captured.out


def test_predict_output_shapes_and_individual_predictions(trained, data):
    x, _ = data
    x_test = x[:6]

    mean, var = trained.predict(x_test)
    assert mean.shape == (6,)
    assert var.shape == (6,)
    assert np.all(var > 0)

    mean2, var2, individual = trained.predict(x_test, return_individual_predictions=True)
    assert individual.shape == (len(trained.sampled_weights), 6)
    np.testing.assert_allclose(mean2, mean)
    np.testing.assert_allclose(var2, var)


def test_predict_matches_predict_single_with_one_sample(trained, data):
    x, _ = data
    trained.sampled_weights = [trained.network_weights]
    x_test = x[:5]

    mean, _ = trained.predict(x_test)
    single = trained.predict_single(x_test, sample_index=0)

    assert single.shape == (5, 2)
    np.testing.assert_allclose(mean, single[:, 0], rtol=1e-6, atol=1e-8)


def test_f_gradient_matches_finite_differences(trained):
    x0 = np.array([0.2, -0.3])
    weights = trained.sampled_weights[0]
    trained.sampled_weights = [weights]

    grad = trained.f_gradient(x0, weights=weights)

    eps = 1e-6
    numeric = np.zeros(2)
    for j in range(2):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        f_plus = trained.predict_single(x_plus[None, :], sample_index=0)[0, 0]
        f_minus = trained.predict_single(x_minus[None, :], sample_index=0)[0, 0]
        numeric[j] = (f_plus - f_minus) / (2 * eps)

    assert grad.shape == (2,)
    np.testing.assert_allclose(grad, numeric, rtol=1e-4, atol=1e-6)


def test_predictive_mean_gradient_is_mean_of_sample_gradients(trained):
    x0 = np.array([0.2, -0.3])

    g = trained.predictive_mean_gradient(x0)
    expected = np.mean(
        [trained.f_gradient(x0, weights=w) for w in trained.sampled_weights], axis=0
    )

    assert g.shape == (2,)
    np.testing.assert_allclose(g, expected, rtol=1e-6, atol=1e-10)


def test_predictive_variance_gradient_is_zero_for_single_sample(trained):
    x0 = np.array([0.2, -0.3])
    trained.sampled_weights = trained.sampled_weights[:1]

    g = trained.predictive_variance_gradient(x0)

    assert g.shape == (2,)
    np.testing.assert_allclose(g, np.zeros(2), atol=1e-10)


def test_train_and_evaluate_returns_learning_curves(data):
    x, y = data
    bnn = Bohamiann()

    n_steps, ll_curve, mse_curve = bnn.train_and_evaluate(
        x[:20], y[:20], x[20:], y[20:],
        num_steps=20,
        validate_every_n_steps=10,
        keep_every=5,
        num_burn_in_steps=5,
        batch_size=8,
    )

    assert n_steps == [15, 25]
    assert len(ll_curve) == 2
    assert len(mse_curve) == 2
    assert np.all(np.isfinite(ll_curve))
    assert np.all(np.isfinite(mse_curve))
