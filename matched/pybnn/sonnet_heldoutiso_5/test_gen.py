import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def _make_data(n=30, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2, 2, size=(n, d))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] + 0.05 * rng.randn(n)
    return x, y


def _trained_model(**kwargs):
    torch.manual_seed(0)
    np.random.seed(0)
    init_kwargs = kwargs.pop("init_kwargs", {})
    model = Bohamiann(print_every_n_steps=10, **init_kwargs)
    x, y = _make_data()
    params = dict(num_steps=30, num_burn_in_steps=10, keep_every=5, batch_size=10, lr=1e-3)
    params.update(kwargs)
    model.train(x, y, **params)
    return model, x, y


def test_get_default_network_output_shape():
    net = get_default_network(input_dimensionality=3).double()
    out = net(torch.zeros(7, 3, dtype=torch.double))
    assert out.shape == (7, 2)


def test_nll_matches_manual_computation():
    pred = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.double)
    target = torch.tensor([0.0, 4.0], dtype=torch.double)
    # log variance 0 -> variance 1; nll = mean(0.5 * (t - m)^2)
    expected = np.mean(0.5 * np.array([1.0, 4.0]))
    assert nll(pred, target).item() == pytest.approx(expected, rel=1e-6)


def test_nll_includes_log_variance_term():
    pred = torch.tensor([[0.0, 2.0]], dtype=torch.double)
    target = torch.tensor([0.0], dtype=torch.double)
    # zero error, nll = 0.5 * log_var
    assert nll(pred, target).item() == pytest.approx(1.0, rel=1e-6)


def test_nll_is_lower_for_better_mean():
    target = torch.tensor([1.0, 1.0], dtype=torch.double)
    good = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.double)
    bad = torch.tensor([[3.0, 0.0], [3.0, 0.0]], dtype=torch.double)
    assert nll(good, target).item() < nll(bad, target).item()


def test_init_defaults():
    model = Bohamiann()
    assert model.is_trained is False
    assert model.sampled_weights == []
    assert model.sampler is None
    assert model.sampling_method == "adaptive_sghmc"
    assert model.use_double_precision is True
    assert model.do_normalize_input is True
    assert model.do_normalize_output is True


def test_normalize_input_and_output_methods():
    model = Bohamiann()
    x = np.random.RandomState(1).randn(50, 3) * 4 + 7
    xn, m, s = model.normalize_input(x)
    assert np.allclose(xn.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(xn.std(axis=0), 1, atol=1e-8)
    assert np.allclose(m, x.mean(axis=0))
    assert np.allclose(s, x.std(axis=0))
    xn2, _, _ = model.normalize_input(x, m, s)
    assert np.allclose(xn, xn2)
    yn, ym, ys = model.normalize_output(x[:, 0])
    assert np.allclose(yn * ys + ym, x[:, 0])


def test_train_collects_expected_number_of_samples():
    model, x, y = _trained_model()
    assert model.is_trained
    assert len(model.sampled_weights) == 3
    assert model.sampler is not None
    # each sample is a tuple of arrays matching network parameters
    n_params = len(list(model.model.parameters()))
    for w in model.sampled_weights:
        assert len(w) == n_params


def test_train_resets_samples_when_not_continuing():
    model, x, y = _trained_model()
    model.train(x, y, num_steps=30, num_burn_in_steps=10, keep_every=5,
                batch_size=10, lr=1e-3)
    assert len(model.sampled_weights) == 3


def test_continue_training_appends_samples():
    model, x, y = _trained_model()
    before = len(model.sampled_weights)
    model.train(x, y, num_steps=20, num_burn_in_steps=0, keep_every=5,
                batch_size=10, lr=1e-3, continue_training=True)
    assert len(model.sampled_weights) == before + 3


@pytest.mark.parametrize("method", ["adaptive_sghmc", "sgld", "sghmc", "preconditioned_sgld"])
def test_sampling_methods_train_and_predict(method):
    model, x, y = _trained_model(init_kwargs={"sampling_method": method})
    assert model.is_trained
    assert len(model.sampled_weights) >= 1
    mean, var = model.predict(x[:5])
    assert mean.shape == (5,)
    assert var.shape == (5,)


def test_predict_shapes_and_positive_variance():
    model, x, y = _trained_model()
    mean, var = model.predict(x[:7])
    assert mean.shape == (7,)
    assert var.shape == (7,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert np.all(var > 0)


def test_predict_return_individual_predictions():
    model, x, y = _trained_model()
    mean, var, indiv = model.predict(x[:4], return_individual_predictions=True)
    assert indiv.shape == (len(model.sampled_weights), 4)
    # mean of individual predictions equals the predictive mean
    assert np.allclose(indiv.mean(axis=0), mean)


def test_predict_single_shape_and_consistency():
    model, x, y = _trained_model()
    out = model.predict_single(x[:6], sample_index=0)
    assert out.shape == (6, 2)
    # mean of predict_single over all samples equals predict mean
    means = np.mean([model.predict_single(x[:6], i)[:, 0]
                     for i in range(len(model.sampled_weights))], axis=0)
    mean, _ = model.predict(x[:6])
    assert np.allclose(means, mean)


def test_predict_without_normalization():
    model, x, y = _trained_model(
        init_kwargs={"normalize_input": False, "normalize_output": False})
    mean, var = model.predict(x[:5])
    assert mean.shape == (5,)
    assert np.all(var > 0)


def test_float32_training_and_prediction():
    model, x, y = _trained_model(init_kwargs={"use_double_precision": False})
    mean, var = model.predict(x[:5])
    assert mean.shape == (5,)
    assert np.all(np.isfinite(mean))
    assert next(model.model.parameters()).dtype == torch.float32


def test_train_accepts_column_vector_targets():
    torch.manual_seed(0)
    model = Bohamiann()
    x, y = _make_data()
    model.train(x, y[:, None], num_steps=15, num_burn_in_steps=5, keep_every=5,
                batch_size=10, lr=1e-3)
    assert model.is_trained
    assert model.y.shape == (x.shape[0],)


def test_train_invalid_inputs_raise():
    model = Bohamiann()
    x, y = _make_data()
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=5, batch_size=0)
    with pytest.raises(AssertionError):
        model.train(x, np.stack([y, y], axis=1), num_steps=5)


def test_batch_size_larger_than_dataset_is_handled():
    torch.manual_seed(0)
    model = Bohamiann()
    x, y = _make_data(n=8)
    model.train(x, y, num_steps=12, num_burn_in_steps=4, keep_every=4,
                batch_size=100, lr=1e-3)
    assert model.is_trained
    assert len(model.sampled_weights) >= 1


def test_custom_get_network_is_used():
    calls = []

    def factory(input_dimensionality):
        calls.append(input_dimensionality)
        return get_default_network(input_dimensionality)

    torch.manual_seed(0)
    model = Bohamiann(get_network=factory)
    x, y = _make_data(d=3)
    model.train(x, y, num_steps=6, num_burn_in_steps=2, keep_every=2,
                batch_size=10, lr=1e-3)
    assert calls == [3]


def test_network_weights_roundtrip():
    model, x, y = _trained_model()
    weights = model.network_weights
    new = [np.zeros_like(w) for w in weights]
    model.network_weights = new
    assert all(np.all(w == 0) for w in model.network_weights)
    model.network_weights = list(weights)
    for a, b in zip(model.network_weights, weights):
        assert np.allclose(a, b)


def test_f_gradient_matches_finite_difference():
    model, x, y = _trained_model()
    weights = model.sampled_weights[0]
    point = np.array([0.3, -0.4])
    grad = model.f_gradient(point, weights)
    assert grad.shape == (2,)

    h = 1e-5
    numeric = np.zeros(2)
    for j in range(2):
        dp = point.copy()
        dm = point.copy()
        dp[j] += h
        dm[j] -= h
        fp = model.predict_single(dp[None, :], 0)[0, 0]
        fm = model.predict_single(dm[None, :], 0)[0, 0]
        numeric[j] = (fp - fm) / (2 * h)
    assert np.allclose(grad, numeric, rtol=1e-3, atol=1e-5)


def test_predictive_gradients_shapes():
    model, x, y = _trained_model()
    point = np.array([0.1, 0.2])
    gm = model.predictive_mean_gradient(point)
    gv = model.predictive_variance_gradient(point)
    assert gm.shape == (2,)
    assert gv.shape == (2,)
    assert np.all(np.isfinite(gm))
    assert np.all(np.isfinite(gv))


def test_train_and_evaluate_returns_learning_curves():
    torch.manual_seed(0)
    model = Bohamiann()
    x, y = _make_data(n=30)
    xv, yv = _make_data(n=10, seed=5)
    n_steps, ll, mse = model.train_and_evaluate(
        x, y, xv, yv, num_steps=20, validate_every_n_steps=10, keep_every=5,
        num_burn_in_steps=10, lr=1e-3, batch_size=10)
    assert n_steps == [20, 30]
    assert len(ll) == 2
    assert len(mse) == 2
    assert all(np.isfinite(mse))
    assert len(model.sampled_weights) >= 1
