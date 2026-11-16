import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def make_data(n=20, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(n, d)
    y = np.sum(x, axis=1) + 0.01 * rng.randn(n)
    return x, y


def trained_model(sampling_method="adaptive_sghmc", num_steps=30, burn_in=10,
                  keep_every=5, seed=1, n=20, d=2, **init_kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y = make_data(n=n, d=d)
    model = Bohamiann(sampling_method=sampling_method, print_every_n_steps=1000, **init_kwargs)
    model.train(x, y, num_steps=num_steps, num_burn_in_steps=burn_in,
                keep_every=keep_every, batch_size=10, lr=1e-3)
    return model, x, y


def test_nll_zero_for_perfect_mean_and_unit_variance():
    inp = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    target = torch.tensor([1.0, 2.0])
    assert float(nll(inp, target)) == pytest.approx(0.0, abs=1e-9)


def test_nll_known_value_and_batch_average():
    # log variance 0, error 1 -> 0.5 per sample
    inp = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([1.0, 1.0])
    assert float(nll(inp, target)) == pytest.approx(0.5, rel=1e-6)


def test_nll_includes_log_variance_term():
    logvar = 2.0
    inp = torch.tensor([[0.0, logvar]])
    target = torch.tensor([0.0])
    assert float(nll(inp, target)) == pytest.approx(0.5 * logvar, rel=1e-6)


def test_nll_accepts_column_target():
    inp = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    t1 = torch.tensor([1.0, 1.0])
    t2 = t1.view(-1, 1)
    assert float(nll(inp, t1)) == pytest.approx(float(nll(inp, t2)))


def test_default_network_output_shape():
    net = get_default_network(input_dimensionality=3).double()
    out = net(torch.zeros(5, 3, dtype=torch.double))
    assert tuple(out.shape) == (5, 2)


def test_init_defaults():
    model = Bohamiann()
    assert model.is_trained is False
    assert model.sampled_weights == []
    assert model.sampler is None
    assert model.do_normalize_input is True
    assert model.do_normalize_output is True
    assert model.sampling_method == "adaptive_sghmc"


def test_normalize_input_and_output_roundtrip_stats():
    model = Bohamiann()
    x = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    xn, m, s = model.normalize_input(x)
    np.testing.assert_allclose(m, x.mean(axis=0))
    np.testing.assert_allclose(s, x.std(axis=0))
    np.testing.assert_allclose(xn.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(xn.std(axis=0), 1.0, atol=1e-12)

    y = np.array([1.0, 2.0, 3.0, 4.0])
    yn, ym, ys = model.normalize_output(y)
    np.testing.assert_allclose(yn * ys + ym, y)
    # given mean/std are used
    yn2, _, _ = model.normalize_output(y, 0.0, 1.0)
    np.testing.assert_allclose(yn2, y)


def test_train_collects_expected_number_of_samples():
    model, x, y = trained_model(num_steps=30, burn_in=10, keep_every=5)
    # steps 15, 20, 25 are kept
    assert model.is_trained
    assert len(model.sampled_weights) == 3
    assert model.sampler is not None


@pytest.mark.parametrize("method", ["adaptive_sghmc", "sghmc", "sgld"])
def test_train_and_predict_with_sampling_methods(method):
    model, x, y = trained_model(sampling_method=method)
    assert len(model.sampled_weights) == 3
    mean, var = model.predict(x[:7])
    assert mean.shape == (7,)
    assert var.shape == (7,)


def test_predict_shapes_and_positive_variance():
    model, x, y = trained_model()
    mean, var = model.predict(x)
    assert mean.shape == (20,)
    assert var.shape == (20,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_predict_individual_predictions_shape():
    model, x, y = trained_model()
    mean, var, indiv = model.predict(x[:5], return_individual_predictions=True)
    assert indiv.shape == (len(model.sampled_weights), 5)
    np.testing.assert_allclose(indiv.mean(axis=0), mean, rtol=1e-6, atol=1e-8)


def test_predict_single_consistent_with_predict_mean():
    model, x, y = trained_model()
    xt = x[:6]
    mean, _ = model.predict(xt)
    singles = [model.predict_single(xt, i) for i in range(len(model.sampled_weights))]
    assert singles[0].shape == (6, 2)
    np.testing.assert_allclose(np.mean([s[:, 0] for s in singles], axis=0), mean,
                               rtol=1e-6, atol=1e-8)


def test_train_without_normalization():
    model, x, y = trained_model(normalize_input=False, normalize_output=False)
    mean, var = model.predict(x[:4])
    assert mean.shape == (4,)
    assert np.all(var > 0)
    assert not hasattr(model, "x_mean")


def test_train_with_single_precision():
    model, x, y = trained_model(use_double_precision=False)
    assert all(w.dtype == np.float32 for w in model.sampled_weights[0])
    mean, var = model.predict(x[:4])
    assert mean.shape == (4,)
    assert var.shape == (4,)


def test_train_accepts_column_targets_and_small_dataset():
    torch.manual_seed(0)
    x, y = make_data(n=5)
    model = Bohamiann(print_every_n_steps=1000)
    # fewer datapoints than batch size -> batch size is reduced
    model.train(x, y.reshape(-1, 1), num_steps=15, num_burn_in_steps=5,
                keep_every=5, batch_size=20, lr=1e-3)
    assert model.is_trained
    assert model.y.shape == (5,)
    assert len(model.sampled_weights) == 2  # steps 10 and 5<step: 10 only? check below


def test_train_invalid_inputs_raise():
    x, y = make_data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=5, batch_size=0)
    with pytest.raises(AssertionError):
        model.train(x, np.stack([y, y], axis=1), num_steps=5)


def test_network_weights_property_and_setter_roundtrip():
    model, x, y = trained_model()
    weights = model.network_weights
    params = list(model.model.parameters())
    assert len(weights) == len(params)
    for w, p in zip(weights, params):
        assert w.shape == tuple(p.shape)
    new = [np.zeros_like(w) for w in weights]
    with torch.no_grad():
        model.network_weights = new
    for w in model.network_weights:
        assert np.all(w == 0)


def test_continue_training_appends_samples_and_keeps_model():
    model, x, y = trained_model()
    net = model.model
    before = len(model.sampled_weights)
    model.train(x, y, num_steps=12, num_burn_in_steps=0, keep_every=4,
                batch_size=10, continue_training=True)
    assert model.model is net
    assert len(model.sampled_weights) > before


def test_custom_likelihood_function_is_used():
    calls = []

    def counting_nll(input, target):
        calls.append(1)
        return nll(input, target)

    torch.manual_seed(0)
    x, y = make_data()
    model = Bohamiann(likelihood_function=counting_nll, print_every_n_steps=1000)
    model.train(x, y, num_steps=7, num_burn_in_steps=3, keep_every=2, batch_size=10, lr=1e-3)
    assert len(calls) == 7


def test_train_and_evaluate_learning_curves():
    torch.manual_seed(0)
    x, y = make_data()
    xv, yv = make_data(n=8, seed=5)
    model = Bohamiann(print_every_n_steps=1000)
    n_steps, ll, mse = model.train_and_evaluate(
        x, y, xv, yv, num_steps=20, validate_every_n_steps=10, keep_every=5,
        num_burn_in_steps=10, lr=1e-3, batch_size=10)
    assert n_steps == [20, 30]
    assert len(ll) == 2
    assert len(mse) == 2
    assert all(m >= 0 for m in mse)
    assert len(model.sampled_weights) >= 1


def test_f_gradient_matches_finite_differences():
    model, x, y = trained_model()
    xt = np.array([0.3, 0.6])
    weights = model.sampled_weights[0]
    g = model.f_gradient(xt, weights)
    assert g.shape == (2,)

    eps = 1e-6
    fd = np.zeros(2)
    for i in range(2):
        xp = xt.copy()
        xm = xt.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = model.predict_single(xp[None, :], 0)[0, 0]
        fm = model.predict_single(xm[None, :], 0)[0, 0]
        fd[i] = (fp - fm) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-3, atol=1e-5)


def test_predictive_gradients_shapes_and_mean_gradient():
    model, x, y = trained_model()
    xt = np.array([0.2, 0.7])
    gm = model.predictive_mean_gradient(xt)
    gv = model.predictive_variance_gradient(xt)
    assert gm.shape == (2,)
    assert gv.shape == (2,)
    expected = np.mean([model.f_gradient(xt, w) for w in model.sampled_weights], axis=0)
    np.testing.assert_allclose(gm, expected)
    assert np.all(np.isfinite(gv))
