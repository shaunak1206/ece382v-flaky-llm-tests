import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def make_data(n=30, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(n, d)
    y = np.sin(3 * x[:, 0]) + 0.5 * x[:, 1] + 0.01 * rng.randn(n)
    return x, y


def expected_num_samples(num_steps, burn_in, keep_every):
    return len([s for s in range(num_steps) if s > burn_in and (s - burn_in) % keep_every == 0])


def train_small(model=None, x=None, y=None, num_steps=50, burn_in=10, keep_every=10,
                batch_size=10, **kwargs):
    if model is None:
        model = Bohamiann(print_every_n_steps=1000)
    if x is None:
        x, y = make_data()
    torch.manual_seed(1)
    model.train(x, y, num_steps=num_steps, num_burn_in_steps=burn_in,
                keep_every=keep_every, batch_size=batch_size, lr=1e-3, **kwargs)
    return model


# ---------------------------------------------------------------- nll

def test_nll_perfect_prediction_zero_logvar_is_zero():
    inp = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert float(nll(inp, target)) == pytest.approx(0.0, abs=1e-9)


def test_nll_known_value_with_error():
    inp = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)
    # 0.5 * (error^2 / var) + 0.5 * log var = 0.5
    assert float(nll(inp, target)) == pytest.approx(0.5, abs=1e-6)


def test_nll_is_batch_average_and_includes_log_variance():
    logvar = np.log(4.0)
    inp = torch.tensor([[0.0, logvar], [0.0, logvar]], dtype=torch.float64)
    target = torch.zeros(2, dtype=torch.float64)
    assert float(nll(inp, target)) == pytest.approx(0.5 * logvar, abs=1e-9)


# ---------------------------------------------------------------- network

def test_default_network_output_shape_and_params():
    net = get_default_network(input_dimensionality=3).double()
    out = net(torch.rand(7, 3, dtype=torch.float64))
    assert out.shape == (7, 2)
    assert len(list(net.parameters())) >= 7  # 3 linear layers (w+b) + log variance


# ---------------------------------------------------------------- init

def test_init_defaults():
    m = Bohamiann()
    assert m.is_trained is False
    assert m.sampled_weights == []
    assert m.sampler is None
    assert m.sampling_method == "adaptive_sghmc"
    assert m.do_normalize_input and m.do_normalize_output
    assert m.use_double_precision is True


# ---------------------------------------------------------------- train

def test_train_collects_expected_number_of_samples():
    m = train_small(num_steps=50, burn_in=10, keep_every=10)
    assert m.is_trained
    assert len(m.sampled_weights) == expected_num_samples(50, 10, 10) == 3


def test_sampled_weights_match_parameter_shapes():
    m = train_small()
    shapes = [p.shape for p in m.model.parameters()]
    for weights in m.sampled_weights:
        assert len(weights) == len(shapes)
        for w, s in zip(weights, shapes):
            assert isinstance(w, np.ndarray)
            assert w.shape == tuple(s)
            assert np.all(np.isfinite(w))


def test_retraining_clears_previous_samples_but_continue_training_keeps_them():
    m = train_small(num_steps=50, burn_in=10, keep_every=10)
    n = len(m.sampled_weights)
    x, y = make_data()
    torch.manual_seed(2)
    m.train(x, y, num_steps=30, num_burn_in_steps=0, keep_every=10, batch_size=10,
            lr=1e-3, continue_training=True)
    assert len(m.sampled_weights) == n + expected_num_samples(30, 0, 10)
    torch.manual_seed(2)
    m.train(x, y, num_steps=50, num_burn_in_steps=10, keep_every=10, batch_size=10, lr=1e-3)
    assert len(m.sampled_weights) == 3


def test_train_invalid_batch_size_raises():
    x, y = make_data()
    with pytest.raises(AssertionError):
        Bohamiann().train(x, y, num_steps=5, batch_size=0)


def test_train_invalid_target_shape_raises():
    x, _ = make_data()
    y_bad = np.zeros((x.shape[0], 2))
    with pytest.raises(AssertionError):
        Bohamiann().train(x, y_bad, num_steps=5)


def test_train_accepts_column_vector_targets_and_small_dataset():
    x, y = make_data(n=8)
    m = train_small(x=x, y=y[:, None], batch_size=20, num_steps=30, burn_in=5, keep_every=5)
    assert m.y.shape == (8,)
    assert len(m.sampled_weights) == expected_num_samples(30, 5, 5)


@pytest.mark.parametrize("method", ["sgld", "sghmc", "adaptive_sghmc", "preconditioned_sgld"])
def test_all_sampling_methods_train_and_predict(method):
    m = Bohamiann(sampling_method=method, print_every_n_steps=1000)
    train_small(model=m, num_steps=30, burn_in=5, keep_every=5)
    assert m.sampler is not None
    assert len(m.sampled_weights) == expected_num_samples(30, 5, 5)
    x, _ = make_data(n=4)
    mean, var = m.predict(x)
    assert mean.shape == (4,)
    assert var.shape == (4,)


def test_train_without_normalization_and_single_precision():
    m = Bohamiann(normalize_input=False, normalize_output=False,
                  use_double_precision=False, print_every_n_steps=1000)
    train_small(model=m)
    assert not hasattr(m, "x_mean")
    assert m.sampled_weights[0][0].dtype == np.float32
    x, _ = make_data(n=5)
    mean, var = m.predict(x.astype(np.float32))
    assert mean.shape == (5,)
    assert np.all(np.isfinite(mean))


def test_training_is_reproducible_with_seed():
    x, y = make_data()
    preds = []
    for _ in range(2):
        torch.manual_seed(123)
        m = Bohamiann(print_every_n_steps=1000)
        m.train(x, y, num_steps=30, num_burn_in_steps=5, keep_every=5, batch_size=10, lr=1e-3)
        preds.append(m.predict(x[:5])[0])
    np.testing.assert_allclose(preds[0], preds[1])


# ---------------------------------------------------------------- predict

def test_predict_shapes_and_positive_variance():
    m = train_small()
    x, _ = make_data(n=6)
    mean, var = m.predict(x)
    assert mean.shape == (6,)
    assert var.shape == (6,)
    assert np.all(var > 0)
    assert np.all(np.isfinite(mean))


def test_predict_individual_predictions_shape_and_mean_consistency():
    m = train_small()
    x, _ = make_data(n=6)
    mean, var, indiv = m.predict(x, return_individual_predictions=True)
    assert indiv.shape == (len(m.sampled_weights), 6)
    np.testing.assert_allclose(indiv.mean(axis=0), mean, rtol=1e-6, atol=1e-8)


def test_predict_single_matches_individual_prediction():
    m = train_small()
    x, _ = make_data(n=4)
    _, _, indiv = m.predict(x, return_individual_predictions=True)
    out = m.predict_single(x, sample_index=1)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out[:, 0], indiv[1], rtol=1e-6, atol=1e-8)


def test_network_weights_property_and_setter_roundtrip():
    m = train_small()
    original = m.network_weights
    new = [w + 1.0 for w in original]
    with torch.no_grad():
        m.network_weights = new
    for w, n in zip(m.network_weights, new):
        np.testing.assert_allclose(w, n)


# ---------------------------------------------------------------- gradients

def test_f_gradient_matches_finite_differences():
    m = train_small()
    x0 = np.array([0.4, 0.6])
    weights = m.sampled_weights[0]
    g = m.f_gradient(x0, weights)
    assert g.shape == (2,)
    eps = 1e-6
    fd = np.zeros(2)
    for i in range(2):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = m.predict_single(xp[None, :], 0)[0, 0]
        fm = m.predict_single(xm[None, :], 0)[0, 0]
        fd[i] = (fp - fm) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-3, atol=1e-5)


def test_predictive_gradients_shapes():
    m = train_small()
    x0 = np.array([0.3, 0.7])
    gm = m.predictive_mean_gradient(x0)
    gv = m.predictive_variance_gradient(x0)
    assert gm.shape == (2,)
    assert gv.shape == (2,)
    assert np.all(np.isfinite(gm))
    assert np.all(np.isfinite(gv))


# ---------------------------------------------------------------- train_and_evaluate

def test_train_and_evaluate_returns_learning_curves():
    x, y = make_data(n=30)
    xv, yv = make_data(n=10, seed=5)
    m = Bohamiann(print_every_n_steps=1000)
    torch.manual_seed(0)
    steps, ll, mse = m.train_and_evaluate(
        x, y, xv, yv, num_steps=20, validate_every_n_steps=10,
        keep_every=5, num_burn_in_steps=10, lr=1e-3, batch_size=10)
    assert steps == [20, 30]
    assert len(ll) == 2 and len(mse) == 2
    assert all(np.isfinite(v) for v in ll + mse)
    assert all(v >= 0 for v in mse)
    assert m.is_trained
    assert len(m.sampled_weights) >= 1
