import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def make_data(n=30, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(n, d)
    y = np.sin(3 * x).sum(axis=1)
    return x, y


def train_small(model=None, x=None, y=None, **kwargs):
    torch.manual_seed(0)
    np.random.seed(0)
    if x is None:
        x, y = make_data()
    if model is None:
        model = Bohamiann(print_every_n_steps=10)
    params = dict(num_steps=60, num_burn_in_steps=20, keep_every=10,
                  lr=1e-3, batch_size=10)
    params.update(kwargs)
    model.train(x, y, **params)
    return model, x, y


# ---------- nll ----------

def test_nll_zero_when_perfect_prediction_unit_variance():
    inp = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.double)
    target = torch.tensor([1.0, 2.0], dtype=torch.double)
    assert nll(inp, target).item() == pytest.approx(0.0, abs=1e-12)


def test_nll_known_value_with_error():
    inp = torch.tensor([[0.0, 0.0]], dtype=torch.double)
    target = torch.tensor([1.0], dtype=torch.double)
    # 0.5 * (1)^2 / 1 + 0.5 * 0
    assert nll(inp, target).item() == pytest.approx(0.5, rel=1e-6)


def test_nll_log_variance_term():
    log_var = float(np.log(4.0))
    inp = torch.tensor([[1.0, log_var], [3.0, log_var]], dtype=torch.double)
    target = torch.tensor([1.0, 3.0], dtype=torch.double)
    assert nll(inp, target).item() == pytest.approx(0.5 * log_var, rel=1e-6)


def test_nll_returns_scalar_and_accepts_column_targets():
    inp = torch.randn(5, 2, dtype=torch.double)
    target = torch.randn(5, 1, dtype=torch.double)
    out = nll(inp, target)
    assert out.dim() == 0
    assert torch.isfinite(out)


# ---------- network ----------

def test_default_network_output_shape():
    net = get_default_network(3).double()
    out = net(torch.randn(7, 3, dtype=torch.double))
    assert out.shape == (7, 2)


# ---------- construction ----------

def test_init_defaults():
    m = Bohamiann()
    assert m.is_trained is False
    assert m.sampled_weights == []
    assert m.sampler is None
    assert m.sampling_method == "adaptive_sghmc"
    assert m.use_double_precision is True
    assert m.do_normalize_input is True
    assert m.do_normalize_output is True
    assert m.likelihood_function is nll


# ---------- normalization ----------

def test_normalize_input_and_output_statistics():
    m = Bohamiann()
    x, _ = make_data(n=50, d=3)
    xn, mean, std = m.normalize_input(x)
    assert np.allclose(xn.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(xn.std(axis=0), 1, atol=1e-8)
    assert np.allclose(mean, x.mean(axis=0))
    assert np.allclose(std, x.std(axis=0))

    y = x[:, 0]
    yn, ymean, ystd = m.normalize_output(y)
    assert yn.mean() == pytest.approx(0, abs=1e-8)
    assert yn.std() == pytest.approx(1, abs=1e-8)


def test_normalize_with_given_mean_std():
    m = Bohamiann()
    x = np.array([[1.0, 2.0], [3.0, 6.0]])
    xn, mean, std = m.normalize_input(x, np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    assert np.allclose(xn, [[0.0, 0.0], [1.0, 1.0]])


# ---------- training ----------

def test_train_collects_expected_number_of_samples():
    model, x, y = train_small()
    # steps 0..59, kept when step > 20 and (step - 20) % 10 == 0 -> 30, 40, 50
    assert model.is_trained
    assert len(model.sampled_weights) == 3
    assert hasattr(model, "x_mean") and hasattr(model, "y_std")


@pytest.mark.parametrize("method", ["adaptive_sghmc", "sgld", "sghmc", "preconditioned_sgld"])
def test_train_all_sampling_methods(method):
    model, x, y = train_small(Bohamiann(sampling_method=method))
    assert model.sampler is not None
    assert len(model.sampled_weights) == 3
    n_params = len(list(model.model.parameters()))
    assert all(len(w) == n_params for w in model.sampled_weights)


def test_train_accepts_column_targets():
    x, y = make_data()
    model, _, _ = train_small(x=x, y=y[:, None])
    assert model.y.shape == (30,)
    assert model.is_trained


def test_train_rejects_bad_target_shape():
    x, y = make_data()
    with pytest.raises(AssertionError):
        Bohamiann().train(x, np.stack([y, y], axis=1), num_steps=5, num_burn_in_steps=1)


def test_train_rejects_invalid_batch_size():
    x, y = make_data()
    with pytest.raises(AssertionError):
        Bohamiann().train(x, y, num_steps=5, num_burn_in_steps=1, batch_size=0)


def test_train_with_fewer_points_than_batch_size():
    x, y = make_data(n=5)
    model, _, _ = train_small(x=x, y=y, batch_size=20)
    mu, var = model.predict(x)
    assert mu.shape == (5,)
    assert var.shape == (5,)


def test_continue_training_keeps_samples_and_adds_more():
    model, x, y = train_small()
    first_sampler = model.sampler
    before = len(model.sampled_weights)
    model.train(x, y, num_steps=20, num_burn_in_steps=0, keep_every=10,
                lr=1e-3, batch_size=10, continue_training=True)
    assert model.sampler is first_sampler
    assert len(model.sampled_weights) == before + 1


def test_retraining_clears_samples():
    model, x, y = train_small()
    train_small(model, x, y, num_steps=40, num_burn_in_steps=20, keep_every=10)
    # steps 0..39: kept at 30 only
    assert len(model.sampled_weights) == 1


def test_custom_get_network_called_with_input_dimensionality():
    calls = []

    def get_net(input_dimensionality):
        calls.append(input_dimensionality)
        return get_default_network(input_dimensionality)

    x, y = make_data(d=4)
    train_small(Bohamiann(get_network=get_net), x=x, y=y)
    assert calls == [4]


def test_verbose_training_prints_progress(capsys):
    x, y = make_data()
    model = Bohamiann(print_every_n_steps=10, normalize_output=False)
    train_small(model, x, y, verbose=True)
    out = capsys.readouterr().out
    assert "Step" in out
    assert "NLL" in out


# ---------- weights ----------

def test_network_weights_roundtrip():
    model, _, _ = train_small()
    weights = model.network_weights
    params = list(model.model.parameters())
    assert len(weights) == len(params)
    for w, p in zip(weights, params):
        assert isinstance(w, np.ndarray)
        assert w.shape == tuple(p.shape)

    new = [np.zeros_like(w) for w in weights]
    with torch.no_grad():
        model.network_weights = new
    assert all(np.all(w == 0) for w in model.network_weights)
    with torch.no_grad():
        model.network_weights = list(weights)
    for a, b in zip(model.network_weights, weights):
        assert np.array_equal(a, b)


# ---------- prediction ----------

def test_predict_shapes_and_positive_variance():
    model, x, y = train_small()
    mu, var = model.predict(x)
    assert mu.shape == (30,)
    assert var.shape == (30,)
    assert np.all(np.isfinite(mu))
    assert np.all(var > 0)


def test_predict_individual_predictions():
    model, x, y = train_small()
    mu, var, indiv = model.predict(x[:7], return_individual_predictions=True)
    assert indiv.shape == (len(model.sampled_weights), 7)
    assert np.allclose(indiv.mean(axis=0), mu)


def test_predict_single_matches_sample_shape_and_mean_consistency():
    model, x, y = train_small()
    xt = x[:6]
    singles = np.array([model.predict_single(xt, i) for i in range(len(model.sampled_weights))])
    assert singles.shape == (len(model.sampled_weights), 6, 2)
    mu, _ = model.predict(xt)
    assert np.allclose(singles[:, :, 0].mean(axis=0), mu)


def test_predict_without_normalization():
    x, y = make_data()
    model = Bohamiann(normalize_input=False, normalize_output=False)
    train_small(model, x, y)
    assert not hasattr(model, "x_mean")
    mu, var = model.predict(x)
    assert mu.shape == (30,)
    assert np.all(var > 0)


def test_float32_training_and_prediction():
    x, y = make_data()
    model = Bohamiann(use_double_precision=False)
    train_small(model, x, y)
    assert all(w.dtype == np.float32 for w in model.sampled_weights[0])
    mu, var = model.predict(x)
    assert mu.shape == (30,)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(var))


# ---------- gradients ----------

def test_f_gradient_matches_finite_differences():
    model, x, y = train_small()
    weights = model.sampled_weights[0]
    x0 = x[0].copy()
    g = model.f_gradient(x0, weights)
    assert g.shape == (2,)

    eps = 1e-6
    num = np.zeros(2)
    for j in range(2):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += eps
        xm[j] -= eps
        fp = model.predict_single(xp[None, :], 0)[0, 0]
        fm = model.predict_single(xm[None, :], 0)[0, 0]
        num[j] = (fp - fm) / (2 * eps)
    assert np.allclose(g, num, rtol=1e-3, atol=1e-5)


def test_predictive_gradients_shapes():
    model, x, y = train_small()
    gm = model.predictive_mean_gradient(x[0])
    gv = model.predictive_variance_gradient(x[0])
    assert gm.shape == (2,)
    assert gv.shape == (2,)
    assert np.all(np.isfinite(gm))
    assert np.all(np.isfinite(gv))


# ---------- train_and_evaluate ----------

def test_train_and_evaluate_learning_curves():
    x, y = make_data()
    xv, yv = make_data(n=10, seed=1)
    torch.manual_seed(0)
    model = Bohamiann(print_every_n_steps=10)
    n_steps, ll, mse = model.train_and_evaluate(
        x, y, xv, yv, num_steps=20, validate_every_n_steps=10, keep_every=5,
        num_burn_in_steps=10, lr=1e-3, batch_size=10)
    assert n_steps == [20, 30]
    assert len(ll) == 2 and len(mse) == 2
    assert all(m >= 0 for m in mse)
    assert model.is_trained
    assert len(model.sampled_weights) >= 1
