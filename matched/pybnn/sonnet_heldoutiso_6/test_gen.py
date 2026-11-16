import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def make_data(n=30, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(n, d)
    y = np.sin(3 * x[:, 0]) + 0.5 * x[:, 1] if d > 1 else np.sin(3 * x[:, 0])
    return x, y


def train_small(model=None, x=None, y=None, num_steps=30, burn_in=10, keep_every=5,
                batch_size=10, **kwargs):
    if x is None:
        x, y = make_data()
    if model is None:
        model = Bohamiann(print_every_n_steps=1000)
    model.train(x, y, num_steps=num_steps, num_burn_in_steps=burn_in,
                keep_every=keep_every, batch_size=batch_size, **kwargs)
    return model, x, y


# ---------------- nll ----------------

def test_nll_perfect_mean_unit_variance_is_zero():
    inp = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert nll(inp, target).item() == pytest.approx(0.0, abs=1e-9)


def test_nll_known_value():
    # error = 1, log variance = 0 -> 0.5 * 1 / 1 = 0.5
    inp = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)
    assert nll(inp, target).item() == pytest.approx(0.5, abs=1e-8)


def test_nll_includes_log_variance_and_averages_over_batch():
    logvar = 2.0
    inp = torch.tensor([[0.0, logvar], [0.0, logvar]], dtype=torch.float64)
    target = torch.tensor([0.0, 0.0], dtype=torch.float64)
    # zero error -> 0.5 * logvar for each sample, averaged
    assert nll(inp, target).item() == pytest.approx(0.5 * logvar, abs=1e-8)


# ---------------- network ----------------

def test_default_network_output_shape():
    net = get_default_network(3)
    out = net(torch.randn(4, 3))
    assert out.shape == (4, 2)
    # second column is the (input independent) log variance
    assert torch.allclose(out[:, 1], out[0, 1].expand(4))


# ---------------- construction ----------------

def test_init_defaults():
    m = Bohamiann()
    assert not m.is_trained
    assert m.sampled_weights == []
    assert m.sampler is None
    assert m.sampling_method == "adaptive_sghmc"
    assert m.use_double_precision


# ---------------- training ----------------

def test_train_collects_expected_number_of_samples():
    model, _, _ = train_small()
    assert model.is_trained
    # steps 15, 20, 25 are kept
    assert len(model.sampled_weights) == 3


def test_train_no_samples_during_burn_in_only():
    model, _, _ = train_small(num_steps=10, burn_in=10)
    assert model.is_trained
    assert len(model.sampled_weights) == 0


@pytest.mark.parametrize("method", ["adaptive_sghmc", "sgld", "sghmc"])
def test_sampling_methods_produce_finite_predictions(method):
    torch.manual_seed(1)
    model = Bohamiann(sampling_method=method, print_every_n_steps=1000)
    model, x, _ = train_small(model=model, lr=1e-3)
    mu, var = model.predict(x[:5])
    assert mu.shape == (5,)
    assert var.shape == (5,)
    assert np.all(np.isfinite(mu))
    assert np.all(var > 0)


def test_train_without_normalization_and_single_precision():
    torch.manual_seed(2)
    x, y = make_data()
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    model = Bohamiann(normalize_input=False, normalize_output=False,
                      use_double_precision=False, print_every_n_steps=1000)
    model, _, _ = train_small(model=model, x=x, y=y, lr=1e-3)
    mu, var = model.predict(x[:4])
    assert mu.shape == (4,)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(var))


def test_train_accepts_column_targets():
    x, y = make_data()
    model, _, _ = train_small(x=x, y=y[:, None])
    assert model.y.shape == (30,)
    assert model.is_trained


def test_train_rejects_bad_target_shape():
    x, y = make_data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, np.stack([y, y], axis=1), num_steps=2, num_burn_in_steps=1)


def test_train_rejects_invalid_batch_size():
    x, y = make_data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=2, num_burn_in_steps=1, batch_size=0)


def test_train_with_fewer_points_than_batch_size():
    x, y = make_data(n=5)
    model = Bohamiann(print_every_n_steps=1000)
    model.train(x, y, num_steps=12, num_burn_in_steps=4, keep_every=2, batch_size=20)
    assert model.is_trained
    assert len(model.sampled_weights) > 0


def test_continue_training_appends_and_retraining_clears():
    model, x, y = train_small()
    assert len(model.sampled_weights) == 3
    model.train(x, y, num_steps=20, num_burn_in_steps=0, keep_every=5,
                batch_size=10, continue_training=True)
    assert len(model.sampled_weights) == 6
    model.train(x, y, num_steps=30, num_burn_in_steps=10, keep_every=5, batch_size=10)
    assert len(model.sampled_weights) == 3


def test_verbose_training_prints_progress(capsys):
    x, y = make_data()
    model = Bohamiann(print_every_n_steps=5)
    model.train(x, y, num_steps=30, num_burn_in_steps=10, keep_every=5,
                batch_size=10, verbose=True)
    out = capsys.readouterr().out
    assert "Step" in out
    assert "NLL" in out


def test_train_and_evaluate_learning_curves():
    x, y = make_data()
    xv, yv = make_data(n=10, seed=5)
    model = Bohamiann(print_every_n_steps=1000)
    n_steps, ll, mse = model.train_and_evaluate(
        x, y, xv, yv, num_steps=20, validate_every_n_steps=10,
        keep_every=5, num_burn_in_steps=10, batch_size=10)
    assert n_steps == [20, 30]
    assert len(ll) == 2 and len(mse) == 2
    assert all(np.isfinite(ll)) and all(np.isfinite(mse))
    assert len(model.sampled_weights) >= 1


# ---------------- weights ----------------

def test_network_weights_roundtrip():
    model, _, _ = train_small()
    weights = model.network_weights
    assert isinstance(weights, tuple)
    assert all(isinstance(w, np.ndarray) for w in weights)
    zeros = [np.zeros_like(w) for w in weights]
    with torch.no_grad():
        model.network_weights = zeros
    assert all(np.all(w == 0) for w in model.network_weights)
    with torch.no_grad():
        model.network_weights = list(weights)
    for a, b in zip(weights, model.network_weights):
        np.testing.assert_allclose(a, b)


# ---------------- prediction ----------------

def test_predict_shapes_and_individual_predictions():
    model, x, _ = train_small()
    mu, var, indiv = model.predict(x[:7], return_individual_predictions=True)
    assert mu.shape == (7,)
    assert var.shape == (7,)
    assert indiv.shape == (3, 7)
    np.testing.assert_allclose(mu, indiv.mean(axis=0), rtol=1e-8, atol=1e-8)
    assert np.all(var > 0)
    # total variance is at least the variance of the ensemble means
    assert np.all(var >= indiv.var(axis=0) - 1e-10)


def test_predict_single_matches_individual_prediction():
    model, x, _ = train_small()
    xt = x[:6]
    _, _, indiv = model.predict(xt, return_individual_predictions=True)
    for i in range(3):
        single = model.predict_single(xt, i)
        assert single.shape == (6, 2)
        np.testing.assert_allclose(single[:, 0], indiv[i], rtol=1e-8, atol=1e-8)


def test_training_is_reproducible_with_seed():
    x, y = make_data()
    results = []
    for _ in range(2):
        torch.manual_seed(123)
        model, _, _ = train_small(x=x, y=y)
        results.append(model.predict(x[:5])[0])
    np.testing.assert_allclose(results[0], results[1], rtol=1e-8, atol=1e-8)


# ---------------- gradients ----------------

def test_f_gradient_matches_finite_differences():
    torch.manual_seed(3)
    model, x, _ = train_small()
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
    np.testing.assert_allclose(g, num, rtol=1e-3, atol=1e-5)


def test_predictive_gradients_shapes_and_mean_consistency():
    torch.manual_seed(4)
    model, x, _ = train_small()
    x0 = x[1]
    gm = model.predictive_mean_gradient(x0)
    gv = model.predictive_variance_gradient(x0)
    assert gm.shape == (2,)
    assert gv.shape == (2,)
    assert np.all(np.isfinite(gm)) and np.all(np.isfinite(gv))
    indiv = [model.f_gradient(x0, w) for w in model.sampled_weights]
    np.testing.assert_allclose(gm, np.mean(indiv, axis=0), rtol=1e-8, atol=1e-10)
