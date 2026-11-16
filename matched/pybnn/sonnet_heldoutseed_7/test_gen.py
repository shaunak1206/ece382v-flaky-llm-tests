import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def make_data(n=30, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(n, d)
    y = np.sin(3 * x.sum(axis=1)) + 0.05 * rng.randn(n)
    return x, y


def train_small(model=None, n=30, d=2, num_steps=60, num_burn_in_steps=20,
                keep_every=10, batch_size=10, **kwargs):
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_data(n, d)
    if model is None:
        model = Bohamiann(print_every_n_steps=1000)
    model.train(x, y, num_steps=num_steps, num_burn_in_steps=num_burn_in_steps,
                keep_every=keep_every, batch_size=batch_size, lr=1e-3, **kwargs)
    return model, x, y


def test_nll_matches_manual_computation():
    inp = torch.tensor([[0.5, 0.2], [1.0, -0.3], [-0.2, 0.7]], dtype=torch.float64)
    target = torch.tensor([0.1, 1.5, 0.3], dtype=torch.float64)
    expected = 0.0
    for (m, lv), t in zip(inp.numpy(), target.numpy()):
        expected += (t - m) ** 2 / (2 * np.exp(lv)) + 0.5 * lv
    expected /= 3
    result = nll(inp, target)
    assert result.dim() == 0
    assert result.item() == pytest.approx(expected, rel=1e-6)


def test_nll_perfect_prediction_unit_variance_is_zero():
    target = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)
    inp = torch.stack([target, torch.zeros_like(target)], dim=1)
    assert nll(inp, target).item() == pytest.approx(0.0, abs=1e-8)


def test_nll_increases_with_error():
    target = torch.zeros(4, dtype=torch.float64)
    good = torch.stack([target, torch.zeros_like(target)], dim=1)
    bad = torch.stack([target + 2.0, torch.zeros_like(target)], dim=1)
    assert nll(bad, target).item() > nll(good, target).item()


def test_default_network_output_shape():
    net = get_default_network(input_dimensionality=3)
    out = net(torch.rand(7, 3))
    assert out.shape == (7, 2)


def test_init_defaults():
    model = Bohamiann()
    assert model.is_trained is False
    assert model.sampled_weights == []
    assert model.sampler is None
    assert model.sampling_method == "adaptive_sghmc"
    assert model.use_double_precision is True


def test_train_sets_state_and_collects_samples():
    model, x, y = train_small()
    assert model.is_trained is True
    # samples kept at steps 30, 40, 50
    assert len(model.sampled_weights) == 3
    assert model.sampler is not None
    np.testing.assert_array_equal(model.X, x)
    assert model.y.shape == (30,)


def test_sampled_weights_match_parameters():
    model, _, _ = train_small()
    n_params = len(list(model.model.parameters()))
    for weights in model.sampled_weights:
        assert len(weights) == n_params
        for w, p in zip(weights, model.model.parameters()):
            assert w.shape == tuple(p.shape)


def test_network_weights_setter_roundtrip():
    model, _, _ = train_small()
    new_weights = [np.zeros_like(w) + 0.25 for w in model.network_weights]
    with torch.no_grad():
        model.network_weights = new_weights
    for w in model.network_weights:
        assert np.allclose(w, 0.25)


def test_normalization_statistics_stored():
    model, x, y = train_small()
    assert np.allclose(model.x_mean, np.mean(x, axis=0))
    assert np.allclose(model.x_std, np.std(x, axis=0), rtol=1e-3)
    assert np.allclose(model.y_mean, np.mean(y))


def test_predict_shapes_and_positive_variance():
    model, _, _ = train_small()
    x_test = np.random.RandomState(1).rand(5, 2)
    mu, var = model.predict(x_test)
    assert mu.shape == (5,)
    assert var.shape == (5,)
    assert np.all(np.isfinite(mu))
    assert np.all(var > 0)


def test_predict_individual_predictions_shape():
    model, _, _ = train_small()
    x_test = np.random.RandomState(1).rand(4, 2)
    mu, var, individual = model.predict(x_test, return_individual_predictions=True)
    assert individual.shape == (len(model.sampled_weights), 4)
    assert mu.shape == (4,)
    assert var.shape == (4,)


def test_predict_single_matches_predict_for_single_sample():
    model, _, _ = train_small(num_steps=20, num_burn_in_steps=5, keep_every=10)
    assert len(model.sampled_weights) == 1
    x_test = np.random.RandomState(2).rand(6, 2)
    mu, var = model.predict(x_test)
    single = model.predict_single(x_test, 0)
    assert single.shape == (6, 2)
    np.testing.assert_allclose(single[:, 0], mu, rtol=1e-6, atol=1e-8)


def test_train_without_normalization():
    model = Bohamiann(normalize_input=False, normalize_output=False,
                      print_every_n_steps=1000)
    model, _, _ = train_small(model=model)
    mu, var = model.predict(np.random.RandomState(3).rand(3, 2))
    assert mu.shape == (3,)
    assert np.all(var > 0)


def test_train_single_precision():
    model = Bohamiann(use_double_precision=False, print_every_n_steps=1000)
    model, _, _ = train_small(model=model)
    assert model.network_weights[0].dtype == np.float32
    mu, var = model.predict(np.random.RandomState(3).rand(3, 2))
    assert mu.shape == (3,)
    assert np.all(np.isfinite(var))


@pytest.mark.parametrize("method", ["adaptive_sghmc", "sgld", "sghmc", "preconditioned_sgld"])
def test_sampling_methods(method):
    model = Bohamiann(sampling_method=method, print_every_n_steps=1000)
    model, _, _ = train_small(model=model)
    assert model.is_trained
    assert len(model.sampled_weights) == 3
    mu, var = model.predict(np.random.RandomState(4).rand(2, 2))
    assert mu.shape == (2,)


def test_train_accepts_column_targets():
    torch.manual_seed(0)
    x, y = make_data()
    model = Bohamiann(print_every_n_steps=1000)
    model.train(x, y[:, None], num_steps=30, num_burn_in_steps=10, keep_every=10,
                batch_size=10, lr=1e-3)
    assert model.y.shape == (30,)
    assert len(model.sampled_weights) >= 1


def test_train_rejects_invalid_inputs():
    x, y = make_data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, np.stack([y, y], axis=1), num_steps=5, num_burn_in_steps=2)
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=5, num_burn_in_steps=2, batch_size=0)


def test_continue_training_keeps_samples():
    model, x, y = train_small(num_steps=30, num_burn_in_steps=10, keep_every=10)
    n_before = len(model.sampled_weights)
    assert n_before >= 1
    model.train(x, y, num_steps=30, num_burn_in_steps=0, keep_every=10,
                batch_size=10, lr=1e-3, continue_training=True)
    assert len(model.sampled_weights) > n_before


def test_retraining_clears_samples():
    model, x, y = train_small()
    model.train(x, y, num_steps=20, num_burn_in_steps=5, keep_every=10,
                batch_size=10, lr=1e-3)
    assert len(model.sampled_weights) == 1


def test_f_gradient_shape_and_finite_difference():
    model, _, _ = train_small()
    x0 = np.array([0.4, 0.6])
    weights = model.sampled_weights[0]
    g = model.f_gradient(x0, weights)
    assert g.shape == (2,)

    eps = 1e-6
    fd = np.zeros(2)
    for i in range(2):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = model.predict_single(xp[None, :], 0)[0, 0]
        fm = model.predict_single(xm[None, :], 0)[0, 0]
        fd[i] = (fp - fm) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-3, atol=1e-5)


def test_predictive_gradients_shapes():
    model, _, _ = train_small()
    x0 = np.array([0.3, 0.7])
    gm = model.predictive_mean_gradient(x0)
    gv = model.predictive_variance_gradient(x0)
    assert gm.shape == (2,)
    assert gv.shape == (2,)
    assert np.all(np.isfinite(gm))
    assert np.all(np.isfinite(gv))


def test_train_and_evaluate_returns_learning_curves():
    torch.manual_seed(0)
    x, y = make_data()
    xv, yv = make_data(n=10, seed=5)
    model = Bohamiann(print_every_n_steps=1000)
    n_steps, ll, mse = model.train_and_evaluate(
        x, y, xv, yv, num_steps=20, validate_every_n_steps=10, keep_every=5,
        num_burn_in_steps=10, lr=1e-3, batch_size=10)
    assert n_steps == [20, 30]
    assert len(ll) == 2
    assert len(mse) == 2
    assert all(np.isfinite(ll))
    assert all(m >= 0 for m in mse)
    assert len(model.sampled_weights) == 2
