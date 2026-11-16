import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def make_data(n=30, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, size=(n, d))
    y = np.sum(x, axis=1) + 0.01 * rng.randn(n)
    return x, y


def trained_model(sampling_method="adaptive_sghmc", use_double_precision=True,
                  normalize_input=True, normalize_output=True, n=30, d=2):
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_data(n=n, d=d)
    model = Bohamiann(sampling_method=sampling_method,
                      use_double_precision=use_double_precision,
                      normalize_input=normalize_input,
                      normalize_output=normalize_output,
                      print_every_n_steps=10)
    model.train(x, y, num_steps=60, keep_every=10, num_burn_in_steps=20,
                batch_size=10, lr=1e-3)
    return model, x, y


# ---------------------------------------------------------------- nll

def test_nll_zero_when_perfect_and_unit_variance():
    inp = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.double)
    target = torch.tensor([1.0, 2.0], dtype=torch.double)
    assert float(nll(inp, target)) == pytest.approx(0.0, abs=1e-9)


def test_nll_value_with_squared_error():
    inp = torch.tensor([[0.0, 0.0]], dtype=torch.double)
    target = torch.tensor([2.0], dtype=torch.double)
    # 0.5 * 4 / 1 + 0.5 * 0 = 2
    assert float(nll(inp, target)) == pytest.approx(2.0, abs=1e-6)


def test_nll_includes_log_variance_term_and_averages_over_batch():
    log_var = float(np.log(4.0))
    inp = torch.tensor([[1.0, log_var], [1.0, log_var]], dtype=torch.double)
    target = torch.tensor([1.0, 1.0], dtype=torch.double)
    assert float(nll(inp, target)) == pytest.approx(0.5 * log_var, abs=1e-9)


def test_nll_accepts_column_target():
    inp = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.double)
    t_vec = torch.tensor([1.0, 3.0], dtype=torch.double)
    t_col = t_vec.view(-1, 1)
    assert float(nll(inp, t_vec)) == pytest.approx(float(nll(inp, t_col)))


def test_nll_is_differentiable():
    inp = torch.tensor([[0.5, 0.1], [0.2, -0.3]], dtype=torch.double, requires_grad=True)
    target = torch.tensor([1.0, 0.0], dtype=torch.double)
    loss = nll(inp, target)
    loss.backward()
    assert inp.grad is not None
    assert torch.all(torch.isfinite(inp.grad))


# ------------------------------------------------------- default network

def test_default_network_output_shape():
    net = get_default_network(3).double()
    out = net(torch.randn(5, 3, dtype=torch.double))
    assert out.shape == (5, 2)
    assert torch.all(torch.isfinite(out))


def test_default_network_log_variance_column_constant():
    net = get_default_network(4).double()
    out = net(torch.randn(6, 4, dtype=torch.double))
    assert torch.allclose(out[:, 1], out[0, 1].expand(6))


# ------------------------------------------------------------ init state

def test_init_state():
    model = Bohamiann()
    assert model.is_trained is False
    assert model.sampled_weights == []
    assert model.sampler is None
    assert model.do_normalize_input is True
    assert model.do_normalize_output is True
    assert model.sampling_method == "adaptive_sghmc"


# -------------------------------------------------------------- training

def test_train_collects_expected_number_of_samples():
    model, x, y = trained_model()
    assert model.is_trained
    # steps 30, 40, 50 are kept
    assert len(model.sampled_weights) == 3
    n_params = len(list(model.model.parameters()))
    for w in model.sampled_weights:
        assert len(w) == n_params
        assert all(isinstance(a, np.ndarray) for a in w)


def test_train_stores_data_and_normalization_statistics():
    model, x, y = trained_model()
    np.testing.assert_allclose(model.X, x)
    np.testing.assert_allclose(model.y, y)
    np.testing.assert_allclose(model.x_mean, x.mean(axis=0))
    np.testing.assert_allclose(model.x_std, x.std(axis=0))
    assert model.y_mean == pytest.approx(y.mean())
    assert model.y_std == pytest.approx(y.std())


def test_train_accepts_column_targets():
    torch.manual_seed(1)
    x, y = make_data()
    model = Bohamiann(print_every_n_steps=10)
    model.train(x, y.reshape(-1, 1), num_steps=30, keep_every=5,
                num_burn_in_steps=10, batch_size=10)
    assert model.y.shape == (x.shape[0],)
    assert model.is_trained


def test_train_rejects_invalid_targets_and_batch_size():
    x, y = make_data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, np.stack([y, y], axis=1), num_steps=5, num_burn_in_steps=1)
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=5, num_burn_in_steps=1, batch_size=0)


def test_continue_training_keeps_and_extends_samples():
    model, x, y = trained_model()
    before = len(model.sampled_weights)
    model.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=0,
                batch_size=10, continue_training=True)
    # steps 5, 10, 15 are kept
    assert len(model.sampled_weights) == before + 3


def test_retraining_without_continue_clears_samples():
    model, x, y = trained_model()
    model.train(x, y, num_steps=30, keep_every=5, num_burn_in_steps=20, batch_size=10)
    # steps 25 only
    assert len(model.sampled_weights) == 1


def test_batch_size_larger_than_dataset_is_handled():
    torch.manual_seed(2)
    x, y = make_data(n=8)
    model = Bohamiann()
    model.train(x, y, num_steps=15, keep_every=5, num_burn_in_steps=5, batch_size=20)
    assert model.is_trained
    assert len(model.sampled_weights) == 2  # steps 10 and... 15 not reached -> check below


@pytest.mark.parametrize("method", ["adaptive_sghmc", "sgld"])
def test_sampling_methods_produce_samples(method):
    model, x, y = trained_model(sampling_method=method)
    assert len(model.sampled_weights) == 3
    mean, var = model.predict(x[:4])
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))


# ------------------------------------------------------------ prediction

def test_predict_shapes_and_positive_variance():
    model, x, y = trained_model()
    mean, var = model.predict(x[:7])
    assert mean.shape == (7,)
    assert var.shape == (7,)
    assert np.all(var > 0)


def test_predict_individual_predictions_shape():
    model, x, y = trained_model()
    mean, var, indiv = model.predict(x[:5], return_individual_predictions=True)
    assert indiv.shape == (len(model.sampled_weights), 5)
    assert mean.shape == (5,)


def test_predict_is_deterministic():
    model, x, y = trained_model()
    m1, v1 = model.predict(x[:5])
    m2, v2 = model.predict(x[:5])
    np.testing.assert_allclose(m1, m2)
    np.testing.assert_allclose(v1, v2)


def test_predict_without_normalization():
    model, x, y = trained_model(normalize_input=False, normalize_output=False)
    mean, var = model.predict(x[:5])
    assert mean.shape == (5,)
    assert np.all(var > 0)


def test_predict_single_precision():
    model, x, y = trained_model(use_double_precision=False)
    mean, var = model.predict(x[:5])
    assert mean.shape == (5,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_predict_single_shape_and_consistency_with_ensemble_mean():
    model, x, y = trained_model()
    xt = x[:6]
    out = model.predict_single(xt, 0)
    assert out.shape == (6, 2)
    singles = np.array([model.predict_single(xt, i)[:, 0]
                        for i in range(len(model.sampled_weights))])
    mean, _ = model.predict(xt)
    np.testing.assert_allclose(singles.mean(axis=0), mean, rtol=1e-6, atol=1e-8)


# ------------------------------------------------------------- gradients

def test_f_gradient_matches_finite_differences():
    model, x, y = trained_model()
    weights = model.sampled_weights[0]
    x0 = x[0].copy()
    grad = model.f_gradient(x0, weights)
    assert grad.shape == (x.shape[1],)

    eps = 1e-6
    numeric = np.zeros_like(x0)
    for j in range(x0.shape[0]):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        fp = model.predict_single(xp[None, :], 0)[0, 0]
        fm = model.predict_single(xm[None, :], 0)[0, 0]
        numeric[j] = (fp - fm) / (2 * eps)
    np.testing.assert_allclose(grad, numeric, rtol=1e-3, atol=1e-5)


def test_predictive_gradients_shapes_and_mean_gradient():
    model, x, y = trained_model()
    x0 = x[1]
    g_mean = model.predictive_mean_gradient(x0)
    assert g_mean.shape == (x.shape[1],)
    expected = np.mean([model.f_gradient(x0, w) for w in model.sampled_weights], axis=0)
    np.testing.assert_allclose(g_mean, expected)

    g_var = model.predictive_variance_gradient(x0)
    assert g_var.shape == (x.shape[1],)
    assert np.all(np.isfinite(g_var))


# ------------------------------------------------------ train_and_evaluate

def test_train_and_evaluate_learning_curves():
    torch.manual_seed(3)
    x, y = make_data()
    xv, yv = make_data(n=10, seed=5)
    model = Bohamiann(print_every_n_steps=10)
    n_steps, ll, mse = model.train_and_evaluate(
        x, y, xv, yv, num_steps=20, validate_every_n_steps=10,
        keep_every=5, num_burn_in_steps=10, batch_size=10, lr=1e-3)
    assert n_steps == [20, 30]
    assert len(ll) == 2
    assert len(mse) == 2
    assert all(np.isfinite(ll))
    assert all(m >= 0 for m in mse)
    assert model.is_trained
