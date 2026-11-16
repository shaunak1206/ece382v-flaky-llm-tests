import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, nll, get_default_network


NUM_STEPS = 30
BURN_IN = 10
KEEP_EVERY = 5
EXPECTED_SAMPLES = 3  # steps 15, 20, 25


def _data(n=20, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, d))
    y = np.sin(3 * x[:, 0]) + 0.5 * x[:, 1] + 0.05 * rng.randn(n)
    return x, y


def _train(bnn, x, y, **kwargs):
    params = dict(num_steps=NUM_STEPS, num_burn_in_steps=BURN_IN,
                  keep_every=KEEP_EVERY, batch_size=10, lr=1e-3)
    params.update(kwargs)
    bnn.train(x, y, **params)
    return bnn


def test_nll_known_values():
    np.random.seed(0)
    torch.manual_seed(0)
    inp = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)
    # 0.5 * 0.25 / 1 + 0.5 * 0
    assert nll(inp, target).item() == pytest.approx(0.125, abs=1e-9)

    inp = torch.tensor([[0.0, float(np.log(4.0))]], dtype=torch.float64)
    target = torch.tensor([2.0], dtype=torch.float64)
    expected = 0.5 * 4.0 / 4.0 + 0.5 * np.log(4.0)
    assert nll(inp, target).item() == pytest.approx(expected, abs=1e-9)


def test_nll_averages_over_batch_and_target_shapes():
    np.random.seed(0)
    torch.manual_seed(0)
    inp = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    t_vec = torch.tensor([1.0, 1.0], dtype=torch.float64)
    t_col = t_vec.view(-1, 1)
    # row losses: 0.5 and 0.0 -> average 0.25
    assert nll(inp, t_vec).item() == pytest.approx(0.25, abs=1e-9)
    assert nll(inp, t_col).item() == pytest.approx(0.25, abs=1e-9)


def test_nll_is_differentiable():
    np.random.seed(0)
    torch.manual_seed(0)
    inp = torch.tensor([[0.5, 0.0], [0.1, 0.2]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1.0, -1.0], dtype=torch.float64)
    loss = nll(inp, target)
    loss.backward()
    assert inp.grad is not None
    assert inp.grad.shape == inp.shape
    assert torch.all(torch.isfinite(inp.grad))
    # d nll / d mean for first row: -(t - m) / var / batch = -0.5 / 2
    assert inp.grad[0, 0].item() == pytest.approx(-0.25, abs=1e-9)


def test_get_default_network_output_shape():
    np.random.seed(0)
    torch.manual_seed(0)
    net = get_default_network(input_dimensionality=3)
    out = net(torch.zeros(7, 3))
    assert out.shape == (7, 2)
    assert torch.all(torch.isfinite(out))


def test_init_defaults():
    np.random.seed(0)
    torch.manual_seed(0)
    bnn = Bohamiann()
    assert bnn.is_trained is False
    assert bnn.sampled_weights == []
    assert bnn.sampler is None
    assert bnn.sampling_method == "adaptive_sghmc"
    assert bnn.use_double_precision is True
    assert bnn.do_normalize_input is True
    assert bnn.do_normalize_output is True


@pytest.mark.parametrize("method", ["adaptive_sghmc", "sgld", "sghmc", "preconditioned_sgld"])
def test_train_with_all_sampling_methods(method):
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = Bohamiann(sampling_method=method)
    _train(bnn, x, y)
    assert bnn.is_trained is True
    assert bnn.sampler is not None
    assert len(bnn.sampled_weights) == EXPECTED_SAMPLES
    mu, var = bnn.predict(x[:4])
    assert mu.shape == (4,)
    assert var.shape == (4,)


def test_sample_weights_match_network_structure():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = _train(Bohamiann(), x, y)
    current = bnn.network_weights
    assert isinstance(current, tuple)
    for sample in bnn.sampled_weights:
        assert len(sample) == len(current)
        for a, b in zip(sample, current):
            assert a.shape == b.shape


def test_no_samples_when_steps_within_burn_in():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = _train(Bohamiann(), x, y, num_steps=8, num_burn_in_steps=10)
    assert bnn.is_trained is True
    assert bnn.sampled_weights == []


def test_normalization_statistics_stored():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = _train(Bohamiann(), x, y)
    np.testing.assert_allclose(bnn.x_mean, x.mean(axis=0), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(bnn.x_std, x.std(axis=0), rtol=1e-9, atol=1e-12)
    assert float(bnn.y_mean) == pytest.approx(float(y.mean()), abs=1e-9)
    assert float(bnn.y_std) == pytest.approx(float(y.std()), abs=1e-9)


def test_normalize_input_method_gives_zero_mean_unit_var():
    np.random.seed(0)
    torch.manual_seed(0)
    x, _ = _data(n=50)
    bnn = Bohamiann()
    x_norm, m, s = bnn.normalize_input(x)
    np.testing.assert_allclose(x_norm.mean(axis=0), np.zeros(2), atol=1e-9)
    np.testing.assert_allclose(x_norm.std(axis=0), np.ones(2), atol=1e-9)
    # reusing given statistics gives the same result
    x_norm2, _, _ = bnn.normalize_input(x, m, s)
    np.testing.assert_allclose(x_norm, x_norm2, atol=1e-12)


def test_train_without_normalization():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = Bohamiann(normalize_input=False, normalize_output=False)
    _train(bnn, x, y)
    assert not hasattr(bnn, "x_mean")
    assert not hasattr(bnn, "y_mean")
    mu, var = bnn.predict(x[:3])
    assert mu.shape == (3,)
    assert var.shape == (3,)


def test_train_accepts_column_targets():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = _train(Bohamiann(), x, y.reshape(-1, 1))
    assert bnn.y.shape == (20,)
    assert bnn.is_trained


def test_train_invalid_arguments_raise():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = Bohamiann()
    with pytest.raises(AssertionError):
        bnn.train(x, y, num_steps=5, batch_size=0)
    with pytest.raises(AssertionError):
        bnn.train(x, np.stack([y, y], axis=1), num_steps=5)


def test_batch_size_larger_than_dataset_is_handled():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data(n=6)
    bnn = _train(Bohamiann(), x, y, batch_size=50)
    assert bnn.is_trained
    assert len(bnn.sampled_weights) == EXPECTED_SAMPLES


def test_continue_training_keeps_model_and_accumulates_samples():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = _train(Bohamiann(), x, y)
    model = bnn.model
    sampler = bnn.sampler
    assert len(bnn.sampled_weights) == EXPECTED_SAMPLES
    _train(bnn, x, y, continue_training=True)
    assert bnn.model is model
    assert bnn.sampler is sampler
    assert len(bnn.sampled_weights) == 2 * EXPECTED_SAMPLES
    # retraining from scratch clears the samples
    _train(bnn, x, y)
    assert bnn.model is not model
    assert len(bnn.sampled_weights) == EXPECTED_SAMPLES


def test_custom_network_and_likelihood_are_used():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    dims = []
    calls = []

    def get_net(input_dimensionality):
        dims.append(input_dimensionality)
        return get_default_network(input_dimensionality)

    def lik(input, target):
        calls.append(1)
        return nll(input, target)

    bnn = Bohamiann(get_network=get_net, likelihood_function=lik)
    _train(bnn, x, y)
    assert dims == [2]
    assert len(calls) == NUM_STEPS


def test_predict_shapes_and_consistency():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = _train(Bohamiann(), x, y)
    x_test = x[:5]
    mu, var = bnn.predict(x_test)
    mu2, var2, individual = bnn.predict(x_test, return_individual_predictions=True)
    assert mu.shape == (5,)
    assert var.shape == (5,)
    assert individual.shape == (EXPECTED_SAMPLES, 5)
    np.testing.assert_allclose(mu, mu2, atol=1e-10)
    np.testing.assert_allclose(var, var2, atol=1e-10)
    np.testing.assert_allclose(mu, individual.mean(axis=0), rtol=1e-8, atol=1e-8)
    assert np.all(var > 0)
    # predicting is deterministic
    mu3, var3 = bnn.predict(x_test)
    np.testing.assert_allclose(mu, mu3, atol=1e-12)
    np.testing.assert_allclose(var, var3, atol=1e-12)


def test_predict_single_matches_individual_predictions():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = _train(Bohamiann(), x, y)
    _, _, individual = bnn.predict(x[:4], return_individual_predictions=True)
    for i in range(EXPECTED_SAMPLES):
        single = bnn.predict_single(x[:4], i)
        assert single.shape == (4, 2)
        np.testing.assert_allclose(single[:, 0], individual[i], rtol=1e-8, atol=1e-8)


def test_network_weights_setter_roundtrip():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = _train(Bohamiann(), x, y)
    zeros = [np.zeros_like(w) for w in bnn.network_weights]
    with torch.no_grad():
        bnn.network_weights = zeros
    for w in bnn.network_weights:
        assert np.all(w == 0)
    # returned arrays are copies, not views of the parameters
    w = bnn.network_weights
    w[0][...] = 5.0
    assert np.all(bnn.network_weights[0] == 0)


def test_weight_dtypes_follow_precision_setting():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn64 = _train(Bohamiann(use_double_precision=True), x, y)
    assert all(w.dtype == np.float64 for w in bnn64.network_weights)

    bnn32 = _train(Bohamiann(use_double_precision=False), x, y)
    assert all(w.dtype == np.float32 for w in bnn32.network_weights)
    mu, var = bnn32.predict(x[:3])
    assert mu.shape == (3,)
    assert var.shape == (3,)


def test_f_gradient_matches_finite_differences():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = _train(Bohamiann(), x, y)
    weights = bnn.sampled_weights[0]
    x0 = np.array([0.1, -0.2])
    grad = bnn.f_gradient(x0, weights=weights)
    assert grad.shape == (2,)

    eps = 1e-5
    numeric = np.zeros(2)
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        fp = bnn.predict_single((x0 + e)[None, :], 0)[0, 0]
        fm = bnn.predict_single((x0 - e)[None, :], 0)[0, 0]
        numeric[j] = (fp - fm) / (2 * eps)
    np.testing.assert_allclose(grad, numeric, rtol=1e-4, atol=1e-6)


def test_predictive_gradients_shapes_and_mean_gradient():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = _train(Bohamiann(), x, y)
    x0 = np.array([0.3, 0.4])
    grads = np.array([bnn.f_gradient(x0, weights=w) for w in bnn.sampled_weights])
    dmdx = bnn.predictive_mean_gradient(x0)
    assert dmdx.shape == (2,)
    np.testing.assert_allclose(dmdx, grads.mean(axis=0), rtol=1e-8, atol=1e-10)
    dvdx = bnn.predictive_variance_gradient(x0)
    assert dvdx.shape == (2,)


def test_train_and_evaluate_learning_curves():
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data(n=30)
    xv, yv = _data(n=10, seed=1)
    bnn = Bohamiann()
    n_steps, ll, mse = bnn.train_and_evaluate(
        x, y, xv, yv,
        num_steps=20, validate_every_n_steps=10, keep_every=5,
        num_burn_in_steps=5, lr=1e-3, batch_size=10)
    assert n_steps == [15, 25]
    assert len(ll) == 2
    assert len(mse) == 2
    assert all(m >= 0 for m in mse)
    assert bnn.is_trained
    assert len(bnn.sampled_weights) == 2


def test_verbose_training_prints_progress(capsys):
    np.random.seed(0)
    torch.manual_seed(0)
    x, y = _data()
    bnn = Bohamiann(normalize_output=False, print_every_n_steps=5)
    _train(bnn, x, y, num_steps=12, num_burn_in_steps=8, keep_every=1, verbose=True)
    out = capsys.readouterr().out
    assert "Step" in out
    assert "NLL" in out
    assert "MSE" in out
