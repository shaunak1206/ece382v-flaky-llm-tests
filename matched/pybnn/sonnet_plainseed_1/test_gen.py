import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll


N_POINTS = 16
N_DIM = 2


def make_data(seed=0, n=N_POINTS, d=N_DIM):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, d))
    y = np.sin(3.0 * x[:, 0]) + 0.5 * x[:, 1] + 0.01 * rng.randn(n)
    return x, y


def train_small(model=None, x=None, y=None, num_steps=30, num_burn_in_steps=10,
                keep_every=5, seed=1, **kwargs):
    """Train a tiny model quickly. Produces 3 samples with the defaults."""
    if model is None:
        model = Bohamiann(print_every_n_steps=1000)
    if x is None or y is None:
        x, y = make_data()
    torch.manual_seed(seed)
    np.random.seed(seed)
    kwargs.setdefault("batch_size", 8)
    model.train(x, y, num_steps=num_steps, num_burn_in_steps=num_burn_in_steps,
                keep_every=keep_every, **kwargs)
    return model


def test_nll_zero_for_perfect_mean_and_unit_variance():
    # mean equals target and log variance is 0 -> loss is 0
    inp = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert nll(inp, target).item() == pytest.approx(0.0, abs=1e-9)


def test_nll_known_value():
    # error 2, variance 4: 0.5 * 4 / 4 + 0.5 * log(4)
    inp = torch.tensor([[3.0, float(np.log(4.0))]], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)
    expected = 0.5 + 0.5 * np.log(4.0)
    assert nll(inp, target).item() == pytest.approx(expected, rel=1e-6)


def test_nll_is_averaged_over_batch():
    single = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    batch = torch.tensor([[0.0, 0.0]] * 4, dtype=torch.float64)
    t1 = torch.tensor([1.0], dtype=torch.float64)
    t4 = torch.tensor([1.0] * 4, dtype=torch.float64)
    assert nll(batch, t4).item() == pytest.approx(nll(single, t1).item(), rel=1e-9)


def test_default_network_output_shape_and_initial_log_variance():
    net = get_default_network(3).double()
    x = torch.randn(5, 3, dtype=torch.float64)
    out = net(x)
    assert out.shape == (5, 2)
    assert torch.allclose(out[:, 1], torch.full((5,), float(np.log(1e-3)), dtype=torch.float64))
    # 3 linear layers (weight and bias) plus the appended log variance
    assert len(list(net.parameters())) == 7


def test_constructor_defaults():
    model = Bohamiann()
    assert model.is_trained is False
    assert model.sampled_weights == []
    assert model.sampler is None
    assert model.sampling_method == "adaptive_sghmc"
    assert model.use_double_precision is True
    assert model.do_normalize_input is True
    assert model.do_normalize_output is True
    assert model.print_every_n_steps == 100
    assert model.likelihood_function is nll


def test_network_weights_getter_and_setter_roundtrip():
    model = Bohamiann()
    model.model = get_default_network(3).double()
    weights = model.network_weights
    assert isinstance(weights, tuple)
    assert len(weights) == len(list(model.model.parameters()))
    assert all(isinstance(w, np.ndarray) for w in weights)

    new_weights = [np.full_like(w, 0.25) for w in weights]
    with torch.no_grad():
        model.network_weights = new_weights
    for w in model.network_weights:
        assert np.allclose(w, 0.25)


def test_train_collects_expected_number_of_samples():
    model = train_small()
    # steps 0..29, burn-in 10, keep every 5 -> steps 15, 20, 25
    assert len(model.sampled_weights) == 3
    assert model.is_trained is True
    assert model.sampler is not None


def test_train_stores_data_and_normalization_statistics():
    x, y = make_data()
    model = train_small(x=x, y=y)
    assert model.X is x
    assert np.allclose(model.y, y)
    assert np.allclose(model.x_mean, x.mean(axis=0))
    assert np.allclose(model.x_std, x.std(axis=0))
    assert model.y_mean == pytest.approx(y.mean())
    assert model.y_std == pytest.approx(y.std())


def test_train_accepts_column_vector_targets():
    x, y = make_data()
    model = train_small(x=x, y=y.reshape(-1, 1))
    assert model.y.shape == (N_POINTS,)
    assert len(model.sampled_weights) == 3


def test_train_rejects_invalid_batch_size():
    x, y = make_data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=5, num_burn_in_steps=1, batch_size=0)


def test_train_rejects_multi_column_targets():
    x, y = make_data()
    model = Bohamiann()
    y_bad = np.stack([y, y], axis=1)
    with pytest.raises(AssertionError):
        model.train(x, y_bad, num_steps=5, num_burn_in_steps=1, batch_size=4)


def test_train_with_batch_size_larger_than_dataset():
    x, y = make_data(n=6)
    model = train_small(x=x, y=y, batch_size=50)
    assert model.is_trained
    assert len(model.sampled_weights) == 3


@pytest.mark.parametrize("method", ["adaptive_sghmc", "sgld", "sghmc", "preconditioned_sgld"])
def test_train_with_each_sampling_method(method):
    model = train_small(model=Bohamiann(sampling_method=method, print_every_n_steps=1000))
    assert model.is_trained
    assert len(model.sampled_weights) == 3
    mean, var = model.predict(make_data(seed=3, n=4)[0])
    assert mean.shape == (4,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))


def test_continue_training_keeps_previous_samples():
    model = train_small()
    first_sampler = model.sampler
    assert len(model.sampled_weights) == 3
    train_small(model=model, continue_training=True)
    assert model.sampler is first_sampler
    assert len(model.sampled_weights) == 6


def test_retraining_without_continue_clears_samples():
    model = train_small()
    assert len(model.sampled_weights) == 3
    train_small(model=model, num_steps=20, num_burn_in_steps=10, keep_every=5)
    # steps 0..19 -> samples at steps 15 only
    assert len(model.sampled_weights) == 1


def test_predict_shapes_and_positive_variance():
    model = train_small()
    x_test = make_data(seed=5, n=7)[0]
    mean, var = model.predict(x_test)
    assert mean.shape == (7,)
    assert var.shape == (7,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_predict_is_deterministic_for_fixed_samples():
    model = train_small()
    x_test = make_data(seed=6, n=5)[0]
    mean1, var1 = model.predict(x_test)
    mean2, var2 = model.predict(x_test)
    assert np.allclose(mean1, mean2)
    assert np.allclose(var1, var2)


def test_predict_individual_predictions_consistent_with_mean():
    model = train_small()
    x_test = make_data(seed=7, n=6)[0]
    mean, var, individual = model.predict(x_test, return_individual_predictions=True)
    assert individual.shape == (3, 6)
    assert np.allclose(individual.mean(axis=0), mean)
    assert var.shape == (6,)


def test_predict_single_matches_predict_for_one_sample():
    model = train_small(num_steps=20, num_burn_in_steps=10, keep_every=5)
    assert len(model.sampled_weights) == 1
    x_test = make_data(seed=8, n=5)[0]
    single = model.predict_single(x_test, 0)
    mean, _ = model.predict(x_test)
    assert single.shape == (5, 2)
    assert np.allclose(single[:, 0], mean)


def test_predict_without_normalization_and_in_single_precision():
    model = Bohamiann(normalize_input=False, normalize_output=False,
                      use_double_precision=False, print_every_n_steps=1000)
    train_small(model=model)
    assert model.model.fc1.weight.dtype == torch.float32
    x_test = make_data(seed=9, n=4)[0]
    mean, var = model.predict(x_test)
    assert mean.shape == (4,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_normalize_helpers():
    model = Bohamiann()
    x, _ = make_data()
    x_norm, mean, std = model.normalize_input(x)
    assert np.allclose(x_norm.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(x_norm.std(axis=0), 1.0)
    # given statistics are used as is
    x_norm2, mean2, std2 = model.normalize_output(np.array([1.0, 3.0]), 2.0, 2.0)
    assert np.allclose(x_norm2, [-0.5, 0.5])
    assert mean2 == 2.0 and std2 == 2.0


@pytest.mark.parametrize("normalize", [False, True])
def test_f_gradient_matches_finite_differences(normalize):
    model = Bohamiann(normalize_input=normalize, normalize_output=normalize,
                      print_every_n_steps=1000)
    train_small(model=model)
    weights = model.sampled_weights[0]
    x0 = np.array([0.3, -0.2])
    grad = model.f_gradient(x0, weights)
    assert grad.shape == (N_DIM,)

    eps = 1e-6
    numeric = np.zeros(N_DIM)
    for i in range(N_DIM):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = model.predict_single(xp[None, :], 0)[0, 0]
        fm = model.predict_single(xm[None, :], 0)[0, 0]
        numeric[i] = (fp - fm) / (2 * eps)
    assert np.allclose(grad, numeric, rtol=1e-3, atol=1e-4)


def test_predictive_gradients_shapes():
    model = train_small()
    x0 = np.array([0.1, 0.4])
    dm = model.predictive_mean_gradient(x0)
    dv = model.predictive_variance_gradient(x0)
    assert dm.shape == (N_DIM,)
    assert dv.shape == (N_DIM,)
    assert np.all(np.isfinite(dm))
    assert np.all(np.isfinite(dv))


def test_train_and_evaluate_returns_learning_curves():
    x, y = make_data()
    x_valid, y_valid = make_data(seed=11, n=6)
    model = Bohamiann(print_every_n_steps=1000)
    torch.manual_seed(2)
    n_steps, ll, mse = model.train_and_evaluate(
        x, y, x_valid, y_valid,
        num_steps=20, validate_every_n_steps=10, keep_every=5,
        num_burn_in_steps=10, batch_size=8)
    assert n_steps == [20, 30]
    assert len(ll) == 2 and len(mse) == 2
    assert all(np.isfinite(v) for v in ll)
    assert all(v >= 0 for v in mse)
    assert model.is_trained


def test_verbose_training_prints_progress(capsys):
    model = Bohamiann(print_every_n_steps=5)
    train_small(model=model, verbose=True)
    out = capsys.readouterr().out
    assert "Step" in out
    assert "NLL" in out
    assert "MSE" in out
