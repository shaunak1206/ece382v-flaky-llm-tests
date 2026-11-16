import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def make_data(n=12, d=2):
    rng = np.random.RandomState(0)
    x = rng.uniform(-1, 1, size=(n, d))
    y = np.sin(x.sum(axis=1))
    return x, y


def trained_model(num_steps=10, keep_every=2, num_burn_in_steps=2, batch_size=5, **kwargs):
    x, y = make_data()
    model = Bohamiann(print_every_n_steps=1000, **kwargs)
    model.train(x, y, num_steps=num_steps, keep_every=keep_every,
                num_burn_in_steps=num_burn_in_steps, batch_size=batch_size)
    return model, x, y


def test_nll_perfect_mean_unit_variance_is_zero():
    torch.manual_seed(0)
    np.random.seed(0)
    inp = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert abs(nll(inp, target).item()) < 1e-9


def test_nll_known_value():
    torch.manual_seed(0)
    np.random.seed(0)
    # mean 0, target 1, log variance 0 -> 0.5 * 1 = 0.5
    inp = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)
    assert abs(nll(inp, target).item() - 0.5) < 1e-6


def test_nll_is_averaged_over_batch():
    torch.manual_seed(0)
    np.random.seed(0)
    inp = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 1.0], dtype=torch.float64)
    single = nll(inp[:1], target[:1]).item()
    both = nll(inp, target).item()
    assert abs(single - both) < 1e-9


def test_nll_accepts_column_target_and_log_variance_term():
    torch.manual_seed(0)
    np.random.seed(0)
    log_var = 2.0
    inp = torch.tensor([[0.5, log_var]], dtype=torch.float64)
    target = torch.tensor([[0.5]], dtype=torch.float64)
    # perfect mean -> only -(-0.5 * log_var) = 0.5 * log_var
    assert abs(nll(inp, target).item() - 0.5 * log_var) < 1e-9


def test_default_network_output_shape():
    torch.manual_seed(0)
    np.random.seed(0)
    net = get_default_network(3).double()
    out = net(torch.zeros(5, 3, dtype=torch.float64))
    assert tuple(out.shape) == (5, 2)


def test_constructor_defaults():
    torch.manual_seed(0)
    np.random.seed(0)
    m = Bohamiann()
    assert m.is_trained is False
    assert m.sampled_weights == []
    assert m.sampler is None
    assert m.sampling_method == "adaptive_sghmc"
    assert m.use_double_precision is True


def test_normalize_helpers():
    torch.manual_seed(0)
    np.random.seed(0)
    x, _ = make_data()
    m = Bohamiann()
    xn, mean, std = m.normalize_input(x)
    assert np.allclose(mean, x.mean(axis=0))
    assert np.allclose(std, x.std(axis=0))
    assert np.allclose(xn.mean(axis=0), 0.0, atol=1e-9)
    xn2, _, _ = m.normalize_output(x, np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert np.allclose(xn2, (x - 1.0) / 2.0)


def test_train_collects_expected_number_of_samples():
    torch.manual_seed(0)
    np.random.seed(0)
    model, x, y = trained_model()
    # steps 0..9, kept when step > 2 and (step - 2) % 2 == 0 -> 4, 6, 8
    assert model.is_trained is True
    assert len(model.sampled_weights) == 3
    assert len(model.sampled_weights[0]) == len(list(model.model.parameters()))
    assert np.allclose(model.x_mean, x.mean(axis=0))
    assert np.allclose(model.y_mean, y.mean())


@pytest.mark.parametrize("method", ["adaptive_sghmc", "sgld", "sghmc", "preconditioned_sgld"])
def test_train_all_sampling_methods(method):
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_data()
    model = Bohamiann(sampling_method=method, print_every_n_steps=1000)
    model.train(x, y, num_steps=10, keep_every=2, num_burn_in_steps=2,
                batch_size=5, lr=1e-4)
    assert model.sampler is not None
    assert model.is_trained
    assert len(model.sampled_weights) == 3
    mean, var = model.predict(x[:4])
    assert mean.shape == (4,)
    assert var.shape == (4,)


def test_train_validates_inputs():
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=2, batch_size=0)
    with pytest.raises(AssertionError):
        model.train(x, np.stack([y, y], axis=1), num_steps=2, batch_size=5)


def test_train_accepts_column_targets_and_large_batch_size():
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_data()
    model = Bohamiann(print_every_n_steps=1000)
    model.train(x, y[:, None], num_steps=6, keep_every=2, num_burn_in_steps=1,
                batch_size=100)
    assert model.y.shape == (x.shape[0],)
    assert model.is_trained
    assert len(model.sampled_weights) >= 1


def test_continue_training_appends_and_restart_clears():
    torch.manual_seed(0)
    np.random.seed(0)
    model, x, y = trained_model()
    before = len(model.sampled_weights)
    model.train(x, y, num_steps=6, keep_every=2, num_burn_in_steps=0,
                batch_size=5, continue_training=True)
    # steps 0..5, kept at step 2 and 4
    assert len(model.sampled_weights) == before + 2
    model.train(x, y, num_steps=10, keep_every=2, num_burn_in_steps=2, batch_size=5)
    assert len(model.sampled_weights) == before


def test_predict_shapes_and_positive_variance():
    torch.manual_seed(0)
    np.random.seed(0)
    model, x, _ = trained_model()
    mean, var = model.predict(x[:5])
    assert mean.shape == (5,)
    assert var.shape == (5,)
    assert np.all(var > 0)
    mean2, var2, indiv = model.predict(x[:5], return_individual_predictions=True)
    assert indiv.shape == (3, 5)
    assert np.allclose(mean, mean2)
    assert np.allclose(var, var2)
    assert np.allclose(indiv.mean(axis=0), mean)


def test_predict_is_deterministic_given_samples():
    torch.manual_seed(0)
    np.random.seed(0)
    model, x, _ = trained_model()
    m1, v1 = model.predict(x)
    m2, v2 = model.predict(x)
    assert np.allclose(m1, m2)
    assert np.allclose(v1, v2)


def test_predict_single_matches_individual_predictions():
    torch.manual_seed(0)
    np.random.seed(0)
    model, x, _ = trained_model()
    _, _, indiv = model.predict(x[:4], return_individual_predictions=True)
    for i in range(len(model.sampled_weights)):
        single = model.predict_single(x[:4], i)
        assert single.shape == (4, 2)
        assert np.allclose(single[:, 0], indiv[i])


def test_without_normalization():
    torch.manual_seed(0)
    np.random.seed(0)
    model, x, _ = trained_model(normalize_input=False, normalize_output=False)
    assert not hasattr(model, "x_mean")
    mean, var = model.predict(x[:3])
    assert mean.shape == (3,)
    assert np.all(var > 0)


def test_float32_training():
    torch.manual_seed(0)
    np.random.seed(0)
    model, x, _ = trained_model(use_double_precision=False)
    assert model.sampled_weights[0][0].dtype == np.float32
    mean, var = model.predict(x[:3])
    assert mean.shape == (3,)
    assert var.shape == (3,)


def test_training_is_reproducible_with_same_seed():
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_data()
    results = []
    for _ in range(2):
        torch.manual_seed(123)
        np.random.seed(123)
        m = Bohamiann(print_every_n_steps=1000)
        m.train(x, y, num_steps=8, keep_every=2, num_burn_in_steps=2, batch_size=5)
        results.append(m.sampled_weights)
    assert len(results[0]) == len(results[1])
    for w1, w2 in zip(results[0], results[1]):
        for a, b in zip(w1, w2):
            assert np.allclose(a, b)


def test_network_weights_property_and_setter():
    torch.manual_seed(0)
    np.random.seed(0)
    model, _, _ = trained_model()
    weights = model.network_weights
    params = list(model.model.parameters())
    assert isinstance(weights, tuple)
    assert len(weights) == len(params)
    for w, p in zip(weights, params):
        assert w.shape == tuple(p.shape)
    with torch.no_grad():
        model.network_weights = [np.zeros_like(w) for w in weights]
    assert all(np.all(w == 0) for w in model.network_weights)


def test_custom_network_and_likelihood_are_used():
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_data()
    net_calls = []
    nll_calls = []

    def get_net(input_dimensionality):
        net_calls.append(input_dimensionality)
        return get_default_network(input_dimensionality)

    def lik(input, target):
        nll_calls.append(1)
        return nll(input, target)

    model = Bohamiann(get_network=get_net, likelihood_function=lik, print_every_n_steps=1000)
    model.train(x, y, num_steps=7, keep_every=2, num_burn_in_steps=1, batch_size=5)
    assert net_calls == [x.shape[1]]
    assert len(nll_calls) == 7


def test_verbose_training_prints_progress(capsys):
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_data()
    model = Bohamiann(print_every_n_steps=4)
    model.train(x, y, num_steps=10, keep_every=2, num_burn_in_steps=2,
                batch_size=5, verbose=True)
    out = capsys.readouterr().out
    assert "Step" in out
    assert "NLL" in out


def test_train_and_evaluate_learning_curves():
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_data()
    x_valid, y_valid = make_data(n=6)
    model = Bohamiann(print_every_n_steps=1000)
    n_steps, ll, mse = model.train_and_evaluate(
        x, y, x_valid, y_valid, num_steps=4, validate_every_n_steps=2,
        keep_every=1, num_burn_in_steps=2, batch_size=5)
    assert n_steps == [4, 6]
    assert len(ll) == 2
    assert len(mse) == 2
    assert model.is_trained
    assert len(model.sampled_weights) == 2


def test_f_gradient_matches_finite_differences():
    torch.manual_seed(0)
    np.random.seed(0)
    model, x, _ = trained_model()
    weights = model.sampled_weights[0]
    x0 = x[0].copy()
    grad = model.f_gradient(x0, weights)
    assert grad.shape == (x.shape[1],)

    eps = 1e-6
    numeric = np.zeros_like(x0)
    for j in range(len(x0)):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        fp = model.predict_single(xp[None, :], 0)[0, 0]
        fm = model.predict_single(xm[None, :], 0)[0, 0]
        numeric[j] = (fp - fm) / (2 * eps)
    assert np.allclose(grad, numeric, atol=1e-4, rtol=1e-3)


def test_predictive_gradients_shapes_and_mean_consistency():
    torch.manual_seed(0)
    np.random.seed(0)
    model, x, _ = trained_model()
    x0 = x[1]
    grads = np.array([model.f_gradient(x0, w) for w in model.sampled_weights])
    dmdx = model.predictive_mean_gradient(x0)
    assert dmdx.shape == (x.shape[1],)
    assert np.allclose(dmdx, grads.mean(axis=0))
    dvdx = model.predictive_variance_gradient(x0)
    assert dvdx.shape == (x.shape[1],)
    assert np.all(np.isfinite(dvdx))
