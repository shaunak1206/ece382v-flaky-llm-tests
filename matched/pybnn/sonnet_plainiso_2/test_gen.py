import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll
from pybnn.sampler import AdaptiveSGHMC, SGLD, SGHMC


def make_data(n=30, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, size=(n, d))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] if d > 1 else np.sin(x[:, 0])
    return x, y


def train_small(model=None, n=30, d=2, num_steps=60, keep_every=10, burn_in=20, **kwargs):
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = make_data(n, d)
    if model is None:
        model = Bohamiann(print_every_n_steps=10)
    model.train(x, y, num_steps=num_steps, keep_every=keep_every,
                num_burn_in_steps=burn_in, batch_size=10, **kwargs)
    return model, x, y


def test_get_default_network_output_shape():
    net = get_default_network(3).double()
    out = net(torch.zeros(5, 3, dtype=torch.double))
    assert out.shape == (5, 2)


def test_get_default_network_log_var_initial_value():
    net = get_default_network(2).double()
    out = net(torch.randn(4, 2, dtype=torch.double))
    assert np.allclose(out[:, 1].detach().numpy(), np.log(1e-3))


def test_nll_known_value():
    # mean 0, log variance 0 -> variance 1; target 1 -> 0.5 * 1 = 0.5 nll
    inp = torch.tensor([[0.0, 0.0]], dtype=torch.double)
    target = torch.tensor([1.0], dtype=torch.double)
    val = nll(inp, target)
    assert val.item() == pytest.approx(0.5, abs=1e-8)


def test_nll_perfect_prediction_depends_only_on_variance():
    inp = torch.tensor([[2.0, 0.0], [3.0, 0.0]], dtype=torch.double)
    target = torch.tensor([2.0, 3.0], dtype=torch.double)
    assert nll(inp, target).item() == pytest.approx(0.0, abs=1e-8)


def test_nll_larger_error_gives_larger_loss():
    target = torch.tensor([0.0, 0.0], dtype=torch.double)
    small = nll(torch.tensor([[0.1, 0.0], [0.1, 0.0]], dtype=torch.double), target)
    large = nll(torch.tensor([[2.0, 0.0], [2.0, 0.0]], dtype=torch.double), target)
    assert large.item() > small.item()


def test_init_defaults():
    model = Bohamiann()
    assert model.is_trained is False
    assert model.sampled_weights == []
    assert model.sampler is None
    assert model.do_normalize_input is True
    assert model.do_normalize_output is True
    assert model.sampling_method == "adaptive_sghmc"
    assert model.use_double_precision is True
    assert model.likelihood_function is nll


def test_init_custom_flags():
    model = Bohamiann(normalize_input=False, normalize_output=False,
                      sampling_method="sgld", use_double_precision=False,
                      print_every_n_steps=7)
    assert model.do_normalize_input is False
    assert model.do_normalize_output is False
    assert model.sampling_method == "sgld"
    assert model.use_double_precision is False
    assert model.print_every_n_steps == 7


def test_normalize_input_and_output_helpers():
    model = Bohamiann()
    x = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    xn, m, s = model.normalize_input(x)
    assert np.allclose(xn.mean(axis=0), 0.0)
    assert np.allclose(xn.std(axis=0), 1.0)
    assert np.allclose(m, [3.0, 20.0])
    assert np.allclose(s, x.std(axis=0))

    y = np.array([1.0, 2.0, 3.0])
    yn, ym, ys = model.normalize_output(y, 2.0, 4.0)
    assert np.allclose(yn, (y - 2.0) / 4.0)
    assert ym == 2.0 and ys == 4.0


def test_train_collects_expected_number_of_samples():
    model, x, y = train_small()
    # steps 30, 40, 50 are kept (step > burn_in and (step - burn_in) % keep_every == 0)
    assert len(model.sampled_weights) == 3
    assert model.is_trained is True


def test_train_stores_data_and_normalization_stats():
    model, x, y = train_small()
    assert model.X is x
    assert np.array_equal(model.y, y)
    assert np.allclose(model.x_mean, x.mean(axis=0))
    assert np.allclose(model.x_std, x.std(axis=0))
    assert model.y_mean == pytest.approx(y.mean())
    assert model.y_std == pytest.approx(y.std())


def test_train_accepts_column_vector_targets():
    torch.manual_seed(0)
    x, y = make_data()
    model = Bohamiann()
    model.train(x, y.reshape(-1, 1), num_steps=30, keep_every=5,
                num_burn_in_steps=10, batch_size=10)
    assert model.y.shape == (x.shape[0],)
    assert model.is_trained


def test_train_rejects_bad_target_shape():
    x, _ = make_data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, np.zeros((x.shape[0], 2)), num_steps=5, num_burn_in_steps=1)


def test_train_rejects_invalid_batch_size():
    x, y = make_data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=5, num_burn_in_steps=1, batch_size=0)


@pytest.mark.parametrize("method,cls", [
    ("adaptive_sghmc", AdaptiveSGHMC),
    ("sgld", SGLD),
    ("sghmc", SGHMC),
])
def test_sampling_method_selects_sampler(method, cls):
    model = Bohamiann(sampling_method=method)
    train_small(model=model, num_steps=25, keep_every=5, burn_in=10)
    assert isinstance(model.sampler, cls)
    assert len(model.sampled_weights) >= 1


def test_predict_output_shapes_and_positive_variance():
    model, x, y = train_small()
    x_test = np.random.RandomState(1).uniform(-1, 1, size=(7, 2))
    mean, var = model.predict(x_test)
    assert mean.shape == (7,)
    assert var.shape == (7,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_predict_individual_predictions_shape():
    model, x, y = train_small()
    x_test = x[:5]
    mean, var, indiv = model.predict(x_test, return_individual_predictions=True)
    assert indiv.shape == (len(model.sampled_weights), 5)
    assert np.allclose(indiv.mean(axis=0), mean)


def test_predict_single_shape_and_matches_ensemble_for_one_sample():
    model, x, y = train_small(num_steps=40, keep_every=100, burn_in=20)
    # keep_every large -> exactly zero or one sample; force a known sample list
    model.sampled_weights = [model.sampled_weights[0]] if model.sampled_weights else [model.network_weights]
    x_test = x[:4]
    single = model.predict_single(x_test, 0)
    assert single.shape == (4, 2)
    mean, _ = model.predict(x_test)
    assert np.allclose(single[:, 0], mean)


def test_network_weights_roundtrip():
    model, x, y = train_small()
    weights = model.network_weights
    assert len(weights) == len(list(model.model.parameters()))
    new_weights = [np.zeros_like(w) for w in weights]
    with torch.no_grad():
        model.network_weights = new_weights
    for w in model.network_weights:
        assert np.all(w == 0)
    with torch.no_grad():
        model.network_weights = list(weights)
    for a, b in zip(weights, model.network_weights):
        assert np.array_equal(a, b)


def test_continue_training_keeps_previous_samples():
    model, x, y = train_small()
    before = len(model.sampled_weights)
    sampler = model.sampler
    model.train(x, y, num_steps=60, keep_every=10, num_burn_in_steps=0,
                batch_size=10, continue_training=True)
    assert model.sampler is sampler
    assert len(model.sampled_weights) > before


def test_retraining_without_continue_clears_samples():
    model, x, y = train_small()
    model.train(x, y, num_steps=15, keep_every=100, num_burn_in_steps=10, batch_size=10)
    assert len(model.sampled_weights) == 0


def test_float32_mode_without_normalization():
    torch.manual_seed(0)
    x, y = make_data()
    model = Bohamiann(normalize_input=False, normalize_output=False,
                      use_double_precision=False,
                      get_network=lambda input_dimensionality: _FloatNet(input_dimensionality))
    model.train(x, y, num_steps=30, keep_every=5, num_burn_in_steps=10, batch_size=10)
    mean, var = model.predict(x[:3])
    assert mean.shape == (3,)
    assert np.all(var > 0)


class _FloatNet(torch.nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.fc = torch.nn.Linear(n_inputs, 1)

    def forward(self, x):
        mean = self.fc(x)
        return torch.cat((mean, torch.full_like(mean, -2.0)), dim=1)


def test_f_gradient_shape_and_predictive_mean_gradient():
    model, x, y = train_small()
    point = x[0]
    g = model.f_gradient(point, model.sampled_weights[0])
    assert g.shape == (2,)
    mg = model.predictive_mean_gradient(point)
    assert mg.shape == (2,)
    expected = np.mean([model.f_gradient(point, w) for w in model.sampled_weights], axis=0)
    assert np.allclose(mg, expected)


def test_train_and_evaluate_returns_learning_curves():
    torch.manual_seed(0)
    x, y = make_data(40)
    model = Bohamiann()
    n_steps, ll, mse = model.train_and_evaluate(
        x[:30], y[:30], x[30:], y[30:],
        num_steps=40, validate_every_n_steps=20, keep_every=5,
        num_burn_in_steps=20, batch_size=10)
    assert n_steps == [40, 60]
    assert len(ll) == 2 and len(mse) == 2
    assert all(np.isfinite(ll)) and all(m >= 0 for m in mse)
