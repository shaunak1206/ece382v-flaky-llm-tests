import numpy as np
import pytest
import torch
import torch.nn as nn

import pybnn.bohamiann as boh


class TinyNet(nn.Module):
    def __init__(self, input_dimensionality):
        super().__init__()
        self.fc = nn.Linear(input_dimensionality, 2, bias=True)

    def forward(self, x):
        return self.fc(x)


class DummySampler:
    def __init__(self, *args, **kwargs):
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1


class DummyOptimizerNoOp:
    def __init__(self, *args, **kwargs):
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1


def make_bohamiann(normalize_input=False, normalize_output=False, use_double_precision=True):
    return boh.Bohamiann(
        get_network=lambda input_dimensionality: TinyNet(input_dimensionality),
        normalize_input=normalize_input,
        normalize_output=normalize_output,
        use_double_precision=use_double_precision,
        print_every_n_steps=1,
        likelihood_function=lambda input, target: torch.mean((input[:, 0] - target.view(-1)) ** 2),
    )


def set_identity_weights(model, input_dim):
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        # mean = x0, log_var = x1
        model.fc.weight[0, 0] = 1.0
        if input_dim > 1:
            model.fc.weight[1, 1] = 1.0


def test_nll_matches_manual_value():
    inp = torch.tensor([[1.0, 0.0], [2.0, np.log(2.0)]], dtype=torch.float64)
    target = torch.tensor([1.0, 0.0], dtype=torch.float64)

    got = boh.nll(inp, target)

    mse1 = (1.0 - 1.0) ** 2
    mse2 = (0.0 - 2.0) ** 2
    ll1 = -mse1 * 0.5 * np.exp(-0.0) - 0.5 * 0.0
    ll2 = -mse2 * 0.5 * np.exp(-np.log(2.0)) - 0.5 * np.log(2.0)
    expected = -(ll1 + ll2) / 2.0

    assert torch.isclose(got, torch.tensor(expected, dtype=torch.float64))


def test_get_default_network_forward_shape():
    net = boh.get_default_network(3)
    x = torch.randn(4, 3)
    y = net(x)
    assert y.shape == (4, 2)


def test_network_weights_property_roundtrip():
    bnn = make_bohamiann()
    bnn.model = TinyNet(2).double()
    original = bnn.network_weights

    new_weights = [np.ones_like(w) for w in original]
    bnn.network_weights = new_weights

    roundtrip = bnn.network_weights
    for got, expected in zip(roundtrip, new_weights):
        assert np.allclose(got, expected)


def test_normalize_input_and_output_identity_roundtrip():
    bnn = make_bohamiann()
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    xn, m, s = bnn.normalize_input(x)
    xr, _, _ = boh.zero_mean_unit_var_denormalization(xn, m, s), m, s
    assert np.allclose(xr, x)

    y = np.array([1.0, 3.0, 5.0])
    yn, ym, ys = bnn.normalize_output(y)
    yr = boh.zero_mean_unit_var_denormalization(yn, ym, ys)
    assert np.allclose(yr, y)


def test_train_rejects_invalid_batch_size():
    bnn = make_bohamiann()
    x = np.zeros((4, 2))
    y = np.zeros(4)
    with pytest.raises(AssertionError):
        bnn.train(x, y, num_steps=1, batch_size=0)


def test_train_accepts_targets_as_column_vector():
    bnn = make_bohamiann()
    x = np.zeros((3, 2))
    y = np.zeros((3, 1))
    y_flat = np.zeros(3)

    # Patch internals to avoid expensive training
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)
    monkeypatch.setattr(boh, "weight_prior", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(boh, "log_variance_prior", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(
        boh,
        "infinite_dataloader",
        lambda loader: iter([(next(iter(loader))[0], next(iter(loader))[1])]),
    )
    bnn.train(x, y, num_steps=1, batch_size=1, verbose=False)
    assert bnn.is_trained
    assert bnn.y.shape == (3,)
    bnn2 = make_bohamiann()
    monkeypatch.undo()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)
    monkeypatch.setattr(boh, "weight_prior", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(boh, "log_variance_prior", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(
        boh,
        "infinite_dataloader",
        lambda loader: iter([(next(iter(loader))[0], next(iter(loader))[1])]),
    )
    bnn2.train(x, y_flat, num_steps=1, batch_size=1, verbose=False)
    assert bnn2.y.shape == (3,)
    monkeypatch.undo()


def test_train_initializes_sampler_for_all_supported_methods():
    x = np.zeros((2, 2))
    y = np.zeros(2)

    for method in ["adaptive_sghmc", "sgld", "preconditioned_sgld", "sghmc"]:
        bnn = make_bohamiann()
        bnn.sampling_method = method

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
        monkeypatch.setattr(boh, "SGLD", DummySampler)
        monkeypatch.setattr(boh, "SGHMC", DummySampler)
        monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)
        monkeypatch.setattr(boh, "weight_prior", lambda *args, **kwargs: torch.tensor(0.0))
        monkeypatch.setattr(boh, "log_variance_prior", lambda *args, **kwargs: torch.tensor(0.0))
        monkeypatch.setattr(boh, "infinite_dataloader", lambda loader: iter([(next(iter(loader))[0], next(iter(loader))[1])]))

        bnn.train(x, y, num_steps=1, batch_size=1, verbose=False)
        assert isinstance(bnn.sampler, DummySampler)

        monkeypatch.undo()


def test_train_collects_sampled_weights_after_burn_in():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])

    bnn = make_bohamiann()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "weight_prior", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(boh, "log_variance_prior", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(boh, "infinite_dataloader", lambda loader: iter([(next(iter(loader))[0], next(iter(loader))[1])]))

    bnn.train(x, y, num_steps=3, keep_every=1, num_burn_in_steps=0, batch_size=1, verbose=False)
    assert bnn.is_trained
    assert len(bnn.sampled_weights) >= 1

    monkeypatch.undo()


def test_train_can_continue_training_without_clearing_samples():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])

    bnn = make_bohamiann()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "weight_prior", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(boh, "log_variance_prior", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(boh, "infinite_dataloader", lambda loader: iter([(next(iter(loader))[0], next(iter(loader))[1])]))

    bnn.train(x, y, num_steps=2, keep_every=1, num_burn_in_steps=0, batch_size=1, verbose=False)
    first_len = len(bnn.sampled_weights)
    bnn.train(x, y, num_steps=1, keep_every=1, num_burn_in_steps=0, batch_size=1, verbose=False, continue_training=True)
    assert len(bnn.sampled_weights) >= first_len

    monkeypatch.undo()


def test_predict_without_normalization_returns_mean_and_variance():
    bnn = make_bohamiann(normalize_input=False, normalize_output=False, use_double_precision=True)
    bnn.model = TinyNet(2).double()
    set_identity_weights(bnn.model, 2)

    # two samples: x0=1 and x0=3, log_var=0
    w1 = tuple(p.detach().cpu().numpy().copy() for p in bnn.model.parameters())
    with torch.no_grad():
        bnn.model.fc.bias[0] = 2.0
    w2 = tuple(p.detach().cpu().numpy().copy() for p in bnn.model.parameters())

    bnn.sampled_weights = [w1, w2]
    mu, var = bnn.predict(np.array([[1.0, 0.0], [3.0, 0.0]]))
    assert mu.shape == (2,)
    assert var.shape == (2,)
    assert np.allclose(mu, np.array([1.5, 3.5]))
    assert np.allclose(var, np.array([1.25, 1.25]))


def test_predict_with_return_individual_predictions():
    bnn = make_bohamiann(normalize_input=False, normalize_output=False, use_double_precision=True)
    bnn.model = TinyNet(2).double()
    set_identity_weights(bnn.model, 2)

    w1 = tuple(p.detach().cpu().numpy().copy() for p in bnn.model.parameters())
    with torch.no_grad():
        bnn.model.fc.bias[0] = 2.0
    w2 = tuple(p.detach().cpu().numpy().copy() for p in bnn.model.parameters())
    bnn.sampled_weights = [w1, w2]

    mu, var, individual = bnn.predict(np.array([[1.0, 0.0]]), return_individual_predictions=True)
    assert mu.shape == (1,)
    assert var.shape == (1,)
    assert individual.shape == (2, 1)
    assert np.allclose(individual[:, 0], np.array([1.0, 2.0]))


def test_predict_single_returns_single_network_output():
    bnn = make_bohamiann(normalize_input=False, normalize_output=False, use_double_precision=True)
    bnn.model = TinyNet(2).double()
    set_identity_weights(bnn.model, 2)
    w = tuple(p.detach().cpu().numpy().copy() for p in bnn.model.parameters())
    bnn.sampled_weights = [w]

    out = bnn.predict_single(np.array([4.0, 7.0]), sample_index=0)
    assert out.shape == (1, 2)
    assert np.allclose(out[0, 0], 4.0)
    assert np.allclose(out[0, 1], 7.0)


def test_f_gradient_matches_linear_model_gradient():
    bnn = make_bohamiann(normalize_input=False, normalize_output=False, use_double_precision=True)
    bnn.model = TinyNet(2).double()
    set_identity_weights(bnn.model, 2)
    w = tuple(p.detach().cpu().numpy().copy() for p in bnn.model.parameters())

    g = bnn.f_gradient(np.array([3.0, 5.0]), weights=w)
    assert g.shape == (2,)
    assert np.allclose(g, np.array([1.0, 0.0]), atol=1e-6)


def test_predictive_mean_gradient_averages_samples():
    bnn = make_bohamiann(normalize_input=False, normalize_output=False, use_double_precision=True)
    bnn.model = TinyNet(2).double()
    set_identity_weights(bnn.model, 2)
    w1 = tuple(p.detach().cpu().numpy().copy() for p in bnn.model.parameters())
    with torch.no_grad():
        bnn.model.fc.weight.zero_()
        bnn.model.fc.weight[0, 1] = 1.0
    w2 = tuple(p.detach().cpu().numpy().copy() for p in bnn.model.parameters())
    bnn.sampled_weights = [w1, w2]

    g = bnn.predictive_mean_gradient(np.array([3.0, 5.0]))
    assert np.allclose(g, np.array([0.5, 0.5]), atol=1e-6)


def test_predictive_variance_gradient_zero_for_constant_variance_and_linear_mean():
    bnn = make_bohamiann(normalize_input=False, normalize_output=False, use_double_precision=True)
    bnn.model = TinyNet(2).double()
    with torch.no_grad():
        bnn.model.fc.weight.zero_()
        bnn.model.fc.weight[0, 0] = 1.0
        bnn.model.fc.weight[1, 1] = 0.0
        bnn.model.fc.bias.zero_()
    w1 = tuple(p.detach().cpu().numpy().copy() for p in bnn.model.parameters())
    bnn.sampled_weights = [w1, w1]

    g = bnn.predictive_variance_gradient(np.array([2.0, 3.0]))
    assert g.shape == (2,)
    assert np.allclose(g, np.zeros(2), atol=1e-6)


def test_train_and_evaluate_returns_learning_curves():
    x_train = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0.0, 1.0])
    x_valid = np.array([[2.0, 2.0]])
    y_valid = np.array([2.0])

    bnn = make_bohamiann()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "weight_prior", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(boh, "log_variance_prior", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(boh, "infinite_dataloader", lambda loader: iter([(next(iter(loader))[0], next(iter(loader))[1])]))

    steps, lls, mses = bnn.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=2, validate_every_n_steps=1, keep_every=1,
        num_burn_in_steps=0, batch_size=1, verbose=False
    )
    assert len(steps) == len(lls) == len(mses) == 2
    assert steps == [1, 2]

    monkeypatch.undo()
