import numpy as np
import pytest
import torch
import torch.nn as nn

import pybnn.bohamiann as boh


class DummySampler:
    def __init__(self, params, **kwargs):
        self.params = list(params)
        self.kwargs = kwargs
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.step_calls += 1


class SimpleNet(nn.Module):
    def __init__(self, input_dimensionality):
        super().__init__()
        self.fc = nn.Linear(input_dimensionality, 2, bias=True)

    def forward(self, x):
        return self.fc(x)


def simple_network_factory(input_dimensionality):
    return SimpleNet(input_dimensionality)


def finite_infinite_dataloader(loader):
    for batch in loader:
        yield batch


def make_data(n=6, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float64)
    y = (x[:, 0] * 0.5 - x[:, 1] * 0.25 + 0.1).astype(np.float64)
    return x, y


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)
    monkeypatch.setattr(boh, "infinite_dataloader", finite_infinite_dataloader)
    monkeypatch.setattr(boh, "weight_prior", lambda params, dtype=None: torch.tensor(0.0, dtype=torch.float64 if dtype == np.float64 else torch.float32))
    monkeypatch.setattr(boh, "log_variance_prior", lambda x: torch.tensor(0.0, dtype=x.dtype))
    yield


def test_nll_zero_error_matches_log_variance():
    inp = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0], dtype=torch.float64)
    loss = boh.nll(inp, target)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_nll_increases_with_prediction_error():
    inp_good = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    inp_bad = torch.tensor([[3.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0], dtype=torch.float64)
    assert boh.nll(inp_bad, target).item() > boh.nll(inp_good, target).item()


def test_network_weights_getter_and_setter_roundtrip():
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False)
    model.model = simple_network_factory(2).double()

    original = model.network_weights
    new_weights = [np.ones_like(w) * 0.123 for w in original]
    model.network_weights = new_weights

    after = model.network_weights
    assert len(after) == len(new_weights)
    for a, b in zip(after, new_weights):
        assert np.allclose(a, b)


def test_train_initializes_model_and_sampler_adaptive_sghmc():
    x, y = make_data()
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False,
                         sampling_method="adaptive_sghmc", use_double_precision=True)
    model.train(x, y, num_steps=1, num_burn_in_steps=0, batch_size=2, keep_every=1)

    assert model.is_trained is True
    assert isinstance(model.sampler, DummySampler)
    assert model.sampler.kwargs["num_burn_in_steps"] == 0
    assert model.sampler.kwargs["scale_grad"] == pytest.approx(float(x.shape[0]))
    assert model.model is not None


def test_train_uses_requested_sampling_method_sgld():
    x, y = make_data()
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False,
                         sampling_method="sgld")
    model.train(x, y, num_steps=1, num_burn_in_steps=0, batch_size=2, keep_every=1)

    assert isinstance(model.sampler, DummySampler)
    assert model.sampler.kwargs["scale_grad"] == x.shape[0]


def test_train_collects_sampled_weights_after_burn_in():
    x, y = make_data()
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False,
                         sampling_method="sgld")
    model.train(x, y, num_steps=5, num_burn_in_steps=1, keep_every=2, batch_size=2)

    assert model.is_trained is True
    assert len(model.sampled_weights) >= 1
    assert all(isinstance(w, tuple) for w in model.sampled_weights)


def test_train_clears_sampled_weights_when_not_continuing():
    x, y = make_data()
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False,
                         sampling_method="sgld")
    model.sampled_weights = [("dummy",)]
    model.train(x, y, num_steps=1, num_burn_in_steps=0, batch_size=2, keep_every=1, continue_training=False)
    assert model.sampled_weights != [("dummy",)]


def test_train_accepts_y_as_column_vector():
    x, y = make_data()
    y2 = y[:, None]
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False,
                         sampling_method="sgld")
    model.train(x, y2, num_steps=1, num_burn_in_steps=0, batch_size=2, keep_every=1)
    assert model.is_trained is True
    assert model.y.shape == (x.shape[0],)


def test_train_rejects_invalid_batch_size():
    x, y = make_data()
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False)
    with pytest.raises(AssertionError):
        model.train(x, y, batch_size=0, num_steps=1, num_burn_in_steps=0)


def test_train_reduces_batch_size_when_dataset_too_small():
    x, y = make_data(n=2, d=2)
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False)
    model.train(x, y, batch_size=5, num_steps=1, num_burn_in_steps=0)
    assert model.is_trained is True


def test_predict_returns_mean_and_variance_without_normalization():
    x, y = make_data()
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False,
                         sampling_method="sgld")
    model.train(x, y, num_steps=1, num_burn_in_steps=0, batch_size=2, keep_every=1)
    if not model.sampled_weights:
        model.sampled_weights = [model.network_weights]

    mu, var = model.predict(x[:3])
    assert mu.shape == (3,)
    assert var.shape == (3,)
    assert np.all(np.isfinite(mu))
    assert np.all(var >= 0)


def test_predict_return_individual_predictions():
    x, y = make_data()
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False,
                         sampling_method="sgld")
    model.train(x, y, num_steps=1, num_burn_in_steps=0, batch_size=2, keep_every=1)
    if len(model.sampled_weights) < 2:
        model.sampled_weights = [model.network_weights, model.network_weights]

    mu, var, individual = model.predict(x[:2], return_individual_predictions=True)
    assert mu.shape == (2,)
    assert var.shape == (2,)
    assert individual.shape[1] == 2


def test_predict_single_matches_single_network_output_shape():
    x, y = make_data()
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False,
                         sampling_method="sgld")
    model.train(x, y, num_steps=1, num_burn_in_steps=0, batch_size=2, keep_every=1)
    if not model.sampled_weights:
        model.sampled_weights = [model.network_weights]

    out = model.predict_single(x[0], 0)
    assert out.shape == (1, 2)


def test_f_gradient_shape_without_normalization():
    x, y = make_data()
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False,
                         sampling_method="sgld")
    model.train(x, y, num_steps=1, num_burn_in_steps=0, batch_size=2, keep_every=1)
    weights = model.network_weights
    g = model.f_gradient(x[0], weights)
    assert g.shape == (x.shape[1],)


def test_predictive_mean_and_variance_gradients_shape():
    x, y = make_data()
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False,
                         sampling_method="sgld")
    model.train(x, y, num_steps=1, num_burn_in_steps=0, batch_size=2, keep_every=1)
    if len(model.sampled_weights) < 2:
        model.sampled_weights = [model.network_weights, model.network_weights]

    mg = model.predictive_mean_gradient(x[0])
    vg = model.predictive_variance_gradient(x[0])
    assert mg.shape == (x.shape[1],)
    assert vg.shape == (x.shape[1],)


def test_train_and_evaluate_returns_learning_curves():
    x, y = make_data(n=8, d=2)
    x_valid, y_valid = make_data(n=4, d=2, seed=1)
    model = boh.Bohamiann(get_network=simple_network_factory, normalize_input=False, normalize_output=False,
                         sampling_method="sgld")
    steps, ll, mse = model.train_and_evaluate(
        x, y, x_valid, y_valid,
        num_steps=4,
        validate_every_n_steps=2,
        keep_every=1,
        num_burn_in_steps=0,
        batch_size=2,
    )
    assert steps == [2, 4]
    assert len(ll) == 2
    assert len(mse) == 2
    assert all(np.isfinite(v) for v in ll)
    assert all(np.isfinite(v) for v in mse)
