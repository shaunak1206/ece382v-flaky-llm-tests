import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, nll, get_default_network


def _make_data(n=30, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(n, d)
    y = np.sin(x.sum(axis=1)) + 0.01 * rng.randn(n)
    return x, y


def _trained_model(**init_kwargs):
    x, y = _make_data()
    torch.manual_seed(0)
    model = Bohamiann(print_every_n_steps=1000, **init_kwargs)
    model.train(x, y, num_steps=30, keep_every=5, num_burn_in_steps=10,
                batch_size=10, lr=1e-3)
    return model, x, y


def test_nll_known_value():
    # mean 0, log variance 0 (variance 1), target 1 -> 0.5 * 1 + 0.5 * 0 = 0.5
    inp = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([1.0])
    loss = nll(inp, target)
    assert loss.item() == pytest.approx(0.5, abs=1e-6)


def test_nll_is_averaged_over_batch():
    inp = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([1.0, 1.0])
    single = nll(inp[:1], target[:1])
    both = nll(inp, target)
    assert both.item() == pytest.approx(single.item(), abs=1e-6)


def test_nll_penalizes_larger_error():
    inp = torch.tensor([[0.0, 0.0]])
    small = nll(inp, torch.tensor([0.1]))
    large = nll(inp, torch.tensor([2.0]))
    assert large.item() > small.item()


def test_default_network_output_shape():
    net = get_default_network(input_dimensionality=3)
    out = net(torch.randn(7, 3))
    assert out.shape == (7, 2)


def test_init_state():
    model = Bohamiann(sampling_method="sgld", use_double_precision=False,
                      normalize_input=False, normalize_output=False)
    assert model.is_trained is False
    assert model.sampled_weights == []
    assert model.sampler is None
    assert model.sampling_method == "sgld"
    assert model.use_double_precision is False
    assert model.do_normalize_input is False
    assert model.do_normalize_output is False


def test_normalize_input_and_output():
    model = Bohamiann()
    x = np.random.RandomState(1).rand(50, 3) * 5 + 2
    x_norm, mean, std = model.normalize_input(x)
    assert np.allclose(x_norm.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(x_norm.std(axis=0), 1, atol=1e-8)
    assert np.allclose(mean, x.mean(axis=0))
    assert np.allclose(std, x.std(axis=0))

    y = x[:, 0]
    y_norm, y_mean, y_std = model.normalize_output(y)
    assert np.allclose(y_norm * y_std + y_mean, y)

    # reusing given statistics
    x2_norm, m2, s2 = model.normalize_input(x[:5], mean, std)
    assert np.allclose(x2_norm, x_norm[:5])


def test_train_collects_expected_number_of_samples():
    model, _, _ = _trained_model()
    assert model.is_trained is True
    # steps 15, 20, 25 are kept
    assert len(model.sampled_weights) == 3


def test_network_weights_shapes_match_parameters():
    model, _, _ = _trained_model()
    weights = model.network_weights
    params = list(model.model.parameters())
    assert len(weights) == len(params)
    for w, p in zip(weights, params):
        assert w.shape == tuple(p.shape)
    for sample in model.sampled_weights:
        assert len(sample) == len(params)


def test_network_weights_setter_roundtrip():
    model, _, _ = _trained_model()
    original = model.network_weights
    new_weights = [np.zeros_like(w) for w in original]
    with torch.no_grad():
        model.network_weights = new_weights
    for w in model.network_weights:
        assert np.all(w == 0)
    with torch.no_grad():
        model.network_weights = list(original)
    for a, b in zip(model.network_weights, original):
        assert np.allclose(a, b)


def test_predict_shapes_and_positive_variance():
    model, x, _ = _trained_model()
    mean, var = model.predict(x[:7])
    assert mean.shape == (7,)
    assert var.shape == (7,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0)


def test_predict_individual_predictions():
    model, x, _ = _trained_model()
    mean, var, indiv = model.predict(x[:5], return_individual_predictions=True)
    assert indiv.shape == (len(model.sampled_weights), 5)
    assert mean.shape == (5,)
    assert var.shape == (5,)


def test_predict_single_shape():
    model, x, _ = _trained_model()
    out = model.predict_single(x[:4], sample_index=0)
    assert out.shape == (4, 2)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("method", ["sgld", "sghmc", "adaptive_sghmc"])
def test_train_with_different_sampling_methods(method):
    model, x, _ = _trained_model(sampling_method=method)
    assert model.is_trained
    assert len(model.sampled_weights) == 3
    mean, var = model.predict(x[:3])
    assert mean.shape == (3,)
    assert np.all(np.isfinite(mean))


def test_train_single_precision_without_normalization():
    x, y = _make_data()
    torch.manual_seed(0)
    model = Bohamiann(use_double_precision=False, normalize_input=False,
                      normalize_output=False)
    model.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=5,
                batch_size=10, lr=1e-3)
    assert model.is_trained
    assert all(w.dtype == np.float32 for w in model.sampled_weights[0])
    mean, var = model.predict(x[:4])
    assert mean.shape == (4,)
    assert np.all(var > 0)


def test_train_accepts_column_vector_targets():
    x, y = _make_data()
    torch.manual_seed(0)
    model = Bohamiann()
    model.train(x, y[:, None], num_steps=15, keep_every=5, num_burn_in_steps=5,
                batch_size=10, lr=1e-3)
    assert model.y.shape == (x.shape[0],)
    assert len(model.sampled_weights) >= 1


def test_train_invalid_inputs_raise():
    x, y = _make_data()
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=5, batch_size=0)
    with pytest.raises(AssertionError):
        model.train(x, np.stack([y, y], axis=1), num_steps=5, batch_size=10)


def test_continue_training_keeps_and_extends_samples():
    model, x, y = _trained_model()
    first_model = model.model
    assert len(model.sampled_weights) == 3
    model.train(x, y, num_steps=30, keep_every=5, num_burn_in_steps=10,
                batch_size=10, lr=1e-3, continue_training=True)
    assert model.model is first_model
    assert len(model.sampled_weights) == 6


def test_retraining_clears_previous_samples():
    model, x, y = _trained_model()
    model.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=10,
                batch_size=10, lr=1e-3)
    # steps 15 only
    assert len(model.sampled_weights) == 1


def test_batch_size_larger_than_dataset_is_clipped():
    x, y = _make_data(n=8)
    torch.manual_seed(0)
    model = Bohamiann()
    model.train(x, y, num_steps=12, keep_every=2, num_burn_in_steps=4,
                batch_size=50, lr=1e-3)
    assert model.is_trained
    assert len(model.sampled_weights) >= 1


def test_gradients_have_input_dimensionality():
    model, x, _ = _trained_model()
    g = model.f_gradient(x[0], model.sampled_weights[0])
    assert g.shape == (2,)
    assert np.all(np.isfinite(g))

    gm = model.predictive_mean_gradient(x[0])
    assert gm.shape == (2,)
    expected = np.mean([model.f_gradient(x[0], w) for w in model.sampled_weights], axis=0)
    assert np.allclose(gm, expected)

    gv = model.predictive_variance_gradient(x[0])
    assert gv.shape == (2,)
    assert np.all(np.isfinite(gv))


def test_train_and_evaluate_returns_learning_curves():
    x, y = _make_data()
    x_valid, y_valid = _make_data(n=10, seed=5)
    torch.manual_seed(0)
    model = Bohamiann()
    n_steps, ll, mse = model.train_and_evaluate(
        x, y, x_valid, y_valid,
        num_steps=20, validate_every_n_steps=10, keep_every=5,
        num_burn_in_steps=10, lr=1e-3, batch_size=10)
    assert n_steps == [20, 30]
    assert len(ll) == 2
    assert len(mse) == 2
    assert all(np.isfinite(v) for v in ll)
    assert all(v >= 0 for v in mse)
    assert len(model.sampled_weights) >= 1
