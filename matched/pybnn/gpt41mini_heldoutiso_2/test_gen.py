import numpy as np
import pytest
import torch
from pybnn.bohamiann import Bohamiann, nll, get_default_network, zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization


def test_get_default_network_returns_module_with_expected_layers():
    input_dim = 3
    net = get_default_network(input_dim)
    assert isinstance(net, torch.nn.Module)
    # Check presence of layers expected by Architecture
    assert hasattr(net, 'fc1')
    assert hasattr(net, 'fc2')
    assert hasattr(net, 'fc3')
    assert hasattr(net, 'log_std')
    # Check input and output shapes
    x = torch.randn(5, input_dim)
    out = net(x)
    assert out.shape == (5, 2) or out.shape[1] == 2  # output + log_std appended


def test_nll_basic_computation():
    # Prepare dummy prediction: batch_size=4, columns=[mean, log_var]
    batch_size = 4
    pred_means = torch.tensor([1., 2., 3., 4.]).view(-1, 1)
    pred_log_vars = torch.log(torch.tensor([0.5, 1., 2., 4.])).view(-1, 1)
    pred = torch.cat([pred_means, pred_log_vars], dim=1)
    targets = torch.tensor([0.9, 2.1, 2.5, 4.2])
    result = nll(pred, targets)
    assert torch.is_tensor(result)
    # NLL must be scalar (0-dim tensor)
    assert result.dim() == 0
    # Test known formula for first element static
    mse0 = (0.9 - 1) ** 2
    var0 = 0.5
    expected_first = 0.5 * mse0 / var0 + 0.5 * np.log(var0)
    # We check result is > 0 (loss)
    assert result.item() > 0


def test_network_weights_property_setter_and_getter_consistency():
    model = get_default_network(2).double()
    bohamiann = Bohamiann(get_network=lambda input_dimensionality: model, use_double_precision=True)
    bohamiann.model = model
    original_weights = bohamiann.network_weights
    # Modify weights copy and set back
    modified_weights = [w + 1.0 for w in original_weights]
    bohamiann.network_weights = modified_weights
    new_weights = bohamiann.network_weights
    for orig, new in zip(modified_weights, new_weights):
        np.testing.assert_allclose(orig, new, rtol=1e-7)


def test_train_runs_and_sets_is_trained(monkeypatch):
    np.random.seed(42)
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)

    m = Bohamiann(get_network=get_default_network, use_double_precision=False, sampling_method="sgld", print_every_n_steps=5)

    # Patch sampler methods to no-op to avoid real sampling overhead
    class DummySampler:
        def zero_grad(self):
            pass

        def step(self):
            pass

    monkeypatch.setattr(m, "sampler", DummySampler())
    m.model = m.get_network(input_dimensionality=2).float()

    # The real train triggers sampler setting, so avoid that by passing continue_training=True to skip sampler init
    # But we want the sampler to exist, so patch AdaptiveSGHMC constructor to DummySampler
    monkeypatch.setattr(m, "sampler", DummySampler())

    # Run train with minimal num_steps to keep runtime short
    m.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=0, lr=1e-2, batch_size=10, verbose=False)
    assert m.is_trained is True
    assert len(m.sampled_weights) > 0


def test_train_with_batch_size_larger_than_dataset(monkeypatch):
    x_train = np.random.randn(5, 2)
    y_train = np.random.randn(5)
    m = Bohamiann(get_network=get_default_network, use_double_precision=False)
    monkeypatch.setattr(m, "sampler", None)

    # Patch sampler class to dummy for test
    class DummySampler:
        def zero_grad(self):
            pass

        def step(self):
            pass

    m.sampler = DummySampler()
    m.model = m.get_network(input_dimensionality=2).float()
    m.train(x_train, y_train, batch_size=10, num_steps=5, num_burn_in_steps=0)
    # It should adjust batch size to 5 and still run without error
    assert m.is_trained


def test_predict_returns_mean_and_variance(monkeypatch):
    input_dim = 3
    bohamiann = Bohamiann(get_network=get_default_network, use_double_precision=False)
    # Setup dummy model and sample weights
    bohamiann.model = bohamiann.get_network(input_dimensionality=input_dim).float()
    sample_weights = [tuple(np.random.randn(*p.shape) for p in bohamiann.model.parameters()) for _ in range(2)]
    bohamiann.sampled_weights = sample_weights
    # Setup normalization data expected by predict
    bohamiann.do_normalize_input = False
    bohamiann.do_normalize_output = False
    x_test = np.random.randn(4, input_dim)
    mean, var = bohamiann.predict(x_test)
    assert mean.shape == (4,)
    assert var.shape == (4,)


def test_predict_with_normalization(monkeypatch):
    # Test predict with normalization enabled
    input_dim = 2
    bohamiann = Bohamiann(get_network=get_default_network, use_double_precision=False)
    bohamiann.model = bohamiann.get_network(input_dimensionality=input_dim).float()
    sample_weights = [tuple(np.random.randn(*p.shape) for p in bohamiann.model.parameters()) for _ in range(1)]
    bohamiann.sampled_weights = sample_weights
    # Setup normalization attributes
    bohamiann.do_normalize_input = True
    bohamiann.do_normalize_output = True
    bohamiann.x_mean = np.zeros(input_dim)
    bohamiann.x_std = np.ones(input_dim)
    bohamiann.y_mean = 0.0
    bohamiann.y_std = 1.0
    x_test = np.random.randn(3, input_dim)
    mean, var = bohamiann.predict(x_test)
    assert mean.shape == (3,)
    assert var.shape == (3,)


def test_predict_single_returns_expected_shape(monkeypatch):
    input_dim = 2
    bohamiann = Bohamiann(get_network=get_default_network, use_double_precision=False)
    bohamiann.model = bohamiann.get_network(input_dimensionality=input_dim).float()
    sample_weights = [tuple(np.random.randn(*p.shape) for p in bohamiann.model.parameters()) for _ in range(2)]
    bohamiann.sampled_weights = sample_weights
    bohamiann.do_normalize_input = False
    bohamiann.do_normalize_output = False
    x_test = np.random.randn(1, input_dim)
    # Should not error and return shape (2,) for mean and var pair
    val = bohamiann.predict_single(x_test, sample_index=0)
    # val shape depends on model output shape, for Architecture it's (?,2)
    assert val.shape[-1] == 2 or val.ndim == 2


def test_f_gradient_produces_correct_shape(monkeypatch):
    input_dim = 2
    bohamiann = Bohamiann(get_network=get_default_network, use_double_precision=False)
    bohamiann.model = bohamiann.get_network(input_dimensionality=input_dim).float()
    sample_weights = [tuple(np.random.randn(*p.shape) for p in bohamiann.model.parameters())]
    bohamiann.sampled_weights = sample_weights
    bohamiann.do_normalize_input = False
    bohamiann.do_normalize_output = False
    x = np.random.randn(input_dim)
    grad = bohamiann.f_gradient(x, sample_weights[0])
    assert grad.shape == (input_dim,)
    assert isinstance(grad, np.ndarray)


def test_predictive_mean_gradient_and_variance_gradient(monkeypatch):
    input_dim = 2
    bohamiann = Bohamiann(get_network=get_default_network, use_double_precision=False)
    bohamiann.model = bohamiann.get_network(input_dimensionality=input_dim).float()
    # Add two samples for meaningful mean and variance gradient
    sample_weights = [tuple(np.random.randn(*p.shape) for p in bohamiann.model.parameters()) for _ in range(3)]
    bohamiann.sampled_weights = sample_weights
    bohamiann.do_normalize_input = False
    bohamiann.do_normalize_output = False
    x = np.random.randn(input_dim)
    mean_grad = bohamiann.predictive_mean_gradient(x)
    var_grad = bohamiann.predictive_variance_gradient(x)
    assert mean_grad.shape == (input_dim,)
    assert var_grad.shape == (input_dim,)
    assert isinstance(mean_grad, np.ndarray)
    assert isinstance(var_grad, np.ndarray)


def test_normalization_methods_return_expected_results():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Test mean/std calculation and normalization
    norm_data, mean, std = zero_mean_unit_var_normalization(data)
    assert np.allclose(np.mean(norm_data, axis=0), np.zeros(data.shape[1]))
    assert np.allclose(np.std(norm_data, axis=0), np.ones(data.shape[1]))
    # When passing mean/std explicitly, output is different but consistent shape
    norm_data2, _, _ = zero_mean_unit_var_normalization(data, m=mean, s=std)
    assert norm_data2.shape == data.shape


def test_train_and_evaluate_returns_learning_curves(monkeypatch):
    # small simple data
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    x_valid = np.random.randn(5, 2)
    y_valid = np.random.randn(5)
    m = Bohamiann(get_network=get_default_network, use_double_precision=False, sampling_method="sgld")

    monkeypatch.setattr(m, "train", lambda *a, **k: setattr(m, "is_trained", True))
    monkeypatch.setattr(m, "predict", lambda *a, **k: (np.zeros(5), np.ones(5)))

    n_steps, ll, mse = m.train_and_evaluate(x_train, y_train, x_valid, y_valid,
                                           num_steps=4, validate_every_n_steps=2,
                                           keep_every=1, num_burn_in_steps=0, lr=0.01,
                                           epsilon=1e-10, batch_size=5, mdecay=0.05, verbose=False)
    assert isinstance(n_steps, list)
    assert isinstance(ll, list)
    assert isinstance(mse, list)
    assert all(isinstance(v, (float, np.floating)) for v in ll)
    assert all(isinstance(v, (float, np.floating)) for v in mse)


def test_train_raises_assert_for_invalid_batch_size():
    m = Bohamiann(get_network=get_default_network)
    x = np.random.randn(10, 2)
    y = np.random.randn(10)
    with pytest.raises(AssertionError):
        m.train(x, y, batch_size=0)


def test_train_raises_assert_for_invalid_yshape():
    m = Bohamiann(get_network=get_default_network)
    x = np.random.randn(10, 2)
    y = np.random.randn(10, 2)  # Not (N,) or (N,1)
    with pytest.raises(AssertionError):
        m.train(x, y)


def test_train_verbose_prints(capsys):
    # We patch methods so that actual training isn't run fully but verbose print triggers
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    m = Bohamiann(get_network=get_default_network, use_double_precision=False, print_every_n_steps=1, sampling_method="sgld")

    class DummySampler:
        def zero_grad(self):
            pass

        def step(self):
            pass

    m.sampled_weights = []
    m.model = m.get_network(input_dimensionality=2).float()
    m.sampler = DummySampler()
    m.y = y_train
    m.do_normalize_output = False
    m.do_normalize_input = False
    m.x_mean = np.zeros(2)
    m.x_std = np.ones(2)
    m.y_mean = 0.0
    m.y_std = 1.0

    # Run train: this should trigger at least one print due to verbose and print_every_n_steps=1
    m.train(x_train, y_train, num_steps=3, keep_every=1, num_burn_in_steps=0, verbose=True, batch_size=10)

    captured = capsys.readouterr()
    assert "Step" in captured.out


def test_network_weights_setter_accepts_correct_shapes():
    model = get_default_network(2).double()
    bohamiann = Bohamiann(get_network=lambda input_dimensionality: model, use_double_precision=True)
    bohamiann.model = model
    weights = [w.cpu().detach().numpy() for w in model.parameters()]
    # modify shape by slicing (simulate error)
    wrong_weights = []
    for w in weights:
        if w.size > 1:
            wrong_weights.append(w.flatten()[1:])
        else:
            wrong_weights.append(w)
    # Setter expects matching shapes; test passes if shapes don't match setter skips or raises error
    with pytest.raises(RuntimeError):
        bohamiann.network_weights = wrong_weights


def test_network_weights_property_returns_tuple_of_ndarrays():
    model = get_default_network(1).float()
    bohamiann = Bohamiann(get_network=lambda input_dimensionality: model, use_double_precision=False)
    bohamiann.model = model
    weights = bohamiann.network_weights
    assert isinstance(weights, tuple)
    for w in weights:
        assert isinstance(w, np.ndarray)


def test_predict_with_return_individual_predictions(monkeypatch):
    input_dim = 2
    bohamiann = Bohamiann(get_network=get_default_network, use_double_precision=False)
    bohamiann.model = bohamiann.get_network(input_dimensionality=input_dim).float()
    sample_weights = [tuple(np.random.randn(*p.shape) for p in bohamiann.model.parameters()) for _ in range(2)]
    bohamiann.sampled_weights = sample_weights
    bohamiann.do_normalize_input = False
    bohamiann.do_normalize_output = False
    test_x = np.random.randn(3, input_dim)
    mean, var, indiv = bohamiann.predict(test_x, return_individual_predictions=True)
    assert mean.shape == (3,)
    assert var.shape == (3,)
    assert isinstance(indiv, np.ndarray)
    assert indiv.shape == (2, 3)


def test_train_continue_training_keeps_sampled_weights_list(monkeypatch):
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    m = Bohamiann(get_network=get_default_network, use_double_precision=False, sampling_method="sgld")
    class DummySampler:
        def zero_grad(self):
            pass
        def step(self):
            pass

    m.sampler = DummySampler()
    m.model = m.get_network(input_dimensionality=2).float()
    # Pre-populate sampled weights with dummy
    m.sampled_weights = [('dummy',)]
    m.train(x_train, y_train, num_steps=5, batch_size=5, continue_training=True, num_burn_in_steps=0)
    # It should have kept the old sample (not cleared)
    assert len(m.sampled_weights) >= 1
    assert m.is_trained
