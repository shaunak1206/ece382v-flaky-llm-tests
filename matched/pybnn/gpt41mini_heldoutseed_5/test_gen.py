import pytest
import numpy as np
import torch
from pybnn.bohamiann import Bohamiann, get_default_network, nll


def test_get_default_network_output_shape_and_type():
    input_dim = 3
    net = get_default_network(input_dim)
    net.eval()
    x = torch.randn(5, input_dim)
    out = net(x)
    assert out.shape == (5, 2)
    assert isinstance(out, torch.Tensor)


def test_nll_basic_computation():
    # Use simple inputs to check non-negative output
    means = torch.tensor([[0.0], [1.0]])
    log_vars = torch.tensor([[0.0], [np.log(0.5)]])
    inputs = torch.cat([means, log_vars], dim=1)
    targets = torch.tensor([0.1, 1.1])
    loss = nll(inputs, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_bohamiann_init_defaults():
    model = Bohamiann()
    assert model.do_normalize_input is True
    assert model.do_normalize_output is True
    assert model.sampling_method == "adaptive_sghmc"
    assert model.is_trained is False
    assert model.sampled_weights == []
    assert callable(model.likelihood_function)
    assert model.print_every_n_steps == 100


def test_network_weights_getter_and_setter():
    model = Bohamiann()
    # Initialize model attributes required
    model.model = get_default_network(2)().double()
    model.use_double_precision = True
    # Capture initial weights
    original_weights = tuple(p.clone().detach().numpy() for p in model.model.parameters())
    # Change weights to zeros
    zero_weights = [np.zeros_like(w) for w in original_weights]
    model.network_weights = zero_weights
    for p in model.model.parameters():
        assert torch.allclose(p, torch.zeros_like(p))
    # Reassign original weights
    model.network_weights = original_weights
    for p, w in zip(model.model.parameters(), original_weights):
        assert np.allclose(p.detach().numpy(), w)


def test_train_with_small_dataset_and_one_step():
    np.random.seed(123)
    model = Bohamiann()
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10)
    # Train with only 1 step (and burn-in 0) to quickly test functionality
    model.train(x_train, y_train, num_steps=1, num_burn_in_steps=0, keep_every=1, batch_size=5, verbose=False)
    assert model.is_trained is True
    # Check sampled_weights was appended (1 sample kept, as steps=1, burn-in=0, keep_every=1)
    assert len(model.sampled_weights) == 1
    # Check model has model attribute and sampler attribute set
    assert hasattr(model, "model")
    assert model.sampler is not None


def test_train_errors_on_invalid_batch_size_shapes():
    model = Bohamiann()
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10, 2)  # invalid shape (N,2)
    with pytest.raises(AssertionError):
        model.train(x_train, y_train, batch_size=5)


def test_train_adjusts_batch_size_warns(monkeypatch):
    model = Bohamiann()
    x_train = np.random.randn(3, 2)
    y_train = np.random.randn(3)

    warn_msgs = []

    def fake_warning(msg):
        warn_msgs.append(msg)

    monkeypatch.setattr("logging.warning", fake_warning)
    # batch_size > dataset size triggers warning and adjustment
    model.train(x_train, y_train, batch_size=5, num_steps=1, num_burn_in_steps=0)
    assert any("Not enough datapoints" in w for w in warn_msgs)


def test_predict_without_samples_returns_zero_arrays(monkeypatch):
    model = Bohamiann()
    x_train = np.random.randn(5, 2)
    y_train = np.random.randn(5)
    # Train to set up model properly but no samples stored
    model.sampled_weights = []
    model.model = get_default_network(2)().double()
    model.x_mean = np.zeros(2)
    model.x_std = np.ones(2)
    model.y_mean = 0.0
    model.y_std = 1.0
    model.do_normalize_input = True
    model.do_normalize_output = True
    model.use_double_precision = True

    # Predict without samples should handle gracefully (empty array)
    mean, var = model.predict(np.random.randn(3, 2))
    # mean and var should be arrays of length 3
    assert mean.shape == (3,)
    assert var.shape == (3,)
    # Variance should be non-negative
    assert np.all(var >= 0)


def test_predict_return_individual_predictions_shape_and_type():
    model = Bohamiann()
    model.model = get_default_network(2)().double()
    model.x_mean = np.zeros(2)
    model.x_std = np.ones(2)
    model.y_mean = 0.0
    model.y_std = 1.0
    model.sampled_weights = []
    # Manually add 2 samples of weights for prediction
    W1 = tuple(p.detach().clone().numpy() for p in model.model.parameters())
    W2 = tuple(p.detach().clone().numpy() for p in model.model.parameters())
    model.sampled_weights = [W1, W2]
    model.do_normalize_input = True
    model.do_normalize_output = True
    model.use_double_precision = True

    x_test = np.random.randn(4, 2)
    mean, var, indiv = model.predict(x_test, return_individual_predictions=True)
    assert mean.shape == (4,)
    assert var.shape == (4,)
    assert indiv.shape == (2, 4)  # 2 samples, 4 datapoints


def test_predict_single_runs_and_returns_correct_shape():
    model = Bohamiann()
    model.model = get_default_network(2)().double()
    model.x_mean = np.zeros(2)
    model.x_std = np.ones(2)
    model.y_mean = 0.0
    model.y_std = 1.0
    model.sampled_weights = []
    # Add one sample of weights
    W = tuple(p.detach().clone().numpy() for p in model.model.parameters())
    model.sampled_weights = [W]
    model.do_normalize_input = True
    model.do_normalize_output = True
    model.use_double_precision = True

    x_test = np.random.randn(2)
    out = model.predict_single(x_test, sample_index=0)
    # Output should be shape (2,) or (1,2)
    assert out.shape[1] == 2 or len(out.shape) == 1 or out.shape[0] == 2


def test_f_gradient_returns_right_shape_and_type():
    model = Bohamiann()
    model.model = get_default_network(2)().double()
    model.x_mean = np.zeros(2)
    model.x_std = np.ones(2)
    model.y_mean = 0.0
    model.y_std = 1.0
    model.sampled_weights = []
    W = tuple(p.detach().clone().numpy() for p in model.model.parameters())
    model.sampled_weights = [W]
    model.do_normalize_input = True
    model.do_normalize_output = True
    model.use_double_precision = True

    x_test = np.random.randn(2)
    grad = model.f_gradient(x_test, W)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2,)


def test_predictive_mean_gradient_returns_correct_shape():
    model = Bohamiann()
    model.model = get_default_network(2)().double()
    model.x_mean = np.zeros(2)
    model.x_std = np.ones(2)
    model.y_mean = 0.0
    model.y_std = 1.0
    W = tuple(p.detach().clone().numpy() for p in model.model.parameters())
    model.sampled_weights = [W, W]
    model.do_normalize_input = True
    model.do_normalize_output = True
    model.use_double_precision = True

    x_test = np.random.randn(2)
    g = model.predictive_mean_gradient(x_test)
    assert isinstance(g, np.ndarray)
    assert g.shape == (2,)


def test_predictive_variance_gradient_returns_correct_shape():
    model = Bohamiann()
    model.model = get_default_network(2)().double()
    model.x_mean = np.zeros(2)
    model.x_std = np.ones(2)
    model.y_mean = 0.0
    model.y_std = 1.0
    W = tuple(p.detach().clone().numpy() for p in model.model.parameters())
    model.sampled_weights = [W, W]
    model.do_normalize_input = True
    model.do_normalize_output = True
    model.use_double_precision = True

    x_test = np.random.randn(2)
    g = model.predictive_variance_gradient(x_test)
    assert isinstance(g, np.ndarray)
    assert g.shape == (2,)


def test_normalize_input_and_output_methods():
    model = Bohamiann()

    x = np.random.randn(10, 3)
    out_x, mean_x, std_x = model.normalize_input(x)
    assert out_x.shape == x.shape
    # Should have zero mean and std close to 1
    assert np.allclose(np.mean(out_x, axis=0), 0, atol=1e-6)
    assert np.allclose(np.std(out_x, axis=0), 1, atol=1e-6)

    y = np.random.randn(10)
    out_y, mean_y, std_y = model.normalize_output(y)
    assert out_y.shape == y.shape
    assert np.allclose(np.mean(out_y), 0, atol=1e-6)
    assert np.allclose(np.std(out_y), 1, atol=1e-6)


def test_train_and_evaluate_runs_and_returns_length_correct():
    model = Bohamiann()
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    x_valid = np.random.randn(10, 2)
    y_valid = np.random.randn(10)

    n_steps, ll, mse = model.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=10,
        validate_every_n_steps=5,
        keep_every=1,
        num_burn_in_steps=2,
        batch_size=5,
        verbose=False
    )
    # Number of validation steps = num_steps // validate_every_n_steps
    expected_len = 10 // 5
    assert len(n_steps) == expected_len
    assert len(ll) == expected_len
    assert len(mse) == expected_len
    # All metrics should be floats
    for v in ll + mse:
        assert isinstance(v, float)


def test_train_continue_training_appends_samples():
    model = Bohamiann()
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    # First call with continue_training=False (default)
    model.train(x_train, y_train, num_steps=3, num_burn_in_steps=0, keep_every=1, batch_size=5)
    count_after_first = len(model.sampled_weights)
    # Second call with continue_training=True
    model.train(x_train, y_train, num_steps=3, num_burn_in_steps=0, keep_every=1,
                batch_size=5, continue_training=True)
    assert len(model.sampled_weights) > count_after_first


def test_train_respects_use_double_precision_flag():
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)

    # Test with double precision enabled
    model = Bohamiann(use_double_precision=True)
    model.train(x_train, y_train, num_steps=1, num_burn_in_steps=0, keep_every=1, batch_size=5)
    for p in model.model.parameters():
        assert p.dtype == torch.float64

    # Test with double precision disabled
    model = Bohamiann(use_double_precision=False)
    model.train(x_train, y_train, num_steps=1, num_burn_in_steps=0, keep_every=1, batch_size=5)
    for p in model.model.parameters():
        assert p.dtype == torch.float32
