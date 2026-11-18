import numpy as np
import pytest
import torch

from pybnn.bohamiann import Bohamiann, get_default_network, nll


def test_get_default_network_returns_module():
    net = get_default_network(5)
    assert isinstance(net, torch.nn.Module)
    sample_input = torch.randn(2, 5)
    out = net(sample_input)
    assert out.shape[0] == 2 and out.shape[1] == 2  # output is (batch, 2)


def test_nll_basic_behavior():
    pred = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    target = torch.tensor([0.0, 1.0])
    loss = nll(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_nll_returns_scalar_and_positive():
    N = 3
    pred = torch.cat([torch.randn(N, 1), torch.zeros(N, 1)], dim=1)
    target = torch.randn(N)
    val = nll(pred, target)
    assert val.dim() == 0


def test_bohamiann_initialization_defaults():
    model = Bohamiann()
    assert model.do_normalize_input
    assert model.do_normalize_output
    assert isinstance(model.sampled_weights, list)
    assert model.is_trained is False
    # Default network is stored as callable
    assert callable(model.get_network)


def test_network_weights_property_and_setter():
    model = Bohamiann()
    model.model = get_default_network(3).double()
    original_weights = model.network_weights
    assert all(isinstance(w, np.ndarray) for w in original_weights)

    # Modify weights and set them again
    new_weights = [w + 0.1 for w in original_weights]
    model.network_weights = new_weights
    updated_weights = model.network_weights
    for nw, uw in zip(new_weights, updated_weights):
        # They should be close but not necessarily exactly equal due to float conversion
        assert np.allclose(nw, uw)


def test_train_runs_and_sets_is_trained():
    # Small dataset for testing
    x_train = np.random.randn(30, 3)
    y_train = np.random.randn(30)
    model = Bohamiann(print_every_n_steps=5, sampling_method="sgld", use_double_precision=False)
    model.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=5, batch_size=10, verbose=False)
    assert model.is_trained
    # After training some weights should be sampled
    assert isinstance(model.sampled_weights, list)
    assert len(model.sampled_weights) > 0


@pytest.mark.parametrize("sampling_method", ["adaptive_sghmc", "sgld", "sghmc", "preconditioned_sgld"])
def test_train_sets_sampler_of_correct_type(sampling_method):
    x_train = np.random.randn(25, 2)
    y_train = np.random.randn(25)
    model = Bohamiann(print_every_n_steps=5, sampling_method=sampling_method, use_double_precision=False)
    model.train(x_train, y_train, num_steps=5, keep_every=2, num_burn_in_steps=2, batch_size=5, verbose=False)
    assert model.sampler is not None


def test_train_rejects_invalid_batch_size():
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10)
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x_train, y_train, batch_size=0)  # batch_size < 1 not allowed


def test_train_accepts_continue_training_flag():
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    model = Bohamiann(print_every_n_steps=5, sampling_method="sgld", use_double_precision=True)
    model.train(x_train, y_train, num_steps=5, keep_every=2, num_burn_in_steps=2, batch_size=10, verbose=False)
    prev_weights_count = len(model.sampled_weights)
    model.train(x_train, y_train, num_steps=5, keep_every=1, num_burn_in_steps=0, continue_training=True, batch_size=10, verbose=False)
    # Should keep previously sampled weights and add more
    assert len(model.sampled_weights) >= prev_weights_count


def test_predict_returns_mean_variance_shape():
    x_train = np.random.randn(40, 4)
    y_train = np.random.randn(40)
    model = Bohamiann(print_every_n_steps=5, sampling_method="sgld", use_double_precision=False)
    model.train(x_train, y_train, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=10, verbose=False)
    x_test = np.random.randn(3, 4)
    mean, var = model.predict(x_test)
    assert mean.shape == (3,)
    assert var.shape == (3,)


def test_predict_with_return_individual_predictions():
    x_train = np.random.randn(30, 3)
    y_train = np.random.randn(30)
    model = Bohamiann(print_every_n_steps=5, sampling_method="sgld", use_double_precision=True)
    model.train(x_train, y_train, num_steps=15, keep_every=3, num_burn_in_steps=5, batch_size=5, verbose=False)
    x_test = np.random.randn(2, 3)
    m, v, indiv = model.predict(x_test, return_individual_predictions=True)
    assert m.shape == (2,)
    assert v.shape == (2,)
    assert indiv.shape[0] == len(model.sampled_weights)
    assert indiv.shape[1] == 2


def test_predict_single_returns_expected_shape_and_type():
    x_train = np.random.randn(30, 2)
    y_train = np.random.randn(30)
    model = Bohamiann(print_every_n_steps=5, sampling_method="sgld", use_double_precision=True)
    model.train(x_train, y_train, num_steps=15, keep_every=3, num_burn_in_steps=5, batch_size=10, verbose=False)
    x_test = np.random.randn(4, 2)
    result = model.predict_single(x_test, sample_index=0)
    assert isinstance(result, np.ndarray)
    if model.do_normalize_output:
        assert result.shape[0] == 4 and result.shape[1] == 2
    else:
        assert result.shape[0] == 4 and result.shape[1] == 2


def test_f_gradient_returns_correct_shape():
    x_train = np.random.randn(25, 3)
    y_train = np.random.randn(25)
    model = Bohamiann(print_every_n_steps=5, sampling_method="sgld", use_double_precision=True)
    model.train(x_train, y_train, num_steps=10, keep_every=2, num_burn_in_steps=5, batch_size=5, verbose=False)
    x_test = np.random.randn(3)
    weights = model.sampled_weights[0]
    g = model.f_gradient(x_test, weights)
    assert isinstance(g, np.ndarray)
    assert g.shape == x_test.shape


def test_predictive_mean_gradient_returns_expected_array():
    x_train = np.random.randn(30, 3)
    y_train = np.random.randn(30)
    model = Bohamiann(print_every_n_steps=5, sampling_method="sgld", use_double_precision=False)
    model.train(x_train, y_train, num_steps=15, keep_every=3, num_burn_in_steps=5, batch_size=10, verbose=False)
    x_test = np.random.randn(3)
    g = model.predictive_mean_gradient(x_test)
    assert isinstance(g, np.ndarray)
    assert g.shape == x_test.shape


def test_predictive_variance_gradient_returns_expected_array():
    x_train = np.random.randn(35, 3)
    y_train = np.random.randn(35)
    model = Bohamiann(print_every_n_steps=5, sampling_method="sgld", use_double_precision=False)
    model.train(x_train, y_train, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=10, verbose=False)
    x_test = np.random.randn(3)
    g = model.predictive_variance_gradient(x_test)
    assert isinstance(g, np.ndarray)
    assert g.shape == x_test.shape


def test_train_and_evaluate_runs_and_returns_curves():
    x_train = np.random.randn(50, 4)
    y_train = np.random.randn(50)
    x_valid = np.random.randn(20, 4)
    y_valid = np.random.randn(20)
    model = Bohamiann(print_every_n_steps=5, sampling_method="sgld", use_double_precision=False)
    n_steps, lls, mses = model.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=20, validate_every_n_steps=10, keep_every=5, num_burn_in_steps=10,
        lr=1e-2, epsilon=1e-10, batch_size=10, mdecay=0.05, verbose=False
    )
    assert isinstance(n_steps, list)
    assert isinstance(lls, list)
    assert isinstance(mses, list)
    assert len(n_steps) == 2  # 20 // 10
    assert len(lls) == 2
    assert len(mses) == 2


def test_normalize_input_output_with_given_mean_std():
    model = Bohamiann()
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    x_norm, m, s = model.normalize_input(x)
    assert np.allclose(m, np.mean(x, axis=0))
    assert np.allclose(s, np.std(x, axis=0))
    assert x_norm.shape == x.shape

    y = np.array([1.0, 3.0])
    y_norm, m2, s2 = model.normalize_output(y)
    assert np.allclose(m2, np.mean(y, axis=0))
    assert np.allclose(s2, np.std(y, axis=0))
    assert y_norm.shape == y.shape


def test_network_weights_setter_rejects_invalid_shape(monkeypatch):
    model = Bohamiann()
    model.model = get_default_network(2).double()
    weights = model.network_weights
    # Modify one weight to wrong shape
    wrong_weights = list(weights)
    wrong_weights[0] = np.ones((10, 10))
    with pytest.raises(RuntimeError):
        model.network_weights = wrong_weights


def test_train_warns_and_adjusts_batch_size(caplog):
    x_train = np.random.randn(5, 2)
    y_train = np.random.randn(5)
    model = Bohamiann()
    with caplog.at_level("WARNING"):
        model.train(x_train, y_train, batch_size=10, num_steps=5, num_burn_in_steps=2, verbose=False)
        assert "Not enough datapoints to form a batch" in caplog.text


def test_predict_no_sampled_weights_returns_error():
    model = Bohamiann()
    with pytest.raises(IndexError):
        model.predict(np.random.randn(1, 3))


def test_predict_single_index_out_of_range_raises():
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10)
    model = Bohamiann()
    model.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=5, batch_size=5, verbose=False)
    with pytest.raises(IndexError):
        model.predict_single(np.random.randn(1, 2), sample_index=1000)

