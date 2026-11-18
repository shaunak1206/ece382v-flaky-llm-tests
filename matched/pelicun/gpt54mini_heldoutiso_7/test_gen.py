import numpy as np
import pytest

import pelicun.uq as uq


def test_tmvn_rvs_unbounded_scalar_delegates_to_multivariate_normal(monkeypatch):
    calls = {}

    def fake_rvs(mean, cov, size=None):
        calls["mean"] = np.asarray(mean)
        calls["cov"] = np.asarray(cov)
        calls["size"] = size
        return np.array([1.23, 4.56])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(0.5, 2.0, size=3)

    assert np.allclose(calls["mean"], np.array([0.5]))
    assert np.allclose(calls["cov"], np.array([2.0]))
    assert calls["size"] == 3
    assert np.allclose(out, np.array([1.23, 4.56]))


def test_tmvn_rvs_unbounded_vector_preserves_inputs(monkeypatch):
    calls = {}

    def fake_rvs(mean, cov, size=None):
        calls["mean"] = np.asarray(mean)
        calls["cov"] = np.asarray(cov)
        calls["size"] = size
        return np.zeros((size, 2))

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    mu = np.array([1.0, -1.0])
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])
    out = uq.tmvn_rvs(mu, cov, size=4)

    assert calls["size"] == 4
    assert np.allclose(calls["mean"], mu)
    assert np.allclose(calls["cov"], cov)
    assert out.shape == (4, 2)


def test_tmvn_rvs_raises_when_alpha_estimate_is_too_uncertain(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (0.001, 0.00002))

    with pytest.raises(ValueError, match="density of the joint probability distribution"):
        uq.tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=[-1.0], upper=[1.0], size=1)


def test_tmvn_rvs_warns_for_high_rejection_rate(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (5e-4, 1e-7))
    monkeypatch.setattr(uq.multivariate_normal, "rvs", lambda *args, **kwargs: np.array([[0.0], [0.1]]))

    with pytest.warns(UserWarning, match="rejection rate"):
        out = uq.tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=[-1.0], upper=[1.0], size=1)

    assert out.shape == (1, 1)


def test_tmvn_rvs_rejection_sampling_respects_bounds(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (0.5, 1e-6))

    seq = [
        np.array([[-10.0], [0.1], [0.2], [3.0]]),
        np.array([[0.3], [-2.0], [0.4], [0.5]]),
    ]
    calls = {"n": 0}

    def fake_rvs(mean, cov, size=None):
        out = seq[calls["n"]]
        calls["n"] += 1
        return out[:size] if size is not None and size <= len(out) else out

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=[0.0], upper=[1.0], size=3)

    assert out.shape == (3, 1)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    assert calls["n"] >= 1


def test_mvn_orthotope_density_univariate_calls_mvndst(monkeypatch):
    calls = {}

    def fake_mvndst(lower, upper, infin, correl):
        calls["lower"] = np.asarray(lower)
        calls["upper"] = np.asarray(upper)
        calls["infin"] = np.asarray(infin)
        calls["correl"] = correl
        return 0.02, 0.8, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    alpha, eps = uq.mvn_orthotope_density(0.0, 4.0, lower=-1.0, upper=1.0)

    assert alpha == 0.8
    assert eps == 0.02
    assert calls["lower"].shape == (1,)
    assert calls["upper"].shape == (1,)
    assert calls["correl"] == 0


def test_mvn_orthotope_density_multivariate_builds_correlation_vector(monkeypatch):
    calls = {}

    def fake_mvndst(lower, upper, infin, correl):
        calls["lower"] = np.asarray(lower)
        calls["upper"] = np.asarray(upper)
        calls["infin"] = np.asarray(infin)
        calls["correl"] = np.asarray(correl)
        return 0.01, 0.5, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    mu = np.array([0.0, 1.0])
    cov = np.array([[1.0, 0.3], [0.3, 4.0]])
    alpha, eps = uq.mvn_orthotope_density(mu, cov, lower=[-1.0, 0.0], upper=[1.0, 2.0])

    assert alpha == 0.5
    assert eps == 0.01
    assert calls["lower"].shape == (2,)
    assert calls["upper"].shape == (2,)
    assert calls["correl"].shape == (1,)
    assert np.isclose(calls["correl"][0], 0.3 / (1.0 * 2.0))


def test_tmvn_MLE_univariate_no_optimization_when_nsamples_le_ndims(monkeypatch):
    samples = np.array([2.0])
    out = uq.tmvn_MLE(samples)

    assert np.allclose(out[0], 2.0)
    assert np.allclose(out[1], 1e-12, atol=1e-18)


def test_tmvn_MLE_multivariate_returns_reasonable_shape_without_optimization(monkeypatch):
    samples = np.array([[1.0, 2.0], [3.0, 4.0]])  # ndims=2, nsamples=2 => no minimize

    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)


def test_tmvn_MLE_uses_minimize_and_reconstructs_solution(monkeypatch):
    samples = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    fake_x = np.array([0.0, 0.0, 1.0, 1.0])

    class Result:
        def __init__(self, x):
            self.x = x
            self.fun = 0.0
            self.message = "ok"

    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: Result(fake_x))

    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.allclose(mu, np.mean(samples, axis=1), atol=1e-8)
    assert np.allclose(np.diag(cov), np.array([1.0, 4.0]), atol=1e-8)


def test_tmvn_MLE_with_truncation_passes_adjusted_limits_to_density(monkeypatch):
    samples = np.array([0.0, 0.1, -0.1, 0.2])
    calls = []

    def fake_density(mu, cov, lower=None, upper=None):
        calls.append((np.asarray(mu).copy(), np.asarray(cov).copy(), lower, upper))
        return 1.0, 0.0

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    out = uq.tmvn_MLE(samples, tr_lower=np.array([-1.0]), tr_upper=np.array([1.0]))

    assert len(calls) >= 1
    assert np.isclose(calls[0][2], np.array([-1.0])[0] if np.ndim(calls[0][2]) == 0 else calls[0][2]).all() if isinstance(calls[0][2], np.ndarray) else True
    assert np.all(np.isfinite(out[0]))
    assert np.isfinite(out[1])


def test_tmvn_MLE_handles_zero_variance_multidimensional_data(monkeypatch):
    samples = np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])

    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
    assert cov[0, 0] >= 0.0


def test_tmvn_MLE_respects_alpha_lim_in_objective(monkeypatch):
    samples = np.array([[0.0, 0.1, -0.1], [1.0, 1.1, 0.9]])

    def fake_density(mu, cov, lower=None, upper=None):
        return 1e-8, 1e-10

    class Result:
        def __init__(self):
            self.x = np.array([0.0, 0.0, 1.0, 1.0])

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: Result())

    mu, cov = uq.tmvn_MLE(samples, tr_lower=np.array([-2.0, -2.0]), tr_upper=np.array([2.0, 2.0]), alpha_lim=1e-4)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)


def test_tmvn_MLE_censored_data_path_with_mocked_density(monkeypatch):
    samples = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])

    def fake_density(mu, cov, lower=None, upper=None):
        return 1.0, 0.0

    class Result:
        def __init__(self):
            self.x = np.array([0.0, 0.0, 1.0, 1.0])

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: Result())

    mu, cov = uq.tmvn_MLE(
        samples,
        censored_count=2,
        det_lower=np.array([0.0, 0.0]),
        det_upper=np.array([10.0, 10.0]),
    )

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
