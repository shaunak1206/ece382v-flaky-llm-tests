import warnings

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import pelicun.uq as uq


class DummyOptimizeResult:
    def __init__(self, x, fun=0.0, message="ok"):
        self.x = np.asarray(x)
        self.fun = fun
        self.message = message


def test_tmvn_rvs_untruncated_scalar_calls_multivariate_normal(monkeypatch):
    called = {}

    def fake_rvs(mean, cov, size=None):
        called["mean"] = np.asarray(mean)
        called["cov"] = np.asarray(cov)
        called["size"] = size
        return np.array([1.23, 4.56])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=2.0, COV=3.0, size=2)

    assert np.allclose(called["mean"], [2.0])
    assert np.allclose(called["cov"], [3.0])
    assert called["size"] == 2
    assert np.allclose(out, [1.23, 4.56])


def test_tmvn_rvs_untruncated_multivariate_returns_shape(monkeypatch):
    def fake_rvs(mean, cov, size=None):
        assert np.allclose(mean, [0.0, 1.0])
        assert np.allclose(cov, [[1.0, 0.2], [0.2, 2.0]])
        return np.arange(size * 2, dtype=float).reshape(size, 2)

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=[0.0, 1.0], COV=[[1.0, 0.2], [0.2, 2.0]], size=3)

    assert out.shape == (3, 2)
    assert np.allclose(out, np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))


def test_tmvn_rvs_truncated_raises_when_alpha_too_small(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-6, 1e-4))

    with pytest.raises(ValueError, match="density of the joint probability distribution"):
        uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=1)


def test_tmvn_rvs_truncated_warns_when_rejection_rate_high(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-4, 1e-8))

    def fake_rvs(mean, cov, size=None):
        # Always within bounds, so sampling finishes immediately.
        return np.zeros(size)

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    with pytest.warns(UserWarning, match="rejection rate"):
        out = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=4)

    assert out.shape == (4,)
    assert np.all(out == 0.0)


def test_tmvn_rvs_truncated_multivariate_all_samples_within_bounds(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (0.5, 1e-8))

    def fake_rvs(mean, cov, size=None):
        return np.array(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [2.0, 2.0],   # out of bounds
                [-2.0, 0.0],  # out of bounds
                [0.25, 0.25],
            ]
        )

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2), lower=[-1.0, -1.0], upper=[1.0, 1.0], size=3)

    assert out.shape == (3, 2)
    assert np.all(out > -1.0)
    assert np.all(out < 1.0)


def test_tmvn_rvs_truncated_uses_default_infinite_bounds(monkeypatch):
    captured = {}

    def fake_density(mu, COV, lower=None, upper=None):
        captured["lower"] = np.asarray(lower)
        captured["upper"] = np.asarray(upper)
        return 1.0, 1e-8

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    def fake_rvs(mean, cov, size=None):
        return np.zeros((size, 2))

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=[0.0, 1.0], COV=np.eye(2), lower=None, upper=[1.0, 2.0], size=2)

    assert np.isneginf(captured["lower"]).all()
    assert np.allclose(captured["upper"], [1.0, 2.0])
    assert out.shape == (2, 2)


def test_tmvn_MLE_1d_no_truncation_returns_expected_estimates(monkeypatch):
    monkeypatch.setattr(uq, "minimize", lambda fun, x0, args=(), **kwargs: DummyOptimizeResult(x0))

    samples = np.array([1.0, 3.0])
    mu, cov = uq.tmvn_MLE(samples)

    assert mu == pytest.approx(2.1)
    assert cov == pytest.approx(2.0)


def test_tmvn_MLE_1d_single_sample_uses_direct_estimate(monkeypatch):
    # nsamples == ndims => no Nelder-Mead step
    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: pytest.fail("minimize should not be called"))

    samples = np.array([5.0])
    mu, cov = uq.tmvn_MLE(samples)

    assert mu == pytest.approx(5.0)
    assert cov == pytest.approx(0.0)


def test_tmvn_MLE_2d_no_truncation_returns_positive_covariance(monkeypatch):
    monkeypatch.setattr(uq, "minimize", lambda fun, x0, args=(), **kwargs: DummyOptimizeResult(x0))

    samples = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.diag(cov) > 0.0)
    assert np.all(np.isfinite(cov))


def test_tmvn_MLE_handles_zero_variance_dimension(monkeypatch):
    monkeypatch.setattr(uq, "minimize", lambda fun, x0, args=(), **kwargs: DummyOptimizeResult(x0))

    samples = np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]])
    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.isfinite(mu).all()
    assert np.isfinite(cov).all()
    assert cov[0, 0] > 0.0
    assert cov[1, 1] > 0.0


def test_tmvn_MLE_warns_on_nan_correlation_estimate(monkeypatch):
    warnings_seen = []

    def fake_show_warning(msg):
        warnings_seen.append(msg)

    monkeypatch.setattr(uq, "show_warning", fake_show_warning)
    monkeypatch.setattr(uq, "minimize", lambda fun, x0, args=(), **kwargs: DummyOptimizeResult(x0))

    # First dimension is constant, so corrcoef can become nan.
    samples = np.array([[1.0, 1.0, 1.0], [5.0, 6.0, 7.0]])
    uq.tmvn_MLE(samples)

    assert any("correlation matrix" in str(msg) for msg in warnings_seen)


def test_tmvn_MLE_truncated_calls_mvn_orthotope_density(monkeypatch):
    density_calls = []

    def fake_density(mu, COV, lower=None, upper=None):
        density_calls.append((np.asarray(mu), np.asarray(COV), np.asarray(lower), np.asarray(upper)))
        return 0.8, 1e-8

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq, "minimize", lambda fun, x0, args=(), **kwargs: DummyOptimizeResult(x0))

    samples = np.array([1.0, 2.0, 3.0, 4.0])
    mu, cov = uq.tmvn_MLE(samples, tr_lower=0.0, tr_upper=10.0)

    assert len(density_calls) >= 1
    assert mu == pytest.approx(2.6, rel=1e-6)
    assert cov == pytest.approx(2.5, rel=1e-6)


def test_tmvn_MLE_with_censoring_uses_detection_limits(monkeypatch):
    calls = []

    def fake_density(mu, COV, lower=None, upper=None):
        calls.append((np.asarray(lower), np.asarray(upper)))
        return 0.9, 1e-8

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "pdf", lambda *args, **kwargs: np.ones(4))
    monkeypatch.setattr(uq, "minimize", lambda fun, x0, args=(), **kwargs: DummyOptimizeResult(x0))

    samples = np.array([1.0, 1.2, 1.1, 0.9])
    mu, cov = uq.tmvn_MLE(
        samples,
        censored_count=2,
        det_lower=0.5,
        det_upper=1.5,
    )

    assert len(calls) >= 2
    assert mu == pytest.approx(np.mean(samples), rel=1e-6)
    assert cov == pytest.approx(np.var(samples, ddof=0), rel=1e-6)


def test_tmvn_MLE_alpha_lim_can_force_nontrivial_path(monkeypatch):
    # Ensure the alpha_lim branch can be taken without numerical issues.
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (0.5, 1e-8))
    monkeypatch.setattr(uq.multivariate_normal, "pdf", lambda *args, **kwargs: np.ones(5))
    monkeypatch.setattr(uq, "minimize", lambda fun, x0, args=(), **kwargs: DummyOptimizeResult(x0))

    samples = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    mu, cov = uq.tmvn_MLE(samples, tr_lower=-1.0, tr_upper=3.0, alpha_lim=1e-4)

    assert np.isfinite(mu)
    assert np.isfinite(cov)
    assert cov > 0.0
