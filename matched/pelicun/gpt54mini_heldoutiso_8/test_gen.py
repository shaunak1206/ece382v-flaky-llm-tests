import numpy as np
import pytest
from scipy.stats import multivariate_normal

import pelicun.uq as uq


def test_tmvn_rvs_untruncated_scalar_calls_scipy_and_returns_shape(monkeypatch):
    calls = {}

    def fake_rvs(mean, cov, size=None, **kwargs):
        calls["mean"] = np.asarray(mean)
        calls["cov"] = np.asarray(cov)
        calls["size"] = size
        return np.arange(size).reshape(size, 1).astype(float)

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(1.5, 2.0, size=4)

    assert out.shape == (4, 1)
    np.testing.assert_allclose(calls["mean"], np.array([1.5]))
    np.testing.assert_allclose(calls["cov"], np.array([2.0]))
    assert calls["size"] == 4
    np.testing.assert_allclose(out[:, 0], np.arange(4, dtype=float))


def test_tmvn_rvs_untruncated_multivariate_returns_scipy_output(monkeypatch):
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])

    def fake_rvs(mean, cov, size=None, **kwargs):
        assert size == 2
        return expected.copy()

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs([0.0, 0.0], np.eye(2), size=2)

    np.testing.assert_allclose(out, expected)


def test_tmvn_rvs_truncated_rejection_sampling_accepts_only_inside_bounds(monkeypatch):
    def fake_density(mu, COV, lower=None, upper=None):
        return 0.8, 0.01

    raw_batches = [
        np.array([[2.0], [0.5], [0.2], [1.0], [0.7]]),
        np.array([[0.4], [0.6], [0.8], [0.9], [1.1]]),
    ]
    call_count = {"i": 0}

    def fake_rvs(mean, cov, size=None, **kwargs):
        arr = raw_batches[call_count["i"]]
        call_count["i"] += 1
        return arr

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs([0.0], [1.0], lower=[0.3], upper=[0.95], size=3)

    assert out.shape == (3, 1)
    assert np.all((out > 0.3) & (out < 0.95))
    np.testing.assert_allclose(out[:, 0], np.array([0.5, 0.4, 0.6]))


def test_tmvn_rvs_raises_when_alpha_too_small(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (0.0001, 0.01))

    with pytest.raises(ValueError, match="density of the joint probability distribution"):
        uq.tmvn_rvs([0.0], [1.0], lower=[1.0], upper=[2.0], size=1)


def test_tmvn_rvs_warns_for_high_rejection_rate(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (5e-4, 1e-8))

    def fake_rvs(mean, cov, size=None, **kwargs):
        return np.array([[0.0], [0.0], [0.0], [0.0]])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    with pytest.warns(UserWarning, match="rejection rate for sampling"):
        out = uq.tmvn_rvs([0.0], [1.0], lower=[-1.0], upper=[1.0], size=2)

    assert out.shape == (2, 1)


def test_tmvn_orthotope_density_scalar_delegates_to_mvndst(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["lower"] = np.asarray(lower)
        captured["upper"] = np.asarray(upper)
        captured["infin"] = np.asarray(infin)
        captured["correl"] = correl
        return 0.012, 0.34, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    alpha, eps = uq.mvn_orthotope_density(1.0, 4.0, lower=0.0, upper=2.0)

    assert alpha == 0.34
    assert eps == 0.012
    np.testing.assert_allclose(captured["lower"], np.array([-0.5]))
    np.testing.assert_allclose(captured["upper"], np.array([0.5]))
    np.testing.assert_allclose(captured["infin"], np.array([2.0]))
    assert captured["correl"] == 0


def test_tmvn_orthotope_density_multivariate_provides_lower_triangle_correlations(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["lower"] = np.asarray(lower)
        captured["upper"] = np.asarray(upper)
        captured["infin"] = np.asarray(infin)
        captured["correl"] = np.asarray(correl)
        return 0.1, 0.2, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    mu = np.array([0.0, 1.0])
    cov = np.array([[1.0, 0.3], [0.3, 4.0]])
    alpha, eps = uq.mvn_orthotope_density(mu, cov, lower=[-1.0, 0.0], upper=[1.0, 2.0])

    assert alpha == 0.1
    assert eps == 0.2
    np.testing.assert_allclose(captured["correl"], np.array([0.3 / (1.0 * 2.0)]))


def test_tmvn_mle_univariate_returns_sample_mean_and_variance_for_ordinary_data():
    samples = np.array([1.0, 2.0, 3.0, 4.0])

    mu, cov = uq.tmvn_MLE(samples)

    assert np.isclose(mu, np.mean(samples))
    assert np.isclose(cov, np.var(samples, ddof=0))


def test_tmvn_mle_multivariate_returns_sample_mean_and_covariance_for_ordinary_data():
    samples = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])

    mu, cov = uq.tmvn_MLE(samples)

    np.testing.assert_allclose(mu, np.array([2.0, 6.0]), rtol=1e-12, atol=1e-12)
    expected_cov = np.cov(samples)
    np.testing.assert_allclose(cov, expected_cov, rtol=1e-12, atol=1e-12)


def test_tmvn_mle_uses_minimum_sigma_for_constant_univariate_data():
    samples = np.array([5.0, 5.0, 5.0])

    mu, cov = uq.tmvn_MLE(samples)

    assert np.isclose(mu, 5.0)
    assert np.isclose(cov, 0.0)


def test_tmvn_mle_handles_constant_multivariate_dimension_without_nan():
    samples = np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]])

    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.isfinite(mu).all()
    assert np.isfinite(cov).all()
    assert cov[0, 0] >= 0.0


def test_tmvn_mle_truncated_data_calls_mvn_orthotope_density(monkeypatch):
    call_args = []

    def fake_density(mu, cov, lower=None, upper=None):
        call_args.append((np.asarray(mu).copy(), np.asarray(cov).copy(), lower, upper))
        return 1.0, 0.0

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    samples = np.array([0.2, 0.4, 0.6, 0.8])
    mu, cov = uq.tmvn_MLE(samples, tr_lower=np.array([-1.0]), tr_upper=np.array([1.0]))

    assert len(call_args) >= 1
    assert np.isfinite(mu)
    assert np.isfinite(cov)


def test_tmvn_mle_with_censored_data_calls_density_for_detection_limits(monkeypatch):
    calls = []

    def fake_density(mu, cov, lower=None, upper=None):
        calls.append((np.asarray(mu).copy(), np.asarray(cov).copy(), lower, upper))
        return 1.0, 0.0

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    samples = np.array([1.0, 1.2, 1.4, 1.6])
    mu, cov = uq.tmvn_MLE(
        samples,
        censored_count=2,
        det_lower=np.array([0.5]),
        det_upper=np.array([1.8]),
    )

    assert len(calls) >= 1
    assert np.isfinite(mu)
    assert np.isfinite(cov)


def test_tmvn_mle_rejects_too_small_truncation_probability(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-8, 1e-4))

    samples = np.array([0.0, 0.1, 0.2])

    mu, cov = uq.tmvn_MLE(samples, tr_lower=np.array([10.0]), tr_upper=np.array([11.0]))

    assert np.isfinite(mu)
    assert np.isfinite(cov)


def test_tmvn_mle_multivariate_output_is_positive_semidefinite():
    samples = np.array([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]])

    mu, cov = uq.tmvn_MLE(samples)

    eigvals = np.linalg.eigvals(cov)
    assert np.all(eigvals >= -1e-10)


def test_tmvn_rvs_single_dimension_returns_column_vector(monkeypatch):
    def fake_density(mu, COV, lower=None, upper=None):
        return 0.9, 0.01

    def fake_rvs(mean, cov, size=None, **kwargs):
        return np.array([[0.2], [0.3], [0.4]])

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(0.0, 1.0, lower=0.0, upper=1.0, size=2)

    assert out.shape == (2, 1)
    assert np.all((out > 0.0) & (out < 1.0))
