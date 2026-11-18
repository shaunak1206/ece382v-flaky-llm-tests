import warnings

import numpy as np
import pytest

import pelicun.uq as uq


def test_tmvn_rvs_no_bounds_univariate_calls_mvn_rvs(monkeypatch):
    calls = {}

    def fake_rvs(mean, cov, size=None):
        calls["mean"] = np.asarray(mean)
        calls["cov"] = np.asarray(cov)
        calls["size"] = size
        return np.array([1.5, 2.5, 3.5])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=2.0, COV=3.0, size=3)

    assert np.allclose(calls["mean"], np.array([2.0]))
    assert np.allclose(calls["cov"], np.array([3.0]))
    assert calls["size"] == 3
    assert np.allclose(out, np.array([1.5, 2.5, 3.5]))


def test_tmvn_rvs_no_bounds_multivariate_preserves_shape(monkeypatch):
    def fake_rvs(mean, cov, size=None):
        mean = np.asarray(mean)
        return np.tile(mean, (size, 1))

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    mu = np.array([1.0, -1.0])
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])

    out = uq.tmvn_rvs(mu=mu, COV=cov, size=4)

    assert out.shape == (4, 2)
    assert np.allclose(out, np.array([[1.0, -1.0]] * 4))


def test_tmvn_rvs_returns_samples_within_bounds(monkeypatch):
    raw = [
        np.array([[0.0], [0.5], [1.5], [0.8], [0.3], [0.9]]),
    ]

    def fake_rvs(mean, cov, size=None):
        return raw.pop(0)

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    out = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, upper=1.0, size=3)

    assert out.shape == (3, 1)
    assert np.all((out > 0.0) & (out < 1.0))
    assert np.allclose(out.flatten(), np.array([0.0, 0.5, 0.8]))


def test_tmvn_rvs_raises_on_too_small_alpha_estimate(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-8, 2e-8))

    with pytest.raises(ValueError, match="density of the joint probability distribution"):
        uq.tmvn_rvs(mu=0.0, COV=1.0, lower=10.0, upper=11.0, size=1)


def test_tmvn_rvs_warns_for_high_rejection_rate(monkeypatch):
    def fake_density(*args, **kwargs):
        return 1e-4, 1e-10

    raw = [
        np.array([[0.0], [0.2], [0.4], [0.6], [0.8], [1.2], [0.7], [0.9], [0.1], [0.3]]),
    ]

    def fake_rvs(mean, cov, size=None):
        return raw.pop(0)

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, upper=1.0, size=2)

    assert any("rejection rate" in str(wi.message) for wi in w)
    assert out.shape == (2, 1)
    assert np.all((out > 0.0) & (out < 1.0))


def test_tmvn_rvs_scalar_mu_and_cov_are_promoted(monkeypatch):
    def fake_density(*args, **kwargs):
        return 1.0, 0.0

    def fake_rvs(mean, cov, size=None):
        assert np.asarray(mean).shape == (1,)
        assert np.asarray(cov).shape == (1,)
        return np.array([[2.0], [3.0]])

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=1.0, COV=4.0, lower=0.0, upper=10.0, size=2)

    assert out.shape == (2, 1)
    assert np.allclose(out.flatten(), np.array([2.0, 3.0]))


def test_mvn_orthotope_density_univariate_calls_mvndst(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["lower"] = np.asarray(lower)
        captured["upper"] = np.asarray(upper)
        captured["infin"] = np.asarray(infin)
        captured["correl"] = correl
        return 0.05, 0.95, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    alpha, eps = uq.mvn_orthotope_density(mu=0.0, COV=1.0, lower=-1.0, upper=1.0)

    assert alpha == pytest.approx(0.95)
    assert eps == pytest.approx(0.05)
    assert captured["correl"] == 0
    assert np.all(captured["infin"] == 2.0)


def test_tmvn_MLE_univariate_returns_sample_mean_and_variance_when_optimization_skipped():
    samples = np.array([1.0, 3.0])

    mu, cov = uq.tmvn_MLE(samples)

    assert mu == pytest.approx(2.0)
    assert cov == pytest.approx(1.0)


def test_tmvn_MLE_multivariate_two_samples_returns_sample_moments():
    samples = np.array([[1.0, 3.0], [2.0, 4.0]])

    mu, cov = uq.tmvn_MLE(samples)

    assert np.allclose(mu, np.array([1.5, 3.5]))
    assert cov.shape == (2, 2)
    assert cov[0, 0] == pytest.approx(0.25)
    assert cov[1, 1] == pytest.approx(0.25)
    assert cov[0, 1] == pytest.approx(0.25)
    assert cov[1, 0] == pytest.approx(0.25)


def test_tmvn_MLE_multivariate_with_constant_dimension_uses_small_variance():
    samples = np.array([[1.0, 1.0], [2.0, 4.0]])

    mu, cov = uq.tmvn_MLE(samples)

    assert np.allclose(mu, np.array([1.5, 2.5]))
    assert cov.shape == (2, 2)
    assert cov[0, 0] > 0.0
    assert cov[1, 1] > 0.0


def test_tmvn_MLE_accepts_1d_input_array():
    samples = np.array([1.0])

    mu, cov = uq.tmvn_MLE(samples)

    assert mu == pytest.approx(1.0)
    assert cov > 0.0


def test_tmvn_MLE_truncated_single_sample_uses_bounds(monkeypatch):
    calls = []

    def fake_density(mu, cov, lower=None, upper=None):
        calls.append((np.asarray(mu), np.asarray(cov), lower, upper))
        return 1.0, 0.0

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    samples = np.array([1.0])
    mu, cov = uq.tmvn_MLE(samples, tr_lower=np.array([0.0]), tr_upper=np.array([2.0]))

    assert calls, "mvn_orthotope_density was not called"
    assert mu == pytest.approx(1.0)
    assert cov > 0.0


def test_tmvn_MLE_with_censored_data_and_bounds_runs(monkeypatch):
    def fake_density(*args, **kwargs):
        return 1.0, 0.0

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    samples = np.array([1.0, 2.0])
    mu, cov = uq.tmvn_MLE(
        samples,
        censored_count=1,
        det_lower=np.array([0.0]),
        det_upper=np.array([3.0]),
        tr_lower=np.array([0.0]),
        tr_upper=np.array([4.0]),
    )

    assert np.isfinite(mu)
    assert np.isfinite(cov)
    assert cov > 0.0


def test_tmvn_MLE_multivariate_rejects_nan_correlations_without_crashing(monkeypatch):
    def fake_warning(msg):
        return None

    monkeypatch.setattr(uq, "show_warning", fake_warning)

    # One dimension is constant, so corrcoef will contain NaNs.
    samples = np.array([[1.0, 1.0], [2.0, 3.0]])

    mu, cov = uq.tmvn_MLE(samples)

    assert np.allclose(mu, np.array([1.0, 2.5]))
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov))


def test_tmvn_MLE_univariate_with_one_sample_has_tiny_variance():
    samples = np.array([5.0])

    mu, cov = uq.tmvn_MLE(samples)

    assert mu == pytest.approx(5.0)
    assert cov == pytest.approx(1e-12)
