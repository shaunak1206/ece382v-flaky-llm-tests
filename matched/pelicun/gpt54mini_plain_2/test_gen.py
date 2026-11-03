import numpy as np
import pytest

import pelicun.uq as uq


def test_tmvn_rvs_univariate_no_bounds_returns_multivariate_normal_rvs(monkeypatch):
    calls = []

    def fake_rvs(mean, cov, size=None):
        calls.append((np.asarray(mean), np.asarray(cov), size))
        return np.array([1.23, 4.56])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=0.0, COV=1.5, size=2)

    assert out.tolist() == [1.23, 4.56]
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.array([0.0]))
    np.testing.assert_array_equal(calls[0][1], np.array([1.5]))
    assert calls[0][2] == 2


def test_tmvn_rvs_multivariate_no_bounds_passes_mean_cov_and_size(monkeypatch):
    calls = []

    def fake_rvs(mean, cov, size=None):
        calls.append((np.asarray(mean), np.asarray(cov), size))
        return np.ones((3, 2))

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    mu = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.1], [0.1, 2.0]])
    out = uq.tmvn_rvs(mu=mu, COV=cov, size=3)

    assert out.shape == (3, 2)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], mu)
    np.testing.assert_array_equal(calls[0][1], cov)
    assert calls[0][2] == 3


def test_tmvn_rvs_scalar_inputs_with_bounds_uses_density_and_filters(monkeypatch):
    density_calls = []

    def fake_density(mu, cov, lower=None, upper=None):
        density_calls.append((np.asarray(mu), np.asarray(cov), lower, upper))
        return 1.0, 0.0

    raw = np.array([-1.0, 0.2, 0.6, 1.5])

    def fake_rvs(mean, cov, size=None):
        return raw[:size]

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, upper=1.0, size=2)

    assert out.shape == (2,)
    np.testing.assert_array_equal(out, np.array([0.2, 0.6]))
    assert len(density_calls) == 1
    np.testing.assert_array_equal(density_calls[0][0], np.array([0.0]))
    np.testing.assert_array_equal(density_calls[0][1], np.array([1.0]))


def test_tmvn_rvs_rejects_too_small_alpha_estimate(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-6, 2e-6))

    with pytest.raises(ValueError, match="density of the joint probability distribution"):
        uq.tmvn_rvs(mu=np.array([0.0]), COV=np.array([1.0]), lower=-1.0, upper=1.0, size=1)


def test_tmvn_rvs_warns_for_high_rejection_rate(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (5e-4, 0.0))
    monkeypatch.setattr(
        uq.multivariate_normal,
        "rvs",
        lambda mean, cov, size=None: np.array([0.0, 0.0, 0.0, 0.0]),
    )

    with pytest.warns(UserWarning, match="rejection rate"):
        out = uq.tmvn_rvs(mu=np.array([0.0]), COV=np.array([1.0]), lower=-1.0, upper=1.0, size=1)

    assert out.shape == (1,)


def test_tmvn_rvs_samples_until_requested_size(monkeypatch):
    density_calls = []

    def fake_density(mu, cov, lower=None, upper=None):
        density_calls.append(1)
        return 0.5, 0.0

    raw_batches = [
        np.array([[-1.0], [0.1], [0.2], [2.0]]),
        np.array([[0.3], [0.4], [3.0], [4.0]]),
    ]
    call_idx = {"i": 0}

    def fake_rvs(mean, cov, size=None):
        arr = raw_batches[call_idx["i"]]
        call_idx["i"] += 1
        return arr[:size]

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=np.array([0.0]), COV=np.array([1.0]), lower=0.0, upper=1.0, size=3)

    assert out.shape == (3,)
    np.testing.assert_allclose(out, np.array([0.1, 0.2, 0.3]))
    assert len(density_calls) == 1
    assert call_idx["i"] >= 2


def test_tmvn_orthotope_density_univariate_calls_mvndst_with_standardized_limits(monkeypatch):
    calls = []

    def fake_mvndst(lower, upper, infin, correl):
        calls.append((np.asarray(lower), np.asarray(upper), np.asarray(infin), correl))
        return 0.01, 0.9, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    alpha, eps = uq.mvn_orthotope_density(mu=1.0, COV=4.0, lower=0.0, upper=3.0)

    assert alpha == pytest.approx(0.9)
    assert eps == pytest.approx(0.01)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], np.array([-0.5]))
    np.testing.assert_allclose(calls[0][1], np.array([1.0]))
    np.testing.assert_allclose(calls[0][2], np.array([2.0]))
    assert calls[0][3] == 0


def test_tmvn_orthotope_density_multivariate_flattens_lower_triangle_correlation(monkeypatch):
    calls = []

    def fake_mvndst(lower, upper, infin, correl):
        calls.append((np.asarray(lower), np.asarray(upper), np.asarray(infin), np.asarray(correl)))
        return 0.02, 0.8, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.3], [0.3, 4.0]])
    alpha, eps = uq.mvn_orthotope_density(mu=mu, COV=cov, lower=[-1.0, -2.0], upper=[1.0, 2.0])

    assert alpha == pytest.approx(0.8)
    assert eps == pytest.approx(0.02)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][3], np.array([0.15]))


def test_tmvn_MLE_univariate_returns_sample_mean_and_variance_for_untruncated_data():
    samples = np.array([1.0, 2.0, 3.0, 4.0])

    mu, cov = uq.tmvn_MLE(samples)

    assert mu == pytest.approx(np.mean(samples))
    assert cov == pytest.approx(np.var(samples, ddof=0))


def test_tmvn_MLE_univariate_constant_sample_uses_negligible_initial_variance():
    samples = np.array([5.0, 5.0, 5.0])

    mu, cov = uq.tmvn_MLE(samples)

    assert mu == pytest.approx(5.0)
    assert cov > 0.0
    assert cov == pytest.approx(1e-12, rel=0.2, abs=1e-15)


def test_tmvn_MLE_multivariate_untruncated_returns_mean_and_covariance(monkeypatch):
    samples = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ]
    )

    mu, cov = uq.tmvn_MLE(samples)

    expected_mu = np.mean(samples, axis=1)
    expected_cov = np.cov(samples, bias=True)

    np.testing.assert_allclose(mu, expected_mu)
    np.testing.assert_allclose(cov, expected_cov)


def test_tmvn_MLE_with_truncation_uses_alpha_limit_and_returns_unbiased_estimate(monkeypatch):
    samples = np.array([0.0, 1.0, 2.0, 3.0])
    calls = []

    def fake_density(mu, cov, lower=None, upper=None):
        calls.append((np.asarray(mu), np.asarray(cov), lower, upper))
        return 1.0, 0.0

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    mu, cov = uq.tmvn_MLE(samples, tr_lower=np.array([-1.0]), tr_upper=np.array([5.0]), alpha_lim=1e-6)

    assert len(calls) >= 1
    assert mu == pytest.approx(np.mean(samples))
    assert cov > 0.0


def test_tmvn_MLE_raises_when_truncation_alpha_is_not_estimable(monkeypatch):
    samples = np.array([0.0, 1.0, 2.0])

    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-8, 1e-6))

    with pytest.raises(ValueError, match="cannot be estimated with sufficiently high accuracy"):
        uq.tmvn_MLE(samples, tr_lower=np.array([-10.0]), tr_upper=np.array([10.0]))


def test_tmvn_MLE_handles_censored_data_when_detection_limits_provided(monkeypatch):
    samples = np.array([0.0, 1.0, 2.0, 3.0])
    density_calls = []

    def fake_density(mu, cov, lower=None, upper=None):
        density_calls.append((np.asarray(mu), np.asarray(cov), lower, upper))
        return 1.0, 0.0

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    mu, cov = uq.tmvn_MLE(
        samples,
        censored_count=1,
        det_lower=np.array([-1.0]),
        det_upper=np.array([4.0]),
    )

    assert mu == pytest.approx(np.mean(samples))
    assert cov > 0.0
    assert len(density_calls) >= 1
