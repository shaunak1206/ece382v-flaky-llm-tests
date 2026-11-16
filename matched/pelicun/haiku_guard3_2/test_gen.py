"""Unit tests for tmvn_rvs and tmvn_MLE in pelicun/uq.py."""

import numpy as np
import pytest

import pelicun.uq as uq


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_unbounded_univariate_shape_and_reproducible():
    np.random.seed(12345)
    first = uq.tmvn_rvs(mu=0.0, COV=1.0, size=10)

    np.random.seed(12345)
    second = uq.tmvn_rvs(mu=0.0, COV=1.0, size=10)

    assert first.shape == (10,)
    np.testing.assert_array_equal(first, second)


def test_rvs_unbounded_multivariate_shape():
    np.random.seed(1)
    samples = uq.tmvn_rvs(mu=[0.0, 1.0],
                          COV=[[1.0, 0.3], [0.3, 2.0]],
                          size=50)

    assert samples.shape == (50, 2)
    assert np.all(np.isfinite(samples))


def test_rvs_different_seeds_give_different_samples():
    np.random.seed(1)
    a = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-2.0, upper=2.0, size=20)

    np.random.seed(2)
    b = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-2.0, upper=2.0, size=20)

    assert not np.array_equal(a, b)


def test_rvs_univariate_two_sided_truncation_within_limits():
    np.random.seed(2024)
    samples = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=500)

    assert samples.shape == (500,)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_rvs_univariate_one_sided_mean_matches_truncated_normal_theory():
    # Standard normal truncated to x > 0 has mean sqrt(2/pi)
    np.random.seed(99)
    samples = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, size=20000)

    assert np.all(samples > 0.0)
    assert np.mean(samples) == pytest.approx(np.sqrt(2.0 / np.pi), abs=0.03)


def test_rvs_multivariate_truncation_all_rows_within_limits():
    np.random.seed(5)
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    samples = uq.tmvn_rvs(mu=[0.0, 0.0],
                          COV=[[1.0, 0.5], [0.5, 1.0]],
                          lower=lower, upper=upper, size=300)

    assert samples.shape == (300, 2)
    assert np.all(samples > lower)
    assert np.all(samples < upper)


def test_rvs_lower_only_truncation():
    np.random.seed(77)
    samples = uq.tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2),
                          lower=[0.0, 0.0], upper=None, size=200)

    assert samples.shape == (200, 2)
    assert np.all(samples > 0.0)


def test_rvs_size_one_multivariate_returns_one_row():
    np.random.seed(31)
    samples = uq.tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2),
                          lower=[-1.0, -1.0], upper=[1.0, 1.0], size=1)

    assert samples.shape == (1, 2)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_rvs_accepts_list_inputs():
    np.random.seed(8)
    samples = uq.tmvn_rvs(mu=[1.0, -1.0],
                          COV=[[1.0, 0.0], [0.0, 1.0]],
                          lower=[0.0, -5.0],
                          upper=[2.0, 5.0],
                          size=100)

    assert samples.shape == (100, 2)
    assert np.all(samples[:, 0] > 0.0)
    assert np.all(samples[:, 0] < 2.0)
    assert np.all(samples[:, 1] > -5.0)
    assert np.all(samples[:, 1] < 5.0)


def test_rvs_raises_when_density_estimate_unreliable(monkeypatch):
    np.random.seed(0)
    # Force the orthotope density estimate to be zero with zero error,
    # which triggers the applicability check in tmvn_rvs.
    monkeypatch.setattr(
        uq, "mvn_orthotope_density",
        lambda mu, COV, lower=None, upper=None: (0.0, 0.0))

    with pytest.raises(ValueError, match="too small"):
        uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=5)


def test_rvs_warns_when_rejection_rate_is_high(monkeypatch):
    np.random.seed(7)
    # Fake a very small acceptance density (below 1e-3) so the high
    # rejection-rate warning is emitted, while the actual sampling
    # still accepts a reasonable fraction of draws.
    monkeypatch.setattr(
        uq, "mvn_orthotope_density",
        lambda *args, **kwargs: (1e-4, 0.0))

    with pytest.warns(UserWarning, match="0.999"):
        samples = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, upper=1.0, size=1)

    assert samples.shape == (1,)
    assert 0.0 < samples[0] < 1.0


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_recovers_sample_moments():
    np.random.seed(7)
    data = np.random.normal(loc=2.0, scale=1.5, size=2000)

    mu, COV = uq.tmvn_MLE(data)

    sample_sd = np.std(data)
    assert mu == pytest.approx(np.mean(data), abs=0.05 * sample_sd)
    assert COV == pytest.approx(np.var(data), rel=0.05)


def test_mle_univariate_scale_equivariance():
    np.random.seed(11)
    data = np.random.normal(size=300)

    mu1, COV1 = uq.tmvn_MLE(data)
    mu2, COV2 = uq.tmvn_MLE(100.0 * data)

    assert mu2 == pytest.approx(100.0 * mu1, rel=1e-3, abs=1e-2)
    assert COV2 == pytest.approx(1e4 * COV1, rel=1e-3)


def test_mle_single_sample_returns_that_sample():
    np.random.seed(4)
    # With one sample no optimization is performed; the mean is the sample
    # and the variance is the negligible replacement value.
    mu, COV = uq.tmvn_MLE(np.array([2.5]))

    assert mu == pytest.approx(2.5, abs=1e-9)
    assert 0.0 < COV < 1e-11


def test_mle_constant_univariate_sample():
    np.random.seed(6)
    data = np.full(20, 3.0)

    mu, COV = uq.tmvn_MLE(data)

    # Zero variance is replaced by a negligible standard deviation, so the
    # mean stays at the constant value and the variance stays negligible.
    assert mu == pytest.approx(3.0, abs=1e-4)
    assert 0.0 <= COV < 1e-6


def test_mle_multivariate_recovers_means_and_correlation():
    np.random.seed(99)
    cov_true = np.array([[1.0, 0.6], [0.6, 2.0]])
    data = np.random.multivariate_normal(mean=[1.0, -2.0],
                                         cov=cov_true,
                                         size=3000).T

    mu, COV = uq.tmvn_MLE(data)

    assert mu.shape == (2,)
    assert COV.shape == (2, 2)

    sample_sd = np.std(data, axis=1)
    sample_mean = np.mean(data, axis=1)
    for i in range(2):
        assert mu[i] == pytest.approx(sample_mean[i], abs=0.05 * sample_sd[i])

    # The correlation matrix is kept at the sample correlation by the fit
    sig = np.sqrt(np.diag(COV))
    corr = COV / np.outer(sig, sig)
    np.testing.assert_allclose(corr, np.corrcoef(data), atol=1e-6)

    # Variances agree with the sample variances up to a small tolerance
    # (the two common estimators differ by a factor n/(n-1)).
    np.testing.assert_allclose(np.diag(COV),
                               np.var(data, axis=1, ddof=1),
                               rtol=0.05)


def test_mle_multivariate_zero_variance_dimension_warns(monkeypatch):
    np.random.seed(3)
    calls = []
    monkeypatch.setattr(uq, "show_warning",
                        lambda message: calls.append(message))

    data = np.vstack([np.random.normal(size=50), np.full(50, 5.0)])

    mu, COV = uq.tmvn_MLE(data)

    assert any("correlation matrix" in c for c in calls)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(COV))
    assert mu[1] == pytest.approx(5.0, abs=1e-3)
    assert 0.0 < COV[1, 1] < 1e-6


def test_mle_censored_univariate_gives_reasonable_estimates():
    np.random.seed(17)
    data = np.random.normal(size=500)
    censored = data >= 1.0
    observed = data[~censored]
    censored_count = int(np.sum(censored))

    mu, COV = uq.tmvn_MLE(observed,
                          censored_count=censored_count,
                          det_lower=-np.inf,
                          det_upper=1.0)

    assert np.isfinite(mu)
    assert np.isfinite(COV)
    assert abs(mu - 0.0) < 0.25
    assert 0.5 < COV < 2.0


def test_mle_infinite_truncation_matches_untruncated():
    np.random.seed(21)
    data = np.random.normal(loc=1.0, scale=2.0, size=200)

    mu_free, COV_free = uq.tmvn_MLE(data)
    mu_tr, COV_tr = uq.tmvn_MLE(data,
                                tr_lower=-np.inf,
                                tr_upper=np.inf)

    assert mu_tr == pytest.approx(mu_free, rel=1e-4, abs=1e-6)
    assert COV_tr == pytest.approx(COV_free, rel=1e-4)


def test_mle_list_input_matches_array_input():
    np.random.seed(42)
    samples = np.random.normal(size=(2, 100))

    mu_arr, COV_arr = uq.tmvn_MLE(samples)
    mu_list, COV_list = uq.tmvn_MLE(samples.tolist())

    np.testing.assert_allclose(mu_list, mu_arr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(COV_list, COV_arr, rtol=1e-12, atol=1e-12)
