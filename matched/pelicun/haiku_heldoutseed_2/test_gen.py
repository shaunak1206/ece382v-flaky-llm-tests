"""Unit tests for tmvn_rvs and tmvn_MLE in pelicun/uq.py."""

import warnings

import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


@pytest.fixture(autouse=True)
def _seed_numpy_rng():
    """Make every test deterministic by seeding the global NumPy RNG."""
    np.random.seed(20240601)
    yield


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_unbounded_multivariate_shape():
    """Unbounded sampling returns an array of shape (size, ndim)."""
    samples = tmvn_rvs(mu=np.zeros(3), COV=np.eye(3), size=500)
    samples = np.asarray(samples)
    assert samples.shape == (500, 3)
    assert np.all(np.isfinite(samples))


def test_rvs_unbounded_moments_match_input():
    """Sample mean and covariance of unbounded draws approximate the inputs."""
    mu = np.array([1.0, -1.0])
    COV = np.array([[2.0, 0.5],
                    [0.5, 1.0]])
    samples = np.asarray(tmvn_rvs(mu=mu, COV=COV, size=20000))

    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)
    assert np.allclose(np.cov(samples, rowvar=False), COV, atol=0.1)


def test_rvs_scalar_mu_and_cov_returns_requested_size():
    """A scalar mean and variance are handled and produce `size` samples."""
    samples = np.asarray(tmvn_rvs(mu=2.0, COV=0.25, size=5000)).reshape(-1)
    assert samples.size == 5000
    assert abs(samples.mean() - 2.0) < 0.05
    assert abs(samples.var() - 0.25) < 0.03


def test_rvs_bounded_univariate_within_limits():
    """All truncated univariate samples lie strictly inside the limits."""
    samples = np.asarray(
        tmvn_rvs(mu=0.5, COV=1.0, lower=[0.0], upper=[1.0], size=1000)
    ).reshape(-1)
    assert samples.size == 1000
    assert np.all((samples > 0.0) & (samples < 1.0))


def test_rvs_bounded_multivariate_within_box():
    """All truncated bivariate samples lie inside the hyperrectangle."""
    samples = np.asarray(
        tmvn_rvs(mu=np.zeros(2), COV=np.eye(2),
                 lower=[-1.0, -1.0], upper=[1.0, 1.0], size=1000)
    )
    assert samples.shape == (1000, 2)
    assert np.all(np.abs(samples) < 1.0)


def test_rvs_upper_bound_only():
    """With lower=None, only the upper truncation is applied."""
    samples = np.asarray(
        tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=None, upper=[0.0], size=1000)
    ).reshape(-1)
    assert samples.size == 1000
    assert np.all(samples < 0.0)


def test_rvs_scalar_infinite_lower_bound():
    """A scalar -inf lower bound with a finite upper bound works in 1D."""
    samples = np.asarray(
        tmvn_rvs(mu=0.0, COV=1.0, lower=-np.inf, upper=0.0, size=800)
    ).reshape(-1)
    assert samples.size == 800
    assert np.all(samples < 0.0)


def test_rvs_infinite_bounds_match_unbounded_moments():
    """Infinite truncation limits give the same moments as no truncation."""
    mu = np.zeros(2)
    COV = np.eye(2)
    samples = np.asarray(
        tmvn_rvs(mu=mu, COV=COV,
                 lower=[-np.inf, -np.inf], upper=[np.inf, np.inf],
                 size=20000)
    )
    assert samples.shape == (20000, 2)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)
    assert np.allclose(np.cov(samples, rowvar=False), COV, atol=0.05)


def test_rvs_truncated_normal_mean_matches_theory():
    """Half-normal (N(0,1) truncated at 0) has mean sqrt(2/pi)."""
    samples = np.asarray(
        tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=[0.0], upper=[np.inf],
                 size=20000)
    ).reshape(-1)
    assert abs(samples.mean() - np.sqrt(2.0 / np.pi)) < 0.03
    assert abs(samples.var() - (1.0 - 2.0 / np.pi)) < 0.03


def test_rvs_reproducible_with_seed():
    """Re-seeding the global RNG reproduces identical truncated samples."""
    kwargs = dict(mu=[0.0, 0.0], COV=np.eye(2),
                  lower=[0.0, 0.0], upper=[np.inf, np.inf], size=50)
    np.random.seed(11)
    first = np.asarray(tmvn_rvs(**kwargs))
    np.random.seed(11)
    second = np.asarray(tmvn_rvs(**kwargs))
    assert np.array_equal(first, second)


def test_rvs_exact_size_when_rejection_sampling_overshoots():
    """Returned sample count equals `size` even when extra samples are drawn."""
    samples = np.asarray(
        tmvn_rvs(mu=np.zeros(2), COV=np.eye(2),
                 lower=[0.0, 0.0], upper=[np.inf, np.inf], size=7)
    )
    assert samples.shape == (7, 2)
    assert np.all(samples > 0.0)


def test_rvs_zero_density_raises_value_error():
    """A truncation window with essentially no probability mass is rejected."""
    with pytest.raises(ValueError):
        tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=[40.0], upper=[41.0], size=10)


def test_rvs_high_rejection_rate_warns():
    """A window with density below 1e-3 emits a UserWarning but still samples."""
    with pytest.warns(UserWarning):
        samples = tmvn_rvs(mu=[0.0], COV=[[1.0]],
                           lower=[3.5], upper=[np.inf], size=20)
    samples = np.asarray(samples).reshape(-1)
    assert samples.size == 20
    assert np.all(samples > 3.5)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_uncensored_recovers_parameters():
    """Untruncated univariate MLE recovers mean and variance approximately."""
    x = np.random.normal(loc=1.5, scale=0.5, size=2000)
    mu, COV = tmvn_MLE(x)
    assert abs(float(mu) - 1.5) < 0.05
    assert abs(float(COV) - 0.25) / 0.25 < 0.1


def test_mle_accepts_list_input():
    """Samples provided as a Python list are converted and fitted."""
    x = list(np.random.normal(loc=-0.5, scale=1.0, size=500))
    mu, COV = tmvn_MLE(x)
    assert np.isfinite(float(mu))
    assert np.isfinite(float(COV))
    assert float(COV) > 0.0
    assert abs(float(mu) + 0.5) < 0.15


def test_mle_univariate_zero_variance_returns_finite_mean():
    """Constant samples get a negligible variance and the constant as mean."""
    x = np.full(200, 2.0)
    mu, COV = tmvn_MLE(x)
    assert abs(float(mu) - 2.0) < 1e-4
    assert np.isfinite(float(COV))
    assert float(COV) < 1e-6


def test_mle_multivariate_recovers_means_and_variances():
    """Bivariate MLE recovers the means and variances of the generating MVN."""
    true_mu = np.array([1.0, -2.0])
    true_cov = np.array([[1.0, 0.6],
                         [0.6, 4.0]])
    # samples are arranged with one row per feature
    data = np.random.multivariate_normal(true_mu, true_cov, size=3000).T

    mu, COV = tmvn_MLE(data)
    mu = np.asarray(mu)
    COV = np.asarray(COV)

    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(mu, true_mu, atol=0.15)
    assert np.allclose(np.diag(COV), np.diag(true_cov), rtol=0.15)


def test_mle_multivariate_covariance_symmetric_and_psd():
    """Fitted multivariate covariance is symmetric, PSD and keeps the sign of correlation."""
    true_cov = np.array([[1.0, 0.7],
                         [0.7, 1.0]])
    data = np.random.multivariate_normal(np.zeros(2), true_cov, size=1000).T

    _, COV = tmvn_MLE(data)
    COV = np.asarray(COV)

    assert np.allclose(COV, COV.T)
    assert np.all(np.linalg.eigvalsh(COV) >= -1e-10)
    assert COV[0, 1] > 0.0


def test_mle_univariate_censored_recovers_parameters():
    """Right-censored data (values above 1.0) still yields the true parameters."""
    x = np.random.normal(loc=0.0, scale=1.0, size=2000)
    detected = x[x <= 1.0]
    censored_count = int(np.sum(x > 1.0))

    mu, COV = tmvn_MLE(detected,
                       censored_count=censored_count,
                       det_lower=-np.inf, det_upper=1.0)

    assert abs(float(mu)) < 0.15
    assert abs(float(COV) - 1.0) < 0.25


def test_mle_univariate_truncated_recovers_parameters():
    """Samples drawn from a truncated normal are fitted with the truncation known."""
    x = np.ravel(tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=2.0, size=2000))

    mu, COV = tmvn_MLE(x, tr_lower=-1.0, tr_upper=2.0)

    assert abs(float(mu) - 0.0) < 0.15
    assert abs(float(COV) - 1.0) < 0.25
