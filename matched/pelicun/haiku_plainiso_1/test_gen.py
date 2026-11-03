"""Unit tests for pelicun.uq.tmvn_rvs and pelicun.uq.tmvn_MLE."""

import math
import warnings

import numpy as np
import pytest

from pelicun.uq import tmvn_MLE, tmvn_rvs


def _phi(x):
    """Standard normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x):
    """Standard normal distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _std_truncnorm_moments(a, b):
    """Mean and variance of a standard normal truncated to [a, b]."""
    z = _Phi(b) - _Phi(a)
    mean = (_phi(a) - _phi(b)) / z
    var = 1.0 + (a * _phi(a) - b * _phi(b)) / z - mean * mean
    return mean, var


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------


def test_no_bounds_univariate_returns_flat_samples():
    np.random.seed(1)
    samples = tmvn_rvs(0.0, 1.0, size=500)
    assert np.asarray(samples).reshape(-1).shape == (500,)


def test_no_bounds_multivariate_returns_one_row_per_sample():
    np.random.seed(1)
    samples = tmvn_rvs(np.zeros(3), np.eye(3), size=400)
    assert samples.shape == (400, 3)


def test_no_bounds_moments_match_target():
    np.random.seed(0)
    mu = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    samples = tmvn_rvs(mu, cov, size=20000)
    assert samples.shape == (20000, 2)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)
    assert np.allclose(np.cov(samples, rowvar=False), cov, atol=0.1)


def test_univariate_lower_bound_enforced():
    np.random.seed(2)
    samples = tmvn_rvs(0.0, 1.0, lower=0.5, size=500)
    assert samples.shape == (500,)
    assert np.all(samples > 0.5)


def test_univariate_upper_bound_enforced():
    np.random.seed(2)
    samples = tmvn_rvs(0.0, 1.0, upper=-1.0, size=500)
    assert samples.shape == (500,)
    assert np.all(samples < -1.0)


def test_two_sided_univariate_matches_truncated_moments():
    np.random.seed(2)
    samples = tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=5000)
    assert samples.shape == (5000,)
    assert np.all(samples > -1.0) and np.all(samples < 1.0)
    mean, var = _std_truncnorm_moments(-1.0, 1.0)
    assert abs(samples.mean() - mean) < 0.05
    assert abs(samples.var() - var) < 0.03


def test_shifted_scaled_truncation_matches_moments():
    np.random.seed(3)
    mu, sigma = 2.0, 3.0
    samples = tmvn_rvs(mu, sigma ** 2, lower=0.0, upper=6.0, size=20000)
    assert np.all(samples > 0.0) and np.all(samples < 6.0)
    m, v = _std_truncnorm_moments((0.0 - mu) / sigma, (6.0 - mu) / sigma)
    assert abs(samples.mean() - (mu + sigma * m)) < 0.1
    assert abs(samples.var() - sigma ** 2 * v) < 0.3


def test_multivariate_box_bounds_enforced_per_dimension():
    np.random.seed(4)
    samples = tmvn_rvs(
        np.zeros(2), np.eye(2), lower=[-1.0, -1.0], upper=[1.0, 1.0], size=500
    )
    assert samples.shape == (500, 2)
    assert np.all(np.abs(samples) < 1.0)


def test_multivariate_half_bounded_dimension_mean():
    np.random.seed(4)
    samples = tmvn_rvs(
        np.zeros(2),
        np.eye(2),
        lower=[0.0, -np.inf],
        upper=[np.inf, np.inf],
        size=20000,
    )
    assert samples.shape == (20000, 2)
    assert np.all(samples[:, 0] > 0.0)
    # mean of a standard normal truncated to [0, inf) is sqrt(2/pi)
    assert abs(samples[:, 0].mean() - math.sqrt(2.0 / math.pi)) < 0.03
    assert abs(samples[:, 1].mean()) < 0.03


def test_seeded_sampling_is_reproducible():
    np.random.seed(5)
    first = tmvn_rvs(0.0, 1.0, lower=0.5, size=200)
    np.random.seed(5)
    second = tmvn_rvs(0.0, 1.0, lower=0.5, size=200)
    np.testing.assert_array_equal(first, second)


def test_size_one_returns_single_sample_within_bounds():
    np.random.seed(6)
    samples = tmvn_rvs(0.0, 1.0, lower=0.0, size=1)
    assert np.shape(samples) == (1,)
    assert samples[0] > 0.0


def test_low_rejection_density_warns():
    np.random.seed(6)
    with pytest.warns(UserWarning, match="rejection rate"):
        samples = tmvn_rvs(0.0, 1.0, lower=3.2, size=50)
    assert samples.shape == (50,)
    assert np.all(samples > 3.2)


def test_negligible_mass_raises_value_error():
    with pytest.raises(ValueError):
        tmvn_rvs(0.0, 1.0, lower=50.0, size=10)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------


def test_mle_univariate_uncensored_recovers_moments():
    samples = np.random.RandomState(1).normal(2.0, 0.5, size=2000)
    mu, cov = tmvn_MLE(samples)
    assert abs(mu - 2.0) < 0.05
    assert abs(cov - 0.25) < 0.03


def test_mle_multivariate_recovers_mean_and_cov():
    mu_true = np.array([1.0, -1.0])
    cov_true = np.array([[1.0, 0.3], [0.3, 0.5]])
    samples = np.random.RandomState(2).multivariate_normal(
        mu_true, cov_true, size=3000
    )
    # tmvn_MLE expects one row per dimension
    mu, cov = tmvn_MLE(samples.T)
    assert np.allclose(mu, mu_true, atol=0.1)
    assert np.allclose(cov, cov_true, atol=0.1)


def test_mle_return_shapes():
    uni = np.random.RandomState(3).normal(size=500)
    mu1, cov1 = tmvn_MLE(uni)
    assert np.ndim(mu1) == 0
    assert np.ndim(cov1) == 0

    multi = np.random.RandomState(3).normal(size=(3, 500))
    mu3, cov3 = tmvn_MLE(multi)
    assert np.shape(mu3) == (3,)
    assert np.shape(cov3) == (3, 3)


def test_mle_is_deterministic_for_same_input():
    samples = np.random.RandomState(5).normal(1.0, 2.0, size=300)
    mu_a, cov_a = tmvn_MLE(samples)
    mu_b, cov_b = tmvn_MLE(samples)
    assert mu_a == mu_b
    assert cov_a == cov_b


def test_mle_truncated_univariate_recovers_parent():
    np.random.seed(7)
    samples = tmvn_rvs(0.0, 1.0, lower=0.0, size=3000)
    mu, cov = tmvn_MLE(samples, tr_lower=0.0, tr_upper=np.inf)
    assert abs(mu - 0.0) < 0.15
    assert abs(cov - 1.0) < 0.25


def test_mle_censored_univariate_recovers_parent():
    x = np.random.RandomState(11).normal(0.0, 1.0, size=3000)
    limit = 1.0
    censored_count = int(np.sum(x > limit))
    observed = x[x <= limit]
    mu, cov = tmvn_MLE(
        observed,
        censored_count=censored_count,
        det_lower=-np.inf,
        det_upper=limit,
    )
    assert abs(mu - 0.0) < 0.15
    assert abs(cov - 1.0) < 0.3


def test_mle_zero_variance_univariate_stays_finite():
    samples = np.full(50, 3.0)
    mu, cov = tmvn_MLE(samples)
    assert np.isfinite(mu)
    assert np.isfinite(cov)
    assert abs(mu - 3.0) < 1e-4
    assert 0.0 <= cov < 1e-6


def test_mle_constant_dimension_multivariate_stays_finite():
    varying = np.random.RandomState(3).normal(5.0, 1.0, size=200)
    constant = np.full(200, 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, cov = tmvn_MLE(np.vstack([varying, constant]))
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
    assert abs(mu[0] - 5.0) < 0.3
    assert abs(mu[1] - 3.0) < 1e-4


def test_mle_accepts_alpha_lim_on_truncated_data():
    np.random.seed(8)
    samples = tmvn_rvs(0.0, 1.0, lower=0.0, size=2000)
    mu, cov = tmvn_MLE(samples, tr_lower=0.0, tr_upper=np.inf, alpha_lim=1e-4)
    assert np.isfinite(mu)
    assert np.isfinite(cov)
    assert abs(mu) < 0.2
