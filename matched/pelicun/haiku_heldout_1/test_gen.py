"""Unit tests for tmvn_rvs and tmvn_MLE in pelicun/uq.py."""

import warnings

import numpy as np
import pytest
from scipy.stats import truncnorm

from pelicun import uq


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_unbounded_multivariate_moments():
    """Unbounded sampling reproduces the requested mean and covariance."""
    np.random.seed(101)
    mu = np.array([1.0, -2.0])
    COV = np.array([[1.0, 0.3], [0.3, 4.0]])

    samples = uq.tmvn_rvs(mu, COV, size=20000)

    assert samples.shape == (20000, 2)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)
    assert np.allclose(np.cov(samples, rowvar=False), COV, atol=0.15)


def test_rvs_unbounded_scalar_input():
    """A scalar mean and variance are accepted for the univariate case."""
    np.random.seed(102)

    samples = uq.tmvn_rvs(2.0, 0.25, size=5000)
    flat = np.ravel(samples)

    assert flat.size == 5000
    assert abs(flat.mean() - 2.0) < 0.05
    assert abs(flat.std() - 0.5) < 0.05


def test_rvs_unbounded_recovers_correlation():
    """Sample correlation of unbounded draws matches the prescribed one."""
    np.random.seed(103)
    COV = np.array([[1.0, 0.8], [0.8, 1.0]])

    samples = uq.tmvn_rvs(np.zeros(2), COV, size=20000)
    corr = np.corrcoef(samples, rowvar=False)[0, 1]

    assert abs(corr - 0.8) < 0.02


def test_rvs_bounded_returns_exact_size():
    """The number of returned samples equals the requested size."""
    np.random.seed(104)

    samples = uq.tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                          lower=[-1.0], upper=[1.0], size=137)

    assert np.shape(samples) == (137,)


def test_rvs_bounded_multivariate_shape():
    """Bounded multivariate sampling returns a (size, ndim) array."""
    np.random.seed(105)

    samples = uq.tmvn_rvs(np.zeros(2), np.eye(2),
                          lower=[-1.0, -1.0], upper=[1.0, 1.0], size=500)

    assert samples.shape == (500, 2)


def test_rvs_lower_bound_only():
    """Only a lower bound: samples are above it and follow a half-normal mean."""
    np.random.seed(106)

    samples = uq.tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                          lower=[0.0], size=2000)

    assert np.all(samples > 0.0)
    assert abs(np.mean(samples) - np.sqrt(2.0 / np.pi)) < 0.05


def test_rvs_upper_bound_only():
    """Only an upper bound: samples are below it and follow a half-normal mean."""
    np.random.seed(107)

    samples = uq.tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                          upper=[0.0], size=2000)

    assert np.all(samples < 0.0)
    assert abs(np.mean(samples) + np.sqrt(2.0 / np.pi)) < 0.05


def test_rvs_two_sided_1d_matches_truncnorm_moments():
    """Two-sided truncated 1D samples match the analytic truncated normal moments."""
    np.random.seed(108)
    lo, hi = -1.0, 2.0

    samples = uq.tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                          lower=[lo], upper=[hi], size=20000)

    assert np.all(samples > lo) and np.all(samples < hi)
    expected_mean = truncnorm.mean(a=lo, b=hi)
    expected_std = truncnorm.std(a=lo, b=hi)
    assert abs(np.mean(samples) - expected_mean) < 0.03
    assert abs(np.std(samples) - expected_std) < 0.03


def test_rvs_multivariate_bounds_respected():
    """Every sampled row lies inside the multivariate truncation box."""
    np.random.seed(109)

    samples = uq.tmvn_rvs(np.array([0.0, 1.0]),
                          np.array([[1.0, 0.5], [0.5, 1.0]]),
                          lower=[-0.5, 0.0], upper=[1.0, 2.0], size=1000)

    assert samples.shape == (1000, 2)
    assert np.all(samples[:, 0] > -0.5) and np.all(samples[:, 0] < 1.0)
    assert np.all(samples[:, 1] > 0.0) and np.all(samples[:, 1] < 2.0)


def test_rvs_infinite_bounds_act_as_unbounded():
    """Infinite bounds leave a dimension untruncated while the other is truncated."""
    np.random.seed(110)

    samples = uq.tmvn_rvs(np.zeros(2), np.eye(2),
                          lower=[-np.inf, 0.0], upper=[np.inf, np.inf],
                          size=20000)

    assert np.all(samples[:, 1] > 0.0)
    assert abs(samples[:, 1].mean() - np.sqrt(2.0 / np.pi)) < 0.03
    assert abs(samples[:, 0].mean()) < 0.03
    assert abs(samples[:, 0].std() - 1.0) < 0.03


def test_rvs_reproducible_with_global_seed():
    """Identical global seeds yield identical truncated samples."""
    args = dict(mu=np.zeros(2), COV=np.eye(2),
                lower=[-1.0, -1.0], upper=[1.0, 1.0], size=300)

    np.random.seed(111)
    first = uq.tmvn_rvs(**args)
    np.random.seed(111)
    second = uq.tmvn_rvs(**args)

    np.testing.assert_array_equal(first, second)


def test_rvs_negligible_truncation_mass_raises():
    """A truncation box with essentially zero probability mass raises ValueError."""
    with pytest.raises(ValueError, match="too small"):
        uq.tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                    lower=[40.0], upper=[50.0], size=10)


def test_rvs_high_rejection_rate_warns():
    """A very small acceptance probability triggers the rejection-rate warning."""
    np.random.seed(112)

    with pytest.warns(UserWarning, match="rejection rate"):
        samples = uq.tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                              lower=[3.2], upper=[3.3], size=5)

    assert np.all(samples > 3.2) and np.all(samples < 3.3)


def test_rvs_moderate_rejection_does_not_warn():
    """A high acceptance probability does not trigger the rejection-rate warning."""
    np.random.seed(113)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        uq.tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                    lower=[-1.0], upper=[1.0], size=50)

    assert not [w for w in record if "rejection rate" in str(w.message)]


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_untruncated_recovers_moments():
    """Untruncated 1D MLE recovers the mean and variance; COV is a scalar."""
    rs = np.random.RandomState(16)
    data = rs.normal(3.0, 2.0, 2000)

    mu, COV = uq.tmvn_MLE(data)

    assert np.ndim(COV) == 0
    assert abs(float(mu) - 3.0) < 0.2
    assert abs(float(COV) - 4.0) < 0.8


def test_mle_multivariate_untruncated_recovers_moments():
    """Untruncated multivariate MLE recovers mean and covariance."""
    rs = np.random.RandomState(17)
    true_mu = np.array([1.0, -2.0])
    true_cov = np.array([[1.0, 0.6], [0.6, 2.0]])
    data = rs.multivariate_normal(true_mu, true_cov, size=1500)

    # tmvn_MLE expects (ndims, nsamples)
    mu, COV = uq.tmvn_MLE(data.T)

    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(mu, true_mu, atol=0.2)
    assert np.allclose(COV, true_cov, atol=0.3)


def test_mle_multivariate_correlation_matches_sample():
    """The fitted correlation equals the sample correlation of the data."""
    rs = np.random.RandomState(18)
    data = rs.multivariate_normal([0.0, 0.0],
                                  [[1.0, 0.6], [0.6, 2.0]], size=1000)

    _, COV = uq.tmvn_MLE(data.T)

    corr_fit = COV[0, 1] / np.sqrt(COV[0, 0] * COV[1, 1])
    corr_sample = np.corrcoef(data.T)[0, 1]
    assert np.isclose(corr_fit, corr_sample, atol=1e-6)


def test_mle_accepts_nested_list_input():
    """Nested Python lists give the same result as the equivalent array."""
    rs = np.random.RandomState(19)
    data = rs.normal(0.5, 1.5, 300)

    mu_arr, COV_arr = uq.tmvn_MLE(data)
    mu_lst, COV_lst = uq.tmvn_MLE(data.tolist())

    assert np.isclose(mu_arr, mu_lst)
    assert np.isclose(COV_arr, COV_lst)


def test_mle_truncated_recovers_underlying_parameters():
    """Truncated MLE recovers the parameters of the untruncated normal."""
    np.random.seed(20)
    samples = uq.tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                          lower=[-1.0], upper=[2.0], size=2000)

    mu, COV = uq.tmvn_MLE(samples, tr_lower=-1.0, tr_upper=2.0)

    assert abs(float(mu)) < 0.15
    assert abs(float(COV) - 1.0) < 0.25


def test_mle_censored_recovers_parameters():
    """Censored MLE recovers the parameters when data above a limit are censored."""
    rs = np.random.RandomState(21)
    raw = rs.normal(0.0, 1.0, 3000)
    censored = raw > 1.0
    observed = raw[~censored]

    mu, COV = uq.tmvn_MLE(observed,
                          censored_count=int(censored.sum()),
                          det_lower=-np.inf, det_upper=1.0)

    assert abs(float(mu)) < 0.15
    assert abs(float(COV) - 1.0) < 0.2


def test_mle_constant_samples_do_not_crash():
    """Zero-variance data gives finite estimates with the mean preserved."""
    mu, COV = uq.tmvn_MLE(np.full(20, 5.0))

    assert np.isfinite(mu)
    assert np.isfinite(COV)
    assert abs(float(mu) - 5.0) < 1e-3
    assert 0.0 <= float(COV) < 1e-9


def test_mle_returned_cov_symmetric_psd():
    """The fitted 3D covariance matrix is symmetric and positive semidefinite."""
    rs = np.random.RandomState(23)
    true_cov = np.array([[1.0, 0.5, 0.2],
                         [0.5, 1.0, 0.3],
                         [0.2, 0.3, 1.0]])
    data = rs.multivariate_normal(np.zeros(3), true_cov, size=800)

    mu, COV = uq.tmvn_MLE(data.T)

    assert mu.shape == (3,)
    assert COV.shape == (3, 3)
    assert np.allclose(COV, COV.T)
    assert np.all(np.linalg.eigvalsh(COV) > -1e-10)
