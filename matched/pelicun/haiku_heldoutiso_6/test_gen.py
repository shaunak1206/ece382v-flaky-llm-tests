"""Unit tests for tmvn_rvs and tmvn_MLE in pelicun/uq.py."""

import numpy as np
import pytest
from scipy.stats import truncnorm

from pelicun.uq import tmvn_rvs, tmvn_MLE


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_unbounded_scalar_moments():
    """Unbounded univariate sampling reproduces mean and variance."""
    np.random.seed(101)
    samples = tmvn_rvs(mu=2.0, COV=4.0, size=20000)
    assert samples.shape == (20000,)
    assert abs(np.mean(samples) - 2.0) < 0.05
    assert abs(np.var(samples) - 4.0) < 0.2


def test_rvs_unbounded_multivariate_shape():
    """Unbounded multivariate sampling returns (size, ndim) array."""
    np.random.seed(102)
    mu = [0.0, 1.0]
    COV = np.array([[1.0, 0.3], [0.3, 2.0]])
    samples = tmvn_rvs(mu=mu, COV=COV, size=500)
    assert samples.shape == (500, 2)


def test_rvs_unbounded_multivariate_covariance():
    """Unbounded multivariate samples match the prescribed covariance."""
    np.random.seed(103)
    mu = np.array([0.0, 1.0])
    COV = np.array([[1.0, 0.5], [0.5, 2.0]])
    samples = tmvn_rvs(mu=mu, COV=COV, size=20000)
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.05)
    assert np.allclose(np.cov(samples, rowvar=False), COV, atol=0.1)


def test_rvs_lower_bound_only_scalar():
    """A scalar lower bound truncates from below; the mean matches theory."""
    np.random.seed(104)
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=1.0, size=5000)
    assert samples.shape == (5000,)
    assert np.all(samples >= 1.0)
    # mean of standard normal truncated to [1, inf) is phi(1)/(1-Phi(1))
    expected = 0.24197 / (1.0 - 0.84134)
    assert abs(np.mean(samples) - expected) < 0.05


def test_rvs_multivariate_box_bounds_respected():
    """All multivariate samples lie within the truncation box."""
    np.random.seed(105)
    samples = tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2),
                       lower=[-1.0, -1.0], upper=[1.0, 1.0], size=1000)
    assert samples.shape == (1000, 2)
    assert np.all(samples >= -1.0)
    assert np.all(samples <= 1.0)
    assert np.allclose(np.mean(samples, axis=0), 0.0, atol=0.1)


def test_rvs_half_normal_mean():
    """Standard normal truncated at zero has mean sqrt(2/pi)."""
    np.random.seed(106)
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, size=20000)
    assert np.all(samples >= 0.0)
    assert abs(np.mean(samples) - np.sqrt(2.0 / np.pi)) < 0.03


@pytest.mark.parametrize("size", [1, 7, 250])
def test_rvs_returns_requested_size(size):
    """The number of returned samples equals the requested size."""
    np.random.seed(107)
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=-0.5, upper=0.5, size=size)
    assert samples.shape[0] == size
    assert np.all((samples >= -0.5) & (samples <= 0.5))


def test_rvs_infinite_limits_match_unbounded_statistics():
    """Infinite limits give the same statistics as unbounded sampling."""
    np.random.seed(108)
    samples = tmvn_rvs(mu=1.0, COV=2.0, lower=-np.inf, upper=np.inf,
                       size=20000)
    assert samples.shape == (20000,)
    assert abs(np.mean(samples) - 1.0) < 0.05
    assert abs(np.var(samples) - 2.0) < 0.15


def test_rvs_negligible_density_raises():
    """Truncation to a region of essentially zero density raises ValueError."""
    with pytest.raises(ValueError):
        tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2),
                 lower=[50.0, 50.0], upper=[51.0, 51.0], size=10)


def test_rvs_low_acceptance_warns():
    """A rejection rate above 0.999 triggers a UserWarning but still samples."""
    np.random.seed(109)
    with pytest.warns(UserWarning):
        samples = tmvn_rvs(mu=0.0, COV=1.0, lower=3.2, upper=3.3, size=5)
    assert samples.shape == (5,)
    assert np.all((samples >= 3.2) & (samples <= 3.3))


def test_rvs_seed_reproducible():
    """Identical global seeds produce identical samples."""
    np.random.seed(110)
    first = tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2),
                     lower=[-1.0, -1.0], upper=[2.0, 2.0], size=100)
    np.random.seed(110)
    second = tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2),
                      lower=[-1.0, -1.0], upper=[2.0, 2.0], size=100)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_untruncated_recovers_params():
    """Untruncated univariate fit recovers the mean and variance."""
    np.random.seed(201)
    samples = np.random.normal(3.0, 2.0, 2000)
    mu, COV = tmvn_MLE(samples)
    assert abs(mu - 3.0) < 0.15
    assert abs(COV - 4.0) < 0.6


def test_mle_multivariate_untruncated_recovers_params():
    """Untruncated bivariate fit recovers mean and covariance."""
    np.random.seed(202)
    true_mu = np.array([1.0, -2.0])
    true_COV = np.array([[1.0, 0.5], [0.5, 2.0]])
    raw = np.random.multivariate_normal(true_mu, true_COV, size=3000)
    mu, COV = tmvn_MLE(raw.T)  # rows are dimensions
    assert np.allclose(mu, true_mu, atol=0.1)
    assert np.allclose(COV, true_COV, atol=0.2)


def test_mle_multivariate_output_shapes():
    """Multivariate output has mean vector and symmetric square covariance."""
    np.random.seed(203)
    raw = np.random.multivariate_normal([0.0, 0.0], np.eye(2), size=500)
    mu, COV = tmvn_MLE(raw.T)
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(COV, COV.T)


def test_mle_accepts_list_input():
    """Lists are accepted and give the same result as arrays."""
    np.random.seed(204)
    samples = np.random.normal(0.0, 1.0, 500)
    mu_arr, COV_arr = tmvn_MLE(samples)
    mu_lst, COV_lst = tmvn_MLE(list(samples))
    assert np.allclose(mu_arr, mu_lst)
    assert np.allclose(COV_arr, COV_lst)


def test_mle_univariate_truncated_recovers_params():
    """Fit to a truncated normal sample recovers the untruncated parameters."""
    rs = np.random.RandomState(205)
    samples = truncnorm.rvs(a=-1.0, b=2.0, loc=0.0, scale=1.0,
                            size=3000, random_state=rs)
    mu, COV = tmvn_MLE(samples, tr_lower=-1.0, tr_upper=2.0)
    assert abs(mu - 0.0) < 0.15
    assert abs(COV - 1.0) < 0.2


def test_mle_univariate_censored_recovers_params():
    """Right-censored data fit recovers the underlying normal parameters."""
    np.random.seed(206)
    x = np.random.normal(0.0, 1.0, 4000)
    det_upper = 1.0
    kept = x[x <= det_upper]
    censored_count = int(np.sum(x > det_upper))
    mu, COV = tmvn_MLE(kept, censored_count=censored_count,
                       det_lower=-np.inf, det_upper=det_upper)
    assert abs(mu - 0.0) < 0.15
    assert abs(COV - 1.0) < 0.2


def test_mle_roundtrip_with_tmvn_rvs():
    """Samples from tmvn_rvs can be refit by tmvn_MLE to the source params."""
    np.random.seed(207)
    samples = tmvn_rvs(mu=1.0, COV=1.0, lower=-0.5, upper=2.5, size=3000)
    mu, COV = tmvn_MLE(samples, tr_lower=-0.5, tr_upper=2.5)
    assert abs(mu - 1.0) < 0.15
    assert abs(COV - 1.0) < 0.2
