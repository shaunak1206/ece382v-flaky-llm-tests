"""Unit tests for tmvn_rvs and tmvn_MLE in pelicun/uq.py."""

import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_tmvn_rvs_unbounded_univariate_size():
    np.random.seed(0)
    samples = tmvn_rvs(mu=0.0, COV=1.0, size=100)
    assert np.size(samples) == 100


def test_tmvn_rvs_unbounded_multivariate_shape():
    np.random.seed(0)
    samples = tmvn_rvs(mu=np.array([0.0, 1.0]), COV=np.eye(2), size=50)
    assert np.shape(samples) == (50, 2)


def test_tmvn_rvs_unbounded_sample_moments():
    np.random.seed(1)
    mu = np.array([1.0, -2.0])
    COV = np.array([[2.0, 0.5], [0.5, 1.0]])
    samples = tmvn_rvs(mu=mu, COV=COV, size=20000)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)
    assert np.allclose(np.cov(samples, rowvar=False), COV, atol=0.1)


def test_tmvn_rvs_univariate_bounded_within_limits():
    np.random.seed(2)
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=500)
    assert np.size(samples) == 500
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_tmvn_rvs_size_one_returns_single_value():
    np.random.seed(3)
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=1)
    assert np.size(samples) == 1
    assert -1.0 < float(np.ravel(samples)[0]) < 1.0


def test_tmvn_rvs_multivariate_bounded_within_limits():
    np.random.seed(4)
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 2.0])
    samples = tmvn_rvs(mu=np.zeros(2), COV=np.eye(2),
                       lower=lower, upper=upper, size=400)
    assert np.shape(samples) == (400, 2)
    assert np.all(samples > lower)
    assert np.all(samples < upper)


def test_tmvn_rvs_one_sided_lower_bound():
    np.random.seed(5)
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, upper=None, size=300)
    assert np.size(samples) == 300
    assert np.all(samples > 0.0)


def test_tmvn_rvs_infinite_bounds_accepted():
    np.random.seed(6)
    lower = np.array([-np.inf, -np.inf])
    upper = np.array([np.inf, np.inf])
    samples = tmvn_rvs(mu=np.zeros(2), COV=np.eye(2),
                       lower=lower, upper=upper, size=200)
    assert np.shape(samples) == (200, 2)
    assert np.all(np.isfinite(samples))


def test_tmvn_rvs_truncated_mean_matches_theory():
    # Mean of a standard normal truncated to [0, inf) is sqrt(2/pi)
    np.random.seed(7)
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, size=20000)
    assert np.isclose(np.mean(samples), np.sqrt(2.0 / np.pi), atol=0.02)


def test_tmvn_rvs_reversed_bounds_raises():
    with pytest.raises(ValueError):
        tmvn_rvs(mu=0.0, COV=1.0, lower=2.0, upper=1.0, size=10)


def test_tmvn_rvs_high_rejection_rate_warns():
    # P(Z > 3.5) is about 2.3e-4, below the 1e-3 warning threshold
    np.random.seed(8)
    with pytest.warns(UserWarning):
        samples = tmvn_rvs(mu=0.0, COV=1.0, lower=3.5, size=5)
    assert np.size(samples) == 5
    assert np.all(samples > 3.5)


def test_tmvn_rvs_same_seed_reproducible():
    np.random.seed(9)
    first = tmvn_rvs(mu=0.0, COV=1.0, lower=-0.5, upper=2.0, size=50)
    np.random.seed(9)
    second = tmvn_rvs(mu=0.0, COV=1.0, lower=-0.5, upper=2.0, size=50)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_tmvn_MLE_univariate_unbounded_recovers_parameters():
    np.random.seed(10)
    samples = np.random.normal(loc=5.0, scale=2.0, size=2000)
    mu, COV = tmvn_MLE(samples)
    assert np.isclose(mu, 5.0, atol=0.2)
    assert np.isclose(COV, 4.0, rtol=0.25)


def test_tmvn_MLE_univariate_returns_scalars():
    np.random.seed(11)
    samples = np.random.normal(loc=0.0, scale=1.0, size=300)
    mu, COV = tmvn_MLE(samples)
    assert np.ndim(mu) == 0
    assert np.ndim(COV) == 0
    assert np.isfinite(mu)
    assert COV > 0.0


def test_tmvn_MLE_multivariate_shapes_and_symmetry():
    np.random.seed(12)
    # rows are variables, columns are samples
    samples = np.random.normal(size=(2, 500))
    mu, COV = tmvn_MLE(samples)
    assert np.shape(mu) == (2,)
    assert np.shape(COV) == (2, 2)
    assert np.allclose(COV, COV.T)
    assert np.all(np.diag(COV) > 0.0)


def test_tmvn_MLE_multivariate_recovers_parameters():
    np.random.seed(13)
    true_mu = np.array([1.0, -2.0])
    true_COV = np.array([[4.0, 1.2], [1.2, 1.0]])
    raw = np.random.multivariate_normal(true_mu, true_COV, size=3000)
    mu, COV = tmvn_MLE(raw.T)
    assert np.allclose(mu, true_mu, atol=0.15)
    assert np.isclose(COV[0, 0], 4.0, rtol=0.2)
    assert np.isclose(COV[1, 1], 1.0, rtol=0.2)
    assert np.isclose(COV[0, 1], 1.2, atol=0.3)


def test_tmvn_MLE_truncated_recovers_parameters():
    np.random.seed(14)
    raw = np.random.normal(loc=0.0, scale=1.0, size=20000)
    truncated = raw[(raw > -1.0) & (raw < 2.0)][:2000]
    mu, COV = tmvn_MLE(truncated, tr_lower=-1.0, tr_upper=2.0)
    assert np.isclose(mu, 0.0, atol=0.15)
    assert np.isclose(COV, 1.0, rtol=0.25)


def test_tmvn_MLE_alpha_lim_runs_and_is_finite():
    np.random.seed(15)
    raw = np.random.normal(loc=0.0, scale=1.0, size=20000)
    truncated = raw[(raw > -1.0) & (raw < 2.0)][:1000]
    mu, COV = tmvn_MLE(truncated, tr_lower=-1.0, tr_upper=2.0,
                       alpha_lim=1e-4)
    assert np.isfinite(mu)
    assert np.isfinite(COV)
    assert COV > 0.0


def test_tmvn_MLE_censored_recovers_parameters():
    np.random.seed(16)
    raw = np.random.normal(loc=0.0, scale=1.0, size=2000)
    censored = raw > 1.0
    observed = raw[~censored]
    mu, COV = tmvn_MLE(observed, censored_count=int(np.sum(censored)),
                       det_lower=-np.inf, det_upper=1.0)
    assert np.isclose(mu, 0.0, atol=0.15)
    assert np.isclose(COV, 1.0, rtol=0.25)


def test_tmvn_MLE_zero_variance_univariate_is_finite():
    samples = np.full(50, 3.0)
    mu, COV = tmvn_MLE(samples)
    assert np.isfinite(mu)
    assert np.isfinite(COV)
    assert COV >= 0.0
    assert np.isclose(mu, 3.0, atol=1e-3)


def test_tmvn_MLE_list_input_matches_array_input():
    np.random.seed(17)
    data = np.random.normal(loc=2.0, scale=0.5, size=300)
    mu_arr, COV_arr = tmvn_MLE(data)
    mu_lst, COV_lst = tmvn_MLE(list(data))
    assert np.isclose(mu_arr, mu_lst)
    assert np.isclose(COV_arr, COV_lst)


def test_tmvn_MLE_does_not_modify_input():
    np.random.seed(18)
    samples = np.random.normal(size=(2, 200))
    original = samples.copy()
    tmvn_MLE(samples)
    assert np.array_equal(samples, original)
