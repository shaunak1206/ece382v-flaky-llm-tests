"""Unit tests for tmvn_rvs and tmvn_MLE in pelicun/uq.py.

Every test seeds numpy's global random state at the start of its body,
because scipy's multivariate_normal, truncnorm and numpy's own sampling
all draw from that global state when no random_state is given. Floating
point comparisons use explicit tolerances.
"""

import numpy as np
import pytest
from scipy.stats import truncnorm

from pelicun.uq import tmvn_MLE, tmvn_rvs


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_unbounded_univariate_moments():
    np.random.seed(101)
    samples = np.ravel(tmvn_rvs(mu=2.0, COV=4.0, size=20000))

    assert samples.size == 20000
    assert np.mean(samples) == pytest.approx(2.0, abs=0.1)
    assert np.var(samples) == pytest.approx(4.0, rel=0.05)


def test_rvs_unbounded_multivariate_shape_and_moments():
    np.random.seed(102)
    mu = np.array([1.0, -2.0])
    COV = np.array([[1.0, 0.6], [0.6, 4.0]])
    samples = tmvn_rvs(mu=mu, COV=COV, size=20000)

    assert samples.shape == (20000, 2)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)
    assert np.allclose(np.cov(samples, rowvar=False), COV, atol=0.15)


def test_rvs_bounded_univariate_samples_within_limits():
    np.random.seed(103)
    samples = tmvn_rvs(mu=np.array([0.0]), COV=np.array([[1.0]]),
                       lower=np.array([-0.5]), upper=np.array([1.5]),
                       size=500)

    assert samples.shape[0] == 500
    # the rejection step is strict, so this holds for every random outcome
    assert np.all(samples > -0.5)
    assert np.all(samples < 1.5)


def test_rvs_bounded_multivariate_samples_within_limits():
    np.random.seed(104)
    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 2.0])
    samples = tmvn_rvs(mu=mu, COV=COV, lower=lower, upper=upper, size=300)

    assert samples.shape == (300, 2)
    assert np.all(samples > lower)
    assert np.all(samples < upper)


def test_rvs_lower_limit_only_bounds_samples_from_below():
    np.random.seed(105)
    samples = tmvn_rvs(mu=np.array([0.0, 0.0]),
                       COV=np.array([[1.0, 0.0], [0.0, 1.0]]),
                       lower=np.array([0.0, 0.0]), upper=None, size=400)

    assert samples.shape == (400, 2)
    assert np.all(samples > 0.0)


def test_rvs_upper_limit_only_bounds_samples_from_above():
    np.random.seed(106)
    samples = tmvn_rvs(mu=np.array([0.0]), COV=np.array([[1.0]]),
                       lower=None, upper=np.array([0.0]), size=300)

    assert samples.shape[0] == 300
    assert np.all(samples < 0.0)


def test_rvs_returns_exactly_requested_size():
    np.random.seed(107)
    for size in (1, 7, 37):
        samples = tmvn_rvs(mu=np.array([0.0]), COV=np.array([[1.0]]),
                           lower=np.array([0.0]), size=size)
        assert samples.shape[0] == size


def test_rvs_half_normal_mean_matches_theory():
    np.random.seed(108)
    samples = tmvn_rvs(mu=np.array([0.0]), COV=np.array([[1.0]]),
                       lower=np.array([0.0]), size=20000)

    # mean of a standard normal truncated to x > 0 is sqrt(2/pi)
    assert np.mean(samples) == pytest.approx(np.sqrt(2.0 / np.pi), abs=0.03)


def test_rvs_same_seed_reproduces_samples():
    np.random.seed(109)
    args = dict(mu=np.array([0.0, 1.0]),
                COV=np.array([[1.0, 0.3], [0.3, 2.0]]),
                lower=np.array([-1.0, -1.0]),
                upper=np.array([2.0, 2.0]),
                size=200)

    np.random.seed(999)
    first = tmvn_rvs(**args)
    np.random.seed(999)
    second = tmvn_rvs(**args)

    assert np.array_equal(first, second)


def test_rvs_list_limits_give_same_samples_as_arrays():
    np.random.seed(110)
    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.2], [0.2, 1.0]])

    np.random.seed(7)
    from_arrays = tmvn_rvs(mu=mu, COV=COV,
                           lower=np.array([-1.0, -1.0]),
                           upper=np.array([1.0, 1.0]), size=100)
    np.random.seed(7)
    from_lists = tmvn_rvs(mu=mu, COV=COV,
                          lower=[-1.0, -1.0], upper=[1.0, 1.0], size=100)

    assert np.array_equal(from_arrays, from_lists)


def test_rvs_raises_when_truncation_density_is_too_small():
    np.random.seed(111)
    # the probability of the box [8, 9] x [8, 9] is far below the
    # accuracy of the orthotope estimate, so sampling must be refused
    with pytest.raises(ValueError, match="too small"):
        tmvn_rvs(mu=np.array([0.0, 0.0]), COV=np.eye(2),
                 lower=np.array([8.0, 8.0]), upper=np.array([9.0, 9.0]),
                 size=10)


def test_rvs_warns_on_very_high_rejection_rate():
    np.random.seed(112)
    # P(Z > 3.1) is about 9.7e-4, which is below the 1e-3 warning threshold
    with pytest.warns(UserWarning, match="0.999"):
        samples = tmvn_rvs(mu=np.array([0.0]), COV=np.array([[1.0]]),
                           lower=np.array([3.1]), upper=np.array([np.inf]),
                           size=5)

    assert samples.shape[0] == 5
    assert np.all(samples > 3.1)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_recovers_normal_parameters():
    np.random.seed(201)
    data = np.random.normal(loc=5.0, scale=2.0, size=2000)
    mu, COV = tmvn_MLE(data)

    mu_f = float(np.ravel(mu)[0])
    var_f = float(np.ravel(COV)[0])
    assert abs(mu_f - 5.0) < 0.2
    assert abs(var_f - 4.0) / 4.0 < 0.15


def test_mle_univariate_zero_spread_gives_finite_result():
    np.random.seed(202)
    # constant data triggers the negligible-standard-deviation branch
    data = np.full(20, 3.0)
    mu, COV = tmvn_MLE(data)

    mu_f = float(np.ravel(mu)[0])
    var_f = float(np.ravel(COV)[0])
    assert np.isfinite(mu_f) and np.isfinite(var_f)
    assert abs(mu_f - 3.0) < 1e-3
    assert 0.0 <= var_f < 1e-6


def test_mle_multivariate_recovers_mean_and_covariance():
    np.random.seed(203)
    mean_true = np.array([1.0, -0.5])
    cov_true = np.array([[1.0, 0.6], [0.6, 4.0]])
    # rows are features, columns are samples
    data = np.random.multivariate_normal(mean_true, cov_true, size=3000).T
    mu, COV = tmvn_MLE(data)

    assert np.shape(mu) == (2,)
    assert np.shape(COV) == (2, 2)
    assert np.allclose(mu, mean_true, atol=0.1)
    assert np.allclose(COV, cov_true, rtol=0.1, atol=0.1)


def test_mle_multivariate_covariance_is_symmetric_and_psd():
    np.random.seed(204)
    cov_true = np.array([[2.0, -0.4], [-0.4, 1.0]])
    data = np.random.multivariate_normal(np.zeros(2), cov_true, size=500).T
    _, COV = tmvn_MLE(data)

    assert np.allclose(COV, COV.T)
    assert np.all(np.linalg.eigvalsh(COV) >= -1e-12)


def test_mle_does_not_modify_input_samples():
    np.random.seed(205)
    data = np.random.normal(size=(2, 200))
    before = data.copy()
    tmvn_MLE(data)

    assert np.array_equal(data, before)


def test_mle_repeated_calls_on_same_data_are_identical():
    np.random.seed(206)
    data = np.random.normal(loc=1.0, scale=0.5, size=(2, 300))
    first = tmvn_MLE(data)
    second = tmvn_MLE(data)

    assert np.array_equal(np.ravel(first[0]), np.ravel(second[0]))
    assert np.array_equal(np.asarray(first[1]), np.asarray(second[1]))


def test_mle_truncated_univariate_recovers_parameters():
    np.random.seed(207)
    # standard normal truncated to x > -1
    samples = truncnorm.rvs(-1.0, np.inf, loc=0.0, scale=1.0, size=3000)
    mu, COV = tmvn_MLE(samples, tr_lower=-1.0, tr_upper=np.inf)

    mu_f = float(np.ravel(mu)[0])
    var_f = float(np.ravel(COV)[0])
    assert abs(mu_f - 0.0) < 0.15
    assert abs(var_f - 1.0) < 0.2


def test_mle_censored_univariate_recovers_parameters():
    np.random.seed(208)
    x = np.random.normal(loc=0.0, scale=1.0, size=3000)
    det_lower = -1.0
    detected = x[x >= det_lower]
    censored_count = int(np.sum(x < det_lower))

    mu, COV = tmvn_MLE(detected, censored_count=censored_count,
                       det_lower=det_lower, det_upper=np.inf)

    mu_f = float(np.ravel(mu)[0])
    var_f = float(np.ravel(COV)[0])
    assert abs(mu_f - 0.0) < 0.15
    assert abs(var_f - 1.0) < 0.2
