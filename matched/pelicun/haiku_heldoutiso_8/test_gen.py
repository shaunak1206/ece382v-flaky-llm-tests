"""Pytest unit tests for tmvn_rvs and tmvn_MLE in pelicun/uq.py."""

import warnings

import numpy as np
import pytest

from pelicun.uq import tmvn_MLE, tmvn_rvs

HALF_NORMAL_MEAN = np.sqrt(2.0 / np.pi)


def _seed(value):
    """Seed the global NumPy RNG, which scipy's rvs also uses by default."""
    np.random.seed(value)


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_unbounded_multivariate_shape():
    _seed(0)
    mu = np.array([1.0, -2.0])
    COV = np.array([[1.0, 0.5], [0.5, 2.0]])
    samples = tmvn_rvs(mu, COV, size=50)
    assert samples.shape == (50, 2)


def test_rvs_unbounded_multivariate_moments():
    _seed(1)
    mu = np.array([1.0, -2.0])
    COV = np.array([[1.0, 0.5], [0.5, 2.0]])
    samples = tmvn_rvs(mu, COV, size=20000)
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.05)
    assert np.allclose(np.cov(samples, rowvar=False), COV, atol=0.1)


def test_rvs_unbounded_univariate_scalar_moments():
    _seed(2)
    samples = tmvn_rvs(mu=2.0, COV=4.0, size=20000)
    s = np.asarray(samples).reshape(-1)
    assert s.size == 20000
    assert np.mean(s) == pytest.approx(2.0, abs=0.05)
    assert np.var(s) == pytest.approx(4.0, abs=0.15)


def test_rvs_lower_bound_enforced_in_one_dimension():
    _seed(3)
    mu = np.zeros(2)
    COV = np.array([[1.0, 0.3], [0.3, 1.0]])
    samples = tmvn_rvs(mu, COV, lower=[0.0, -np.inf], size=1000)
    assert samples.shape == (1000, 2)
    assert np.all(samples[:, 0] > 0.0)


def test_rvs_two_sided_bounds_enforced():
    _seed(4)
    mu = np.zeros(2)
    COV = np.eye(2)
    lower = [-1.0, -1.0]
    upper = [1.0, 2.0]
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=2000)
    assert samples.shape == (2000, 2)
    assert np.all(samples[:, 0] > -1.0) and np.all(samples[:, 0] < 1.0)
    assert np.all(samples[:, 1] > -1.0) and np.all(samples[:, 1] < 2.0)


def test_rvs_truncated_univariate_half_normal_mean():
    _seed(5)
    mu = np.array([0.0])
    COV = np.array([[1.0]])
    samples = tmvn_rvs(mu, COV, lower=[0.0], size=20000)
    s = np.asarray(samples).reshape(-1)
    assert s.size == 20000
    assert np.all(s > 0.0)
    assert np.mean(s) == pytest.approx(HALF_NORMAL_MEAN, abs=0.03)


def test_rvs_upper_only_truncation_mean():
    _seed(6)
    mu = np.array([0.0])
    COV = np.array([[1.0]])
    samples = tmvn_rvs(mu, COV, upper=[0.0], size=20000)
    s = np.asarray(samples).reshape(-1)
    assert np.all(s < 0.0)
    assert np.mean(s) == pytest.approx(-HALF_NORMAL_MEAN, abs=0.03)


def test_rvs_truncated_multivariate_shape_and_bounds():
    _seed(7)
    mu = np.zeros(2)
    COV = np.array([[1.0, 0.5], [0.5, 1.0]])
    samples = tmvn_rvs(mu, COV, lower=[-1.0, -1.0], upper=[1.0, 1.0],
                       size=3000)
    assert samples.shape == (3000, 2)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_rvs_symmetric_truncation_keeps_zero_mean():
    _seed(8)
    mu = np.zeros(2)
    COV = np.array([[1.0, 0.5], [0.5, 1.0]])
    samples = tmvn_rvs(mu, COV, lower=[-1.0, -1.0], upper=[1.0, 1.0],
                       size=20000)
    assert np.allclose(np.mean(samples, axis=0), 0.0, atol=0.03)


def test_rvs_truncated_size_one_returns_single_row():
    _seed(9)
    mu = np.zeros(2)
    COV = np.eye(2)
    samples = tmvn_rvs(mu, COV, lower=[0.0, 0.0], size=1)
    assert samples.shape == (1, 2)
    assert np.all(samples > 0.0)


def test_rvs_same_seed_gives_same_samples():
    mu = np.zeros(2)
    COV = np.array([[1.0, 0.2], [0.2, 1.0]])
    _seed(10)
    first = tmvn_rvs(mu, COV, lower=[-0.5, -0.5], upper=[1.5, 1.5], size=200)
    _seed(10)
    second = tmvn_rvs(mu, COV, lower=[-0.5, -0.5], upper=[1.5, 1.5], size=200)
    assert np.array_equal(first, second)


def test_rvs_low_acceptance_rate_emits_user_warning():
    _seed(11)
    mu = np.array([0.0])
    COV = np.array([[1.0]])
    # Probability of the interval is roughly 2e-4, well below the 1e-3
    # threshold that triggers the warning.
    with pytest.warns(UserWarning):
        samples = tmvn_rvs(mu, COV, lower=[3.5], upper=[4.0], size=5)
    s = np.asarray(samples).reshape(-1)
    assert s.size == 5
    assert np.all(s > 3.5) and np.all(s < 4.0)


def test_rvs_moderate_acceptance_emits_no_user_warning():
    _seed(12)
    mu = np.array([0.0])
    COV = np.array([[1.0]])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        tmvn_rvs(mu, COV, lower=[0.0], size=100)
    user_warnings = [w for w in record
                     if issubclass(w.category, UserWarning)]
    assert user_warnings == []


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_unbounded_recovers_parameters():
    _seed(13)
    x = np.random.normal(5.0, 2.0, 4000)
    mu, COV = tmvn_MLE(x)
    assert mu == pytest.approx(5.0, abs=0.15)
    assert COV == pytest.approx(4.0, abs=0.4)


def test_mle_univariate_returns_scalars():
    _seed(14)
    x = np.random.normal(0.0, 1.0, 500)
    mu, COV = tmvn_MLE(x)
    assert np.ndim(mu) == 0
    assert np.ndim(COV) == 0
    assert COV > 0.0


def test_mle_multivariate_unbounded_recovers_parameters():
    _seed(15)
    true_mu = np.array([1.0, -1.0])
    true_cov = np.array([[1.0, 0.6], [0.6, 2.0]])
    samples = np.random.multivariate_normal(true_mu, true_cov,
                                            size=3000).T
    mu, COV = tmvn_MLE(samples)
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(mu, true_mu, atol=0.1)
    assert np.allclose(COV, true_cov, atol=0.2)


def test_mle_multivariate_output_shapes_and_symmetry():
    _seed(16)
    true_cov = np.array([[1.0, 0.3], [0.3, 1.5]])
    samples = np.random.multivariate_normal(np.zeros(2), true_cov,
                                            size=200).T
    mu, COV = tmvn_MLE(samples)
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(COV, COV.T)
    assert np.all(np.diag(COV) > 0.0)


def test_mle_univariate_truncated_recovers_parameters():
    _seed(17)
    x = tmvn_rvs(mu=np.array([0.0]), COV=np.array([[1.0]]),
                 lower=[0.0], size=3000)
    x = np.asarray(x).reshape(-1)
    mu, COV = tmvn_MLE(x,
                       tr_lower=np.array([0.0]),
                       tr_upper=np.array([np.inf]))
    assert mu == pytest.approx(0.0, abs=0.15)
    assert COV == pytest.approx(1.0, abs=0.3)


def test_mle_univariate_censored_recovers_parameters():
    _seed(18)
    x = np.random.normal(0.0, 1.0, 4000)
    detection_upper = 1.0
    censored = x > detection_upper
    observed = x[~censored]
    censored_count = int(np.sum(censored))
    assert censored_count > 0

    mu, COV = tmvn_MLE(observed,
                       censored_count=censored_count,
                       det_lower=np.array([-np.inf]),
                       det_upper=np.array([detection_upper]))
    assert mu == pytest.approx(0.0, abs=0.1)
    assert COV == pytest.approx(1.0, abs=0.2)


def test_mle_constant_samples_do_not_crash():
    x = np.full(50, 3.0)
    mu, COV = tmvn_MLE(x)
    assert np.isfinite(mu)
    assert np.isfinite(COV)
    assert mu == pytest.approx(3.0, abs=1e-3)
    assert COV < 1e-6


def test_mle_multivariate_truncated_recovers_mean():
    _seed(19)
    true_mu = np.zeros(2)
    true_cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.5, 1.5])
    samples = tmvn_rvs(true_mu, true_cov, lower=lower, upper=upper,
                       size=1500).T
    mu, COV = tmvn_MLE(samples, tr_lower=lower, tr_upper=upper)
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(mu, true_mu, atol=0.2)
    assert np.allclose(np.diag(COV), np.diag(true_cov), atol=0.3)
