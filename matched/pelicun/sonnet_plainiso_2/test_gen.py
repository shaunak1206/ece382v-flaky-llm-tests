import warnings

import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(12345)
    yield


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_untruncated_univariate_shape_and_moments():
    samples = np.atleast_1d(tmvn_rvs(2.0, 4.0, size=4000))
    assert samples.shape == (4000,)
    assert abs(np.mean(samples) - 2.0) < 0.2
    assert abs(np.std(samples) - 2.0) < 0.2


def test_rvs_untruncated_multivariate_shape():
    mu = np.array([0.0, 1.0, -1.0])
    cov = np.diag([1.0, 2.0, 0.5])
    samples = tmvn_rvs(mu, cov, size=50)
    assert samples.shape == (50, 3)
    assert np.all(np.isfinite(samples))


def test_rvs_untruncated_multivariate_correlation():
    mu = np.zeros(2)
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples = tmvn_rvs(mu, cov, size=4000)
    corr = np.corrcoef(samples.T)[0, 1]
    assert abs(corr - 0.8) < 0.08


def test_rvs_univariate_lower_bound_respected():
    samples = tmvn_rvs(0.0, 1.0, lower=0.0, size=500)
    assert samples.shape == (500,)
    assert np.all(samples > 0.0)
    # half-normal mean is sqrt(2/pi)
    assert abs(np.mean(samples) - np.sqrt(2.0 / np.pi)) < 0.15


def test_rvs_univariate_upper_bound_respected():
    samples = tmvn_rvs(0.0, 1.0, upper=0.5, size=500)
    assert samples.shape == (500,)
    assert np.all(samples < 0.5)
    assert np.mean(samples) < 0.0


def test_rvs_univariate_both_bounds_respected():
    samples = tmvn_rvs(1.0, 1.0, lower=0.0, upper=2.0, size=1000)
    assert samples.shape == (1000,)
    assert np.all((samples > 0.0) & (samples < 2.0))
    # distribution is symmetric about mu = 1
    assert abs(np.mean(samples) - 1.0) < 0.1


def test_rvs_exact_size_returned_when_truncated():
    for size in (1, 2, 7, 33):
        samples = np.atleast_1d(tmvn_rvs(0.0, 1.0, lower=-0.5, upper=0.5,
                                         size=size))
        assert samples.shape[0] == size


def test_rvs_multivariate_bounds_respected():
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    lower = np.array([-1.0, -0.5])
    upper = np.array([1.5, 2.0])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=300)
    assert samples.shape == (300, 2)
    assert np.all(samples[:, 0] > -1.0) and np.all(samples[:, 0] < 1.5)
    assert np.all(samples[:, 1] > -0.5) and np.all(samples[:, 1] < 2.0)


def test_rvs_multivariate_infinite_bounds_in_some_dims():
    mu = np.array([0.0, 0.0])
    cov = np.eye(2)
    lower = np.array([0.0, -np.inf])
    upper = np.array([np.inf, np.inf])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=400)
    assert samples.shape == (400, 2)
    assert np.all(samples[:, 0] > 0.0)
    # the unbounded dimension should still take negative values
    assert np.any(samples[:, 1] < 0.0)


def test_rvs_multivariate_only_upper_bound():
    mu = np.array([0.0, 0.0])
    cov = np.eye(2)
    upper = np.array([0.5, 0.5])
    samples = tmvn_rvs(mu, cov, upper=upper, size=200)
    assert samples.shape == (200, 2)
    assert np.all(samples < 0.5)


def test_rvs_negligible_density_raises_value_error():
    with pytest.raises(ValueError):
        tmvn_rvs(0.0, 1.0, lower=100.0, upper=101.0, size=5)


def test_rvs_truncated_reproducible_with_seed():
    np.random.seed(7)
    a = tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=20)
    np.random.seed(7)
    b = tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=20)
    np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_untruncated_recovers_moments():
    data = np.random.normal(3.0, 2.0, size=300)
    mu, cov = tmvn_MLE(data)
    assert np.isfinite(mu) and np.isfinite(cov)
    assert abs(mu - np.mean(data)) < 0.3
    assert cov > 0.0
    assert abs(np.sqrt(cov) - np.std(data)) < 0.6


def test_mle_multivariate_untruncated_output_shapes_and_moments():
    mean = np.array([1.0, -2.0])
    cov_true = np.array([[1.0, 0.5], [0.5, 2.0]])
    data = np.random.multivariate_normal(mean, cov_true, size=300).T
    assert data.shape == (2, 300)
    mu, cov = tmvn_MLE(data)
    assert np.shape(mu) == (2,)
    assert np.shape(cov) == (2, 2)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-8, atol=1e-10)
    assert np.all(np.diag(cov) > 0.0)
    assert np.all(np.abs(mu - data.mean(axis=1)) < 0.4)


def test_mle_univariate_constant_data_does_not_fail():
    data = np.full(20, 5.0)
    mu, cov = tmvn_MLE(data)
    assert abs(mu - 5.0) < 1e-3
    assert np.isfinite(cov)
    assert cov >= 0.0


def test_mle_single_sample_returns_the_sample_as_mean():
    mu, cov = tmvn_MLE(np.array([3.0]))
    assert mu == pytest.approx(3.0, abs=1e-6)
    assert np.isfinite(cov)


def test_mle_univariate_truncated_stays_near_truth():
    data = np.atleast_1d(tmvn_rvs(0.0, 1.0, lower=-1.0, upper=2.0, size=300))
    mu, cov = tmvn_MLE(data, tr_lower=-1.0, tr_upper=2.0)
    assert np.isfinite(mu) and np.isfinite(cov)
    assert cov > 0.0
    assert abs(mu - 0.0) < 0.7


def test_mle_univariate_censored_data_gives_finite_estimates():
    raw = np.random.normal(0.0, 1.0, size=400)
    det_up = 1.0
    observed = raw[raw <= det_up]
    n_cens = int(np.sum(raw > det_up))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, cov = tmvn_MLE(observed, censored_count=n_cens,
                           det_lower=-np.inf, det_upper=det_up)
    assert np.isfinite(mu) and np.isfinite(cov)
    assert cov > 0.0
    assert abs(mu) < 0.7


def test_mle_multivariate_truncated_returns_valid_covariance():
    mu_true = np.array([0.0, 0.0])
    cov_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    lower = np.array([-1.5, -1.5])
    upper = np.array([2.0, 2.0])
    data = tmvn_rvs(mu_true, cov_true, lower=lower, upper=upper, size=200).T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, cov = tmvn_MLE(data, tr_lower=lower, tr_upper=upper)
    assert np.shape(mu) == (2,)
    assert np.shape(cov) == (2, 2)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(cov))
    assert np.all(np.diag(cov) > 0.0)
    assert np.all(np.abs(mu) < 0.8)
