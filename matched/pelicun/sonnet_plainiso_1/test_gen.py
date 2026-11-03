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

def test_rvs_unbounded_univariate_size():
    samples = tmvn_rvs([0.0], [[1.0]], size=200)
    assert np.asarray(samples).shape == (200,)
    assert np.all(np.isfinite(samples))


def test_rvs_unbounded_scalar_inputs():
    samples = tmvn_rvs(0.0, 1.0, size=150)
    assert np.asarray(samples).size == 150


def test_rvs_unbounded_moments_univariate():
    samples = tmvn_rvs([3.0], [[4.0]], size=5000)
    assert abs(np.mean(samples) - 3.0) < 0.2
    assert abs(np.std(samples) - 2.0) < 0.2


def test_rvs_unbounded_multivariate_shape_and_mean():
    mu = np.array([1.0, -2.0])
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    samples = tmvn_rvs(mu, cov, size=3000)
    assert samples.shape == (3000, 2)
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.15)


def test_rvs_truncated_univariate_within_bounds():
    samples = tmvn_rvs([0.0], [[1.0]], lower=-1.0, upper=2.0, size=500)
    assert samples.shape == (500,)
    assert np.all(samples > -1.0)
    assert np.all(samples < 2.0)


def test_rvs_truncated_exact_sample_count():
    for size in (1, 7, 33):
        samples = tmvn_rvs([0.0], [[1.0]], lower=-0.5, upper=0.5, size=size)
        assert np.asarray(samples).size == size


def test_rvs_symmetric_truncation_preserves_mean():
    samples = tmvn_rvs([2.0], [[1.0]], lower=1.0, upper=3.0, size=4000)
    assert abs(np.mean(samples) - 2.0) < 0.1
    # truncation reduces the spread relative to the untruncated sigma of 1
    assert np.std(samples) < 1.0


def test_rvs_lower_bound_only_shifts_mean_up():
    samples = tmvn_rvs([0.0], [[1.0]], lower=0.0, size=3000)
    assert np.all(samples > 0.0)
    # mean of a half normal is sqrt(2/pi) ~ 0.798
    assert abs(np.mean(samples) - np.sqrt(2.0 / np.pi)) < 0.1


def test_rvs_upper_bound_only():
    samples = tmvn_rvs([0.0], [[1.0]], upper=0.0, size=3000)
    assert np.all(samples < 0.0)
    assert abs(np.mean(samples) + np.sqrt(2.0 / np.pi)) < 0.1


def test_rvs_multivariate_truncated_within_bounds():
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([-1.0, -0.5])
    upper = np.array([1.5, 2.0])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=400)
    assert samples.shape == (400, 2)
    assert np.all(samples > lower)
    assert np.all(samples < upper)


def test_rvs_multivariate_infinite_bound_in_one_dimension():
    mu = np.array([0.0, 0.0])
    cov = np.eye(2)
    lower = np.array([0.0, -np.inf])
    upper = np.array([np.inf, np.inf])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=1000)
    assert samples.shape == (1000, 2)
    assert np.all(samples[:, 0] > 0.0)
    # second dimension is untruncated and independent
    assert abs(np.mean(samples[:, 1])) < 0.15


def test_rvs_reproducible_with_seed():
    np.random.seed(2024)
    a = tmvn_rvs([0.0], [[1.0]], lower=-1.0, upper=1.0, size=50)
    np.random.seed(2024)
    b = tmvn_rvs([0.0], [[1.0]], lower=-1.0, upper=1.0, size=50)
    assert np.array_equal(a, b)


def test_rvs_negligible_density_raises_value_error():
    with pytest.raises(ValueError):
        tmvn_rvs([0.0], [[1.0]], lower=50.0, upper=60.0, size=10)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_untruncated_recovers_moments():
    data = np.random.normal(5.0, 2.0, size=600)
    mu, cov = tmvn_MLE(data)
    assert abs(float(mu) - np.mean(data)) < 0.15
    assert abs(np.sqrt(float(cov)) - np.std(data)) < 0.2
    assert float(cov) > 0.0


def test_mle_univariate_truncated_recovers_underlying_parameters():
    data = tmvn_rvs([0.0], [[1.0]], lower=-1.0, upper=2.0, size=1500)
    mu, cov = tmvn_MLE(data, tr_lower=-1.0, tr_upper=2.0)
    assert abs(float(mu) - 0.0) < 0.4
    assert abs(np.sqrt(float(cov)) - 1.0) < 0.4


def test_mle_alpha_lim_small_does_not_change_result():
    data = tmvn_rvs([0.0], [[1.0]], lower=-1.0, upper=2.0, size=300)
    mu1, cov1 = tmvn_MLE(data, tr_lower=-1.0, tr_upper=2.0)
    mu2, cov2 = tmvn_MLE(data, tr_lower=-1.0, tr_upper=2.0, alpha_lim=1e-6)
    assert np.allclose(mu1, mu2)
    assert np.allclose(cov1, cov2)


def test_mle_univariate_censored_data():
    full = np.random.normal(0.0, 1.0, size=1500)
    limit = 1.5
    observed = full[full < limit]
    censored_count = int(np.sum(full >= limit))
    mu, cov = tmvn_MLE(observed, censored_count=censored_count,
                       det_lower=-np.inf, det_upper=limit)
    assert abs(float(mu)) < 0.4
    assert abs(np.sqrt(float(cov)) - 1.0) < 0.4


def test_mle_multivariate_untruncated_shapes_and_values():
    true_mu = np.array([1.0, -2.0])
    true_cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    raw = np.random.multivariate_normal(true_mu, true_cov, size=600)
    samples = raw.T  # rows are variables, columns are samples
    mu, cov = tmvn_MLE(samples)
    assert np.shape(mu) == (2,)
    assert np.shape(cov) == (2, 2)
    assert np.allclose(mu, np.mean(raw, axis=0), atol=0.15)
    assert np.allclose(cov, cov.T)
    # the correlation structure comes directly from the sample correlation
    sig = np.sqrt(np.diag(cov))
    rho = cov / np.outer(sig, sig)
    assert np.allclose(rho, np.corrcoef(samples), atol=1e-6)
    assert np.all(np.diag(cov) > 0.0)


def test_mle_multivariate_truncated_runs_and_is_valid():
    mu0 = np.array([0.0, 0.0])
    cov0 = np.eye(2)
    lower = np.array([-1.5, -1.5])
    upper = np.array([2.0, 2.0])
    raw = tmvn_rvs(mu0, cov0, lower=lower, upper=upper, size=400)
    mu, cov = tmvn_MLE(raw.T, tr_lower=lower, tr_upper=upper)
    assert np.shape(mu) == (2,)
    assert np.shape(cov) == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
    assert np.allclose(mu, mu0, atol=0.6)
    assert np.all(np.diag(cov) > 0.0)


def test_mle_single_sample_returns_the_sample_with_negligible_variance():
    mu, cov = tmvn_MLE(np.array([3.0]))
    assert abs(float(mu) - 3.0) < 1e-6
    assert 0.0 <= float(cov) < 1e-9


def test_mle_constant_samples_give_near_zero_variance():
    mu, cov = tmvn_MLE(np.full(10, 2.0))
    assert abs(float(mu) - 2.0) < 1e-4
    assert 0.0 <= float(cov) < 1e-6
