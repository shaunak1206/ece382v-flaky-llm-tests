import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_tmvn_rvs_no_bounds_univariate_shape_and_moments():
    np.random.seed(1)
    samples = tmvn_rvs(mu=2.0, COV=4.0, size=20000)
    samples = np.asarray(samples)
    assert samples.shape == (20000,)
    assert abs(np.mean(samples) - 2.0) < 0.1
    assert abs(np.var(samples) - 4.0) < 0.3


def test_tmvn_rvs_no_bounds_multivariate_shape_and_moments():
    np.random.seed(2)
    mu = np.array([1.0, -3.0])
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    samples = tmvn_rvs(mu, cov, size=20000)
    assert samples.shape == (20000, 2)
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.1)
    assert np.allclose(np.cov(samples.T), cov, atol=0.15)


def test_tmvn_rvs_univariate_two_sided_bounds_respected():
    np.random.seed(3)
    samples = tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=1000)
    samples = np.asarray(samples)
    assert samples.shape == (1000,)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_tmvn_rvs_returns_exact_requested_size():
    np.random.seed(4)
    for size in (1, 7, 123):
        samples = tmvn_rvs(0.0, 1.0, lower=-0.5, upper=2.0, size=size)
        assert np.asarray(samples).shape[0] == size


def test_tmvn_rvs_lower_only_half_normal_mean():
    np.random.seed(5)
    samples = np.asarray(tmvn_rvs(0.0, 1.0, lower=0.0, size=20000))
    assert np.all(samples > 0.0)
    # mean of a half-normal distribution is sqrt(2/pi)
    assert abs(np.mean(samples) - np.sqrt(2.0 / np.pi)) < 0.05


def test_tmvn_rvs_upper_only_bound_respected():
    np.random.seed(6)
    samples = np.asarray(tmvn_rvs(1.0, 1.0, upper=1.5, size=2000))
    assert np.all(samples < 1.5)
    assert np.mean(samples) < 1.0


def test_tmvn_rvs_multivariate_bounds_respected():
    np.random.seed(7)
    mu = np.array([0.0, 0.0])
    cov = np.eye(2)
    lower = np.array([-1.0, -0.5])
    upper = np.array([1.0, 2.0])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=2000)
    assert samples.shape == (2000, 2)
    assert np.all(samples[:, 0] > lower[0])
    assert np.all(samples[:, 0] < upper[0])
    assert np.all(samples[:, 1] > lower[1])
    assert np.all(samples[:, 1] < upper[1])


def test_tmvn_rvs_multivariate_infinite_bounds_in_some_dims():
    np.random.seed(8)
    mu = np.array([0.0, 0.0])
    cov = np.eye(2)
    lower = np.array([0.0, -np.inf])
    upper = np.array([np.inf, np.inf])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=3000)
    assert samples.shape == (3000, 2)
    assert np.all(samples[:, 0] > 0.0)
    # the unbounded, uncorrelated dimension keeps its original mean
    assert abs(np.mean(samples[:, 1])) < 0.1


def test_tmvn_rvs_preserves_positive_correlation_under_truncation():
    np.random.seed(9)
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples = tmvn_rvs(mu, cov, lower=np.array([0.0, 0.0]),
                       upper=np.array([np.inf, np.inf]), size=3000)
    assert np.all(samples > 0.0)
    corr = np.corrcoef(samples.T)[0, 1]
    assert corr > 0.3


def test_tmvn_rvs_reproducible_with_seed():
    np.random.seed(10)
    a = tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=50)
    np.random.seed(10)
    b = tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=50)
    assert np.array_equal(a, b)


def test_tmvn_rvs_negligible_probability_region_raises():
    with pytest.raises(ValueError):
        tmvn_rvs(0.0, 1.0, lower=50.0, upper=60.0, size=10)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_tmvn_MLE_univariate_untruncated_recovers_parameters():
    np.random.seed(11)
    data = np.random.normal(5.0, 2.0, size=500)
    mu, cov = tmvn_MLE(data)
    assert abs(float(mu) - np.mean(data)) < 0.2
    assert abs(float(cov) - 4.0) < 1.0
    assert float(cov) > 0.0


def test_tmvn_MLE_accepts_list_input():
    np.random.seed(12)
    data = list(np.random.normal(-1.0, 0.5, size=200))
    mu, cov = tmvn_MLE(data)
    assert np.isfinite(float(mu))
    assert np.isfinite(float(cov))
    assert abs(float(mu) + 1.0) < 0.2


def test_tmvn_MLE_univariate_truncated_fit():
    np.random.seed(13)
    data = np.asarray(tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=1000))
    mu, cov = tmvn_MLE(data, tr_lower=-1.0, tr_upper=1.0)
    assert abs(float(mu)) < 0.3
    assert np.isfinite(float(cov))
    # accounting for truncation, the fitted variance should not be
    # (much) smaller than the variance of the truncated sample
    assert float(cov) >= 0.9 * np.var(data)


def test_tmvn_MLE_univariate_censored_data():
    np.random.seed(14)
    raw = np.random.normal(0.0, 1.0, size=1000)
    det_lower, det_upper = -1.5, 1.5
    inside = raw[(raw > det_lower) & (raw < det_upper)]
    censored_count = len(raw) - len(inside)
    mu, cov = tmvn_MLE(inside, censored_count=censored_count,
                       det_lower=det_lower, det_upper=det_upper)
    assert abs(float(mu)) < 0.3
    assert 0.6 < float(cov) < 1.6


def test_tmvn_MLE_multivariate_untruncated_shapes_and_values():
    np.random.seed(15)
    true_mu = np.array([1.0, -2.0])
    true_cov = np.array([[1.0, 0.6], [0.6, 4.0]])
    data = np.random.multivariate_normal(true_mu, true_cov, size=500).T
    mu, cov = tmvn_MLE(data)

    mu = np.asarray(mu)
    cov = np.asarray(cov)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)
    assert np.allclose(mu, np.mean(data, axis=1), atol=0.3)
    assert np.allclose(np.diag(cov), np.var(data, axis=1), rtol=0.3)

    # the correlation matrix is not fitted; it follows the sample correlation
    sig = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sig, sig)
    assert np.allclose(corr, np.corrcoef(data), atol=1e-6)


def test_tmvn_MLE_multivariate_truncated_returns_valid_output():
    np.random.seed(16)
    samples = tmvn_rvs(np.array([1.0, 1.0]), np.eye(2),
                       lower=np.array([0.0, 0.0]),
                       upper=np.array([np.inf, np.inf]), size=300)
    data = samples.T
    mu, cov = tmvn_MLE(data,
                       tr_lower=np.array([0.0, 0.0]),
                       tr_upper=np.array([np.inf, np.inf]))
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
    assert np.all(np.diag(cov) > 0.0)
    assert np.allclose(cov, cov.T)


def test_tmvn_MLE_constant_samples_give_negligible_variance():
    data = np.full(20, 5.0)
    mu, cov = tmvn_MLE(data)
    assert abs(float(mu) - 5.0) < 1e-3
    assert 0.0 <= float(cov) < 1e-6


def test_tmvn_MLE_single_sample_returns_the_sample():
    mu, cov = tmvn_MLE(np.array([3.0]))
    assert abs(float(mu) - 3.0) < 1e-3
    assert 0.0 <= float(cov) < 1e-6
