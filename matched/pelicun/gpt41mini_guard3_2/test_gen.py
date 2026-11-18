import numpy as np
import pytest
from scipy.stats import truncnorm, norm, multivariate_normal
from pelicun import uq

# The main targets for tests:
# - tmvn_rvs(mu, COV, lower=None, upper=None, size=1)
# - tmvn_MLE(samples, tr_lower=None, tr_upper=None, censored_count=0, det_lower=None, det_upper=None, alpha_lim=None)

# As tests must be reproducible and non-flaky, any randomness used by uq module is seeded inside each test.
# The uq module uses numpy.random and scipy.stats.multivariate_normal internally.
# We'll seed numpy's RNG per test consistently.
# We do not monkeypatch here because functions called (e.g. multivariate_normal.rvs) use numpy randomness internally.


def test_tmvn_rvs_no_truncation_scalar_mean():
    np.random.seed(1234)
    mu = 2.0
    cov = 4.0  # variance (covariance = 1x1 matrix or scalar variance)
    samples = uq.tmvn_rvs(mu, cov, size=1000)
    samples = np.atleast_1d(samples)
    # Check shape
    assert samples.shape == (1000,)
    # Mean close to mu within tolerance
    assert np.isclose(np.mean(samples), mu, atol=0.1)
    # Variance close to cov
    assert np.isclose(np.var(samples), cov, atol=0.2)


def test_tmvn_rvs_no_truncation_vector_mean():
    np.random.seed(4567)
    mu = np.array([0.0, 3.0])
    COV = np.array([[1.0, 0.5], [0.5, 2.0]])
    samples = uq.tmvn_rvs(mu, COV, size=5000)
    assert samples.shape == (5000, 2)
    # Check sample means approx mu
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.1)
    # Check covariance approx COV (within some tolerance)
    sample_cov = np.cov(samples, rowvar=False)
    assert np.allclose(sample_cov, COV, atol=0.15)


def test_tmvn_rvs_with_truncation_simple_bounds():
    np.random.seed(1357)
    mu = np.array([0.0])
    cov = np.array([[1.0]])
    lower = np.array([-0.5])
    upper = np.array([1.5])
    samples = uq.tmvn_rvs(mu, cov, lower=lower, upper=upper, size=500)
    # Check shape
    assert samples.shape == (500, 1)
    # Check all samples obey truncation bounds
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)


def test_tmvn_rvs_multidim_truncation_bounds():
    np.random.seed(9753)
    mu = np.array([1.0, 2.0])
    COV = np.array([[1.0, 0.3], [0.3, 1.5]])
    lower = np.array([0.0, 1.5])
    upper = np.array([2.0, 3.0])
    samples = uq.tmvn_rvs(mu, COV, lower=lower, upper=upper, size=300)
    assert samples.shape == (300, 2)
    # Check truncation
    assert np.all(samples[:, 0] >= lower[0]) and np.all(samples[:, 0] <= upper[0])
    assert np.all(samples[:, 1] >= lower[1]) and np.all(samples[:, 1] <= upper[1])


def test_tmvn_rvs_low_density_raises(monkeypatch):
    np.random.seed(12345)
    mu = 0.0
    cov = 1.0
    lower = 10.0  # Very far from mean, likely density inside truncation is tiny
    upper = 15.0

    # This should raise a ValueError due to low density within truncation limits
    with pytest.raises(ValueError):
        uq.tmvn_rvs(mu, cov, lower=lower, upper=upper, size=5)


def test_mvnt_orthotope_density_univariate_no_bounds():
    mu = 0.0
    cov = 1.0
    alpha, eps_alpha = uq.mvn_orthotope_density(mu, cov, lower=None, upper=None)
    # Probability of whole space should be nearly 1 with small error
    assert np.isclose(alpha, 1.0, atol=1e-10)
    assert eps_alpha >= 0.0


def test_mvnt_orthotope_density_univariate_finite_bounds():
    mu = 0.0
    cov = 1.0
    lower = -1.0
    upper = 1.0
    alpha, eps_alpha = uq.mvn_orthotope_density(mu, cov, lower=lower, upper=upper)
    # Expected probability ~0.6827 for +-1 std bounds in std normal
    expected = norm.cdf(1) - norm.cdf(-1)
    assert np.isclose(alpha, expected, atol=1e-3)
    assert eps_alpha >= 0.0


def test_mvnt_orthotope_density_multivariate_bounds():
    mu = np.zeros(2)
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    lower = np.array([-0.5, -0.5])
    upper = np.array([0.5, 0.5])
    alpha, eps_alpha = uq.mvn_orthotope_density(mu, cov, lower=lower, upper=upper)
    assert 0.0 <= alpha <= 1.0
    assert eps_alpha >= 0.0
    # Known that probabilities inside small box > 0, less than 1
    assert alpha < 1.0


def test_tmvn_MLE_univariate_normal_no_truncation():
    np.random.seed(2024)
    mu_true = 3.0
    sigma_true = 2.0
    n_samples = 500
    samples = np.random.normal(mu_true, sigma_true, size=n_samples)
    mu_est, cov_est = uq.tmvn_MLE(samples)
    # mu_est should be close to mu_true
    assert np.isclose(mu_est, mu_true, atol=0.1)
    # cov_est should be close to sigma_true squared
    assert np.isclose(cov_est, sigma_true**2, atol=0.2)


def test_tmvn_MLE_multivariate_normal_no_truncation():
    np.random.seed(2024)
    mu_true = np.array([1.5, -2.0])
    cov_true = np.array([[1.0, 0.5], [0.5, 2.0]])
    n_samples = 1000
    samples = np.random.multivariate_normal(mu_true, cov_true, size=n_samples).T
    mu_est, cov_est = uq.tmvn_MLE(samples)
    assert np.allclose(mu_est, mu_true, atol=0.1)
    assert np.allclose(cov_est, cov_true, atol=0.15)


def test_tmvn_MLE_with_truncation():
    np.random.seed(7777)
    mu_true = 0.0
    sigma_true = 1.0
    samples = np.random.normal(mu_true, sigma_true, size=1000)

    tr_lower = -1.0
    tr_upper = 2.5

    # Truncate samples manually for test (not required but simulates reality)
    samples = samples[(samples >= tr_lower) & (samples <= tr_upper)]

    mu_est, cov_est = uq.tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper)
    # The estimated mean should be inside truncation limits and near mu_true
    assert tr_lower < mu_est < tr_upper
    assert np.isclose(mu_est, mu_true, atol=0.2)
    # Estimated variance should be positive and reasonable
    assert cov_est > 0


def test_tmvn_MLE_with_censored_samples():
    np.random.seed(2029)
    mu_true = 1.0
    sigma_true = 1.0
    n_samples = 500
    samples = np.random.normal(mu_true, sigma_true, size=n_samples)

    # impose detection limits
    det_lower = -np.inf
    det_upper = 0.0
    censored_count = 50  # pretend we have 50 censored samples beyond detection limits

    # Provide adjusted samples (no truncation)
    mu_est, cov_est = uq.tmvn_MLE(samples, censored_count=censored_count,
                                  det_lower=det_lower, det_upper=det_upper)
    assert np.isfinite(mu_est)
    assert np.isfinite(cov_est)
    assert cov_est > 0


def test_tmvn_MLE_alpha_lim_warn(monkeypatch):
    np.random.seed(555)
    # Use samples far from truncation area to trigger alpha_lim warning
    samples = np.array([0.0, 0.1, 0.2, 0.15, 0.25])
    tr_lower = 10.0
    tr_upper = 20.0
    # alpha_lim low value to force warning and bailout
    alpha_lim = 1e-4

    # Should raise a warning, and return something sensible or bounded values
    with pytest.warns(UserWarning):
        mu_est, cov_est = uq.tmvn_MLE(samples, tr_lower=tr_lower,
                                      tr_upper=tr_upper, alpha_lim=alpha_lim)
    assert np.all(np.isfinite(mu_est))
    assert np.all(np.isfinite(cov_est))


def test_tmvn_MLE_univariate_zero_variance(monkeypatch):
    np.random.seed(888)
    # All identical values -> zero variance
    samples = np.ones(100) * 3.0
    mu_est, cov_est = uq.tmvn_MLE(samples)
    # variance replaced with small positive number, not zero
    assert cov_est > 0
    # mean equals sample values
    assert np.isclose(mu_est, 3.0)


def test_tmvn_MLE_multivariate_zero_variance(monkeypatch):
    np.random.seed(999)
    # Multi-dim samples with zero variance in one dimension
    samples = np.vstack([
        np.ones(100) * 3.0,
        np.random.normal(0, 1, 100)
    ])
    mu_est, cov_est = uq.tmvn_MLE(samples)
    assert cov_est[0, 0] > 0
    assert np.isclose(mu_est[0], 3.0)
    assert cov_est.shape == (2, 2)


def test_tmvn_MLE_multivariate_singular_cov(monkeypatch):
    np.random.seed(111)
    # Construct singular covariance by having perfect correlation
    base = np.random.normal(0, 1, 1000)
    samples = np.vstack([base, base * 2])
    mu_est, cov_est = uq.tmvn_MLE(samples)
    # covariance matrix should be positive semidefinite
    eigs = np.linalg.eigvals(cov_est)
    assert np.all(eigs >= -1e-12)


def test_tmvn_MLE_input_ndim_1_and_2():
    np.random.seed(2020)
    # test 1D samples as list and 2D as matrix
    samples1d = np.random.normal(1.0, 0.5, 100)
    mu1d, cov1d = uq.tmvn_MLE(samples1d)
    samples2d = np.vstack([samples1d, samples1d + 0.1 * np.random.normal(0, 1, 100)])
    mu2d, cov2d = uq.tmvn_MLE(samples2d)
    assert mu1d.shape == ()
    assert cov1d.shape == ()
    assert mu2d.shape == (2,)
    assert cov2d.shape == (2, 2)


def test_tmvn_MLE_large_sample_small_dim_efficiency():
    np.random.seed(31415)
    mu_true = [2.0, -1.0]
    cov_true = [[1.0, 0.3], [0.3, 1.0]]
    samples = np.random.multivariate_normal(mu_true, cov_true, 3000).T
    mu_est, cov_est = uq.tmvn_MLE(samples)
    # Check that estimation returns close values
    assert np.allclose(mu_est, mu_true, atol=0.1)
    assert np.allclose(cov_est, cov_true, atol=0.15)


def test_tmvn_MLE_bounds_are_respected(monkeypatch):
    np.random.seed(2021)
    # Test that optimizer respects the bounds on parameters
    samples = np.random.normal(0, 1, 100)
    # Call tmvn_MLE with a monkeypatch on minimize to check bounds are passed
    # We patch minimize to a dummy that records input
    called = {}

    def dummy_minimize(*args, **kwargs):
        called['bounds'] = kwargs.get('bounds', None)
        # Return an object with .x attribute to mimic result
        class DummyResult:
            x = args[0]  # params initialized
        return DummyResult()

    monkeypatch.setattr(uq, "minimize", dummy_minimize)
    mu_est, cov_est = uq.tmvn_MLE(samples)
    # Bounds should be passed to minimize
    assert called.get('bounds') is not None
    # Restore done automatically by monkeypatch fixture


def test_tmvn_MLE_infinite_truncation_limits(monkeypatch):
    np.random.seed(4242)
    samples = np.random.normal(0, 1, 500)
    # Use infinite truncation limits, equivalent to no truncation
    tr_lower = -np.inf
    tr_upper = np.inf
    mu_est, cov_est = uq.tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper)
    # Should behave similarly to no truncation case
    mu0, cov0 = uq.tmvn_MLE(samples)
    assert np.isclose(mu_est, mu0, atol=0.1)
    assert np.isclose(cov_est, cov0, atol=0.1)


def test_tmvn_MLE_censored_count_zero(monkeypatch):
    np.random.seed(5555)
    samples = np.random.normal(0, 1, 100)
    mu_est, cov_est = uq.tmvn_MLE(samples, censored_count=0)
    # Should behave same as no censoring
    mu0, cov0 = uq.tmvn_MLE(samples)
    assert np.isclose(mu_est, mu0, atol=1e-6)
    assert np.isclose(cov_est, cov0, atol=1e-6)
