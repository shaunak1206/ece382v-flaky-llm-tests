import numpy as np
import pytest
from pelicun.uq import tmvn_rvs, tmvn_MLE, mvn_orthotope_density
from scipy.stats import multivariate_normal

# For reproducibility of random samples
np.random.seed(42)

def test_tmvn_rvs_no_truncation_univariate():
    mu = 0
    cov = 1
    samples = tmvn_rvs(mu, cov, size=100)
    assert samples.shape == (100,)
    # Check mean approx mu
    assert abs(np.mean(samples) - mu) < 0.3

def test_tmvn_rvs_no_truncation_multivariate():
    mu = np.array([0, 1])
    cov = np.array([[1, 0.5], [0.5, 2]])
    samples = tmvn_rvs(mu, cov, size=200)
    assert samples.shape == (200, 2)
    # Check mean approx mu
    mean_sample = np.mean(samples, axis=0)
    assert np.allclose(mean_sample, mu, atol=0.4)

def test_tmvn_rvs_with_truncation_univariate():
    mu = 0
    cov = 1
    lower = -0.5
    upper = 1.5
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=50)
    assert samples.shape == (50,)
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)

def test_tmvn_rvs_with_truncation_multivariate():
    mu = np.array([0, 0])
    cov = np.array([[1.0, 0.2],[0.2, 1.0]])
    lower = np.array([-1, -0.5])
    upper = np.array([1, 1.5])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=100)
    assert samples.shape == (100, 2)
    assert np.all(samples[:, 0] >= lower[0])
    assert np.all(samples[:, 0] <= upper[0])
    assert np.all(samples[:, 1] >= lower[1])
    assert np.all(samples[:, 1] <= upper[1])

def test_tmvn_rvs_raise_value_error_for_too_low_density():
    # Construct very tight truncation limits far from mean to reduce alpha drastically
    mu = 0
    cov = 1
    lower = 10
    upper = 11
    with pytest.raises(ValueError):
        tmvn_rvs(mu, cov, lower=lower, upper=upper, size=5)

def test_mvn_orthotope_density_unbounded():
    mu = np.array([0, 0])
    cov = np.eye(2)
    alpha, eps_alpha = mvn_orthotope_density(mu, cov)
    assert pytest.approx(alpha, abs=1e-6) == 1.0
    assert eps_alpha >= 0

def test_mvn_orthotope_density_with_bounds():
    mu = np.array([0, 0])
    cov = np.eye(2)
    lower = np.array([-1, -1])
    upper = np.array([1, 1])
    alpha, eps_alpha = mvn_orthotope_density(mu, cov, lower=lower, upper=upper)
    # Probability should be > 0 and < 1
    assert 0 < alpha < 1
    assert eps_alpha >= 0

def test_mvn_orthotope_density_univariate_inf_bounds():
    mu = 0
    cov = 1
    lower = -np.inf
    upper = 1
    alpha, eps_alpha = mvn_orthotope_density(mu, cov, lower=lower, upper=upper)
    assert 0 < alpha < 1
    assert eps_alpha >= 0

def test_tmvn_MLE_univariate_no_censoring():
    # Generate data from normal dist
    mu = 1.5
    sigma = 0.4
    samples = np.random.normal(mu, sigma, size=100)
    est_mu, est_cov = tmvn_MLE(samples)
    assert est_cov > 0
    assert abs(est_mu - mu) < 0.3
    # Covariance is variance here
    assert abs(est_cov - sigma**2) < 0.1

def test_tmvn_MLE_multivariate_no_censoring():
    mu = np.array([1, -1])
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    samples = np.random.multivariate_normal(mu, cov, size=150).T
    est_mu, est_cov = tmvn_MLE(samples)
    assert est_mu.shape == (2,)
    assert est_cov.shape == (2,2)
    assert abs(est_mu[0] - mu[0]) < 0.3
    assert abs(est_mu[1] - mu[1]) < 0.3
    # Covariance diagonal approx original variances
    assert abs(est_cov[0,0] - cov[0,0]) < 0.2
    assert abs(est_cov[1,1] - cov[1,1]) < 0.3

def test_tmvn_MLE_multivariate_truncation_limits():
    mu = np.array([0, 0])
    cov = np.array([[1, 0.1],[0.1, 1]])
    samples = np.random.multivariate_normal(mu, cov, size=200).T
    lower = np.array([-0.5, -0.5])
    upper = np.array([0.5, 0.5])
    est_mu, est_cov = tmvn_MLE(samples, tr_lower=lower, tr_upper=upper)
    assert est_mu.shape == (2,)
    assert est_cov.shape == (2,2)
    # Estimates should remain finite and positive definite
    eigvals = np.linalg.eigvals(est_cov)
    assert np.all(eigvals >= 0)

def test_tmvn_MLE_censored_data():
    mu = 2.0
    sigma = 0.5
    # data mostly between 1 and 3
    samples = np.random.normal(mu, sigma, size=100)
    censored_count = 10
    det_lower = 1
    det_upper = None
    est_mu, est_cov = tmvn_MLE(samples, censored_count=censored_count,
                               det_lower=np.array([det_lower]),
                               det_upper=np.array([np.inf]))
    assert est_cov > 0
    assert abs(est_mu - mu) < 0.4

def test_tmvn_MLE_alpha_lim_warning(recwarn):
    mu = 0
    sigma = 1
    # samples close to mu but tight truncation limits far away
    samples = np.random.normal(mu, sigma, size=50)
    tr_lower = np.array([10])
    tr_upper = np.array([20])
    # Provide alpha_lim to trigger warning for below threshold
    _ = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper, alpha_lim=1e-3)
    w = recwarn.pop()
    assert issubclass(w.category, UserWarning)
    assert "density of the joint probability distribution" in str(w.message)

def test_tmvn_MLE_zero_variance_feature():
    # Samples with zero variance in one dimension
    samples = np.vstack([np.ones(50), np.linspace(0, 1, 50)])
    est_mu, est_cov = tmvn_MLE(samples)
    assert est_mu.shape == (2,)
    assert est_cov.shape == (2, 2)
    assert est_cov[0, 0] > 0  # variance replaced for zero variance dim
    assert est_cov[1, 1] > 0

def test_tmvn_MLE_small_sample_size_no_optimization():
    samples = np.array([1.0, 2.0])
    # small number of samples less than ndims
    est_mu, est_cov = tmvn_MLE(samples)
    assert est_cov > 0
    assert est_mu.shape == ()

def test_tmvn_MLE_multivariate_non_pos_semidef_correlation(monkeypatch):
    # Provide samples with weird structure to get nan or non-pos-semidef covariance
    samples = np.array([[1, 1, 1], [2, 2, 2], [3,3,3]])
    # forcibly call correlation with small sample to get warning
    est_mu, est_cov = tmvn_MLE(samples)
    # Should still produce output (fallback to zero correlation)
    assert est_mu.shape[0] == 3
    assert est_cov.shape == (3,3)
    eigvals = np.linalg.eigvals(est_cov)
    assert np.all(eigvals >= 0)

def test_tmvn_MLE_univariate_truncation_none_and_inf():
    mu, cov = 1.0, 1.0
    samples = np.random.normal(mu, np.sqrt(cov), size=100)
    # Test with None and -inf/inf truncation equivalent
    mu1, cov1 = tmvn_MLE(samples, tr_lower=None, tr_upper=None)
    mu2, cov2 = tmvn_MLE(samples, tr_lower=np.array([-np.inf]), tr_upper=np.array([np.inf]))
    assert np.isclose(mu1, mu2, atol=1e-5)
    assert np.isclose(cov1, cov2, atol=1e-5)

def test_tmvn_MLE_mutlidim_truncation_none_and_inf():
    mu = np.array([0, 0])
    cov = np.array([[1,0.1],[0.1,1]])
    samples = np.random.multivariate_normal(mu, cov, size=100).T
    mu1, cov1 = tmvn_MLE(samples, tr_lower=None, tr_upper=None)
    mu2, cov2 = tmvn_MLE(samples, tr_lower=np.array([-np.inf,-np.inf]), tr_upper=np.array([np.inf,np.inf]))
    assert np.allclose(mu1, mu2, atol=1e-5)
    assert np.allclose(cov1, cov2, atol=1e-5)
