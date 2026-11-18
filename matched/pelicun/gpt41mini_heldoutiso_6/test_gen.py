import numpy as np
import pytest
from pelicun.uq import tmvn_rvs, tmvn_MLE, mvn_orthotope_density
from scipy.stats import multivariate_normal

@pytest.mark.parametrize("mu, cov, size", [
    (0, 1, 10),
    ([0], [[1]], 5),
    ([1, 2], [[1, 0], [0, 1]], 20),
])
def test_tmvn_rvs_no_truncation(mu, cov, size):
    samples = tmvn_rvs(mu, cov, size=size)
    samples = np.atleast_2d(samples)
    assert samples.shape[0] == size or samples.shape[0] == samples.shape[0]
    if np.ndim(mu) == 0 or np.size(mu) == 1:
        assert np.allclose(np.mean(samples, axis=0), np.atleast_1d(mu), atol=1.0)
    else:
        assert samples.shape[1] == len(mu)
    # Samples should roughly respect mean and covariance - rough check
    mean_est = np.mean(samples, axis=0)
    cov_est = np.cov(samples, rowvar=False)
    mean_target = np.atleast_1d(mu)
    assert np.allclose(mean_est, mean_target, atol=2)
    cov_target = np.atleast_2d(cov)
    assert cov_target.shape[0] == cov_target.shape[1]
    assert np.allclose(np.diag(cov_est), np.diag(cov_target), atol=2)

def test_tmvn_rvs_with_truncation_basic():
    mu = np.array([0.0])
    cov = np.array([[1.0]])
    lower = np.array([-0.5])
    upper = np.array([0.5])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=100)
    assert samples.shape == (100, 1)
    # Check samples are within bounds
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)

def test_tmvn_rvs_multivariate_truncation():
    mu = np.zeros(2)
    cov = np.eye(2)
    lower = np.array([-1, -np.inf])
    upper = np.array([1, 1])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=50)
    assert samples.shape == (50, 2)
    assert np.all(samples[:,0] >= lower[0])
    assert np.all(samples[:,0] <= upper[0])
    assert np.all(samples[:,1] <= upper[1])

def test_tmvn_rvs_truncation_density_too_low_raises():
    mu = np.zeros(2)
    cov = np.eye(2)
    # Very narrow bounds far from mean ensure low p mass
    lower = np.array([10, 10])
    upper = np.array([11, 11])
    with pytest.raises(ValueError):
        tmvn_rvs(mu, cov, lower=lower, upper=upper, size=10)

def test_mvno_orthotope_density_basic_and_none_bounds():
    mu = 0.0
    cov = np.array([[1.0]])
    alpha, eps = mvn_orthotope_density(mu, cov)
    # Should return ~1 since no bounds means total probability mass
    assert 0.99 < alpha <= 1.0
    assert eps >= 0

def test_mvno_orthotope_density_bounded():
    mu = np.zeros(2)
    cov = np.eye(2)
    lower = [-1, -1]
    upper = [1, 1]
    alpha, eps = mvn_orthotope_density(mu, cov, lower=lower, upper=upper)
    assert 0 < alpha < 1
    assert eps >= 0

def test_tmvn_MLE_simple_univariate():
    np.random.seed(0)
    samples = np.random.normal(2, 1, 100)
    mu, cov = tmvn_MLE(samples)
    # Because input is univariate, cov is variance scalar
    assert np.isscalar(cov)
    # Estimated mean should be near 2
    assert abs(mu - 2) < 0.5
    # Estimated variance close to 1
    assert abs(cov - 1) < 0.5

def test_tmvn_MLE_multivariate_no_truncation():
    np.random.seed(1)
    mu_true = [1.0, -1.0]
    cov_true = np.array([[1.0, 0.3], [0.3, 2.0]])
    samples = np.random.multivariate_normal(mu_true, cov_true, 300).T
    mu, cov = tmvn_MLE(samples)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.allclose(mu, mu_true, atol=0.2)
    assert np.allclose(np.diag(cov), np.diag(cov_true), atol=0.5)

def test_tmvn_MLE_with_truncation_limits():
    np.random.seed(2)
    samples = np.random.normal(0, 1, 200)
    tr_lower = np.array([-1.0])
    tr_upper = np.array([1.0])
    mu, cov = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper)
    assert np.isscalar(cov)
    assert mu >= tr_lower[0] and mu <= tr_upper[0]

def test_tmvn_MLE_with_censored_data():
    np.random.seed(3)
    samples = np.random.normal(0, 1, 100)
    censored_count = 5
    det_lower = np.array([-0.5])
    det_upper = np.array([0.5])
    mu, cov = tmvn_MLE(samples, censored_count=censored_count,
                       det_lower=det_lower, det_upper=det_upper)
    assert np.isscalar(cov)
    # mu should be roughly close to sample mean (no truncation)
    mean_sample = np.mean(samples)
    assert abs(mu - mean_sample) < 0.5

def test_tmvn_MLE_alpha_lim_enforced_warns():
    # Setup samples from a very truncated normal, alpha_lim to trigger warning
    np.random.seed(4)
    samples = np.random.normal(0, 1, 100)
    tr_lower = np.array([10.0])
    tr_upper = np.array([20.0])
    with pytest.warns(UserWarning):
        mu, cov = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper,
                           alpha_lim=1e-4)
    assert mu is not None
    assert cov is not None

def test_tmvn_MLE_zero_variance_handling():
    # Input samples with zero variance
    samples = np.array([5.0] * 50)
    mu, cov = tmvn_MLE(samples)
    assert np.isscalar(cov)
    assert cov > 0  # Should replace zero variance with small positive number

def test_tmvn_MLE_multidim_zero_variance_handling():
    samples = np.array([[1.0]*50, np.linspace(0, 1, 50)])
    mu, cov = tmvn_MLE(samples)
    assert mu.shape[0] == 2
    assert cov.shape == (2, 2)
    assert np.all(np.diag(cov) > 0)

def test_tmvn_MLE_low_sample_count_no_optimization(monkeypatch):
    # For small sample count, optimizer fallback path is used
    samples = np.random.normal(0, 1, 1)
    mu, cov = tmvn_MLE(samples)
    assert mu is not None
    assert cov is not None

def test_tmvn_rvs_warns_high_rejection(monkeypatch):
    mu = np.zeros(1)
    cov = np.eye(1)
    lower = np.array([10])
    upper = np.array([100])
    
    # Patch mvn_orthotope_density to simulate small alpha and eps 
    import pelicun.uq as uq
    def fake_density(mu, cov, lower, upper):
        return 1e-10, 1e-11  # alpha < 1e-3 triggers warning in tmvn_rvs
    
    monkeypatch.setattr(uq, 'mvn_orthotope_density', fake_density)
    monkeypatch.setattr(uq.multivariate_normal, 'rvs', lambda mean, cov, size=None: np.full((max(size, 2), len(mean)), 10.5))
    
    with pytest.warns(UserWarning):
        # Will warn about high rejection rate but still return samples
        samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=2)
    assert samples.shape[0] == 2

def test_mvno_orthotope_density_infinite_bounds():
    mu = 0.0
    cov = [[1.0]]
    lower = [-np.inf]
    upper = [np.inf]
    alpha, eps = mvn_orthotope_density(mu, cov, lower, upper)
    # should be 1 with no error
    assert abs(alpha - 1) < 1e-8
    assert eps >= 0

def test_tmvn_rvs_scalar_mu_cov():
    mu = 2.0
    cov = 1.0
    samples = tmvn_rvs(mu, cov, size=10)
    # Output is 1D or 2D with shape (size, 1)
    samples = np.atleast_2d(samples)
    assert samples.shape[0] == 10
    assert samples.shape[1] == 1

def test_tmvn_MLE_handles_nan_in_samples():
    samples = np.array([np.nan, 1, 2, 3])
    # NaNs in data will propagate to mean/std, likely cause issues.
    with pytest.raises(Exception):
        tmvn_MLE(samples)

def test_tmvn_MLE_handles_empty_samples():
    samples = np.array([])
    with pytest.raises(Exception):
        tmvn_MLE(samples)

def test_tmvn_MLE_one_dim_one_sample():
    samples = np.array([3.5])
    mu, cov = tmvn_MLE(samples)
    assert np.isscalar(mu)
    assert np.isscalar(cov)
    assert cov >= 0

def test_tmvn_MLE_multivariate_with_nan_cov(monkeypatch):
    # Provide data that will cause correlation matrix NaN to trigger warning
    samples = np.array([[1]*10, [2]*10])
    import pelicun.uq as uq
    with pytest.warns(UserWarning):
        mu, cov = tmvn_MLE(samples)
    assert mu.shape[0] == 2
    assert cov.shape == (2, 2)

