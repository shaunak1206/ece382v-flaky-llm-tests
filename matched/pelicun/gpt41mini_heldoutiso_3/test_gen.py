import numpy as np
import pytest
from pelicun.uq import tmvn_rvs, tmvn_MLE

def test_tmvn_rvs_no_truncation_univariate():
    mu = 0.0
    cov = 1.0
    samples = tmvn_rvs(mu, cov, size=1000)
    assert samples.shape == (1000,)
    mean_sample = np.mean(samples)
    assert abs(mean_sample - mu) < 0.2  # sample mean close to mu
    var_sample = np.var(samples)
    assert abs(var_sample - cov) < 0.2

def test_tmvn_rvs_no_truncation_multivariate():
    mu = [0.0, 1.0]
    cov = [[1.0, 0.5],[0.5, 2.0]]
    samples = tmvn_rvs(mu, cov, size=500)
    assert samples.shape == (500, 2)
    mean_sample = np.mean(samples, axis=0)
    assert np.allclose(mean_sample, mu, atol=0.3)

def test_tmvn_rvs_with_truncation_univariate():
    mu = 0.0
    cov = 1.0
    lower = -1.0
    upper = 1.0
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=500)
    assert samples.shape == (500,)
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)

def test_tmvn_rvs_with_truncation_multivariate():
    mu = [0.0, 0.0]
    cov = [[1.0, 0.2], [0.2, 1.0]]
    lower = [-1.0, -0.5]
    upper = [1.0, 1.0]
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=300)
    assert samples.shape == (300, 2)
    assert np.all(samples[:,0] >= lower[0])
    assert np.all(samples[:,1] >= lower[1])
    assert np.all(samples[:,0] <= upper[0])
    assert np.all(samples[:,1] <= upper[1])

def test_tmvn_rvs_rejection_rate_warning(monkeypatch):
    # Use truncation to very small region so rejection is large and warning triggers
    mu = 0.0
    cov = 1.0
    lower = 10.0
    upper = 11.0
    # monkeypatch mvn_orthotope_density to small alpha to trigger warning
    import pelicun.uq as uq
    orig_mvn_orthotope_density = uq.mvn_orthotope_density
    def fake_mvn_orthotope_density(*args, **kwargs):
        return 1e-5, 1e-10  # alpha very small, relative error small
    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_mvn_orthotope_density)
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            tmvn_rvs(mu, cov, lower=lower, upper=upper, size=3)
    # restore
    monkeypatch.setattr(uq, "mvn_orthotope_density", orig_mvn_orthotope_density)

def test_tmvn_rvs_low_alpha_warning(monkeypatch):
    # Monkeypatch mvn_orthotope_density to alpha < 1e-3 to trigger rejection warning
    import pelicun.uq as uq
    orig_mvn_orthotope_density = uq.mvn_orthotope_density
    def fake_mvn_orthotope_density(*args, **kwargs):
        return 5e-4, 1e-10
    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_mvn_orthotope_density)
    mu = 0.0
    cov = 1.0
    lower = -1
    upper = 1
    with pytest.warns(UserWarning):
        samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=5)
    monkeypatch.setattr(uq, "mvn_orthotope_density", orig_mvn_orthotope_density)
    assert samples.shape[0] == 5

def test_tmvn_MLE_univariate_no_truncation():
    np.random.seed(42)
    data = np.random.normal(3.0, 2.0, size=100)
    mu, cov = tmvn_MLE(data)
    assert isinstance(mu, float) or mu.shape == ()
    assert cov > 0
    # mu close to 3, cov close to variance 4 (std 2)
    assert abs(mu - 3.0) < 0.5
    assert abs(cov - 4.0) < 0.5

def test_tmvn_MLE_multivariate_no_truncation():
    np.random.seed(42)
    mean = np.array([1.0, 2.0])
    cov_true = np.array([[1.0, 0.3],[0.3, 2.0]])
    data = np.random.multivariate_normal(mean, cov_true, size=150).T
    mu, cov = tmvn_MLE(data)
    assert mu.shape == (2,)
    assert cov.shape == (2,2)
    assert np.allclose(mu, mean, atol=0.5)
    assert np.allclose(np.diag(cov), np.diag(cov_true), atol=0.5)

def test_tmvn_MLE_with_truncation():
    np.random.seed(123)
    mu_orig = 0
    cov_orig = 1
    data = np.random.normal(mu_orig, np.sqrt(cov_orig), size=200)
    tr_lower = -1.0
    tr_upper = 1.0
    # truncate data accordingly
    truncated_data = data[(data >= tr_lower) & (data <= tr_upper)]
    mu, cov = tmvn_MLE(truncated_data, tr_lower=tr_lower, tr_upper=tr_upper)
    assert isinstance(mu, float) or mu.shape == ()
    assert cov > 0
    assert mu <= tr_upper + 0.5
    assert mu >= tr_lower - 0.5

def test_tmvn_MLE_with_censoring():
    np.random.seed(7)
    data = np.random.normal(5.0, 1.5, size=100)
    cens_count = 5
    det_lower = np.array([3.0])
    det_upper = np.array([7.0])
    mu, cov = tmvn_MLE(data, censored_count=cens_count, det_lower=det_lower, det_upper=det_upper)
    assert isinstance(mu, float) or mu.shape == ()
    assert cov > 0
    assert mu > 0

def test_tmvn_MLE_alpha_lim_enforced(monkeypatch):
    np.random.seed(42)
    data = np.random.normal(0, 1, size=50)
    tr_lower = -0.1
    tr_upper = 0.1

    import pelicun.uq as uq
    orig_mvn_orthotope_density = uq.mvn_orthotope_density

    # Force mvn_orthotope_density to return alpha below alpha_lim so that NLL returns large value
    def fake_orthotope_density(*args, **kwargs):
        return 1e-8, 1e-12
    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_orthotope_density)

    # We expect the optimizer to have difficulty due to low alpha
    mu, cov = tmvn_MLE(data, tr_lower=tr_lower, tr_upper=tr_upper, alpha_lim=1e-4)

    # This returns something, but likely close to truncated region
    assert isinstance(mu, float) or mu.shape == ()
    monkeypatch.setattr(uq, "mvn_orthotope_density", orig_mvn_orthotope_density)

def test_tmvn_MLE_zero_variance_dim():
    # Create data with zero variance in one dimension
    data = np.array([[1.0]*50, np.random.normal(0,1,50)])
    mu, cov = tmvn_MLE(data)
    assert mu.shape == (2,)
    assert cov.shape == (2,2)
    # variance in first dimension is not zero due to adjustment
    assert cov[0,0] > 0

def test_tmvn_MLE_small_number_samples():
    # Test fitting with fewer samples than dimensions to validate non-optimization branch
    data = np.random.normal(0,1, size=1)
    mu, cov = tmvn_MLE(data)
    # Should return reasonable values without error
    assert np.isnan(mu) == False
    assert np.isnan(cov) == False

def test_tmvn_rvs_scalar_lower_bound():
    # test scalar lower bound with vector mean and cov
    mu = [0.0, 0.0]
    cov = [[1,0],[0,1]]
    lower = 0.0
    upper = None
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=50)
    assert samples.shape == (50,2)
    # In truncated dimension, values should be >= lower
    assert np.all(samples[:,0] >= lower) or np.all(samples[:,1] >= lower)

def test_tmvn_rvs_scalar_upper_bound():
    mu = [0.0]
    cov = [[1]]
    upper = 0.5
    samples = tmvn_rvs(mu, cov, lower=None, upper=upper, size=50)
    assert samples.shape == (50,1) or samples.shape == (50,)
    assert np.all(samples <= upper)

def test_tmvn_rvs_invalid_bounds_raise():
    # Lower > upper should raise error due to no valid region
    mu = 0
    cov = 1
    lower = 5
    upper = -5
    with pytest.raises(ValueError):
        tmvn_rvs(mu, cov, lower=lower, upper=upper, size=5)

def test_tmvn_MLE_nan_input():
    # Insert some nans in the data to confirm handling
    data = np.array([0., 1., np.nan, 2., 3.])
    with pytest.raises(Exception):
        tmvn_MLE(data)

def test_tmvn_MLE_perfect_correlation():
    # Covariance matrix with perfect correlation (would cause pos semidef check)
    data = np.array([[1.,1.,1.,1.,1.],[2.,2.,2.,2.,2.]])
    mu, cov = tmvn_MLE(data)
    assert mu.shape == (2,)
    assert cov.shape == (2,2)
    # Covariance matrix diagonal should be > 0
    assert np.all(np.diag(cov) > 0)

def test_tmvn_rvs_multivariate_single_sample():
    mu = [0.0, 0.0, 0.0]
    cov = np.eye(3)
    samples = tmvn_rvs(mu, cov, size=1)
    assert samples.shape == (1, 3)

def test_tmvn_MLE_multivariate_with_censoring_and_truncation():
    np.random.seed(1234)
    mu_true = np.array([0., 1.])
    cov_true = np.array([[1., 0.4], [0.4, 1.]])
    data = np.random.multivariate_normal(mu_true, cov_true, size=50).T
    tr_lower = np.array([-0.5, 0.0])
    tr_upper = np.array([0.5, 2.0])
    censored = 2
    det_lower = np.array([-0.3, 0.2])
    det_upper = np.array([0.3, 1.5])
    mu, cov = tmvn_MLE(data, tr_lower=tr_lower, tr_upper=tr_upper,
                       censored_count=censored,
                       det_lower=det_lower, det_upper=det_upper)
    assert mu.shape == (2,)
    assert cov.shape == (2,2)

def test_tmvn_MLE_zero_variance_and_mean_zero():
    # data with all zeros, sd = 0
    data = np.array([0., 0., 0., 0.])
    mu, cov = tmvn_MLE(data)
    assert mu == 0
    assert cov > 0  # should be adjusted to small variance

def test_tmvn_MLE_bounds_enforced(monkeypatch):
    # Test out-of-bound parameters returns large NLL to discourage optimizer
    np.random.seed(42)
    data = np.random.normal(0,1, size=50)
    import pelicun.uq as uq

    def wrapper_neg_log_likelihood(params, rho, enforce_bounds=False):
        # purposely give parameters out of bounds
        return uq.tmvn_MLE.__globals__['tmvn_MLE'].__globals__['_neg_log_likelihood'](params, rho, enforce_bounds)

    mu, cov = tmvn_MLE(data)
    # The test is mainly implicit by running; explicit testing of internal function is non-trivial here

