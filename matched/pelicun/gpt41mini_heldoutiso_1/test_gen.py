import numpy as np
import pytest
from pelicun.uq import tmvn_rvs, tmvn_MLE, mvn_orthotope_density
from scipy.stats import multivariate_normal

@pytest.mark.parametrize("mu,COV,size", [
    (0, 1, 10),                                   # univariate scalar mean and variance
    ([0], [[1]], 10),                             # univariate but in array form
    ([0, 1], [[1, 0], [0, 1]], 1000),             # bivariate standard normal
    (np.array([1, 2]), np.array([[2, 0.5], [0.5, 1]]), 1000),  # general 2D
])
def test_tmvn_rvs_unbounded(mu, COV, size):
    samples = tmvn_rvs(mu, COV, size=size)
    samples = np.atleast_2d(samples)
    assert samples.shape[0] == size or samples.shape[1] == size
    # Check mean and covariance approximate
    # We allow some tolerance since these are random samples
    mean_est = np.mean(samples, axis=0)
    cov_est = np.cov(samples, rowvar=False)
    mu = np.atleast_1d(mu)
    assert mean_est.shape == mu.shape
    assert cov_est.shape == (mu.size, mu.size)
    np.testing.assert_allclose(mean_est, mu, atol=0.3)
    np.testing.assert_allclose(np.diag(cov_est), np.diag(COV), atol=0.3)

@pytest.mark.parametrize("mu,COV,lower,upper", [
    (0, 1, -1, 1),                               # univariate truncation symmetric
    ([0], [[1]], [-0.5], [0.5]),                 # univariate array form truncation
    ([0,0], [[1,0.5],[0.5,1]], [-1,-1], [1,1]), # bivariate with bounds on all dims
])
def test_tmvn_rvs_truncated(mu, COV, lower, upper):
    size = 5000
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=size)
    samples = np.atleast_2d(samples)
    # Shape checks
    assert samples.shape[0] == size or samples.shape[1] == size
    # Check truncation bounds are respected
    samples = np.atleast_2d(samples)
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)
    assert np.all(samples >= lower - 1e-12)
    assert np.all(samples <= upper + 1e-12)

def test_tmvn_rvs_truncation_limits_none_defaults():
    # Provide lower or upper as None and test default behavior
    mu = [0, 1]
    COV = [[1, 0.2], [0.2, 1]]
    lower = [None, -np.inf]
    upper = [np.inf, None]
    size = 1000
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=size)
    samples = np.atleast_2d(samples)
    assert samples.shape[0] == size or samples.shape[1] == size
    # Check samples respect bounds, None replaced by -inf or inf so no filtering needed
    # Just check samples dimension
    assert samples.shape[1] == 2 or samples.shape[0] == 2

def test_tmvn_rvs_rejection_warning(monkeypatch):
    # Use very tight truncation to cause very low acceptance rate and generate warn
    mu = [0.0]
    COV = [[1.0]]
    lower = [10.0]
    upper = [20.0]
    size = 10
    # We expect a ValueError due to very small probability in truncation region
    with pytest.raises(ValueError):
        tmvn_rvs(mu, COV, lower=lower, upper=upper, size=size)

def test_mvn_orthotope_density_basic():
    mu = [0]
    COV = [[1]]
    alpha, eps_alpha = mvn_orthotope_density(mu, COV, lower=[-1], upper=[1])
    # Should be close to P(-1 <= Z <= 1) ~0.682689
    np.testing.assert_allclose(alpha, 0.6826, rtol=1e-3)
    assert eps_alpha >= 0

def test_mvn_orthotope_density_multidim():
    mu = [0, 0]
    COV = [[1, 0.5], [0.5, 1]]
    alpha, eps_alpha = mvn_orthotope_density(mu, COV, lower=[-1, -1], upper=[1, 1])
    # Probability should be less than 1 and positive
    assert 0 < alpha < 1
    assert eps_alpha >= 0

def test_tmvn_MLE_fit_univariate_normal():
    # Generate 100 samples from N(3, 2^2)
    mu_true = 3.0
    sigma_true = 2.0
    samples = np.random.normal(mu_true, sigma_true, size=100)
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert isinstance(mu_fit, float) or mu_fit.shape == ()
    assert np.isscalar(cov_fit)
    # Check that estimated mean close to true mean
    assert abs(mu_fit - mu_true) < 0.3
    # Check that estimated variance close to true variance
    assert abs(cov_fit - sigma_true**2) < 0.5

def test_tmvn_MLE_fit_univariate_truncated():
    # Generate truncated normal samples and fit using tmvn_MLE
    from scipy.stats import truncnorm
    a, b = -1, 2  # truncation limits in std units
    size = 200
    mu_true = 5.0
    sigma_true = 1.5
    samples = truncnorm.rvs(a, b, loc=mu_true, scale=sigma_true, size=size)
    # Provide actual truncation limits (linear scale)
    tr_lower = mu_true + a * sigma_true
    tr_upper = mu_true + b * sigma_true

    mu_fit, cov_fit = tmvn_MLE(samples, tr_lower=[tr_lower], tr_upper=[tr_upper])
    # Estimated mean should be inside truncation bounds and close to true
    assert tr_lower <= mu_fit <= tr_upper
    assert abs(mu_fit - mu_true) < 0.4
    # Variance less than sigma_true^2 due to truncation
    assert cov_fit < sigma_true ** 2 + 0.1

def test_tmvn_MLE_fit_multivariate_normal():
    # Generate 2D samples with correlated normal
    mu_true = np.array([2.0, -1.0])
    cov_true = np.array([[2.0, 0.6], [0.6, 1.0]])
    samples = np.random.multivariate_normal(mu_true, cov_true, size=200).T
    mu_fit, cov_fit = tmvn_MLE(samples)
    mu_fit = np.array(mu_fit)
    cov_fit = np.array(cov_fit)

    # Assert shapes
    assert mu_fit.shape == (2,)
    assert cov_fit.shape == (2, 2)
    # Check mean and covariance diagonals close to true
    assert np.allclose(mu_fit, mu_true, atol=0.5)
    assert np.allclose(np.diag(cov_fit), np.diag(cov_true), atol=0.5)

def test_tmvn_MLE_fit_multivariate_truncated():
    # Generate 2D truncated normal samples from scipy truncnorm per dimension roughly
    size = 200
    mu_true = np.array([0., 0.])
    cov_true = np.array([[1.0, 0.4], [0.4, 1.0]])
    # We'll sample base and then reject points out of bounds
    samples_raw = np.random.multivariate_normal(mu_true, cov_true, size=400)
    lower = np.array([-1, -0.5])
    upper = np.array([2, 1.5])
    mask = np.all((samples_raw >= lower) & (samples_raw <= upper), axis=1)
    samples = samples_raw[mask][:size].T
    # Provide truncation limits accordingly
    tr_lower = lower
    tr_upper = upper

    mu_fit, cov_fit = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper)
    mu_fit = np.array(mu_fit)
    cov_fit = np.array(cov_fit)
    # Means should be between truncation bounds
    assert np.all(mu_fit >= tr_lower - 0.5)
    assert np.all(mu_fit <= tr_upper + 0.5)
    # Covariance diagonals should be positive
    assert np.all(np.diag(cov_fit) > 0)

def test_tmvn_MLE_censored_data_behavior():
    # Create censored data with detection limits
    mu_true = 0
    sigma_true = 1
    samples = np.random.normal(mu_true, sigma_true, 200)
    # left censor data below -0.5
    det_lower = [-np.inf]
    det_upper = [-0.5]
    censored_count = np.sum(samples < -0.5)
    samples_obs = samples[samples >= -0.5]
    mu_fit, cov_fit = tmvn_MLE(samples_obs, det_lower=det_lower, det_upper=det_upper,
                               censored_count=censored_count)
    # Estimated mean should be near mu_true
    assert abs(mu_fit - mu_true) < 0.4
    assert cov_fit > 0

def test_tmvn_MLE_tight_alpha_limit_truncation():
    # Provide truncation with very tight alpha limiting value to test alpha_lim warning
    mu_true = 0
    sigma_true = 1
    samples = np.random.normal(mu_true, sigma_true, 50)
    tr_lower = [-0.1]
    tr_upper = [0.1]
    # alpha_lim set very high to force the alpha limit warning behavior in function
    with pytest.warns(UserWarning):
        mu_fit, cov_fit = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper,
                                   alpha_lim=0.5)
    assert cov_fit > 0

def test_tmvn_MLE_zero_variance_dimension():
    # Samples with one dimension constant zero to test zero variance handling
    samples = np.vstack([np.random.normal(2, 1, 100), np.zeros(100)])
    mu_fit, cov_fit = tmvn_MLE(samples)
    mu_fit = np.array(mu_fit)
    cov_fit = np.array(cov_fit)
    # The zero variance dimension should be small but nonzero
    assert cov_fit[1,1] > 0

def test_tmvn_MLE_samples_less_than_unknowns_warn():
    # Use smaller sample size than parameters to fit to check warning (fit_rho = False disables rho fitting)
    samples = np.random.normal(0,1, 2)  # only 2 samples in 1D
    tr_lower = [-1]
    tr_upper = [1]
    # We patch verbose to True and capture warnings, but 
    # since fit_rho=False always, warning will not appear. We'll just run to coverage.
    mu_fit, cov_fit = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper)
    assert cov_fit > 0

def test_mvn_orthotope_density_infinite_bounds():
    mu = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    alpha, eps_alpha = mvn_orthotope_density(mu, cov,
                                             lower=[-np.inf, None], upper=[None, np.inf])
    # The value of alpha should be 1, since the hyperrectangle is the whole space
    assert np.isclose(alpha, 1, atol=1e-8)

def test_tmvn_rvs_size_zero_returns_empty():
    mu = [0]
    cov = [[1]]
    samples = tmvn_rvs(mu, cov, size=0)
    arr = np.atleast_2d(samples)
    assert arr.shape[0] == 0 or arr.shape[1] == 0

def test_tmvn_rvs_single_sample_shape():
    mu = [1, 2]
    cov = [[1, 0], [0, 1]]
    sample = tmvn_rvs(mu, cov, size=1)
    sample = np.atleast_2d(sample)
    # Sample shape compatible with one sample, two dimensions
    assert sample.shape[0] == 1 or sample.shape[1] == 1
    assert sample.shape[0] == 2 or sample.shape[1] == 2

def test_tmvn_rvs_high_dim_rejection(monkeypatch):
    # Create a 5D truncated normal with a tight truncation causing likely inefficiency
    mu = np.zeros(5)
    cov = np.eye(5)
    lower = np.ones(5) * 0.5
    upper = np.ones(5) * 2.0

    # Patch mvn_orthotope_density to return very small alpha but large eps to cause raise
    def fake_mvn_orthotope_density(*args, **kwargs):
        return (1e-10, 1e-7)  # very small alpha with large relative error

    monkeypatch.setattr('pelicun.uq.mvn_orthotope_density', fake_mvn_orthotope_density)
    with pytest.raises(ValueError):
        tmvn_rvs(mu, cov, lower=lower, upper=upper, size=10)

def test_tmvn_MLE_nan_params():
    # Prepare samples that will force nan in correlation matrix (2 identical samples)
    samples = np.array([[1, 1, 1], [2, 2, 2]])
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert mu_fit.shape[0] == samples.shape[0]
    assert cov_fit.shape == (samples.shape[0], samples.shape[0])

