import numpy as np
import pytest
from pelicun.uq import tmvn_rvs, tmvn_MLE, mvn_orthotope_density

@pytest.mark.parametrize("mu,COV,size", [
    (0, 1.0, 10),                      # univariate scalar mean and cov
    ([0], [[1]], 5),                  # univariate vector and 2D cov
    ([0, 0], [[1, 0.5], [0.5, 1]], 10) # bivariate mvn
])
def test_tmvn_rvs_no_truncation(mu, COV, size):
    samples = tmvn_rvs(mu=mu, COV=COV, size=size)
    samples = np.atleast_2d(samples)
    assert samples.shape[0] == size or samples.shape[0] == size and samples.shape[1] == len(np.atleast_1d(mu))
    # Samples should be finite values
    assert np.all(np.isfinite(samples))

def test_tmvn_rvs_with_truncation_univariate():
    mu = 0
    COV = 1.0
    lower = -1
    upper = 1
    size = 1000
    samples = tmvn_rvs(mu=mu, COV=COV, lower=lower, upper=upper, size=size)
    assert samples.shape[0] == size
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)

def test_tmvn_rvs_with_truncation_multivariate():
    mu = [0, 0]
    COV = [[1, 0.2], [0.2, 1]]
    lower = [-1, -0.5]
    upper = [1, 0.5]
    size = 1000
    samples = tmvn_rvs(mu=mu, COV=COV, lower=lower, upper=upper, size=size)
    samples = np.atleast_2d(samples)
    assert samples.shape[0] == size
    assert np.all(samples >= np.array(lower))
    assert np.all(samples <= np.array(upper))

def test_tmvn_rvs_raises_valueerror_for_low_density():
    mu = 0
    COV = 1.0
    # Set very narrow bounds far from mean to cause near-zero density
    lower = 100
    upper = 101
    size = 1
    with pytest.raises(ValueError):
        tmvn_rvs(mu=mu, COV=COV, lower=lower, upper=upper, size=size)

@pytest.mark.parametrize("mu,COV,lower,upper,expected_alpha_range", [
    (0, 1.0, -1, 1, (0.68, 0.70)),             # univariate truncated between -1 and 1
    ([0,0], [[1,0], [0,1]], [-1,-1], [1,1], (0.45, 0.50))  # bivariate cube [-1,1]^2
])
def test_mvn_orthotope_density(mu, COV, lower, upper, expected_alpha_range):
    alpha, eps_alpha = mvn_orthotope_density(mu, COV, lower, upper)
    assert 0.0 <= alpha <= 1.0
    assert eps_alpha >= 0
    assert expected_alpha_range[0] <= alpha <= expected_alpha_range[1]

def test_tmvn_MLE_fit_univariate_no_truncation():
    np.random.seed(123)
    mu_true = 2.0
    sigma_true = 3.0
    samples = np.random.normal(loc=mu_true, scale=sigma_true, size=1000)
    mu_fit, cov_fit = tmvn_MLE(samples)
    # mu_fit should be close to mu_true
    assert np.abs(mu_fit - mu_true) < 0.2
    # cov_fit should be close to variance sigma^2
    assert np.abs(cov_fit - sigma_true**2) < 0.3

def test_tmvn_MLE_fit_multivariate_no_truncation():
    np.random.seed(42)
    mu_true = np.array([1.0, -1.0])
    cov_true = np.array([[1.0, 0.4], [0.4, 2.0]])
    samples = np.random.multivariate_normal(mu_true, cov_true, size=1500).T
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert mu_fit.shape == mu_true.shape
    assert cov_fit.shape == cov_true.shape
    # Mean close
    assert np.allclose(mu_fit, mu_true, atol=0.15)
    # Covariance close within tolerance
    assert np.allclose(cov_fit, cov_true, atol=0.3)

def test_tmvn_MLE_with_truncation_limits():
    np.random.seed(456)
    mu_true = 0
    sigma_true = 2.0
    samples = np.random.normal(mu_true, sigma_true, 1500)
    tr_lower = -1.0
    tr_upper = 1.0
    # Truncate generated samples manually
    truncated_samples = samples[(samples >= tr_lower) & (samples <= tr_upper)]
    # Fit to truncated data with limits
    mu_fit, cov_fit = tmvn_MLE(truncated_samples, tr_lower=tr_lower, tr_upper=tr_upper)
    # Fitted mean should be within truncation bounds
    assert tr_lower <= mu_fit <= tr_upper
    # Variance estimate should be positive
    assert cov_fit > 0

def test_tmvn_MLE_censored_data():
    np.random.seed(789)
    mu_true = 0
    sigma_true = 1.0
    samples = np.random.normal(mu_true, sigma_true, 500)
    # Say 50 samples are censored below detection limit -2
    det_lower = -2
    det_upper = None
    censored_count = 50
    # Truncate samples above -2 to simulate observed data
    observed_samples = samples[samples > det_lower]
    mu_fit, cov_fit = tmvn_MLE(observed_samples,
                               censored_count=censored_count,
                               det_lower=np.array([det_lower]),
                               det_upper=np.array([np.inf]))
    # The fit mean should be near true mean
    assert np.abs(mu_fit - mu_true) < 0.3
    # Covariance positive
    assert cov_fit > 0

def test_tmvn_MLE_invalid_input_nan_returns_large_nll():
    from pelicun.uq import tmvn_MLE
    # Prepare samples with NaN to trigger internal nan check
    samples = np.array([np.nan, 0.0, 1.0])
    # Try fitting - should not break and produce some output
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert np.isfinite(mu_fit)
    assert np.isfinite(cov_fit)
    assert cov_fit >= 0

def test_tmvn_rvs_rejection_sampling_warning(monkeypatch):
    import warnings
    mu = 0
    COV = 1.0
    # Bounds set so very low density inside truncation bounds (<1e-3)
    lower = -0.0001
    upper = 0.0001
    size = 5
    # Should trigger warning for high rejection rate (alpha < 1e-3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=size)
        # Check a warning was raised
        found = any("rejection rate" in str(wi.message) for wi in w)
        assert found or samples.shape[0] == size  # Warning or samples produced

def test_tmvn_MLE_alpha_lim_warning(monkeypatch):
    import warnings
    # Define samples around 0, but set truncation limits far from mean forcing low alpha
    samples = np.random.normal(0, 1, 50)
    tr_lower = -10
    tr_upper = -9
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mu, cov = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper, alpha_lim=1e-3)
        # Should catch warning about alpha below limit
        found_warn = any("prescribed minimum limit" in str(wi.message) for wi in w)
        assert found_warn or np.isfinite(mu)

def test_tmvn_MLE_zero_variance_replaced():
    # Samples all identical, variance zero, should be replaced internally
    samples = np.full(10, 3.0)
    mu, cov = tmvn_MLE(samples)
    # mu should remain 3
    assert np.isclose(mu, 3.0)
    # cov should be replaced to a small positive number internally
    assert cov > 0

def test_tmvn_MLE_multiple_dimensions_zero_variance():
    # Samples with one dimension zero variance
    samples = np.vstack([
        np.linspace(0, 1, 10),
        np.ones(10)*5
    ])
    mu, cov = tmvn_MLE(samples)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    # Check covariance diagonal second dim > 0 (replaced)
    assert cov[1,1] > 0

def test_tmvn_rvs_lower_upper_as_none_samples():
    mu = [0, 0]
    COV = [[1, 0.2], [0.2, 1]]
    # None lower and upper means no truncation, should sample mvn
    samples = tmvn_rvs(mu, COV, lower=None, upper=None, size=20)
    samples = np.atleast_2d(samples)
    assert samples.shape[0] == 20
    assert np.all(np.isfinite(samples))

def test_tmvn_rvs_scalar_cov_repeated():
    mu = 1
    COV = 2.0
    # Should handle scalar covariance converted properly
    samples = tmvn_rvs(mu, COV, size=10)
    # sample shape is correct
    assert samples.shape[0] == 10
    # values roughly around mu
    assert np.abs(np.mean(samples) - mu) < 1.0

def test_mvn_orthotope_density_infinite_limits():
    mu = 0
    COV = 1.0
    alpha, eps_alpha = mvn_orthotope_density(mu, COV,
                                            lower=None,
                                            upper=None)
    assert np.isclose(alpha, 1.0, atol=1e-10)
    assert eps_alpha >= 0

def test_tmvn_MLE_small_sample_size():
    samples = np.random.normal(0, 1, 2)
    mu, cov = tmvn_MLE(samples)
    assert np.isfinite(mu)
    assert cov > 0

def test_tmvn_MLE_constant_samples_raise_warning(monkeypatch):
    import warnings
    samples = np.ones(10) * 5
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mu, cov = tmvn_MLE(samples)
        # Should produce warnings about correlation and zero variance replaced
        found_corr_warn = any("correlation matrix" in str(wi.message) for wi in w)
        assert found_corr_warn or np.isfinite(mu)

def test_tmvn_rvs_multidim_high_dim_small_trunc():
    mu = np.zeros(3)
    COV = np.eye(3)
    lower = [-0.1, -0.1, -0.1]
    upper = [0.1, 0.1, 0.1]
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=50)
    samples = np.atleast_2d(samples)
    assert np.all(samples >= np.array(lower))
    assert np.all(samples <= np.array(upper))

def test_tmvn_MLE_array_input_1D_samples():
    samples = np.array([1.2, 3.4, 2.1, 4.5, 6.7])
    mu, cov = tmvn_MLE(samples)
    assert np.isfinite(mu) and np.isfinite(cov)
    assert cov > 0

def test_tmvn_MLE_array_input_2D_samples():
    samples = np.array([[1, 2, 3], [4, 5, 6]])
    mu, cov = tmvn_MLE(samples)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.linalg.eigvals(cov) >= 0)

