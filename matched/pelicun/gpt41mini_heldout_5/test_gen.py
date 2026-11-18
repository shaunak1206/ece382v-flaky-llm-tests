import numpy as np
import pytest
from pelicun.uq import tmvn_rvs, tmvn_MLE, mvn_orthotope_density
from scipy.stats import multivariate_normal

@pytest.mark.parametrize("mu,COV,size", [
    (0, 1, 10),  # univariate scalar mean, scalar cov
    ([0], [[1]], 5),  # univariate as arrays
    ([0, 0], [[1, 0], [0, 1]], 10),  # bivariate standard normal
])
def test_tmvn_rvs_no_truncation(mu, COV, size):
    samples = tmvn_rvs(mu, COV, size=size)
    assert samples.shape[0] == size or samples.shape[0] == size and samples.shape[1] == len(np.atleast_1d(mu))
    # Check sample mean approximately close to mean
    mean_sample = np.mean(samples, axis=0)
    mu_arr = np.atleast_1d(mu)
    assert np.allclose(mean_sample, mu_arr, atol=0.5)

def test_tmvn_rvs_invalid_truncation_bounds_raise_ValueError():
    mu = [0]
    COV = [[1]]
    lower = [10]
    upper = [11]
    with pytest.raises(ValueError):
        tmvn_rvs(mu, COV, lower=lower, upper=upper, size=3)

def test_tmvn_rvs_truncation_samples_within_bounds():
    mu = [0, 0]
    COV = [[1, 0.5],
           [0.5, 1]]
    lower = [-1, -np.inf]
    upper = [np.inf, 1]
    size = 20
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=size)
    assert samples.shape[0] == size
    # Check all samples obey truncation limits
    assert np.all(samples[:, 0] >= lower[0])
    assert np.all(samples[:, 1] <= upper[1])

def test_mvn_orthotope_density_probability_bounds():
    mu = [0, 0]
    COV = [[1, 0], [0, 1]]
    lower = [-1, -np.inf]
    upper = [1, 2]
    alpha, eps_alpha = mvn_orthotope_density(mu, COV, lower, upper)
    # probability must be between 0 and 1
    assert 0 <= alpha <= 1
    # error estimate is >= 0
    assert eps_alpha >= 0

def test_tmvn_rvs_truncation_warning(monkeypatch):
    import warnings

    # We patch mvn_orthotope_density to return very small alpha and tiny eps to trigger warning
    def fake_mvn_orthotope_density(mu, cov, lower, upper):
        return 5e-4, 1e-7  # alpha < 1e-3 but > 0, triggers warning but not error

    monkeypatch.setattr("pelicun.uq.mvn_orthotope_density", fake_mvn_orthotope_density)

    mu = [0]
    COV = [[1]]
    lower = [-1]
    upper = [1]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=3)
        # One warning for rejection rate too high expected
        assert any("rejection rate" in str(warn.message) for warn in w)

def test_tmvn_rvs_multivariate_samples_range():
    mu = [1, 2, 3]
    COV = np.diag([1, 4, 9])
    lower = [0, 1, 2]
    upper = [2, 4, 5]
    size = 30
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=size)
    assert samples.shape[0] == size
    # Check samples within bounds
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)

def test_tmvn_MLE_univariate_no_censoring():
    # Generate normal samples
    true_mu = 5
    true_var = 4
    true_sigma = np.sqrt(true_var)
    np.random.seed(42)
    samples = np.random.normal(true_mu, true_sigma, size=100)

    mu_est, cov_est = tmvn_MLE(samples)
    assert isinstance(mu_est, float) or (isinstance(mu_est, np.ndarray) and mu_est.shape == ())
    assert np.isscalar(cov_est) or (isinstance(cov_est, np.ndarray) and cov_est.shape == ())
    # Estimated mean close to true mean within reasonable tolerance
    assert np.abs(mu_est - true_mu) < 0.5
    # Estimated variance close to true variance
    assert np.abs(cov_est - true_var) < 0.5

def test_tmvn_MLE_bivariate_no_censoring():
    np.random.seed(123)
    mu_true = np.array([1.0, 2.0])
    cov_true = np.array([[1.0, 0.3],
                         [0.3, 0.5]])
    samples = np.random.multivariate_normal(mu_true, cov_true, size=200).T

    mu_est, cov_est = tmvn_MLE(samples)
    assert mu_est.shape == (2,)
    assert cov_est.shape == (2, 2)
    # Mean close to true mean
    assert np.allclose(mu_est, mu_true, atol=0.2)
    # Covariance matrix close to true covariance
    assert np.allclose(cov_est, cov_true, atol=0.3)

def test_tmvn_MLE_truncated_data():
    np.random.seed(123)
    mu_true = 0
    var_true = 1
    samples = np.random.normal(mu_true, np.sqrt(var_true), size=300)
    # Truncate samples at lower -0.5 (simulate truncation)
    samples_trunc = samples[samples > -0.5]
    tr_lower = -0.5
    tr_upper = np.inf

    mu_est, cov_est = tmvn_MLE(samples_trunc, tr_lower=[tr_lower], tr_upper=[tr_upper])
    assert isinstance(mu_est, float) or (isinstance(mu_est, np.ndarray) and mu_est.shape == ())
    # Estimated mean should be > truncation lower bound
    assert mu_est > tr_lower

def test_tmvn_MLE_with_censoring():
    np.random.seed(77)
    mu_true = 3
    sigma_true = 1.5
    samples = np.random.normal(mu_true, sigma_true, size=200)
    # Let's say samples below 2 are censored
    detected_samples = samples[samples >= 2]
    censored_count = np.sum(samples < 2)
    det_lower = [-np.inf]
    det_upper = [2]

    mu_est, cov_est = tmvn_MLE(detected_samples, censored_count=censored_count,
                               det_lower=det_lower, det_upper=det_upper)
    assert mu_est > 2 or np.isclose(mu_est, 2, atol=0.5)
    assert cov_est > 0

def test_tmvn_MLE_multiple_dimensions_censoring():
    np.random.seed(99)
    mu_true = np.array([0, 5])
    cov_true = np.array([[1.0, 0.2],
                         [0.2, 2.0]])
    samples = np.random.multivariate_normal(mu_true, cov_true, 150).T
    # Censor samples below (0, 4)
    mask = (samples[0] >= 0) & (samples[1] >= 4)
    samples_censored = samples[:, mask]
    censored_count = np.sum(~mask)
    det_lower = np.array([-np.inf, -np.inf])
    det_upper = np.array([0, 4])

    mu_est, cov_est = tmvn_MLE(samples_censored, censored_count=censored_count,
                               det_lower=det_lower, det_upper=det_upper)
    assert mu_est.shape == (2,)
    assert cov_est.shape == (2, 2)
    # Means should be greater than detection upper limits roughly
    assert mu_est[0] >= 0 or np.isclose(mu_est[0], 0, atol=0.5)
    assert mu_est[1] >= 4 or np.isclose(mu_est[1], 4, atol=1.0)

def test_tmvn_MLE_univariate_zero_variance_samples():
    # All samples same value
    samples = np.full(50, 10.0)
    mu_est, cov_est = tmvn_MLE(samples)
    # mu_est should be close to sample value
    assert np.isclose(mu_est, 10.0, atol=1e-4)
    # cov_est should not be zero because minimal variance enforced
    assert cov_est > 0

def test_tmvn_MLE_bivariate_zero_variance_1dim():
    samples = np.vstack([np.full(50, 2.0), np.random.normal(0, 1, 50)])
    mu_est, cov_est = tmvn_MLE(samples)
    # mu_est shape 2
    assert mu_est.shape == (2,)
    # cov_est shape (2,2)
    assert cov_est.shape == (2, 2)
    # variance on first dim should be > 0 due to minimal correction
    assert cov_est[0, 0] > 0

def test_tmvn_MLE_raise_warning_for_invalid_corr(monkeypatch):
    # Provide samples with zero variance in one dimension causing nan in correlation
    samples = np.vstack([np.ones(50), np.random.normal(0, 1, 50)])
    # We patch np.corrcoef to produce nan matrix to trigger warning
    def dummy_corrcoef(x):
        return np.array([[1.0, np.nan], [np.nan, 1.0]])
    monkeypatch.setattr("numpy.corrcoef", dummy_corrcoef)

    # This test only checks that the function runs, warnings printed are caught by pytest warning capture
    mu_est, cov_est = tmvn_MLE(samples)
    assert mu_est.shape == (2,)
    assert cov_est.shape == (2, 2)

def test_tmvn_MLE_alpha_limit_blocks_low_prob(monkeypatch):
    # We simulate by patching mvn_orthotope_density to return low alpha
    def fake_mvn_orthotope_density(mu, cov, lower, upper):
        return 1e-7, 1e-9  # very low alpha

    monkeypatch.setattr("pelicun.uq.mvn_orthotope_density", fake_mvn_orthotope_density)

    samples = np.array([0.0, 0.0, 0.0])
    tr_lower = [-1]
    tr_upper = [1]
    with pytest.warns(UserWarning):
        mu_est, cov_est = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper, alpha_lim=1e-4)
        # Should return reasonable outputs despite warning
        assert mu_est is not None
        assert cov_est is not None

def test_tmvn_MLE_return_shape_univariate_multivariate():
    # Univariate data
    samples_uni = np.random.normal(0, 1, 100)
    mu_uni, cov_uni = tmvn_MLE(samples_uni)
    assert np.isscalar(mu_uni) or mu_uni.shape == ()
    assert np.isscalar(cov_uni) or cov_uni.shape == ()

    # Multivariate data
    samples_multi = np.random.multivariate_normal([0, 1], [[1, 0], [0, 1]], 100).T
    mu_multi, cov_multi = tmvn_MLE(samples_multi)
    assert mu_multi.shape == (2,)
    assert cov_multi.shape == (2, 2)

def test_mvn_orthotope_density_scalar_input():
    # Test with scalar inputs
    mu = 0
    cov = np.array([[1]])
    lower = -1
    upper = 1
    alpha, eps_alpha = mvn_orthotope_density(mu, cov, lower, upper)
    assert 0 <= alpha <= 1
    assert eps_alpha >= 0

def test_tmvn_rvs_scalar_bounds():
    mu = 0
    cov = np.array([[1]])
    lower = -0.5
    upper = 0.5
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=10)
    assert samples.shape[0] == 10
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)
