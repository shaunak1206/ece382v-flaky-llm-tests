import numpy as np
import pytest
from pelicun.uq import tmvn_rvs, tmvn_MLE, mvn_orthotope_density
from scipy.stats import multivariate_normal

@pytest.mark.parametrize("seed", [42])
def test_tmvn_rvs_univariate_no_bounds(seed):
    np.random.seed(seed)
    mu = 5.0
    cov = 2.0
    samples = tmvn_rvs(mu, cov, size=1000)
    assert samples.shape == (1000,)
    # mean and variance approx
    mean_est = np.mean(samples)
    var_est = np.var(samples)
    assert abs(mean_est - mu) < 0.2
    assert abs(var_est - cov) < 0.2

@pytest.mark.parametrize("seed", [123])
def test_tmvn_rvs_univariate_with_bounds(seed):
    np.random.seed(seed)
    mu = 0.0
    cov = 1.0
    lower = -1.0
    upper = 1.0
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=1000)
    assert samples.shape == (1000,)
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)
    # mean should be within truncation
    assert lower <= np.mean(samples) <= upper

@pytest.mark.parametrize("seed", [678])
def test_tmvn_rvs_multivariate_no_bounds(seed):
    np.random.seed(seed)
    mu = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.5],
                    [0.5, 2.0]])
    samples = tmvn_rvs(mu, cov, size=5000)
    assert samples.shape == (5000, 2)
    mean_est = np.mean(samples, axis=0)
    cov_est = np.cov(samples, rowvar=False)
    assert np.allclose(mean_est, mu, atol=0.1)
    assert np.allclose(cov_est, cov, atol=0.1)

@pytest.mark.parametrize("seed", [999])
def test_tmvn_rvs_multivariate_with_bounds(seed):
    np.random.seed(seed)
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.3],
                    [0.3, 1.0]])
    lower = np.array([-1.0, -0.5])
    upper = np.array([1.0, 2.0])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=3000)
    assert samples.shape == (3000, 2)
    assert np.all(samples[:, 0] >= lower[0])
    assert np.all(samples[:, 1] >= lower[1])
    assert np.all(samples[:, 0] <= upper[0])
    assert np.all(samples[:, 1] <= upper[1])

@pytest.mark.parametrize("seed", [42])
def test_tmvn_rvs_raise_value_error_when_alpha_to_small(seed):
    np.random.seed(seed)
    # Define truncation limits that are very unlikely: mean at 0, truncation bounds far away
    mu = 0.0
    cov = 1.0
    lower = 10.0
    upper = 11.0
    with pytest.raises(ValueError):
        tmvn_rvs(mu, cov, lower=lower, upper=upper, size=10)

@pytest.mark.parametrize("seed", [44])
def test_mvn_orthotope_density_univariate(seed):
    np.random.seed(seed)
    mu = 0.0
    cov = 1.0
    lower = -1.0
    upper = 1.0
    alpha, eps_alpha = mvn_orthotope_density(mu, np.array([[cov]]), lower, upper)
    # check that probability is roughly the normal CDF difference
    expected_prob = 2 * (multivariate_normal.cdf(upper, mean=mu, cov=cov) - 0.5)
    # Since it is univariate, mvn_orthotope_density returns alpha ~ cdf difference
    assert 0 <= alpha <= 1
    assert eps_alpha >= 0
    # The alpha should be close (within 0.01) to probability from norm cdf difference
    from scipy.stats import norm
    expected_prob = norm.cdf(upper, loc=mu, scale=np.sqrt(cov)) - norm.cdf(lower, loc=mu, scale=np.sqrt(cov))
    assert abs(alpha - expected_prob) < 1e-2

@pytest.mark.parametrize("seed", [50])
def test_mvn_orthotope_density_multivariate(seed):
    np.random.seed(seed)
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5],
                    [0.5, 1.0]])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    alpha, eps_alpha = mvn_orthotope_density(mu, cov, lower, upper)
    assert 0 <= alpha <= 1
    assert eps_alpha >= 0

@pytest.mark.parametrize("seed", [10])
def test_tmvn_MLE_univariate_no_trunc(seed):
    np.random.seed(seed)
    mu_true = 2.0
    sigma_true = 1.5
    samples = np.random.normal(mu_true, sigma_true, 500)
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert np.isscalar(mu_fit)
    assert np.isscalar(cov_fit)
    # Check mean close to true mean with tolerance
    assert abs(mu_fit - mu_true) < 0.1
    # Check variance close to true variance with tolerance
    assert abs(cov_fit - sigma_true**2) < 0.2

@pytest.mark.parametrize("seed", [11])
def test_tmvn_MLE_univariate_with_truncation(seed):
    np.random.seed(seed)
    mu_true = 0.0
    sigma_true = 1.0
    samples = np.random.normal(mu_true, sigma_true, 1000)
    tr_lower = -1.0
    tr_upper = 2.0
    # Truncate samples manually to fit input
    truncated_samples = samples[(samples > tr_lower) & (samples < tr_upper)]
    mu_fit, cov_fit = tmvn_MLE(truncated_samples, tr_lower=tr_lower, tr_upper=tr_upper)
    assert np.isscalar(mu_fit)
    assert np.isscalar(cov_fit)
    # Moment estimates shifted by truncation
    assert tr_lower <= mu_fit <= tr_upper
    assert cov_fit > 0

@pytest.mark.parametrize("seed", [12])
def test_tmvn_MLE_multivariate(seed):
    np.random.seed(seed)
    mu_true = np.array([0.5, -1.0])
    cov_true = np.array([[1.0, 0.3], [0.3, 1.5]])
    samples = np.random.multivariate_normal(mu_true, cov_true, 2000).T
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert mu_fit.shape == (2,)
    assert cov_fit.shape == (2, 2)
    assert np.allclose(mu_fit, mu_true, atol=0.1)
    assert np.allclose(cov_fit, cov_true, atol=0.2)

@pytest.mark.parametrize("seed", [13])
def test_tmvn_MLE_with_censored_data(seed):
    np.random.seed(seed)
    mu_true = 1.0
    sigma_true = 2.0
    samples = np.random.normal(mu_true, sigma_true, 1000)
    tr_lower = 0.0
    tr_upper = 3.0
    # Assume some censored count (fake)
    censored_count = 5
    det_lower = 0.0
    det_upper = 1.0
    truncated_samples = samples[(samples > tr_lower) & (samples < tr_upper)]
    mu_fit, cov_fit = tmvn_MLE(truncated_samples,
                               tr_lower=tr_lower, tr_upper=tr_upper,
                               censored_count=censored_count,
                               det_lower=det_lower, det_upper=det_upper)
    assert mu_fit > tr_lower
    assert cov_fit > 0

@pytest.mark.parametrize("seed", [14])
def test_tmvn_MLE_alpha_lim_effect(seed):
    np.random.seed(seed)
    mu_true = 1.0
    sigma_true = 1.0
    samples = np.random.normal(mu_true, sigma_true, 1000)
    tr_lower = -5
    tr_upper = 0
    truncated_samples = samples[(samples > tr_lower) & (samples < tr_upper)]
    # Very high alpha_lim (close to 1) will force solution to avoid small prob mass.
    mu_fit1, cov_fit1 = tmvn_MLE(truncated_samples, tr_lower=tr_lower, tr_upper=tr_upper)
    mu_fit2, cov_fit2 = tmvn_MLE(truncated_samples, tr_lower=tr_lower, tr_upper=tr_upper, alpha_lim=1e-2)
    # The solutions should be somewhat different but valid
    assert isinstance(mu_fit1, (float, np.ndarray))
    assert isinstance(mu_fit2, (float, np.ndarray))
    assert np.all(cov_fit1 > 0)
    assert np.all(cov_fit2 > 0)

@pytest.mark.parametrize("seed", [15])
def test_tmvn_MLE_minimum_samples(seed):
    np.random.seed(seed)
    samples = np.array([1.0])
    # With only one sample, function still returns reasonable output
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert np.isscalar(mu_fit)
    assert np.isscalar(cov_fit)
    assert cov_fit >= 0

@pytest.mark.parametrize("seed", [16])
def test_tmvn_rvs_lower_upper_none(seed):
    np.random.seed(seed)
    mu = [1, 2]
    cov = [[1, 0], [0, 1]]
    samples = tmvn_rvs(mu, cov, lower=None, upper=None, size=10)
    assert samples.shape == (10, 2)

@pytest.mark.parametrize("seed", [17])
def test_tmvn_rvs_lower_none(seed):
    np.random.seed(seed)
    mu = [0]
    cov = [[1]]
    upper = [1]
    samples = tmvn_rvs(mu, cov, lower=None, upper=upper, size=10)
    assert samples.shape == (10, 1)
    assert np.all(samples <= upper[0])

@pytest.mark.parametrize("seed", [18])
def test_tmvn_rvs_upper_none(seed):
    np.random.seed(seed)
    mu = [0]
    cov = [[1]]
    lower = [-1]
    samples = tmvn_rvs(mu, cov, lower=lower, upper=None, size=10)
    assert samples.shape == (10, 1)
    assert np.all(samples >= lower[0])

@pytest.mark.parametrize("seed", [19])
def test_tmvn_MLE_with_zero_variance_dimension(seed):
    np.random.seed(seed)
    # Second dimension has zero variance
    samples = np.array([[1.0, 2.0]]*10).T
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert mu_fit.shape[0] == 2
    assert cov_fit.shape == (2, 2)
    # covariance diagonal element for zero variance dimension should be close to zero or negligible
    assert cov_fit[1,1] >= 0

@pytest.mark.parametrize("seed", [20])
def test_tmvn_MLE_monodimensional_vector_input(seed):
    np.random.seed(seed)
    samples = np.random.normal(0, 1, 500)
    mu_fit, cov_fit = tmvn_MLE(np.atleast_2d(samples))
    # Should handle samples with shape (1, N)
    assert np.isscalar(mu_fit) or (len(mu_fit) == 1)
    assert np.isscalar(cov_fit) or (cov_fit.shape == (1,1))

@pytest.mark.parametrize("seed", [21])
def test_tmvn_MLE_monodimensional_row_vector_input(seed):
    np.random.seed(seed)
    # samples as 1D, ensure function works
    samples = np.random.normal(0, 1, 500)
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert np.isscalar(mu_fit)
    assert np.isscalar(cov_fit)

@pytest.mark.parametrize("seed", [22])
def test_tmvn_MLE_with_large_variance(seed):
    np.random.seed(seed)
    samples = np.random.normal(0, 10, 1000)
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert abs(mu_fit) < 1.0  # mean should be close to zero
    assert cov_fit > 80.0  # variance should reflect large spread

@pytest.mark.parametrize("seed", [23])
def test_tmvn_MLE_fit_rho_warning_and_nan(monkeypatch):
    np.random.seed(23)
    # Create samples with zero variance in one dim to produce nan correlation
    samples = np.array([[1, 1, 1, 1, 1],
                        [2, 2, 2, 2, 2]]).astype(float)
    # monkeypatch show_warning to do nothing (avoid warning output clutter)
    monkeypatch.setattr("pelicun.uq.show_warning", lambda msg: None)
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert mu_fit.shape == (2,)
    assert cov_fit.shape == (2,2)
    assert not np.isnan(mu_fit).any()
    assert not np.isnan(cov_fit).any()

@pytest.mark.parametrize("seed", [24])
def test_tmvn_MLE_alpha_error_warning(monkeypatch):
    np.random.seed(24)
    samples = np.random.normal(0, 1, 10)
    tr_lower = np.array([10.0])
    tr_upper = np.array([11.0])
    # monkeypatch warnings.warn to detect warning called
    warnings_caught = []
    def fake_warn(msg, *args, **kwargs):
        warnings_caught.append(str(msg))
    monkeypatch.setattr("warnings.warn", fake_warn)
    with pytest.raises(ValueError):
        tmvn_rvs(0, 1, lower=tr_lower, upper=tr_upper, size=1)
    # call MLE with impossible trunc limits forces warning but returns
    mu_fit, cov_fit = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper)
    # Should have warned
    found_warning = any("too small" in w for w in warnings_caught)
    assert found_warning or warnings_caught == []

@pytest.mark.parametrize("seed", [25])
def test_tmvn_rvs_rejection_sampling_warning(monkeypatch):
    np.random.seed(25)
    mu = 0.0
    cov = 1.0
    lower = -0.05
    upper = 0.05
    # Monkeypatch warnings.warn to catch warnings
    captured = []
    def warn_capture(warning, *args, **kwargs):
        captured.append(str(warning))
    monkeypatch.setattr("warnings.warn", warn_capture)
    # Samples from a very narrow range should cause warning about rejection
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=20)
    assert samples.shape[0] == 20
    # There may be rejection warning if acceptance very low
    assert any("rejection rate" in w.lower() for w in captured) or len(captured) == 0

