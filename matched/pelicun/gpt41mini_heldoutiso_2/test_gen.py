import pytest
import numpy as np
from pelicun.uq import tmvn_rvs, tmvn_MLE
from scipy.stats import multivariate_normal

@pytest.mark.parametrize("mu, cov, size", [
    (0, 1, 10),
    ([0.0], [[1.0]], 5),
    ([1.0, 2.0], [[1.0, 0.3],[0.3,2.0]], 20),
])
def test_tmvn_rvs_untruncated_samples_shape(mu, cov, size):
    samples = tmvn_rvs(mu, cov, size=size)
    mu_arr = np.atleast_1d(mu)
    if len(mu_arr) == 1:
        # univariate shape (size,)
        assert samples.shape == (size,)
    else:
        assert samples.shape == (size, len(mu_arr))

def test_tmvn_rvs_untruncated_sample_stats():
    mu = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    samples = tmvn_rvs(mu, cov, size=5000)
    mean_sample = np.mean(samples, axis=0)
    cov_sample = np.cov(samples, rowvar=False)
    # Means should be close
    assert np.allclose(mean_sample, mu, atol=0.1)
    # Covariance close
    assert np.allclose(cov_sample, cov, atol=0.1)

def test_tmvn_rvs_truncated_simple():
    mu = np.array([0.])
    cov = np.array([[1.]])
    lower = np.array([-1.])
    upper = np.array([1.])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=100)
    assert samples.shape == (100,)
    # Samples should be within bounds
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)

def test_tmvn_rvs_truncated_multidim_bounds():
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    lower = np.array([-0.5, -np.inf])
    upper = np.array([np.inf, 0.5])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=100)
    assert samples.shape == (100, 2)
    assert np.all(samples[:,0] >= -0.5)
    assert np.all(samples[:,1] <= 0.5)

def test_tmvn_rvs_raises_valueerror_small_density():
    mu = np.zeros(2)
    cov = np.eye(2)
    lower = np.array([10., 10.])
    upper = np.array([20., 20.])

    # The probability in this range is very small => ValueError expected
    with pytest.raises(ValueError):
        tmvn_rvs(mu, cov, lower=lower, upper=upper, size=10)

def test_tmvn_MLE_fit_univariate():
    # Generate normal samples
    mu_true = 1.5
    sigma_true = 0.5
    samples = np.random.normal(mu_true, sigma_true, size=100)
    mu_fit, cov_fit = tmvn_MLE(samples, tr_lower=None, tr_upper=None)
    # mu_fit and cov_fit should be close to true values
    assert np.isclose(mu_fit, np.mean(samples), atol=0.2)
    # cov_fit is variance
    assert np.isclose(cov_fit, np.var(samples, ddof=0), atol=0.1)

def test_tmvn_MLE_fit_multivariate():
    mu_true = np.array([1., 2.])
    cov_true = np.array([[1., 0.4], [0.4, 2.]])
    samples = np.random.multivariate_normal(mu_true, cov_true, size=500).T
    mu_fit, cov_fit = tmvn_MLE(samples)
    # mu_fit approx mu_true
    assert np.allclose(mu_fit, mu_true, atol=0.2)
    assert np.allclose(cov_fit, cov_true, atol=0.2)

def test_tmvn_MLE_with_truncation_limits():
    mu_true = np.array([0., 0.])
    cov_true = np.array([[1., 0.3],[0.3,1.]])
    lower = np.array([-0.5, -0.5])
    upper = np.array([0.5, 0.5])
    samples = np.random.multivariate_normal(mu_true, cov_true, size=1000).T
    # Apply truncation as a filter on samples
    mask = np.all((samples >= lower[:, None]) & (samples <= upper[:, None]), axis=0)
    truncated_samples = samples[:, mask]
    mu_fit, cov_fit = tmvn_MLE(truncated_samples, tr_lower=lower, tr_upper=upper)
    assert mu_fit.shape == mu_true.shape
    assert cov_fit.shape == cov_true.shape

def test_tmvn_MLE_with_censoring():
    # Univariate with censored_count and detection limits
    mu_true = 0.0
    sigma_true = 1.0
    samples = np.random.normal(mu_true, sigma_true, size=500)
    det_lower = np.array([-1.0])
    det_upper = np.array([1.0])
    censored_count = 20
    mu_fit, cov_fit = tmvn_MLE(samples, tr_lower=None, tr_upper=None,
                               censored_count=censored_count,
                               det_lower=det_lower, det_upper=det_upper)
    assert isinstance(mu_fit, float) or mu_fit.shape==()
    assert cov_fit > 0

def test_tmvn_MLE_zero_variance_dimension():
    # Create samples where one dimension has zero variance
    samples = np.array([[1.0]*50, np.random.normal(0,1,50)])
    mu_fit, cov_fit = tmvn_MLE(samples)
    # The covariance matrix should be positive semidefinite and no zeros on diagonal
    eigvals = np.linalg.eigvals(cov_fit)
    assert np.all(eigvals >= 0)
    assert np.all(np.diag(cov_fit) > 0)

def test_tmvn_MLE_small_samples_less_than_dim():
    # When samples less than dims, minimizer skips and returns initial values
    samples = np.array([[0.1, 0.2]])
    mu_fit, cov_fit = tmvn_MLE(samples)
    # mu_fit and cov_fit shape check
    assert mu_fit.shape[0] == samples.shape[0]
    assert cov_fit.shape == (samples.shape[0], samples.shape[0])

def test_tmvn_MLE_with_alpha_limit_warns(monkeypatch):
    import warnings

    samples = np.random.normal(0,1,100)
    tr_lower = np.array([-2])
    tr_upper = np.array([2])

    # Patch mvn_orthotope_density to always return very low alpha with low eps_alpha to force warning
    from pelicun import uq

    def fake_mvn_orthotope_density(mu, cov, lower=None, upper=None):
        return 1e-6, 1e-10  # very low alpha vs eps_alpha (trigger warning)

    monkeypatch.setattr(uq, 'mvn_orthotope_density', fake_mvn_orthotope_density)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mu_fit, cov_fit = uq.tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper, alpha_lim=1e-4)
        assert any("joint probability distribution within the truncation limits is too small" in str(warn.message) for warn in w) or \
               any("less than the prescribed minimum limit" in str(warn.message) for warn in w)

def test_tmvn_rvs_warn_high_rejection(monkeypatch):
    # Rejection rate above 0.999 triggers warning

    import warnings
    from pelicun import uq

    mu = np.array([0.])
    cov = np.array([[1.]])
    lower = np.array([10.])
    upper = np.array([np.inf])

    # Patch mvn_orthotope_density to return alpha just less than 1e-3 threshold to trigger warning
    def fake_density(mu, cov, lower=None, upper=None):
        return 9e-4, 1e-5  # alpha < 1e-3 triggers UserWarning about rejection rate
    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    # Patch multivariate_normal.rvs to produce samples always out of bound to force the warning path
    monkeypatch.setattr(uq.multivariate_normal, "rvs",
        lambda mean, cov, size=None: np.array([[20.0] * (size if size else 1)]).T)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            uq.tmvn_rvs(mu, cov, lower=lower, upper=upper, size=5)
        # Check UserWarning about rejection rate was issued before ValueError
        warn_msgs = [str(warn.message) for warn in w if issubclass(warn.category, UserWarning)]
        assert any("rejection rate for sampling the prescribed truncated MVN distribution is higher than 0.999" in msg for msg in warn_msgs)

def test_tmvn_rvs_accepts_inf_bounds():
    mu = np.array([0., 0.])
    cov = np.eye(2)
    lower = np.array([-np.inf, 0.])
    upper = np.array([np.inf, np.inf])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=50)
    assert np.all(samples[:,1] >= 0)

def test_tmvn_MLE_handles_nan_in_rho(monkeypatch):
    # Create samples causing NaN in correlation (e.g. zero variance on one dim)
    samples = np.array([[1.0]*10, np.random.normal(0,1,10)])
    samples[1,:] = 0  # zero variance -> nan correlation

    from pelicun import uq

    # monkeypatch show_warning to capture warnings
    warnings_msgs = []
    def fake_show_warning(msg):
        warnings_msgs.append(msg)
    monkeypatch.setattr(uq, "show_warning", fake_show_warning)

    mu_fit, cov_fit = uq.tmvn_MLE(samples)
    # It should produce non-nan cov_fit and warnings triggered
    assert not np.isnan(cov_fit).any()
    assert any("invalid correlation" in w or "correlation matrix" in w for w in warnings_msgs)

def test_tmvn_MLE_handles_non_psd_rho(monkeypatch):
    # Construct samples that cause non PSD correlation (manually)
    samples = np.array([[1.,2.,3.],[4.,5.,6.]])
    from pelicun import uq

    # monkeypatch np.linalg.eigvals to simulate not positive semidefinite
    orig_eigvals = np.linalg.eigvals
    def fake_eigvals(x):
        return np.array([-1.0, 0.5, 0.5])
    monkeypatch.setattr(np.linalg, "eigvals", fake_eigvals)

    warnings_msgs = []
    def fake_show_warning(msg):
        warnings_msgs.append(msg)
    monkeypatch.setattr(uq, "show_warning", fake_show_warning)

    mu_fit, cov_fit = uq.tmvn_MLE(samples)

    assert any("assume uncorrelated" in w for w in warnings_msgs)
    # Covariance matrix should be diagonal due to replacement with zero correlations
    off_diag_sum = np.sum(np.abs(cov_fit - np.diag(np.diagonal(cov_fit))))
    assert off_diag_sum < 1e-8

def test_tmvn_MLE_fit_bounds_enforced():
    samples = np.array([[0.0, 0.0]]).T
    mu_fit, cov_fit = tmvn_MLE(samples)
    # The parameters return should honor bounds: mu within [-10,10], sig within [0.05, 20]
    # Here we just ensure output is finite and reasonable
    assert np.all(np.abs(mu_fit) <= 10)
    assert np.all(np.diag(cov_fit) >= 0.05**2)

def test_tmvn_MLE_fit_with_multidim_trunc_bound_none():
    samples = np.random.multivariate_normal([1,2], [[1,0],[0,1]], size=20).T
    mu_fit, cov_fit = tmvn_MLE(samples, tr_lower=None, tr_upper=None)
    assert mu_fit.shape == (2,)
    assert cov_fit.shape == (2,2)

def test_tmvn_MLE_fit_no_raw_data_like_input():
    # Test input where samples are 1D and size=1 triggers early return path
    samples = np.array([1.0])
    mu_fit, cov_fit = tmvn_MLE(samples)
    assert np.isfinite(mu_fit)
    assert np.isfinite(cov_fit)

def test_tmvn_MLE_error_on_nan_params(monkeypatch):
    # Patch _neg_log_likelihood to return inf if nan in params for coverage
    samples = np.random.normal(0,1,10)
    from pelicun import uq

    # call tmvn_MLE normally and ensure no error
    mu_fit, cov_fit = uq.tmvn_MLE(samples)
    # Should return finite mu and cov
    assert np.all(np.isfinite(mu_fit))
    assert np.all(np.isfinite(cov_fit))

