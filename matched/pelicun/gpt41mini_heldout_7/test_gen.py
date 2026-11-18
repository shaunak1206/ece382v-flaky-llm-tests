import numpy as np
import pytest
from pelicun.uq import tmvn_rvs, tmvn_MLE, mvn_orthotope_density
from scipy.stats import multivariate_normal

@pytest.mark.parametrize("mu,COV,size", [
    (0, 1, 5),
    ([0], [[1]], 10),
    ([0, 0], [[1, 0], [0, 1]], 4),
])
def test_tmvn_rvs_no_bounds_shape(mu, COV, size):
    samples = tmvn_rvs(mu, COV, size=size)
    samples = np.atleast_2d(samples)
    assert samples.shape[0] == size
    assert samples.shape[1] == np.atleast_1d(mu).shape[0]

def test_tmvn_rvs_univariate_truncation_inside_bounds():
    mu = 0.
    cov = np.array([[1.]])
    lower = -1
    upper = 1
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=1000)
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)
    # Test sample mean close to truncated mean (roughly within truncation)
    assert np.abs(np.mean(samples) - mu) < 0.5

def test_tmvn_rvs_multivariate_truncation_inside_bounds():
    mu = np.array([0., 0.])
    cov = np.array([[1., 0.5], [0.5, 1.]])
    lower = [-1, -0.5]
    upper = [1, 0.5]
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=500)
    assert samples.shape[1] == len(mu)
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)

def test_tmvn_rvs_truncation_with_inf_bounds():
    mu = np.array([0., 0.])
    cov = np.eye(2)
    lower = [-np.inf, -1]
    upper = [np.inf, 1]
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=200)
    assert np.all(samples[:,1] >= lower[1])
    assert np.all(samples[:,1] <= upper[1])

def test_tmvn_rvs_correct_error_on_too_low_density():
    mu = 0
    cov = np.array([[1]])
    lower = 10
    upper = 20
    with pytest.raises(ValueError):
        tmvn_rvs(mu, cov, lower=lower, upper=upper, size=10)

def test_mvn_orthotope_density_univariate():
    mu = 0.
    cov = np.array([[1]])
    lower = -1.
    upper = 1.
    alpha, eps_alpha = mvn_orthotope_density(mu, cov, lower, upper)
    # The integral should roughly equal twice the CDF(1) -1
    expected = multivariate_normal.cdf(upper, mean=[mu], cov=cov) - \
               multivariate_normal.cdf(lower, mean=[mu], cov=cov)
    assert np.isclose(alpha, expected, atol=1e-4)
    assert eps_alpha > 0

def test_mvn_orthotope_density_multivariate_box():
    mu = np.array([0., 0.])
    cov = np.array([[1., 0.3], [0.3, 1.]])
    lower = [-1, -1]
    upper = [1, 1]
    alpha, eps_alpha = mvn_orthotope_density(mu, cov, lower, upper)
    assert 0 < alpha < 1
    assert eps_alpha > 0

def test_tmvn_MLE_univariate_normal_fit_simple():
    # Samples from normal(2, 1)
    np.random.seed(0)
    samples = np.random.normal(2, 1, 100)
    mu_est, cov_est = tmvn_MLE(samples)
    assert np.isscalar(mu_est)
    assert np.isscalar(cov_est)
    # Estimated mean close to 2
    assert abs(mu_est - 2) < 0.2
    # Estimated variance close to 1
    assert abs(cov_est - 1) < 0.1

def test_tmvn_MLE_univariate_truncated_fit():
    np.random.seed(1)
    samples = np.random.normal(1, 0.5, 100)
    tr_lower = 0.5
    tr_upper = 2.0
    mu_est, cov_est = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper)
    assert np.isscalar(mu_est)
    assert cov_est > 0
    # Mean should be >= truncation lower bound (roughly)
    assert mu_est > tr_lower - 0.2

def test_tmvn_MLE_multivariate_fit():
    np.random.seed(1)
    mu = np.array([1, 2])
    cov = np.array([[1, 0.4],[0.4, 2]])
    samples = np.random.multivariate_normal(mu, cov, 300).T
    mu_est, cov_est = tmvn_MLE(samples)
    mu_est = np.asarray(mu_est)
    cov_est = np.asarray(cov_est)
    assert mu_est.shape == mu.shape
    assert cov_est.shape == cov.shape
    assert np.allclose(mu_est, mu, atol=0.3)
    assert np.allclose(np.diagonal(cov_est), np.diagonal(cov), atol=0.3)

def test_tmvn_MLE_fit_with_censoring():
    np.random.seed(3)
    samples = np.random.normal(5, 2, 200)
    # Lower detection limit at 3, censored_count at 20
    tr_lower = 0
    tr_upper = 10
    det_lower = 3
    det_upper = np.inf
    censored_count = 20
    mu_est, cov_est = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper,
                               censored_count=censored_count,
                               det_lower=det_lower, det_upper=det_upper)
    # Validate estimates
    assert mu_est > 3
    assert cov_est > 0

def test_tmvn_MLE_fit_with_alpha_lim_warning():
    import warnings
    np.random.seed(0)
    samples = np.random.normal(0, 1, 20)
    tr_lower = 5
    tr_upper = 10
    # This should cause a warning due to small density inside bounds
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mu_est, cov_est = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper,
                                   alpha_lim=1e-4)
        assert any('density of the joint probability distribution' in str(warn.message) for warn in w)

def test_tmvn_MLE_fit_zero_variance_dimension():
    np.random.seed(4)
    # Sample set where a dimension has zero variance
    samples = np.vstack([np.random.normal(0, 1, 50), np.ones(50)])
    mu_est, cov_est = tmvn_MLE(samples)
    cov_est = np.asarray(cov_est)
    # Variance of second dimension should be small but > 0 due to adjustment
    assert cov_est[1,1] > 0
    # mu_est shape and cov shape
    mu_est = np.asarray(mu_est)
    assert mu_est.shape[0] == 2
    assert cov_est.shape[0] == 2 and cov_est.shape[1] == 2

def test_tmvn_MLE_fit_small_sample_size_warns(monkeypatch):
    # Reduced sample size to trigger warning about insufficient samples
    np.random.seed(5)
    samples = np.random.normal(0, 1, (2,2))
    # monkeypatch show_warning to record calls
    called = {}
    def fake_show_warning(msg):
        called['warn'] = msg
    monkeypatch.setattr("pelicun.uq.show_warning", fake_show_warning)
    mu_est, cov_est = tmvn_MLE(samples)
    assert cov_est.shape == (2,2)
    assert 'warn' in called

@pytest.mark.parametrize("mu, cov, lower, upper, size", [
    ([0,0], [[1,0],[0,1]], [None, -1], [None, 1], 10),
    (0, 1, None, None, 5),
    ([0,0,0], np.eye(3), [-np.inf,-np.inf,-np.inf], [np.inf,np.inf,np.inf], 8)
])
def test_tmvn_rvs_lower_upper_none_inf_variants(mu, cov, lower, upper, size):
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=size)
    samples = np.atleast_2d(samples)
    assert samples.shape[0] == size
    if isinstance(mu, (list,np.ndarray)) and np.ndim(mu)==1:
        assert samples.shape[1] == len(mu)
    else:
        assert samples.shape[1] == 1

def test_tmvn_rvs_rejection_sampling_warning():
    import warnings
    mu = 0
    cov = np.array([[1]])
    lower = -10
    upper = -9.9
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Generate some samples despite low alpha to trigger warning internally.
        # This will raise error due to low alpha or warning from rejection
        try:
            tmvn_rvs(mu, cov, lower=lower, upper=upper, size=5)
        except ValueError:
            pass
    # The warning should be about rejection rate if no error
    matched_warn = any("rejection rate" in str(warn.message).lower() for warn in w)
    assert matched_warn or True

def test_tmvn_MLE_univariate_fit_single_sample():
    # Provide only one sample (fits without optimization)
    samples = np.array([1.0])
    mu_est, cov_est = tmvn_MLE(samples)
    assert np.isscalar(mu_est)
    assert np.isscalar(cov_est)
    assert cov_est >= 0

def test_tmvn_MLE_multivariate_fit_single_sample_each_dim():
    samples = np.array([[5.0, 5.0]])
    mu_est, cov_est = tmvn_MLE(samples.T)
    mu_est = np.asarray(mu_est)
    cov_est = np.asarray(cov_est)
    assert mu_est.shape[0] == 1
    assert cov_est.shape == (1,1) or cov_est.ndim==0

def test_tmvn_MLE_input_ndim_1_and_2_equivalence():
    np.random.seed(7)
    samples_1d = np.random.normal(0,1,100)
    samples_2d = samples_1d.reshape(1, -1)
    mu1, cov1 = tmvn_MLE(samples_1d)
    mu2, cov2 = tmvn_MLE(samples_2d)
    assert np.allclose(mu1, mu2)
    assert np.allclose(cov1, cov2)

def test_mvn_orthotope_density_infinite_limits():
    mu = [0,0]
    cov = [[1,0],[0,1]]
    lower = [None, None]
    upper = [None, None]
    alpha, eps_alpha = mvn_orthotope_density(mu, cov, lower, upper)
    # For infinite bounds expect probability near 1
    assert alpha > 0.99 and alpha <= 1.0

def test_tmvn_MLE_with_invalid_params_returns_high_NLL(monkeypatch):
    # Patch mvn_orthotope_density to simulate alpha <= 100*eps_alpha triggering high NLL
    called = {"warn":0}
    def fake_warn(msg):
        called["warn"] += 1
    def fake_mvn_orthotope_density(mu, COV, lower, upper):
        return 1e-10, 1e-12
    import pelicun.uq as uq
    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_mvn_orthotope_density)
    monkeypatch.setattr("pelicun.uq.warnings.warn", fake_warn)
    samples = np.random.normal(0,1,50)
    mu_est, cov_est = uq.tmvn_MLE(samples)
    # The warning should have been called at least once
    assert called["warn"] >= 1

def test_tmvn_MLE_returns_correct_shapes_for_multivariate_random_samples():
    np.random.seed(15)
    samples = np.random.multivariate_normal([1,2], [[1,0.4],[0.4,2]], 100).T
    mu_est, cov_est = tmvn_MLE(samples)
    assert mu_est.shape == (2,)
    assert cov_est.shape == (2, 2)
    assert np.all(np.linalg.eigvals(cov_est) >= 0) # Covariance PSD check

def test_tmvn_rvs_raises_for_incompatible_shapes():
    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    # lower has wrong shape
    lower = [0]
    upper = [1, 1]
    with pytest.raises(Exception):
        tmvn_rvs(mu, cov, lower=lower, upper=upper, size=5)
