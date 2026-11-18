# test_uq_tmvn.py
import numpy as np
import pytest
import warnings

from pelicun.uq import tmvn_rvs, tmvn_MLE

@pytest.mark.filterwarnings("ignore:.*truncation limits is too small.*:UserWarning")
def test_tmvn_rvs_univariate_no_truncation():
    mu = 0
    cov = 1.0
    samples = tmvn_rvs(mu, cov, size=100)
    assert samples.shape == (100,)
    # sample mean close to mu
    assert abs(np.mean(samples) - mu) < 0.3
    # sample std close to sqrt(cov)
    assert abs(np.std(samples) - np.sqrt(cov)) < 0.3

@pytest.mark.filterwarnings("ignore:.*rejection rate for sampling.*:UserWarning")
def test_tmvn_rvs_univariate_truncation():
    mu = 0.0
    cov = 1.0
    lower = -1.0
    upper = 1.0
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=1000)
    assert samples.shape == (1000,)
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)

def test_tmvn_rvs_multivariate_no_truncation():
    mu = np.array([0.0, 2.0])
    cov = np.array([[1.0, 0.5],[0.5, 2.0]])
    samples = tmvn_rvs(mu, cov, size=500)
    assert samples.shape == (500, 2)
    mean_sample = np.mean(samples, axis=0)
    assert np.allclose(mean_sample, mu, atol=0.3)

@pytest.mark.filterwarnings("ignore:.*rejection rate for sampling.*:UserWarning")
def test_tmvn_rvs_multivariate_truncation():
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.2],[0.2, 1.0]])
    lower = np.array([-np.inf, -1.0])
    upper = np.array([1.0, 1.0])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=500)
    assert samples.shape == (500, 2)
    assert np.all(samples[:,0] <= 1.0)
    assert np.all(samples[:,1] >= -1.0)
    assert np.all(samples[:,1] <= 1.0)

def test_tmvn_rvs_low_density_raises():
    # Setup truncation such that density inside trunc bounds is very small
    mu = 0.0
    cov = 1.0
    lower = 10.0
    upper = 11.0
    with pytest.raises(ValueError):
        tmvn_rvs(mu, cov, lower=lower, upper=upper, size=10)

@pytest.mark.filterwarnings("ignore:.*not sufficiently high accuracy.*:UserWarning")
def test_tmvn_MLE_univariate_no_truncation():
    np.random.seed(0)
    data = np.random.normal(loc=2.0, scale=1.5, size=100)
    mu, cov = tmvn_MLE(data)
    assert isinstance(mu, float) or (isinstance(mu, np.ndarray) and mu.shape==())
    assert np.isclose(mu, 2.0, atol=0.3)
    assert np.isclose(np.sqrt(cov), 1.5, atol=0.3)

@pytest.mark.filterwarnings("ignore:.*density of the joint probability distribution within the truncation limits is too small.*:UserWarning")
def test_tmvn_MLE_univariate_truncation():
    np.random.seed(1)
    data = np.random.normal(loc=0.5, scale=1.0, size=50)
    tr_lower = np.array([-1.0])
    tr_upper = np.array([2.0])
    mu, cov = tmvn_MLE(data, tr_lower=tr_lower, tr_upper=tr_upper)
    assert isinstance(mu, float) or (isinstance(mu, np.ndarray) and mu.shape==())
    assert -1.0 < mu < 2.0
    assert cov > 0

@pytest.mark.filterwarnings("ignore:.*is not valid.*:UserWarning")
def test_tmvn_MLE_multivariate_no_truncation():
    np.random.seed(2)
    mean = [1.0, 2.0]
    cov_mat = [[1.0, 0.6],[0.6, 2.0]]
    data = np.random.multivariate_normal(mean, cov_mat, size=200).T
    mu, cov = tmvn_MLE(data)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.allclose(mu, mean, atol=0.3)

@pytest.mark.filterwarnings("ignore:.*density of the joint probability distribution within the truncation limits is too small.*:UserWarning")
def test_tmvn_MLE_multivariate_truncation():
    np.random.seed(3)
    mean = [0, 0]
    cov_mat = [[1.0, 0.3],[0.3, 1.0]]
    data = np.random.multivariate_normal(mean, cov_mat, size=150).T
    tr_lower = [-1.0, -0.5]
    tr_upper = [1.5, 1.0]
    mu, cov = tmvn_MLE(data, tr_lower=tr_lower, tr_upper=tr_upper)
    assert mu.shape == (2,)
    assert cov.shape == (2,2)
    assert np.all(mu > -2)
    assert np.all(mu < 2)
    # Covariance matrix should be positive semidefinite
    eigvals = np.linalg.eigvals(cov)
    assert np.all(eigvals >= -1e-12)  # allow tiny negative numerical noise

def test_tmvn_MLE_with_censoring():
    np.random.seed(4)
    # Create data truncated below at 0
    data = np.random.normal(1, 1, 80)
    data = data[data > 0]
    tr_lower = [0]
    tr_upper = [np.inf]
    det_lower = [0]
    det_upper = [np.inf]
    censored_count = 20
    mu, cov = tmvn_MLE(data, tr_lower=tr_lower, tr_upper=tr_upper,
                       censored_count=censored_count,
                       det_lower=det_lower, det_upper=det_upper)
    assert np.isscalar(mu) or (mu.shape==())
    assert cov > 0
    assert mu > 0

@pytest.mark.filterwarnings("ignore:.*less than the prescribed minimum limit.*:UserWarning")
def test_tmvn_MLE_alpha_lim_warning():
    np.random.seed(5)
    data = np.random.normal(10, 1, 50)
    tr_lower = [8]
    tr_upper = [12]
    # Set alpha_lim large to trigger warning and forced high NLL
    mu, cov = tmvn_MLE(data, tr_lower=tr_lower, tr_upper=tr_upper, alpha_lim=0.1)
    assert np.isscalar(mu) or (mu.shape == ())

@pytest.mark.filterwarnings("ignore:.*density of the joint probability distribution within the detection limits is too small.*:UserWarning")
def test_tmvn_MLE_detection_limits_low_density():
    np.random.seed(6)
    data = np.random.normal(1, 2, 50)
    tr_lower = [-1]
    tr_upper = [3]
    det_lower = [10]
    det_upper = [20]
    censored_count = 10
    mu, cov = tmvn_MLE(data, tr_lower=tr_lower, tr_upper=tr_upper,
                       det_lower=det_lower, det_upper=det_upper,
                       censored_count=censored_count)
    assert isinstance(mu, np.ndarray) or np.isscalar(mu)
    assert cov is not None

def test_tmvn_MLE_zero_variance_dimension():
    # Create data with one dimension having zero variance
    data = np.array([[2, 2, 2],[1, 2, 3]])
    mu, cov = tmvn_MLE(data)
    assert mu.shape[0] == 2
    # Check diagonal variances are positive (zero variance replaced)
    assert np.all(np.diag(cov) > 0)

def test_tmvn_MLE_1d_data_as_2d_array():
    data = np.array([[1, 2, 3, 4]])
    mu, cov = tmvn_MLE(data)
    assert np.isscalar(mu) or mu.shape == ()
    assert np.isscalar(cov) or cov.shape == ()

def test_tmvn_rvs_lower_none_upper_none_defaults():
    mu = 0
    cov = 1
    samples = tmvn_rvs(mu, cov, lower=None, upper=None, size=10)
    assert samples.shape[0] == 10

@pytest.mark.filterwarnings("ignore:.*rejection rate for sampling.*:UserWarning")
def test_tmvn_rvs_warning_for_high_rejection():
    mu = 0
    cov = 1
    lower = -0.001
    upper = 0.001
    # Should issue a warning for high rejection rate sampling
    with warnings.catch_warnings(record=True) as w:
        samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=5)
        assert any("The rejection rate for sampling the prescribed" in str(warn.message) for warn in w)

def test_tmvn_rvs_multivariate_edge_case_one_dim():
    mu = np.array([0])
    cov = np.array([[1]])
    lower = np.array([-1])
    upper = np.array([1])
    samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=10)
    assert samples.shape == (10, 1)
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)
