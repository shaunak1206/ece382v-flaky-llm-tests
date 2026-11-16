import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


def _scalar(value):
    """Return a plain float from a scalar or single-element array."""
    return float(np.ravel(value)[0])


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_tmvn_rvs_unbounded_scalar_returns_requested_size():
    np.random.seed(1)
    samples = np.asarray(tmvn_rvs(0.0, 1.0, size=50))
    assert samples.shape[0] == 50


def test_tmvn_rvs_unbounded_multivariate_shape_and_mean():
    np.random.seed(2)
    mu = np.array([1.0, -2.0])
    COV = np.array([[1.0, 0.3], [0.3, 2.0]])
    samples = tmvn_rvs(mu, COV, size=20000)
    assert samples.shape == (20000, 2)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)


def test_tmvn_rvs_lower_bound_respected_and_half_normal_mean():
    # Standard normal truncated to [0, inf): mean is sqrt(2/pi)
    np.random.seed(3)
    samples = tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                       lower=0.0, upper=None, size=5000).reshape(-1)
    assert samples.shape == (5000,)
    assert np.all(samples >= 0.0)
    assert abs(samples.mean() - np.sqrt(2.0 / np.pi)) < 0.03


def test_tmvn_rvs_upper_bound_only_respected():
    # Standard normal truncated to (-inf, 0]: mean is -sqrt(2/pi)
    np.random.seed(4)
    samples = tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                       lower=None, upper=0.0, size=3000).reshape(-1)
    assert np.all(samples <= 0.0)
    assert abs(samples.mean() + np.sqrt(2.0 / np.pi)) < 0.04


def test_tmvn_rvs_box_truncation_respected_multivariate():
    np.random.seed(5)
    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([-1.0, -0.5])
    upper = np.array([1.0, 2.0])
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=2000)
    assert samples.shape == (2000, 2)
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)


@pytest.mark.parametrize("size", [1, 7, 13, 250])
def test_tmvn_rvs_returns_exact_requested_size(size):
    np.random.seed(6)
    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.2], [0.2, 1.0]])
    samples = tmvn_rvs(mu, COV,
                       lower=np.array([-1.0, -1.0]),
                       upper=np.array([1.0, 1.0]),
                       size=size)
    assert samples.shape == (size, 2)


def test_tmvn_rvs_warns_for_very_high_rejection_rate():
    # P(X > 3.2) is about 6.9e-4, which is below the 1e-3 warning threshold
    np.random.seed(7)
    with pytest.warns(UserWarning, match="rejection rate"):
        samples = tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                           lower=3.2, upper=None, size=10).reshape(-1)
    assert samples.shape == (10,)
    assert np.all(samples >= 3.2)


def test_tmvn_rvs_raises_for_negligible_truncated_density():
    # The probability mass above 40 standard deviations is numerically zero
    with pytest.raises(ValueError, match="too small"):
        tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                 lower=40.0, upper=None, size=10)


def test_tmvn_rvs_is_reproducible_with_same_seed():
    mu = np.array([0.0, 1.0])
    COV = np.array([[1.0, 0.4], [0.4, 1.5]])
    lower = np.array([-2.0, -1.0])
    upper = np.array([2.0, 3.0])

    np.random.seed(8)
    first = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=100)
    np.random.seed(8)
    second = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=100)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_tmvn_MLE_untruncated_univariate_recovers_mean_and_variance():
    rng = np.random.RandomState(10)
    x = rng.normal(5.0, 2.0, 3000)
    mu, COV = tmvn_MLE(x)
    assert abs(_scalar(mu) - 5.0) < 0.15
    assert abs(_scalar(COV) - 4.0) < 0.4


def test_tmvn_MLE_untruncated_univariate_variance_matches_biased_estimate():
    rng = np.random.RandomState(11)
    x = rng.normal(5.0, 2.0, 3000)
    _, COV = tmvn_MLE(x)
    assert np.isclose(_scalar(COV), np.var(x), rtol=0.03)


def test_tmvn_MLE_multivariate_output_shapes():
    rng = np.random.RandomState(12)
    samples = rng.multivariate_normal([0.0, 1.0],
                                      [[1.0, 0.2], [0.2, 2.0]],
                                      size=200).T
    mu, COV = tmvn_MLE(samples)
    assert np.asarray(mu).shape == (2,)
    assert np.asarray(COV).shape == (2, 2)
    assert np.allclose(COV, COV.T)


def test_tmvn_MLE_multivariate_recovers_means_and_variances():
    rng = np.random.RandomState(13)
    true_mu = np.array([1.0, -2.0])
    true_cov = np.array([[1.0, 1.8], [1.8, 9.0]])  # sd 1 and 3, corr 0.6
    samples = rng.multivariate_normal(true_mu, true_cov, size=4000).T

    mu, COV = tmvn_MLE(samples)
    mu = np.asarray(mu)
    COV = np.asarray(COV)

    assert np.allclose(mu, true_mu, atol=0.15)
    assert np.allclose(np.diag(COV), np.diag(true_cov), rtol=0.1)
    corr = COV / np.outer(np.sqrt(np.diag(COV)), np.sqrt(np.diag(COV)))
    assert abs(corr[0, 1] - 0.6) < 0.05


def test_tmvn_MLE_multivariate_correlation_equals_sample_correlation():
    # The correlation matrix is not fitted; it is kept at the sample values
    rng = np.random.RandomState(14)
    samples = rng.multivariate_normal([0.0, 0.0, 0.0],
                                      [[1.0, 0.5, 0.1],
                                       [0.5, 1.0, -0.3],
                                       [0.1, -0.3, 2.0]],
                                      size=1000).T
    _, COV = tmvn_MLE(samples)
    COV = np.asarray(COV)
    sig = np.sqrt(np.diag(COV))
    corr = COV / np.outer(sig, sig)
    assert np.allclose(corr, np.corrcoef(samples), atol=1e-6)


def test_tmvn_MLE_truncated_univariate_recovers_underlying_normal():
    # Observations are a standard normal truncated to [-1, 1]
    rng = np.random.RandomState(15)
    x = rng.normal(0.0, 1.0, 6000)
    obs = x[(x > -1.0) & (x < 1.0)]

    mu, COV = tmvn_MLE(obs, tr_lower=-1.0, tr_upper=1.0)
    assert abs(_scalar(mu)) < 0.1
    assert abs(_scalar(COV) - 1.0) < 0.2


def test_tmvn_MLE_censored_univariate_recovers_underlying_normal():
    # Standard normal censored above at 1.0; values above are replaced by a
    # count of censored samples
    rng = np.random.RandomState(16)
    x = rng.normal(0.0, 1.0, 4000)
    censored = x > 1.0
    observed = x[~censored]
    censored_count = int(censored.sum())

    mu, COV = tmvn_MLE(observed,
                       censored_count=censored_count,
                       det_lower=-np.inf, det_upper=1.0)
    assert abs(_scalar(mu)) < 0.1
    assert abs(_scalar(COV) - 1.0) < 0.2


def test_tmvn_MLE_accepts_list_input_same_as_array():
    rng = np.random.RandomState(17)
    x = rng.normal(2.0, 0.5, 500)

    mu_arr, COV_arr = tmvn_MLE(x)
    mu_lst, COV_lst = tmvn_MLE(list(x))

    assert np.allclose(mu_arr, mu_lst)
    assert np.allclose(COV_arr, COV_lst)


def test_tmvn_MLE_is_deterministic_for_same_input():
    rng = np.random.RandomState(18)
    samples = rng.multivariate_normal([0.0, 1.0],
                                      [[1.0, 0.3], [0.3, 1.0]],
                                      size=300).T
    mu1, COV1 = tmvn_MLE(samples)
    mu2, COV2 = tmvn_MLE(samples)
    assert np.allclose(mu1, mu2)
    assert np.allclose(COV1, COV2)


def test_tmvn_MLE_constant_univariate_samples_give_finite_result():
    # Zero variance is replaced by a negligible variance, so the fitted
    # mean is the constant and the fitted variance is tiny
    samples = np.full(50, 3.0)
    mu, COV = tmvn_MLE(samples)
    assert np.isfinite(_scalar(mu))
    assert np.isfinite(_scalar(COV))
    assert abs(_scalar(mu) - 3.0) < 1e-4
    assert 0.0 < _scalar(COV) < 1e-6


def test_tmvn_MLE_truncated_multivariate_returns_valid_covariance():
    # Loose check that the multivariate truncated fit runs and returns
    # a finite, symmetric, positive definite covariance matrix
    np.random.seed(20)
    samples = tmvn_rvs(np.array([0.0, 0.0]),
                       np.array([[1.0, 0.5], [0.5, 1.0]]),
                       lower=np.array([-1.0, -1.0]),
                       upper=np.array([2.0, 2.0]),
                       size=1500)
    mu, COV = tmvn_MLE(samples.T,
                       tr_lower=np.array([-1.0, -1.0]),
                       tr_upper=np.array([2.0, 2.0]))
    mu = np.asarray(mu)
    COV = np.asarray(COV)
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(COV))
    assert np.allclose(COV, COV.T)
    assert np.all(np.linalg.eigvalsh(COV) > 0.0)
