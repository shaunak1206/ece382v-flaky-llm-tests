import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(12345)
    yield


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_tmvn_rvs_univariate_no_bounds_shape_and_moments():
    samples = tmvn_rvs(mu=2.0, COV=4.0, size=5000)
    assert samples.shape == (5000,)
    assert abs(np.mean(samples) - 2.0) < 0.15
    assert abs(np.std(samples) - 2.0) < 0.15


def test_tmvn_rvs_multivariate_no_bounds_shape_and_moments():
    mu = np.array([0.0, 1.0])
    COV = np.array([[1.0, 0.6], [0.6, 2.0]])
    samples = tmvn_rvs(mu, COV, size=5000)
    assert samples.shape == (5000, 2)
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.15)
    assert np.allclose(np.cov(samples, rowvar=False), COV, atol=0.2)


def test_tmvn_rvs_univariate_two_sided_truncation_respected():
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=1000)
    assert samples.shape == (1000,)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_tmvn_rvs_truncated_symmetric_mean_and_reduced_variance():
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=5000)
    assert abs(np.mean(samples)) < 0.1
    # the variance of a standard normal truncated at +-1 is about 0.29
    assert np.var(samples) < 0.4
    assert np.var(samples) > 0.2


def test_tmvn_rvs_non_zero_mean_scalar_truncation():
    samples = tmvn_rvs(mu=5.0, COV=4.0, lower=4.0, upper=8.0, size=800)
    assert samples.shape == (800,)
    assert samples.min() > 4.0
    assert samples.max() < 8.0


def test_tmvn_rvs_lower_bound_only_half_normal():
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, size=5000)
    assert samples.shape == (5000,)
    assert np.all(samples > 0.0)
    # mean of a half normal is sqrt(2/pi) ~ 0.798
    assert abs(np.mean(samples) - np.sqrt(2.0 / np.pi)) < 0.06


def test_tmvn_rvs_upper_bound_only():
    samples = tmvn_rvs(mu=0.0, COV=1.0, upper=0.0, size=2000)
    assert samples.shape == (2000,)
    assert np.all(samples < 0.0)
    assert np.mean(samples) < -0.6


def test_tmvn_rvs_multivariate_truncation_respected():
    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = [-1.0, -0.5]
    upper = [1.5, 1.0]
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=400)
    assert samples.shape == (400, 2)
    assert np.all(samples[:, 0] > -1.0) and np.all(samples[:, 0] < 1.5)
    assert np.all(samples[:, 1] > -0.5) and np.all(samples[:, 1] < 1.0)


def test_tmvn_rvs_multivariate_infinite_bounds_in_some_dimensions():
    mu = np.array([0.0, 0.0])
    COV = np.eye(2)
    lower = np.array([-np.inf, 0.0])
    upper = np.array([np.inf, np.inf])
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=500)
    assert samples.shape == (500, 2)
    assert np.all(samples[:, 1] > 0.0)
    # the unbounded dimension should still take negative values
    assert np.any(samples[:, 0] < 0.0)


def test_tmvn_rvs_truncation_preserves_positive_correlation():
    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples = tmvn_rvs(mu, COV, lower=[-2.0, -2.0], upper=[2.0, 2.0],
                       size=3000)
    corr = np.corrcoef(samples, rowvar=False)[0, 1]
    assert corr > 0.5


def test_tmvn_rvs_returns_exact_requested_size_for_various_sizes():
    for size in (1, 2, 7, 50):
        samples = tmvn_rvs(mu=0.0, COV=1.0, lower=-0.5, upper=0.5, size=size)
        assert samples.shape[0] == size


def test_tmvn_rvs_reproducible_with_seed():
    np.random.seed(7)
    a = tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=50)
    np.random.seed(7)
    b = tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=50)
    assert np.array_equal(a, b)


def test_tmvn_rvs_raises_when_density_in_limits_is_negligible():
    with pytest.raises(ValueError):
        tmvn_rvs(mu=0.0, COV=1.0, lower=40.0, upper=41.0, size=5)


def test_tmvn_rvs_warns_for_very_low_acceptance_rate():
    with pytest.warns(UserWarning):
        samples = tmvn_rvs(mu=0.0, COV=1.0, lower=3.5, size=3)
    assert samples.shape == (3,)
    assert np.all(samples > 3.5)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_tmvn_MLE_univariate_no_truncation_matches_sample_moments():
    data = np.random.normal(3.0, 2.0, size=500)
    mu, COV = tmvn_MLE(data)
    mu = float(np.squeeze(mu))
    COV = float(np.squeeze(COV))
    assert abs(mu - np.mean(data)) < 0.1 * np.std(data)
    assert COV == pytest.approx(np.var(data), rel=0.15)


def test_tmvn_MLE_accepts_list_input():
    data = list(np.random.normal(-1.0, 0.5, size=200))
    mu, COV = tmvn_MLE(data)
    assert np.isfinite(np.squeeze(mu))
    assert np.squeeze(COV) > 0.0
    assert abs(float(np.squeeze(mu)) - np.mean(data)) < 0.1


def test_tmvn_MLE_multivariate_no_truncation_shapes_and_moments():
    true_mu = np.array([1.0, -2.0])
    true_COV = np.array([[1.0, 0.7], [0.7, 4.0]])
    data = np.random.multivariate_normal(true_mu, true_COV, size=1000).T
    mu, COV = tmvn_MLE(data)

    assert np.asarray(mu).shape == (2,)
    assert np.asarray(COV).shape == (2, 2)
    assert np.allclose(COV, np.transpose(COV))
    assert np.allclose(mu, np.mean(data, axis=1), atol=0.1)
    assert np.allclose(np.diag(COV), np.var(data, axis=1), rtol=0.2)

    # correlation is taken directly from the samples (rho is not fitted)
    sig = np.sqrt(np.diag(COV))
    rho = COV / np.outer(sig, sig)
    assert np.allclose(rho, np.corrcoef(data), atol=1e-6)


def test_tmvn_MLE_univariate_truncated_data_gives_larger_variance_than_sample():
    raw = tmvn_rvs(mu=0.0, COV=1.0, lower=-1.5, upper=1.5, size=2000)
    mu, COV = tmvn_MLE(raw, tr_lower=-1.5, tr_upper=1.5)
    mu = float(np.squeeze(mu))
    COV = float(np.squeeze(COV))
    assert np.isfinite(mu) and np.isfinite(COV)
    assert abs(mu) < 1.0
    # correcting for truncation must inflate the variance of the fit
    assert COV > np.var(raw)


def test_tmvn_MLE_censored_univariate_recovers_underlying_parameters():
    full = np.random.normal(0.0, 1.0, size=1500)
    det_lower, det_upper = -1.5, 1.5
    inside = full[(full > det_lower) & (full < det_upper)]
    censored_count = len(full) - len(inside)
    assert censored_count > 0

    mu, COV = tmvn_MLE(inside, censored_count=censored_count,
                       det_lower=det_lower, det_upper=det_upper)
    mu = float(np.squeeze(mu))
    COV = float(np.squeeze(COV))
    assert abs(mu) < 0.25
    assert 0.7 < COV < 1.4
    assert COV > np.var(inside)


def test_tmvn_MLE_multivariate_truncated_runs_and_returns_valid_covariance():
    mu_true = np.array([0.0, 0.0])
    COV_true = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([-3.0, -3.0])
    upper = np.array([3.0, 3.0])
    raw = tmvn_rvs(mu_true, COV_true, lower=lower, upper=upper, size=300).T

    mu, COV = tmvn_MLE(raw, tr_lower=lower, tr_upper=upper)
    assert np.asarray(mu).shape == (2,)
    assert np.asarray(COV).shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(COV))
    assert np.all(np.linalg.eigvalsh(COV) > 0.0)
    assert np.allclose(mu, 0.0, atol=0.5)


def test_tmvn_MLE_constant_samples_returns_mean_with_negligible_variance():
    data = np.full(10, 5.0)
    mu, COV = tmvn_MLE(data)
    assert float(np.squeeze(mu)) == pytest.approx(5.0, abs=1e-3)
    assert float(np.squeeze(COV)) < 1e-6


def test_tmvn_MLE_single_sample_returns_sample_value():
    mu, COV = tmvn_MLE(np.array([2.0]))
    assert float(np.squeeze(mu)) == pytest.approx(2.0)
    assert float(np.squeeze(COV)) < 1e-6
