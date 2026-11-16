import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_tmvn_rvs_untruncated_univariate_moments():
    np.random.seed(0)
    samples = tmvn_rvs(mu=[1.0], COV=[[4.0]], size=5000)
    assert np.asarray(samples).size == 5000
    assert abs(np.mean(samples) - 1.0) < 0.15
    assert abs(np.var(samples) - 4.0) < 0.5


def test_tmvn_rvs_scalar_mu_and_cov():
    np.random.seed(1)
    samples = tmvn_rvs(mu=2.0, COV=1.0, size=2000)
    assert np.asarray(samples).size == 2000
    assert abs(np.mean(samples) - 2.0) < 0.15


def test_tmvn_rvs_univariate_truncated_within_limits():
    np.random.seed(2)
    samples = np.asarray(tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0,
                                  size=500))
    assert samples.size == 500
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


@pytest.mark.parametrize("size", [1, 5, 37, 200])
def test_tmvn_rvs_returns_exact_size_when_truncated(size):
    np.random.seed(3)
    samples = np.asarray(tmvn_rvs(mu=0.0, COV=1.0, lower=-0.5, upper=1.5,
                                  size=size))
    assert samples.shape[0] == size
    assert np.all(samples > -0.5) and np.all(samples < 1.5)


def test_tmvn_rvs_lower_only_gives_half_normal_mean():
    np.random.seed(4)
    samples = np.asarray(tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, size=5000))
    assert samples.size == 5000
    assert np.all(samples > 0.0)
    # mean of a half normal is sqrt(2/pi) ~ 0.798
    assert abs(np.mean(samples) - np.sqrt(2.0 / np.pi)) < 0.08


def test_tmvn_rvs_upper_only_respects_limit():
    np.random.seed(5)
    samples = np.asarray(tmvn_rvs(mu=0.0, COV=1.0, upper=-0.5, size=1000))
    assert samples.size == 1000
    assert np.all(samples < -0.5)


def test_tmvn_rvs_multivariate_truncated_within_limits():
    np.random.seed(6)
    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([-1.0, -1.5])
    upper = np.array([1.0, 1.5])
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=1000)
    assert samples.shape == (1000, 2)
    assert np.all(samples[:, 0] > -1.0) and np.all(samples[:, 0] < 1.0)
    assert np.all(samples[:, 1] > -1.5) and np.all(samples[:, 1] < 1.5)
    # symmetric truncation around the mean keeps the mean at zero
    assert np.all(np.abs(np.mean(samples, axis=0)) < 0.1)
    # positive correlation is preserved after truncation
    assert np.corrcoef(samples.T)[0, 1] > 0.2


def test_tmvn_rvs_multivariate_untruncated_covariance():
    np.random.seed(7)
    mu = np.array([1.0, -1.0])
    COV = np.array([[2.0, 0.8], [0.8, 1.0]])
    samples = tmvn_rvs(mu, COV, size=8000)
    assert samples.shape == (8000, 2)
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.1)
    assert np.allclose(np.cov(samples.T), COV, atol=0.15)


def test_tmvn_rvs_multivariate_infinite_limits_in_one_dimension():
    np.random.seed(8)
    mu = np.array([0.0, 0.0])
    COV = np.eye(2)
    lower = np.array([0.0, -np.inf])
    upper = np.array([np.inf, np.inf])
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=1500)
    assert samples.shape == (1500, 2)
    assert np.all(samples[:, 0] > 0.0)
    # the unbounded dimension is still centered on zero
    assert abs(np.mean(samples[:, 1])) < 0.12


def test_tmvn_rvs_reproducible_with_seed():
    mu = np.array([0.0, 1.0])
    COV = np.array([[1.0, 0.3], [0.3, 1.0]])
    np.random.seed(123)
    a = tmvn_rvs(mu, COV, lower=[-1.0, 0.0], upper=[1.0, 2.0], size=50)
    np.random.seed(123)
    b = tmvn_rvs(mu, COV, lower=[-1.0, 0.0], upper=[1.0, 2.0], size=50)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_tmvn_MLE_univariate_untruncated_matches_sample_moments():
    np.random.seed(10)
    data = np.random.normal(5.0, 2.0, size=1000)
    mu, COV = tmvn_MLE(data)
    assert abs(mu - np.mean(data)) < 0.1
    assert COV == pytest.approx(np.var(data), rel=0.1)
    assert COV > 0.0


def test_tmvn_MLE_multivariate_untruncated_matches_sample_moments():
    np.random.seed(11)
    true_mu = [1.0, -2.0]
    true_cov = [[1.0, 0.6], [0.6, 2.0]]
    data = np.random.multivariate_normal(true_mu, true_cov, size=600).T
    mu, COV = tmvn_MLE(data)
    assert np.asarray(mu).shape == (2,)
    assert np.asarray(COV).shape == (2, 2)
    assert np.allclose(mu, np.mean(data, axis=1), atol=0.1)
    assert np.allclose(COV, np.cov(data), atol=0.2)
    assert np.allclose(COV, COV.T)


def test_tmvn_MLE_multivariate_preserves_sample_correlation():
    np.random.seed(12)
    data = np.random.multivariate_normal([0.0, 0.0],
                                         [[1.0, 0.7], [0.7, 1.0]],
                                         size=400).T
    mu, COV = tmvn_MLE(data)
    sig = np.sqrt(np.diag(COV))
    corr = COV / np.outer(sig, sig)
    assert corr[0, 1] == pytest.approx(np.corrcoef(data)[0, 1], abs=1e-6)
    assert np.allclose(np.diag(corr), 1.0)


def test_tmvn_MLE_univariate_truncated_recovers_wider_distribution():
    np.random.seed(13)
    samples = np.asarray(tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0,
                                  size=2000))
    mu, COV = tmvn_MLE(samples, tr_lower=-1.0, tr_upper=1.0)
    assert abs(mu) < 0.3
    # truncation reduces the sample variance; the MLE should compensate
    assert COV > np.var(samples)
    assert 0.5 < COV < 2.0


def test_tmvn_MLE_censored_data_upper_detection_limit():
    np.random.seed(14)
    full = np.random.normal(0.0, 1.0, size=2000)
    limit = 1.5
    observed = full[full < limit]
    censored_count = int(np.sum(full >= limit))
    assert censored_count > 0
    mu, COV = tmvn_MLE(observed, censored_count=censored_count,
                       det_lower=-np.inf, det_upper=limit)
    assert abs(mu - 0.0) < 0.2
    assert 0.7 < COV < 1.4
    # accounting for censoring should inflate variance relative to observed
    assert COV > np.var(observed)


def test_tmvn_MLE_constant_samples_give_negligible_variance():
    mu, COV = tmvn_MLE(np.array([3.0, 3.0, 3.0, 3.0, 3.0]))
    assert mu == pytest.approx(3.0, abs=1e-4)
    assert 0.0 <= COV < 1e-6


def test_tmvn_MLE_single_sample_returns_that_sample():
    mu, COV = tmvn_MLE(np.array([2.0]))
    assert mu == pytest.approx(2.0, abs=1e-4)
    assert 0.0 <= COV < 1e-6


def test_tmvn_MLE_accepts_list_input():
    np.random.seed(15)
    data = np.random.normal(1.0, 0.5, size=200)
    mu_a, COV_a = tmvn_MLE(data)
    mu_l, COV_l = tmvn_MLE(data.tolist())
    assert mu_a == pytest.approx(mu_l)
    assert COV_a == pytest.approx(COV_l)


def test_tmvn_MLE_affine_equivariance():
    np.random.seed(16)
    data = np.random.normal(0.0, 1.0, size=500)
    mu1, COV1 = tmvn_MLE(data)
    mu2, COV2 = tmvn_MLE(10.0 * data + 100.0)
    assert mu2 == pytest.approx(10.0 * mu1 + 100.0, rel=1e-3)
    assert COV2 == pytest.approx(100.0 * COV1, rel=1e-3)


def test_tmvn_MLE_alpha_lim_gives_finite_valid_estimates():
    np.random.seed(17)
    samples = np.asarray(tmvn_rvs(mu=0.0, COV=1.0, lower=-1.5, upper=1.5,
                                  size=500))
    mu, COV = tmvn_MLE(samples, tr_lower=-1.5, tr_upper=1.5, alpha_lim=1e-4)
    assert np.isfinite(mu)
    assert np.isfinite(COV)
    assert COV > 0.0
    assert abs(mu) < 0.4


def test_tmvn_MLE_multivariate_truncated():
    np.random.seed(18)
    mu_true = np.array([0.0, 0.0])
    COV_true = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([-1.5, -1.5])
    upper = np.array([1.5, 1.5])
    samples = tmvn_rvs(mu_true, COV_true, lower=lower, upper=upper, size=400)
    mu, COV = tmvn_MLE(samples.T, tr_lower=lower, tr_upper=upper)
    assert np.asarray(mu).shape == (2,)
    assert np.asarray(COV).shape == (2, 2)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(COV))
    assert np.all(np.abs(mu) < 0.5)
    assert np.all(np.diag(COV) > 0.0)
    assert np.allclose(COV, COV.T)
