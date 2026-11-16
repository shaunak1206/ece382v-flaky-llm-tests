import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(12345)


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_untruncated_univariate_shape():
    samples = tmvn_rvs(0.0, 1.0, size=100)
    assert samples.shape == (100,)


def test_rvs_untruncated_univariate_moments():
    samples = tmvn_rvs(5.0, 4.0, size=5000)
    assert np.mean(samples) == pytest.approx(5.0, abs=0.15)
    assert np.var(samples) == pytest.approx(4.0, rel=0.15)


def test_rvs_untruncated_multivariate_shape_and_moments():
    mu = np.array([1.0, -2.0])
    COV = np.array([[1.0, 0.5], [0.5, 2.0]])
    samples = tmvn_rvs(mu, COV, size=5000)
    assert samples.shape == (5000, 2)
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.1)
    assert np.allclose(np.cov(samples.T), COV, atol=0.2)


def test_rvs_list_inputs_univariate():
    samples = tmvn_rvs([0.0], [[1.0]], size=50)
    assert samples.shape == (50,)


def test_rvs_reproducible_with_seed():
    np.random.seed(7)
    a = tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=50)
    np.random.seed(7)
    b = tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=50)
    assert np.array_equal(a, b)


def test_rvs_truncated_univariate_within_limits_and_size():
    samples = tmvn_rvs(0.0, 1.0, lower=-0.5, upper=1.5, size=200)
    assert samples.shape == (200,)
    assert np.all(samples > -0.5)
    assert np.all(samples < 1.5)


def test_rvs_lower_only_truncation_half_normal_mean():
    samples = tmvn_rvs(0.0, 1.0, lower=0.0, size=4000)
    assert samples.shape == (4000,)
    assert np.all(samples > 0.0)
    # mean of a half-normal distribution is sqrt(2/pi)
    assert np.mean(samples) == pytest.approx(np.sqrt(2.0 / np.pi), abs=0.05)


def test_rvs_upper_only_truncation():
    samples = tmvn_rvs(0.0, 1.0, upper=0.0, size=1000)
    assert samples.shape == (1000,)
    assert np.all(samples < 0.0)
    assert np.mean(samples) < 0.0


def test_rvs_symmetric_truncation_moments():
    samples = tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=4000)
    assert np.mean(samples) == pytest.approx(0.0, abs=0.05)
    # variance of N(0,1) truncated at +-1 is about 0.291
    assert 0.25 < np.var(samples) < 0.33


def test_rvs_truncated_multivariate_within_limits():
    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.3], [0.3, 1.0]])
    lower = np.array([-1.0, -0.5])
    upper = np.array([2.0, 1.0])
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=300)
    assert samples.shape == (300, 2)
    assert np.all(samples[:, 0] > -1.0) and np.all(samples[:, 0] < 2.0)
    assert np.all(samples[:, 1] > -0.5) and np.all(samples[:, 1] < 1.0)


def test_rvs_truncated_multivariate_with_infinite_limits():
    mu = np.array([1.0, 1.0])
    COV = np.eye(2)
    lower = np.array([0.0, -np.inf])
    upper = np.array([np.inf, np.inf])
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=500)
    assert samples.shape == (500, 2)
    assert np.all(samples[:, 0] > 0.0)


def test_rvs_truncation_preserves_positive_correlation():
    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples = tmvn_rvs(mu, COV, lower=np.array([-1.0, -1.0]),
                       upper=np.array([1.0, 1.0]), size=3000)
    corr = np.corrcoef(samples.T)[0, 1]
    assert corr > 0.3


def test_rvs_negligible_density_raises():
    with pytest.raises(ValueError):
        tmvn_rvs(0.0, 1.0, lower=50.0, upper=60.0, size=10)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_untruncated_recovers_moments():
    data = np.random.normal(5.0, 2.0, size=500)
    mu, COV = tmvn_MLE(data)
    assert float(mu) == pytest.approx(np.mean(data), abs=0.1)
    assert float(COV) == pytest.approx(np.var(data), rel=0.1)


def test_mle_accepts_list_input():
    data = list(np.random.normal(1.0, 0.5, size=200))
    mu, COV = tmvn_MLE(data)
    assert np.isfinite(float(mu))
    assert float(COV) > 0.0


def test_mle_multivariate_untruncated_shapes_and_values():
    true_mu = np.array([1.0, -3.0])
    true_COV = np.array([[1.0, 0.6], [0.6, 4.0]])
    data = np.random.multivariate_normal(true_mu, true_COV, size=500).T
    mu, COV = tmvn_MLE(data)
    assert np.shape(mu) == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(COV, COV.T)
    assert np.allclose(mu, np.mean(data, axis=1), atol=0.3)
    assert np.allclose(np.diag(COV), np.var(data, axis=1), rtol=0.2)


def test_mle_multivariate_preserves_sample_correlation():
    true_COV = np.array([[1.0, 0.7], [0.7, 1.0]])
    data = np.random.multivariate_normal([0.0, 0.0], true_COV, size=300).T
    mu, COV = tmvn_MLE(data)
    sig = np.sqrt(np.diag(COV))
    corr = COV / np.outer(sig, sig)
    assert corr[0, 1] == pytest.approx(np.corrcoef(data)[0, 1], abs=1e-6)
    assert np.allclose(np.diag(corr), 1.0)


def test_mle_truncated_univariate_corrects_bias():
    np.random.seed(3)
    data = tmvn_rvs(1.0, 1.0, lower=0.0, upper=np.inf, size=2000)
    mu, COV = tmvn_MLE(data, tr_lower=0.0, tr_upper=np.inf)
    sample_mean = np.mean(data)
    assert sample_mean > 1.1
    assert abs(float(mu) - 1.0) < abs(sample_mean - 1.0)
    assert float(mu) == pytest.approx(1.0, abs=0.3)
    assert float(COV) > 0.0


def test_mle_truncated_with_alpha_lim_returns_finite():
    np.random.seed(4)
    data = tmvn_rvs(1.0, 1.0, lower=0.0, upper=np.inf, size=300)
    mu, COV = tmvn_MLE(data, tr_lower=0.0, tr_upper=np.inf, alpha_lim=1e-4)
    assert np.isfinite(float(mu))
    assert np.isfinite(float(COV))
    assert float(COV) > 0.0


def test_mle_censored_univariate():
    np.random.seed(5)
    raw = np.random.normal(0.0, 1.0, size=1000)
    limit = 1.5
    kept = raw[raw < limit]
    censored = int(np.sum(raw >= limit))
    mu, COV = tmvn_MLE(kept, censored_count=censored,
                       det_lower=-np.inf, det_upper=limit)
    assert float(mu) == pytest.approx(0.0, abs=0.25)
    assert float(COV) == pytest.approx(1.0, abs=0.4)


def test_mle_multivariate_truncated_structure():
    np.random.seed(6)
    mu_true = np.array([1.0, 1.0])
    COV_true = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([0.0, 0.0])
    upper = np.array([np.inf, np.inf])
    data = tmvn_rvs(mu_true, COV_true, lower=lower, upper=upper, size=400).T
    mu, COV = tmvn_MLE(data, tr_lower=lower, tr_upper=upper)
    assert np.shape(mu) == (2,)
    assert COV.shape == (2, 2)
    assert np.all(np.diag(COV) > 0.0)
    assert np.allclose(COV, COV.T)
    assert np.all(np.abs(mu - mu_true) < 0.6)


def test_mle_constant_samples_zero_variance():
    data = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    mu, COV = tmvn_MLE(data)
    assert float(mu) == pytest.approx(3.0, abs=1e-3)
    assert 0.0 <= float(COV) < 1e-6


def test_mle_single_sample():
    mu, COV = tmvn_MLE(np.array([2.0]))
    assert float(mu) == pytest.approx(2.0, abs=1e-3)
    assert np.isfinite(float(COV))
