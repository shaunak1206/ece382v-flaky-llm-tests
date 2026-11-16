import numpy as np
import pytest
from scipy.stats import truncnorm

from pelicun.uq import tmvn_rvs, tmvn_MLE


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_tmvn_rvs_unbounded_univariate_size():
    np.random.seed(0)
    samples = tmvn_rvs(0.0, 1.0, size=100)
    assert np.asarray(samples).size == 100


def test_tmvn_rvs_unbounded_multivariate_shape():
    np.random.seed(1)
    samples = tmvn_rvs([0.0, 1.0], np.eye(2), size=500)
    assert np.asarray(samples).shape == (500, 2)


def test_tmvn_rvs_unbounded_moments_match_target():
    np.random.seed(2)
    samples = np.ravel(tmvn_rvs(2.0, 4.0, size=20000))
    assert np.mean(samples) == pytest.approx(2.0, abs=0.1)
    assert np.var(samples, ddof=1) == pytest.approx(4.0, rel=0.05)


def test_tmvn_rvs_univariate_two_sided_bounds_respected():
    np.random.seed(3)
    samples = np.ravel(tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=1000))
    assert samples.size == 1000
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_tmvn_rvs_multivariate_bounds_respected():
    np.random.seed(4)
    samples = tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2),
                       lower=[-1.0, -1.0], upper=[1.0, 1.0], size=200)
    samples = np.asarray(samples)
    assert samples.shape == (200, 2)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_tmvn_rvs_truncated_moments_match_truncnorm():
    np.random.seed(5)
    samples = np.ravel(tmvn_rvs(0.0, 1.0, lower=-1.0, upper=2.0,
                                size=20000))
    expected_mean = truncnorm.mean(-1.0, 2.0)
    expected_std = truncnorm.std(-1.0, 2.0)
    assert np.mean(samples) == pytest.approx(expected_mean, abs=0.02)
    assert np.std(samples) == pytest.approx(expected_std, abs=0.02)


def test_tmvn_rvs_returns_exact_size_when_rejection_needed():
    np.random.seed(6)
    samples = np.ravel(tmvn_rvs(0.0, 1.0, lower=-0.5, upper=0.5, size=37))
    assert samples.size == 37
    assert np.all(samples > -0.5)
    assert np.all(samples < 0.5)


def test_tmvn_rvs_negligible_density_raises():
    # Truncation window far in the tail: the density is numerically zero
    with pytest.raises(ValueError):
        tmvn_rvs(0.0, 1.0, lower=50.0, upper=51.0, size=10)


def test_tmvn_rvs_high_rejection_rate_warns():
    # Probability mass in [3.0, 3.2] is about 6.6e-4, below the 1e-3 threshold
    np.random.seed(7)
    with pytest.warns(UserWarning):
        samples = np.ravel(tmvn_rvs(0.0, 1.0, lower=3.0, upper=3.2, size=10))
    assert samples.size == 10
    assert np.all(samples > 3.0)
    assert np.all(samples < 3.2)


def test_tmvn_rvs_reproducible_with_seed():
    np.random.seed(8)
    first = np.ravel(tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=50))
    np.random.seed(8)
    second = np.ravel(tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=50))
    np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_tmvn_MLE_univariate_unbounded_recovers_parameters():
    np.random.seed(11)
    samples = np.random.normal(loc=2.0, scale=3.0, size=2000)
    mu, COV = tmvn_MLE(samples)
    mu = float(np.ravel(mu)[0])
    var = float(np.ravel(COV)[0])
    assert mu == pytest.approx(2.0, abs=0.2)
    assert var == pytest.approx(9.0, rel=0.1)


def test_tmvn_MLE_univariate_zero_variance_does_not_fail():
    samples = np.full(20, 5.0)
    mu, COV = tmvn_MLE(samples)
    mu = float(np.ravel(mu)[0])
    var = float(np.ravel(COV)[0])
    assert np.isfinite(mu) and np.isfinite(var)
    assert mu == pytest.approx(5.0, abs=1e-3)
    assert var < 1e-3


def test_tmvn_MLE_multivariate_output_shapes():
    np.random.seed(12)
    samples = np.random.randn(3, 200)
    mu, COV = tmvn_MLE(samples)
    assert np.asarray(mu).shape == (3,)
    assert np.asarray(COV).shape == (3, 3)


def test_tmvn_MLE_multivariate_correlation_and_variance_match_samples():
    np.random.seed(13)
    true_cov = np.array([[1.0, 0.6], [0.6, 2.0]])
    samples = np.random.multivariate_normal([1.0, -1.0], true_cov,
                                            size=2000).T
    mu, COV = tmvn_MLE(samples)
    COV = np.asarray(COV)

    # correlation is taken from the sample correlation (not fitted)
    sample_corr = np.corrcoef(samples)[0, 1]
    model_corr = COV[0, 1] / np.sqrt(COV[0, 0] * COV[1, 1])
    assert model_corr == pytest.approx(sample_corr, abs=1e-6)

    # covariance is symmetric
    assert np.allclose(COV, COV.T)

    # variances are close to the unbiased sample variances
    sample_var = np.var(samples, axis=1, ddof=1)
    assert np.diag(COV) == pytest.approx(sample_var, rel=0.1)

    # means are close to the true means
    assert np.asarray(mu) == pytest.approx([1.0, -1.0], abs=0.2)


def test_tmvn_MLE_univariate_truncated_recovers_parameters():
    np.random.seed(14)
    samples = truncnorm.rvs(a=-1.0, b=2.0, loc=0.0, scale=1.0, size=3000)
    mu, COV = tmvn_MLE(samples, tr_lower=-1.0, tr_upper=2.0)
    mu = float(np.ravel(mu)[0])
    var = float(np.ravel(COV)[0])
    assert mu == pytest.approx(0.0, abs=0.1)
    assert var == pytest.approx(1.0, rel=0.15)


def test_tmvn_MLE_univariate_censored_recovers_parameters():
    np.random.seed(15)
    x = np.random.normal(0.0, 1.0, size=3000)
    observed = x[(x >= -1.0) & (x <= 1.5)]
    censored = x.size - observed.size
    assert censored > 0

    mu, COV = tmvn_MLE(observed, det_lower=-1.0, det_upper=1.5,
                       censored_count=censored)
    mu = float(np.ravel(mu)[0])
    var = float(np.ravel(COV)[0])
    assert mu == pytest.approx(0.0, abs=0.1)
    assert var == pytest.approx(1.0, rel=0.15)


def test_tmvn_MLE_one_sided_truncation_is_ignored_unless_both_bounds_set():
    # Only one truncation limit is provided, so no truncation is applied
    np.random.seed(16)
    samples = np.random.normal(1.0, 2.0, size=500)
    mu_full, COV_full = tmvn_MLE(samples)
    mu_one, COV_one = tmvn_MLE(samples, tr_lower=None, tr_upper=2.0)
    np.testing.assert_allclose(mu_one, mu_full)
    np.testing.assert_allclose(COV_one, COV_full)


def test_tmvn_MLE_does_not_modify_input():
    np.random.seed(17)
    samples = np.random.normal(0.0, 1.0, size=(2, 100))
    original = samples.copy()
    tmvn_MLE(samples)
    np.testing.assert_array_equal(samples, original)


def test_tmvn_MLE_scale_equivariance():
    np.random.seed(18)
    samples = np.random.normal(0.5, 1.5, size=400)
    mu1, COV1 = tmvn_MLE(samples)
    mu2, COV2 = tmvn_MLE(10.0 * samples)
    np.testing.assert_allclose(mu2, 10.0 * np.asarray(mu1),
                               rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(COV2, 100.0 * np.asarray(COV1), rtol=1e-3)


def test_tmvn_MLE_list_input_treated_as_univariate():
    np.random.seed(19)
    samples = list(np.random.normal(0.0, 1.0, size=200))
    mu, COV = tmvn_MLE(samples)
    assert np.ravel(mu).size == 1
    assert np.ravel(COV).size == 1
    assert float(np.ravel(COV)[0]) > 0.0
