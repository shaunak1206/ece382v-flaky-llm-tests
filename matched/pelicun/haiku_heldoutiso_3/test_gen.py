"""Unit tests for tmvn_rvs and tmvn_MLE in pelicun/uq.py."""

import numpy as np
import pytest
from scipy.stats import truncnorm

from pelicun.uq import tmvn_rvs, tmvn_MLE


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_unbounded_univariate_sample_count():
    np.random.seed(1)
    samples = tmvn_rvs(mu=2.0, COV=1.5, size=500)
    assert np.asarray(samples).size == 500


def test_rvs_unbounded_multivariate_shape():
    np.random.seed(2)
    samples = tmvn_rvs(mu=[0.0, 1.0, 2.0], COV=np.eye(3) * 2.0, size=200)
    assert samples.shape == (200, 3)


def test_rvs_unbounded_moments():
    np.random.seed(3)
    mu = np.array([1.0, -1.0])
    COV = np.array([[1.0, 0.6], [0.6, 2.0]])
    samples = tmvn_rvs(mu, COV, size=20000)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)
    assert np.allclose(np.cov(samples.T), COV, atol=0.1)


def test_rvs_two_sided_univariate_bounds_respected():
    np.random.seed(4)
    samples = tmvn_rvs(mu=0.5, COV=1.0, lower=0.0, upper=1.0, size=1000)
    assert samples.shape == (1000,)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_rvs_truncated_univariate_mean_matches_truncnorm():
    np.random.seed(5)
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, upper=2.0, size=20000)
    expected = truncnorm.mean(0.0, 2.0)
    assert np.mean(samples) == pytest.approx(expected, abs=0.02)


def test_rvs_lower_bound_only_is_respected():
    np.random.seed(6)
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=1.0, size=500)
    assert np.all(samples > 1.0)


def test_rvs_upper_bound_only_is_respected():
    np.random.seed(7)
    samples = tmvn_rvs(mu=0.0, COV=1.0, upper=-1.0, size=500)
    assert np.all(samples < -1.0)


def test_rvs_multivariate_bounds_respected_per_dimension():
    np.random.seed(8)
    samples = tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2),
                       lower=[-1.0, 0.0], upper=[1.0, 2.0], size=2000)
    assert samples.shape == (2000, 2)
    assert np.all((samples[:, 0] > -1.0) & (samples[:, 0] < 1.0))
    assert np.all((samples[:, 1] > 0.0) & (samples[:, 1] < 2.0))


def test_rvs_infinite_limits_leave_dimension_unbounded():
    np.random.seed(9)
    samples = tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2),
                       lower=[-np.inf, 0.0], upper=[np.inf, np.inf],
                       size=3000)
    assert samples.shape == (3000, 2)
    assert np.all(samples[:, 1] > 0.0)
    assert samples[:, 0].min() < -1.0
    assert samples[:, 0].max() > 1.0


def test_rvs_returns_requested_size_and_shape():
    np.random.seed(10)
    COV = np.array([[1.0, 0.5], [0.5, 1.0]])
    samples = tmvn_rvs(mu=[0.0, 1.0], COV=COV,
                       lower=[-1.0, -1.0], upper=[1.0, 3.0], size=37)
    assert samples.shape == (37, 2)


def test_rvs_same_seed_gives_same_samples():
    np.random.seed(11)
    first = tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=2.0, size=100)
    np.random.seed(11)
    second = tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=2.0, size=100)
    np.testing.assert_array_equal(first, second)


def test_rvs_raises_for_negligible_probability_mass():
    # The truncation box lies 50 standard deviations out, so the density
    # inside it cannot be estimated and the function must refuse to sample.
    with pytest.raises(ValueError, match="too small"):
        tmvn_rvs(mu=0.0, COV=1.0, lower=50.0, upper=51.0, size=10)


def test_rvs_warns_when_rejection_rate_is_high():
    # P(X > 3.2) for a standard normal is about 6.9e-4, which is below the
    # 1e-3 threshold that triggers the high-rejection warning.
    np.random.seed(13)
    with pytest.warns(UserWarning, match="rejection rate"):
        samples = tmvn_rvs(mu=0.0, COV=1.0, lower=3.2, size=5)
    assert samples.shape == (5,)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_unbounded_recovers_parameters():
    data = np.random.RandomState(20).normal(2.0, 1.5, 5000)
    mu, COV = tmvn_MLE(data)
    assert np.ndim(mu) == 0
    assert np.ndim(COV) == 0
    assert COV > 0
    assert mu == pytest.approx(2.0, abs=0.1)
    assert COV == pytest.approx(2.25, rel=0.15)


def test_mle_multivariate_unbounded_recovers_moments_and_psd():
    rs = np.random.RandomState(22)
    true_mu = np.array([1.0, -0.5])
    true_cov = np.array([[1.0, 0.5], [0.5, 4.0]])
    data = rs.multivariate_normal(true_mu, true_cov, size=5000).T
    mu, COV = tmvn_MLE(data)

    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(COV, COV.T)
    assert np.allclose(mu, true_mu, atol=0.1)
    assert np.allclose(np.diag(COV), np.diag(true_cov), rtol=0.15)
    assert np.all(np.linalg.eigvalsh(COV) > -1e-10)


def test_mle_multivariate_preserves_sample_correlation():
    rs = np.random.RandomState(23)
    data = rs.multivariate_normal([0.0, 0.0],
                                  [[1.0, 0.7], [0.7, 2.0]], size=1000).T
    _, COV = tmvn_MLE(data)
    sig = np.sqrt(np.diag(COV))
    corr = COV / np.outer(sig, sig)
    np.testing.assert_allclose(corr, np.corrcoef(data), atol=1e-6)


def test_mle_zero_variance_dimension_gives_finite_estimates():
    rs = np.random.RandomState(24)
    data = np.vstack([rs.normal(0.0, 1.0, 500), np.full(500, 5.0)])
    mu, COV = tmvn_MLE(data)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(COV))
    assert mu[1] == pytest.approx(5.0, abs=1e-3)


def test_mle_truncated_univariate_recovers_parameters():
    rs = np.random.RandomState(25)
    # samples from N(1, 1) truncated to [0, 2]
    data = truncnorm.rvs(-1.0, 1.0, loc=1.0, scale=1.0,
                         size=4000, random_state=rs)
    mu, COV = tmvn_MLE(data, tr_lower=0.0, tr_upper=2.0)
    assert mu == pytest.approx(1.0, abs=0.1)
    assert COV == pytest.approx(1.0, rel=0.2)


def test_mle_truncated_multivariate_round_trip():
    np.random.seed(26)
    mu_true = np.array([1.0, 1.0])
    COV_true = np.array([[1.0, 0.3], [0.3, 1.0]])
    lo = np.array([0.0, 0.0])
    hi = np.array([2.0, 2.0])
    samples = tmvn_rvs(mu_true, COV_true, lower=lo, upper=hi, size=3000)
    mu_hat, COV_hat = tmvn_MLE(samples.T, tr_lower=lo, tr_upper=hi)
    assert mu_hat.shape == (2,)
    assert COV_hat.shape == (2, 2)
    assert np.allclose(mu_hat, mu_true, atol=0.2)


def test_mle_censored_univariate_recovers_parameters():
    x = np.random.RandomState(27).normal(0.0, 1.0, 4000)
    det_upper = 1.0
    observed = x[x <= det_upper]
    censored = int(x.size - observed.size)
    mu, COV = tmvn_MLE(observed, det_lower=-np.inf, det_upper=det_upper,
                       censored_count=censored)
    assert mu == pytest.approx(0.0, abs=0.1)
    assert COV == pytest.approx(1.0, rel=0.2)


def test_mle_alpha_lim_does_not_disturb_well_posed_fit():
    rs = np.random.RandomState(28)
    data = truncnorm.rvs(-1.0, 1.0, loc=1.0, scale=1.0,
                         size=4000, random_state=rs)
    mu, COV = tmvn_MLE(data, tr_lower=0.0, tr_upper=2.0, alpha_lim=1e-4)
    assert mu == pytest.approx(1.0, abs=0.15)
    assert COV == pytest.approx(1.0, rel=0.25)


def test_mle_is_affine_equivariant_for_univariate_data():
    x = np.random.RandomState(29).normal(0.0, 1.0, 1000)
    a, b = 3.0, 10.0
    mu1, COV1 = tmvn_MLE(x)
    mu2, COV2 = tmvn_MLE(a * x + b)
    assert mu2 == pytest.approx(a * mu1 + b, abs=1e-2 * a)
    assert COV2 == pytest.approx(a * a * COV1, rel=1e-2)


def test_mle_accepts_nested_list_input():
    data = np.random.RandomState(30).normal(0.0, 1.0, (2, 300))
    mu, COV = tmvn_MLE(data.tolist())
    assert np.shape(mu) == (2,)
    assert np.shape(COV) == (2, 2)
