import warnings

import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_tmvn_rvs_univariate_no_bounds_shape_and_moments():
    np.random.seed(1)
    samples = tmvn_rvs(mu=3.0, COV=4.0, size=4000)
    assert samples.shape == (4000,)
    assert abs(np.mean(samples) - 3.0) < 0.15
    assert abs(np.std(samples) - 2.0) < 0.15


def test_tmvn_rvs_multivariate_no_bounds_shape_and_correlation():
    np.random.seed(2)
    mu = np.array([1.0, -1.0])
    COV = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples = tmvn_rvs(mu=mu, COV=COV, size=5000)
    assert samples.shape == (5000, 2)
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.1)
    corr = np.corrcoef(samples.T)[0, 1]
    assert abs(corr - 0.8) < 0.05


def test_tmvn_rvs_univariate_two_sided_truncation_respected():
    np.random.seed(3)
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=1000)
    assert samples.shape == (1000,)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)
    # symmetric truncation keeps the mean near zero
    assert abs(np.mean(samples)) < 0.1


def test_tmvn_rvs_exact_sample_count_returned():
    np.random.seed(4)
    for size in (1, 7, 50):
        samples = tmvn_rvs(mu=0.0, COV=1.0, lower=-0.5, upper=2.0, size=size)
        assert samples.shape[0] == size


def test_tmvn_rvs_lower_bound_only():
    np.random.seed(5)
    samples = tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, size=2000)
    assert np.all(samples > 0.0)
    # mean of a half-normal is sqrt(2/pi)
    assert abs(np.mean(samples) - np.sqrt(2.0 / np.pi)) < 0.1


def test_tmvn_rvs_upper_bound_only():
    np.random.seed(6)
    samples = tmvn_rvs(mu=0.0, COV=1.0, upper=0.0, size=2000)
    assert np.all(samples < 0.0)
    assert abs(np.mean(samples) + np.sqrt(2.0 / np.pi)) < 0.1


def test_tmvn_rvs_multivariate_truncation_respected():
    np.random.seed(7)
    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([-1.0, -1.5])
    upper = np.array([1.0, 2.0])
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=500)
    assert samples.shape == (500, 2)
    assert np.all(samples[:, 0] > -1.0) and np.all(samples[:, 0] < 1.0)
    assert np.all(samples[:, 1] > -1.5) and np.all(samples[:, 1] < 2.0)


def test_tmvn_rvs_multivariate_with_infinite_bounds():
    np.random.seed(8)
    mu = np.array([0.0, 0.0])
    COV = np.eye(2)
    samples = tmvn_rvs(mu, COV, lower=[0.0, -np.inf],
                       upper=[np.inf, np.inf], size=500)
    assert samples.shape == (500, 2)
    assert np.all(samples[:, 0] > 0.0)
    # the second dimension is unrestricted and independent -> has negatives
    assert np.any(samples[:, 1] < 0.0)


def test_tmvn_rvs_reproducible_with_seed():
    np.random.seed(42)
    s1 = tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=20)
    np.random.seed(42)
    s2 = tmvn_rvs(0.0, 1.0, lower=-1.0, upper=1.0, size=20)
    assert np.array_equal(s1, s2)


def test_tmvn_rvs_negligible_density_raises_value_error():
    with pytest.raises(ValueError):
        tmvn_rvs(mu=0.0, COV=1.0, lower=50.0, upper=60.0, size=5)


def test_tmvn_rvs_low_acceptance_warns_but_samples():
    np.random.seed(9)
    with pytest.warns(UserWarning):
        samples = tmvn_rvs(mu=0.0, COV=1.0, lower=3.5, size=5)
    assert samples.shape[0] == 5
    assert np.all(samples > 3.5)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_tmvn_MLE_univariate_untruncated_recovers_moments():
    np.random.seed(10)
    data = np.random.normal(5.0, 2.0, size=500)
    mu, COV = tmvn_MLE(data)
    assert np.ndim(mu) == 0
    assert np.ndim(COV) == 0
    std = np.std(data)
    assert abs(mu - np.mean(data)) < 0.1 * std
    assert abs(COV - np.var(data)) < 0.2 * np.var(data)
    assert COV > 0.0


def test_tmvn_MLE_multivariate_untruncated_shapes_and_values():
    np.random.seed(11)
    true_mu = np.array([1.0, -2.0])
    true_COV = np.array([[1.0, 0.5], [0.5, 4.0]])
    data = np.random.multivariate_normal(true_mu, true_COV, size=1000).T
    mu, COV = tmvn_MLE(data)
    mu = np.asarray(mu)
    COV = np.asarray(COV)
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(COV, COV.T)
    assert np.all(np.diag(COV) > 0.0)
    assert np.allclose(mu, np.mean(data, axis=1), atol=0.2)
    sample_COV = np.cov(data)
    assert np.allclose(np.diag(COV), np.diag(sample_COV), rtol=0.2)
    assert np.allclose(COV[0, 1], sample_COV[0, 1], atol=0.3)


def test_tmvn_MLE_univariate_affine_equivariance():
    np.random.seed(12)
    data = np.random.normal(0.0, 1.0, size=300)
    a, b = 3.0, 100.0
    mu1, COV1 = tmvn_MLE(data)
    mu2, COV2 = tmvn_MLE(a * data + b)
    assert mu2 == pytest.approx(a * mu1 + b, rel=1e-3)
    assert COV2 == pytest.approx(a ** 2 * COV1, rel=1e-3)


def test_tmvn_MLE_univariate_truncated_corrects_for_truncation():
    np.random.seed(13)
    true_mu, true_sig = 10.0, 2.0
    lower, upper = 8.0, 20.0
    data = tmvn_rvs(true_mu, true_sig ** 2, lower=lower, upper=upper,
                    size=1000)
    mu, COV = tmvn_MLE(data, tr_lower=lower, tr_upper=upper)
    # the naive sample mean is biased upward by the left truncation; the
    # truncation-aware fit should land closer to the true value
    assert mu < np.mean(data)
    assert abs(mu - true_mu) < 1.0
    assert COV > 0.0
    assert abs(np.sqrt(COV) - true_sig) < 1.0


def test_tmvn_MLE_univariate_censored_data():
    np.random.seed(14)
    full = np.random.normal(0.0, 1.0, size=1000)
    det_upper = 1.0
    kept = full[full < det_upper]
    censored_count = int(np.sum(full >= det_upper))
    mu, COV = tmvn_MLE(kept, censored_count=censored_count,
                       det_lower=-np.inf, det_upper=det_upper)
    assert np.isfinite(mu) and np.isfinite(COV)
    assert abs(mu) < 0.3
    assert abs(COV - 1.0) < 0.5


def test_tmvn_MLE_multivariate_truncated_returns_valid_estimates():
    np.random.seed(15)
    true_mu = np.array([0.0, 0.0])
    true_COV = np.array([[1.0, 0.3], [0.3, 1.0]])
    lower = np.array([-2.0, -2.0])
    upper = np.array([2.0, 2.0])
    data = tmvn_rvs(true_mu, true_COV, lower=lower, upper=upper, size=400).T
    mu, COV = tmvn_MLE(data, tr_lower=lower, tr_upper=upper)
    mu = np.asarray(mu)
    COV = np.asarray(COV)
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(COV))
    assert np.allclose(COV, COV.T)
    assert np.all(np.diag(COV) > 0.0)
    assert np.all(np.abs(mu) < 0.5)
    assert np.all(np.sqrt(np.diag(COV)) < 2.5)


def test_tmvn_MLE_with_alpha_lim_runs_and_is_finite():
    np.random.seed(16)
    data = tmvn_rvs(0.0, 1.0, lower=-2.0, upper=2.0, size=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, COV = tmvn_MLE(data, tr_lower=-2.0, tr_upper=2.0,
                           alpha_lim=1e-2)
    assert np.isfinite(mu)
    assert np.isfinite(COV)
    assert COV > 0.0
    assert abs(mu) < 0.5
