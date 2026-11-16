import warnings

import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_tmvn_rvs_unbounded_multivariate_shape():
    np.random.seed(0)
    mu = np.array([1.0, -2.0])
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    samples = tmvn_rvs(mu, cov, size=25)
    assert samples.shape == (25, 2)
    assert np.all(np.isfinite(samples))


def test_tmvn_rvs_unbounded_scalar_inputs():
    np.random.seed(1)
    samples = tmvn_rvs(0.0, 1.0, size=5)
    assert samples.shape == (5,)
    assert np.all(np.isfinite(samples))


def test_tmvn_rvs_unbounded_sample_statistics():
    np.random.seed(2)
    mu = np.array([1.0, -2.0])
    cov = np.array([[1.0, 1.2], [1.2, 4.0]])
    samples = tmvn_rvs(mu, cov, size=20000)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.1)
    assert np.allclose(np.cov(samples.T), cov, atol=0.2)


def test_tmvn_rvs_reproducible_with_seed():
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    np.random.seed(123)
    first = tmvn_rvs(mu, cov, lower=[-1.0, -1.0], upper=[1.0, 1.0], size=30)
    np.random.seed(123)
    second = tmvn_rvs(mu, cov, lower=[-1.0, -1.0], upper=[1.0, 1.0], size=30)
    assert np.array_equal(first, second)


def test_tmvn_rvs_bounded_univariate_within_limits():
    np.random.seed(3)
    samples = tmvn_rvs([0.0], [[1.0]], lower=[-1.0], upper=[1.0], size=200)
    assert samples.shape == (200,)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_tmvn_rvs_bounded_multivariate_within_limits_and_size():
    np.random.seed(4)
    mu = np.array([10.0, -5.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = [9.0, -6.0]
    upper = [11.0, -4.0]
    for size in (1, 7, 50):
        samples = tmvn_rvs(mu, cov, lower=lower, upper=upper, size=size)
        assert samples.shape == (size, 2)
        assert np.all(samples > np.array(lower))
        assert np.all(samples < np.array(upper))


def test_tmvn_rvs_one_sided_limits():
    np.random.seed(5)
    mu = np.zeros(2)
    cov = np.eye(2)
    low_only = tmvn_rvs(mu, cov, lower=[0.5, -0.5], size=300)
    assert low_only.shape == (300, 2)
    assert np.all(low_only[:, 0] > 0.5)
    assert np.all(low_only[:, 1] > -0.5)

    up_only = tmvn_rvs(mu, cov, upper=[0.0, 0.0], size=300)
    assert up_only.shape == (300, 2)
    assert np.all(up_only < 0.0)


def test_tmvn_rvs_half_normal_mean():
    np.random.seed(6)
    samples = tmvn_rvs([0.0], [[1.0]], lower=[0.0], size=20000)
    assert samples.shape == (20000,)
    assert np.all(samples > 0.0)
    assert samples.mean() == pytest.approx(np.sqrt(2.0 / np.pi), abs=0.03)


def test_tmvn_rvs_infinite_limit_in_one_dimension():
    np.random.seed(7)
    samples = tmvn_rvs(np.zeros(2), np.eye(2),
                       lower=[-np.inf, 0.0], upper=[np.inf, np.inf],
                       size=4000)
    assert samples.shape == (4000, 2)
    assert np.all(samples[:, 1] > 0.0)
    assert samples[:, 1].mean() == pytest.approx(np.sqrt(2.0 / np.pi),
                                                 abs=0.1)
    assert abs(samples[:, 0].mean()) < 0.1
    # the first dimension is untruncated, so it must have negative values
    assert np.any(samples[:, 0] < 0.0)


def test_tmvn_rvs_correlated_truncation_shifts_other_dimension():
    np.random.seed(8)
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples = tmvn_rvs(np.zeros(2), cov,
                       lower=[0.0, -np.inf], upper=[np.inf, np.inf],
                       size=5000)
    assert np.all(samples[:, 0] > 0.0)
    # E[x2 | x1 > 0] = 0.8 * sqrt(2/pi) ~ 0.64
    assert 0.5 < samples[:, 1].mean() < 0.8


def test_tmvn_rvs_negligible_density_raises():
    np.random.seed(9)
    with pytest.raises(ValueError):
        tmvn_rvs([0.0], [[1.0]], lower=[40.0], upper=[50.0], size=3)


def test_tmvn_rvs_tail_sampling_warns_but_respects_limits():
    np.random.seed(10)
    with pytest.warns(UserWarning):
        samples = tmvn_rvs([0.0], [[1.0]], lower=[3.5], upper=[10.0], size=3)
    assert samples.shape == (3,)
    assert np.all(samples > 3.5)
    assert np.all(samples < 10.0)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_tmvn_MLE_univariate_untruncated_matches_sample_moments():
    np.random.seed(11)
    samples = np.random.normal(5.0, 2.0, size=200)
    mu, cov = tmvn_MLE(samples)
    assert np.ndim(mu) == 0
    assert np.ndim(cov) == 0
    assert mu == pytest.approx(np.mean(samples), abs=0.1)
    assert cov == pytest.approx(np.var(samples), rel=0.1)


def test_tmvn_MLE_single_sample_returns_sample_with_tiny_variance():
    np.random.seed(12)
    mu, cov = tmvn_MLE(np.array([3.0]))
    assert mu == pytest.approx(3.0, abs=1e-9)
    assert 0.0 < cov < 1e-10


def test_tmvn_MLE_constant_samples_give_near_zero_variance():
    np.random.seed(13)
    mu, cov = tmvn_MLE(np.full(10, 3.0))
    assert mu == pytest.approx(3.0, abs=1e-4)
    assert 0.0 <= cov < 1e-9


def test_tmvn_MLE_affine_equivariance_univariate():
    np.random.seed(14)
    x = np.random.normal(0.0, 1.0, size=150)
    a, b = 3.0, 10.0
    mu1, cov1 = tmvn_MLE(x)
    mu2, cov2 = tmvn_MLE(a * x + b)
    assert mu2 == pytest.approx(a * mu1 + b, abs=0.2)
    assert cov2 == pytest.approx(a ** 2 * cov1, rel=0.1)


def test_tmvn_MLE_accepts_list_input():
    np.random.seed(15)
    x = np.random.normal(1.0, 0.5, size=50)
    mu1, cov1 = tmvn_MLE(x)
    mu2, cov2 = tmvn_MLE(x.tolist())
    assert mu1 == pytest.approx(mu2, rel=1e-9, abs=1e-12)
    assert cov1 == pytest.approx(cov2, rel=1e-9, abs=1e-12)


def test_tmvn_MLE_multivariate_untruncated_matches_sample_moments():
    np.random.seed(16)
    true_mu = [1.0, -2.0]
    true_cov = [[1.0, 1.2], [1.2, 4.0]]
    samples = np.random.multivariate_normal(true_mu, true_cov, size=400).T
    mu, cov = tmvn_MLE(samples)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov.T)
    assert np.allclose(mu, samples.mean(axis=1), atol=0.1)
    assert np.allclose(cov, np.cov(samples), rtol=0.1, atol=0.05)


def test_tmvn_MLE_multivariate_truncated_gives_sane_estimates():
    np.random.seed(17)
    cov_true = np.array([[1.0, 0.5], [0.5, 1.0]])
    data = tmvn_rvs(np.zeros(2), cov_true, lower=[-1.0, -1.0],
                    upper=[2.0, 2.0], size=300)
    assert data.shape == (300, 2)
    mu, cov = tmvn_MLE(data.T,
                       tr_lower=np.array([-1.0, -1.0]),
                       tr_upper=np.array([2.0, 2.0]))
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)
    assert np.all(np.diag(cov) > 0.0)
    assert np.all(np.abs(mu) < 1.0)
    assert np.all(np.diag(cov) > 0.3)
    assert np.all(np.diag(cov) < 3.0)


def test_tmvn_MLE_univariate_truncated_corrects_mean_of_truncated_data():
    np.random.seed(18)
    data = tmvn_rvs([0.0], [[1.0]], lower=[0.0], upper=[5.0], size=500)
    naive_mean = np.mean(data)
    mu, cov = tmvn_MLE(data, tr_lower=0.0, tr_upper=5.0)
    assert np.isfinite(mu)
    assert np.isfinite(cov)
    assert cov > 0.0
    # the truncated sample mean is about 0.8; the fit must move below it
    assert mu < naive_mean


def test_tmvn_MLE_censored_univariate_inflates_variance():
    np.random.seed(19)
    x = np.random.normal(0.0, 1.0, size=1000)
    observed = x[(x > -1.0) & (x < 1.0)]
    censored_count = 1000 - len(observed)
    assert censored_count > 0
    mu, cov = tmvn_MLE(observed, censored_count=censored_count,
                       det_lower=-1.0, det_upper=1.0)
    assert abs(mu) < 0.35
    assert 0.5 < cov < 2.0
    assert cov > 1.5 * np.var(observed)


def test_tmvn_MLE_small_alpha_lim_does_not_change_result():
    np.random.seed(20)
    data = tmvn_rvs([0.0], [[1.0]], lower=[-1.0], upper=[2.0], size=200)
    mu1, cov1 = tmvn_MLE(data, tr_lower=-1.0, tr_upper=2.0)
    mu2, cov2 = tmvn_MLE(data, tr_lower=-1.0, tr_upper=2.0, alpha_lim=1e-4)
    assert mu2 == pytest.approx(mu1, rel=1e-6, abs=1e-9)
    assert cov2 == pytest.approx(cov1, rel=1e-6, abs=1e-9)


def test_tmvn_MLE_unreachable_alpha_lim_warns():
    np.random.seed(21)
    data = tmvn_rvs([0.0], [[1.0]], lower=[-1.0], upper=[2.0], size=100)
    with pytest.warns(UserWarning):
        mu, cov = tmvn_MLE(data, tr_lower=-1.0, tr_upper=2.0, alpha_lim=1.0)
    assert np.isfinite(mu)
    assert np.isfinite(cov)
