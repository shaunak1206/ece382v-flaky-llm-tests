"""Unit tests for tmvn_rvs and tmvn_MLE in pelicun.uq."""

import numpy as np
import pytest

import pelicun.uq as uq
from pelicun.uq import tmvn_MLE, tmvn_rvs

HALF_NORMAL_MEAN = np.sqrt(2.0 / np.pi)
HALF_NORMAL_VAR = 1.0 - 2.0 / np.pi


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_unbounded_shape():
    np.random.seed(1)
    samples = tmvn_rvs([0.0, 0.0], np.eye(2), size=50)
    assert samples.shape == (50, 2)


def test_rvs_unbounded_moments():
    np.random.seed(2)
    mu = np.array([1.0, -2.0])
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    samples = tmvn_rvs(mu, cov, size=20000)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.05)
    assert np.allclose(np.cov(samples, rowvar=False), cov, atol=0.1)


def test_rvs_accepts_list_inputs():
    np.random.seed(3)
    samples = tmvn_rvs(
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
        size=25,
    )
    assert samples.shape == (25, 2)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_rvs_box_samples_inside_bounds():
    np.random.seed(4)
    samples = tmvn_rvs(
        np.array([0.0, 0.0]),
        np.eye(2),
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        size=500,
    )
    assert samples.shape == (500, 2)
    assert np.all(samples >= -1.0)
    assert np.all(samples <= 1.0)


def test_rvs_returns_exactly_requested_rows():
    np.random.seed(5)
    one = tmvn_rvs([0.0, 0.0], np.eye(2), lower=[-1.0, -1.0],
                   upper=[1.0, 1.0], size=1)
    many = tmvn_rvs([0.0, 0.0], np.eye(2), lower=[-1.0, -1.0],
                    upper=[1.0, 1.0], size=37)
    assert one.shape == (1, 2)
    assert many.shape == (37, 2)


def test_rvs_one_sided_lower_bound():
    np.random.seed(6)
    samples = tmvn_rvs([0.0, 0.0], np.eye(2), lower=[0.0, -np.inf], size=300)
    assert samples.shape == (300, 2)
    assert np.all(samples[:, 0] > 0.0)


def test_rvs_one_sided_upper_bound():
    np.random.seed(7)
    samples = tmvn_rvs([0.0, 0.0], np.eye(2), upper=[np.inf, 0.0], size=300)
    assert samples.shape == (300, 2)
    assert np.all(samples[:, 1] < 0.0)


def test_rvs_quadrant_mean_matches_half_normal():
    np.random.seed(8)
    samples = tmvn_rvs([0.0, 0.0], np.eye(2), lower=[0.0, 0.0], size=20000)
    assert np.all(samples > 0.0)
    assert np.allclose(samples.mean(axis=0), HALF_NORMAL_MEAN, atol=0.03)


def test_rvs_quadrant_variance_matches_half_normal():
    np.random.seed(9)
    samples = tmvn_rvs([0.0, 0.0], np.eye(2), lower=[0.0, 0.0], size=20000)
    assert abs(np.var(samples[:, 0]) - HALF_NORMAL_VAR) < 0.03
    assert abs(np.var(samples[:, 1]) - HALF_NORMAL_VAR) < 0.03


def test_rvs_correlated_quadrant_stays_in_bounds():
    np.random.seed(10)
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples = tmvn_rvs([0.0, 0.0], cov, lower=[0.0, 0.0], size=5000)
    assert samples.shape == (5000, 2)
    assert np.all(samples >= 0.0)
    assert np.corrcoef(samples, rowvar=False)[0, 1] > 0.3


def test_rvs_raises_when_density_estimate_is_unreliable(monkeypatch):
    def fake_density(mu, COV, lower=None, upper=None):
        return 0.0, 0.0

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    with pytest.raises(ValueError):
        tmvn_rvs([0.0, 0.0], np.eye(2), lower=[-1.0, -1.0],
                 upper=[1.0, 1.0], size=10)


def test_rvs_warns_when_density_is_very_small(monkeypatch):
    def fake_density(mu, COV, lower=None, upper=None):
        return 5e-4, 1e-9

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    np.random.seed(12)
    with pytest.warns(UserWarning):
        samples = tmvn_rvs([0.0, 0.0], np.eye(2), lower=[-1.0, -1.0],
                           upper=[1.0, 1.0], size=10)
    assert samples.shape == (10, 2)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_rvs_is_reproducible_with_same_seed():
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    np.random.seed(13)
    first = tmvn_rvs([0.0, 0.0], cov, lower=[-0.5, -0.5], upper=[2.0, 2.0],
                     size=100)
    np.random.seed(13)
    second = tmvn_rvs([0.0, 0.0], cov, lower=[-0.5, -0.5], upper=[2.0, 2.0],
                      size=100)
    np.testing.assert_array_equal(first, second)


def test_rvs_three_dimensional_box():
    np.random.seed(14)
    samples = tmvn_rvs(np.zeros(3), np.eye(3), lower=-0.5 * np.ones(3),
                       upper=0.5 * np.ones(3), size=200)
    assert samples.shape == (200, 3)
    assert np.all(samples >= -0.5)
    assert np.all(samples <= 0.5)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_untruncated_multivariate_matches_sample_moments():
    np.random.seed(15)
    true_mu = np.array([1.0, -2.0])
    true_cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    samples = np.random.multivariate_normal(true_mu, true_cov, size=4000).T
    mu, COV = tmvn_MLE(samples)
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(mu, samples.mean(axis=1), atol=0.05)
    assert np.allclose(COV, np.cov(samples), rtol=0.05)
    assert np.allclose(mu, true_mu, atol=0.1)
    assert np.allclose(COV, true_cov, atol=0.15)


def test_mle_untruncated_univariate_matches_sample_moments():
    np.random.seed(16)
    samples = 3.0 + 2.0 * np.random.standard_normal(4000)
    mu, COV = tmvn_MLE(samples)
    assert np.ndim(mu) == 0
    assert np.ndim(COV) == 0
    assert np.isclose(mu, samples.mean(), atol=0.05)
    assert np.isclose(COV, samples.var(), rtol=0.05)
    assert np.isclose(mu, 3.0, atol=0.2)
    assert np.isclose(COV, 4.0, rtol=0.1)


def test_mle_three_dimensional_output_is_symmetric_positive():
    np.random.seed(17)
    cov = np.array([[1.0, 0.4, 0.1], [0.4, 2.0, -0.3], [0.1, -0.3, 0.5]])
    samples = np.random.multivariate_normal(np.zeros(3), cov, size=1000).T
    mu, COV = tmvn_MLE(samples)
    assert mu.shape == (3,)
    assert COV.shape == (3, 3)
    assert np.allclose(COV, COV.T)
    assert np.all(np.diag(COV) > 0.0)
    assert np.allclose(COV, cov, atol=0.35)


def test_mle_censored_multivariate_recovers_mean():
    np.random.seed(18)
    raw = np.random.multivariate_normal([0.0, 0.0], np.eye(2), size=6000).T
    observed_mask = np.all(raw <= 1.0, axis=0)
    observed = raw[:, observed_mask]
    censored_count = int(np.sum(~observed_mask))
    assert censored_count > 0
    mu, COV = tmvn_MLE(observed,
                       censored_count=censored_count,
                       det_lower=np.array([-np.inf, -np.inf]),
                       det_upper=np.array([1.0, 1.0]))
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(COV))
    assert np.all(np.abs(mu) < 0.15)


def test_mle_truncated_multivariate_recovers_mean():
    np.random.seed(19)
    raw = np.random.multivariate_normal([0.0, 0.0], np.eye(2), size=6000).T
    kept = raw[:, raw[0] <= 1.0]
    mu, COV = tmvn_MLE(kept,
                       tr_lower=np.array([-np.inf, -np.inf]),
                       tr_upper=np.array([1.0, np.inf]))
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(COV))
    assert np.all(np.abs(mu) < 0.15)


def test_mle_constant_dimension_keeps_its_value():
    np.random.seed(20)
    samples = np.vstack([np.random.standard_normal(200), np.full(200, 3.0)])
    mu, COV = tmvn_MLE(samples)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(COV))
    assert abs(mu[1] - 3.0) < 1e-3
    assert COV[1, 1] < 1e-8


def test_mle_does_not_modify_input():
    np.random.seed(21)
    samples = np.random.multivariate_normal([0.0, 1.0],
                                            [[1.0, 0.3], [0.3, 1.5]],
                                            size=500).T
    original = samples.copy()
    tmvn_MLE(samples)
    assert np.array_equal(samples, original)
