import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_no_bounds_scalar_inputs_shape():
    np.random.seed(0)
    samples = tmvn_rvs(0., 1., size=50)
    assert samples.shape == (50,)
    assert np.all(np.isfinite(samples))


def test_rvs_no_bounds_univariate_moments():
    np.random.seed(1)
    samples = tmvn_rvs([3.], [[4.]], size=20000)
    assert samples.shape == (20000,)
    assert np.mean(samples) == pytest.approx(3.0, abs=0.1)
    assert np.var(samples) == pytest.approx(4.0, rel=0.1)


def test_rvs_no_bounds_multivariate_shape_and_moments():
    np.random.seed(2)
    mu = np.array([1., -2.])
    COV = np.array([[1.0, 0.6], [0.6, 2.0]])
    samples = tmvn_rvs(mu, COV, size=20000)
    assert samples.shape == (20000, 2)
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.1)
    assert np.allclose(np.cov(samples, rowvar=False), COV, atol=0.15)


def test_rvs_univariate_samples_within_limits():
    np.random.seed(3)
    samples = tmvn_rvs([0.], [[1.]], lower=[-1.], upper=[1.], size=500)
    assert samples.shape == (500,)
    assert np.all(samples > -1.)
    assert np.all(samples < 1.)


def test_rvs_multivariate_samples_within_limits():
    np.random.seed(4)
    mu = np.array([0., 0.])
    COV = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([-1., -0.5])
    upper = np.array([1.5, 1.0])
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=400)
    assert samples.shape == (400, 2)
    assert np.all(samples[:, 0] > lower[0])
    assert np.all(samples[:, 0] < upper[0])
    assert np.all(samples[:, 1] > lower[1])
    assert np.all(samples[:, 1] < upper[1])


def test_rvs_exact_sample_count_for_various_sizes():
    np.random.seed(5)
    for size in (2, 7, 33, 120):
        samples = tmvn_rvs([0., 0.], np.eye(2), lower=[-1., -1.],
                           upper=[1., 1.], size=size)
        assert samples.shape == (size, 2)


def test_rvs_lower_only_bound_half_normal_mean():
    np.random.seed(6)
    samples = tmvn_rvs([0.], [[1.]], lower=[0.], upper=None, size=5000)
    assert np.all(samples > 0.)
    # mean of a half normal distribution is sqrt(2/pi)
    assert np.mean(samples) == pytest.approx(np.sqrt(2. / np.pi), abs=0.05)


def test_rvs_upper_only_bound_multivariate():
    np.random.seed(7)
    samples = tmvn_rvs([0., 0.], np.eye(2), lower=None, upper=[0.5, 0.5],
                       size=300)
    assert samples.shape == (300, 2)
    assert np.all(samples < 0.5)


def test_rvs_infinite_bounds_in_some_dimensions():
    np.random.seed(8)
    lower = [-np.inf, 0.]
    upper = [np.inf, np.inf]
    samples = tmvn_rvs([0., 0.], np.eye(2), lower=lower, upper=upper,
                       size=300)
    assert samples.shape == (300, 2)
    assert np.all(samples[:, 1] > 0.)
    # first dimension is unbounded and should have values on both sides
    assert samples[:, 0].min() < 0. < samples[:, 0].max()


def test_rvs_truncation_preserves_positive_correlation():
    np.random.seed(9)
    COV = np.array([[1.0, 0.7], [0.7, 1.0]])
    samples = tmvn_rvs([0., 0.], COV, lower=[-2., -2.], upper=[2., 2.],
                       size=3000)
    corr = np.corrcoef(samples, rowvar=False)[0, 1]
    assert corr > 0.4


def test_rvs_reproducible_with_seed():
    np.random.seed(10)
    s1 = tmvn_rvs([0., 0.], np.eye(2), lower=[-1., -1.], upper=[1., 1.],
                  size=25)
    np.random.seed(10)
    s2 = tmvn_rvs([0., 0.], np.eye(2), lower=[-1., -1.], upper=[1., 1.],
                  size=25)
    assert np.array_equal(s1, s2)


def test_rvs_negligible_probability_mass_raises():
    with pytest.raises(ValueError):
        tmvn_rvs([0.], [[1.]], lower=[50.], upper=[60.], size=10)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_untruncated_matches_sample_moments():
    np.random.seed(11)
    data = np.random.normal(5.0, 2.0, size=500)
    mu, COV = tmvn_MLE(data)
    assert np.ndim(mu) == 0
    assert np.ndim(COV) == 0
    assert float(mu) == pytest.approx(np.mean(data), abs=0.3)
    assert float(COV) == pytest.approx(np.var(data), rel=0.15)


def test_mle_multivariate_untruncated_shapes_and_values():
    np.random.seed(12)
    true_mu = np.array([1., -2.])
    true_COV = np.array([[1.0, 0.6], [0.6, 2.0]])
    data = np.random.multivariate_normal(true_mu, true_COV, size=800).T
    mu, COV = tmvn_MLE(data)
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(COV, COV.T)
    assert np.allclose(mu, np.mean(data, axis=1), atol=0.3)
    assert np.allclose(COV, np.cov(data), rtol=0.2, atol=0.2)
    # correlation coefficient is taken from the data
    sig = np.sqrt(np.diag(COV))
    corr = COV[0, 1] / (sig[0] * sig[1])
    assert corr == pytest.approx(np.corrcoef(data)[0, 1], abs=0.05)


def test_mle_truncated_fit_inflates_variance_relative_to_sample():
    np.random.seed(13)
    lower = np.array([-1.5, -1.5])
    upper = np.array([1.5, 1.5])
    data = tmvn_rvs([0., 0.], np.eye(2), lower=lower, upper=upper,
                    size=500).T
    mu, COV = tmvn_MLE(data, tr_lower=lower, tr_upper=upper)
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(COV))
    # variance of truncated data underestimates the parent variance
    assert np.all(np.diag(COV) > np.var(data, axis=1))
    # the fitted mean should stay near the (symmetric) true mean
    assert np.allclose(mu, 0., atol=0.4)


def test_mle_censored_data_returns_valid_estimates():
    np.random.seed(14)
    data = np.random.multivariate_normal([0., 0.], np.eye(2), size=200).T
    mu, COV = tmvn_MLE(data, censored_count=5,
                       det_lower=np.array([-3., -3.]),
                       det_upper=np.array([3., 3.]))
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(COV))
    assert np.all(np.diag(COV) > 0.)
