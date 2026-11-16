import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


def _cov2():
    return np.array([[1.0, 0.5], [0.5, 1.0]])


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_rvs_untruncated_scalar_inputs_count():
    np.random.seed(1)
    samples = tmvn_rvs(0.0, 1.0, size=25)
    assert np.asarray(samples).size == 25
    assert np.all(np.isfinite(samples))


def test_rvs_untruncated_multivariate_shape():
    np.random.seed(2)
    samples = tmvn_rvs(np.array([0.0, 1.0]), _cov2(), size=30)
    assert samples.shape == (30, 2)
    assert np.all(np.isfinite(samples))


def test_rvs_untruncated_mean_and_cov_close_to_target():
    np.random.seed(3)
    mu = np.array([10.0, -5.0])
    cov = np.array([[4.0, 1.0], [1.0, 2.0]])
    samples = tmvn_rvs(mu, cov, size=20000)
    assert np.allclose(samples.mean(axis=0), mu, atol=0.15)
    assert np.allclose(np.cov(samples.T), cov, atol=0.25)


def test_rvs_same_seed_gives_same_samples():
    mu = np.array([0.0, 0.0])
    lower = [-1.0, -1.0]
    upper = [1.0, 1.0]
    np.random.seed(4)
    a = tmvn_rvs(mu, _cov2(), lower=lower, upper=upper, size=20)
    np.random.seed(4)
    b = tmvn_rvs(mu, _cov2(), lower=lower, upper=upper, size=20)
    assert np.array_equal(a, b)


def test_rvs_univariate_truncated_within_bounds_and_exact_count():
    np.random.seed(5)
    samples = tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                       lower=[-1.0], upper=[1.0], size=50)
    assert samples.shape == (50,)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_rvs_multivariate_truncated_within_bounds():
    np.random.seed(6)
    samples = tmvn_rvs(np.array([0.0, 0.0]), _cov2(),
                       lower=[-1.0, -0.5], upper=[1.5, 1.0], size=200)
    assert samples.shape == (200, 2)
    assert np.all(samples[:, 0] > -1.0) and np.all(samples[:, 0] < 1.5)
    assert np.all(samples[:, 1] > -0.5) and np.all(samples[:, 1] < 1.0)


@pytest.mark.parametrize("size", [1, 2, 7, 33])
def test_rvs_truncated_returns_requested_size(size):
    np.random.seed(7)
    samples = tmvn_rvs(np.array([0.0, 0.0]), _cov2(),
                       lower=[-1.0, -1.0], upper=[1.0, 1.0], size=size)
    assert samples.shape[0] == size


def test_rvs_only_lower_bound():
    np.random.seed(8)
    samples = tmvn_rvs(np.array([0.0, 0.0]), np.eye(2),
                       lower=[0.0, 0.0], size=100)
    assert samples.shape == (100, 2)
    assert np.all(samples > 0.0)


def test_rvs_only_upper_bound():
    np.random.seed(9)
    samples = tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                       upper=[0.5], size=100)
    assert samples.shape == (100,)
    assert np.all(samples < 0.5)


def test_rvs_infinite_bound_in_one_dimension_only():
    np.random.seed(10)
    samples = tmvn_rvs(np.array([0.0, 0.0]), np.eye(2),
                       lower=[-np.inf, 0.0], upper=[np.inf, 1.0], size=100)
    assert samples.shape == (100, 2)
    assert np.all(samples[:, 1] > 0.0)
    assert np.all(samples[:, 1] < 1.0)


def test_rvs_symmetric_truncation_statistics():
    np.random.seed(11)
    samples = tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                       lower=[-1.0], upper=[1.0], size=3000)
    assert abs(np.mean(samples)) < 0.1
    # truncation must shrink the variance well below that of the parent
    assert np.var(samples) < 0.6


def test_rvs_unreachable_limits_raise_value_error():
    np.random.seed(12)
    with pytest.raises(ValueError, match="too small"):
        tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                 lower=[40.0], upper=[50.0], size=3)


def test_rvs_low_acceptance_region_warns_but_samples():
    np.random.seed(13)
    with pytest.warns(UserWarning):
        samples = tmvn_rvs(np.array([0.0]), np.array([[1.0]]),
                           lower=[3.2], size=2)
    assert samples.shape == (2,)
    assert np.all(samples > 3.2)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_mle_univariate_untruncated_recovers_moments():
    np.random.seed(20)
    data = np.random.normal(5.0, 2.0, size=500)
    mu, cov = tmvn_MLE(data)
    assert np.isclose(mu, np.mean(data), atol=0.2)
    assert np.isclose(cov, np.var(data), rtol=0.1)


def test_mle_multivariate_untruncated_recovers_moments():
    np.random.seed(21)
    true_cov = np.array([[4.0, 1.2], [1.2, 1.0]])
    data = np.random.multivariate_normal([1.0, -2.0], true_cov, size=500).T
    mu, cov = tmvn_MLE(data)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.allclose(mu, data.mean(axis=1), atol=0.1)
    assert np.allclose(cov, np.cov(data), rtol=0.1, atol=0.1)
    assert np.allclose(cov, cov.T)


def test_mle_accepts_list_input_same_as_array():
    np.random.seed(22)
    data = np.random.normal(1.0, 0.5, size=100)
    mu_a, cov_a = tmvn_MLE(data)
    mu_l, cov_l = tmvn_MLE(data.tolist())
    assert np.isclose(mu_a, mu_l)
    assert np.isclose(cov_a, cov_l)


def test_mle_affine_equivariance_univariate():
    np.random.seed(23)
    data = np.random.normal(0.0, 1.0, size=200)
    mu1, cov1 = tmvn_MLE(data)
    mu2, cov2 = tmvn_MLE(3.0 * data + 100.0)
    assert np.isclose(mu2, 3.0 * mu1 + 100.0, atol=0.1)
    assert np.isclose(cov2, 9.0 * cov1, rtol=0.05)


def test_mle_single_sample_skips_optimization():
    mu, cov = tmvn_MLE(np.array([3.0]))
    assert np.isclose(mu, 3.0)
    assert 0.0 <= cov < 1e-10


def test_mle_multivariate_truncated_inflates_variance():
    np.random.seed(24)
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    raw = tmvn_rvs(np.array([0.0, 0.0]), np.eye(2),
                   lower=list(lower), upper=list(upper), size=300)
    data = raw.T
    mu, cov = tmvn_MLE(data, tr_lower=lower, tr_upper=upper)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)
    assert np.all(np.abs(mu) < 0.5)
    # accounting for truncation must give variances above the raw sample's
    assert np.all(np.diag(cov) > np.var(data, axis=1))


def test_mle_multivariate_truncated_with_alpha_lim_is_finite():
    np.random.seed(25)
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    raw = tmvn_rvs(np.array([0.0, 0.0]), np.eye(2),
                   lower=list(lower), upper=list(upper), size=200)
    mu, cov = tmvn_MLE(raw.T, tr_lower=lower, tr_upper=upper, alpha_lim=1e-4)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
    assert np.all(np.diag(cov) > 0.0)


def test_mle_multivariate_censored_returns_valid_covariance():
    np.random.seed(26)
    data = np.random.multivariate_normal([0.0, 0.0], _cov2(), size=200).T
    mu, cov = tmvn_MLE(data, censored_count=5,
                       det_lower=np.array([-3.0, -3.0]),
                       det_upper=np.array([3.0, 3.0]))
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)
