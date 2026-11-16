import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_tmvn_rvs_no_bounds_scalar_inputs():
    np.random.seed(1)
    samples = tmvn_rvs(2.0, 0.25, size=4000)
    assert samples.shape == (4000,)
    assert abs(np.mean(samples) - 2.0) < 0.05
    assert abs(np.std(samples) - 0.5) < 0.05


def test_tmvn_rvs_no_bounds_multivariate_correlation():
    np.random.seed(2)
    mu = [1.0, -1.0]
    COV = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples = tmvn_rvs(mu, COV, size=5000)
    assert samples.shape == (5000, 2)
    assert np.allclose(np.mean(samples, axis=0), mu, atol=0.1)
    corr = np.corrcoef(samples.T)[0, 1]
    assert abs(corr - 0.8) < 0.05


def test_tmvn_rvs_univariate_within_bounds():
    np.random.seed(3)
    samples = tmvn_rvs([0.0], [[1.0]], lower=[-1.0], upper=[1.0], size=1000)
    assert samples.shape == (1000,)
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_tmvn_rvs_univariate_symmetric_truncation_moments():
    np.random.seed(4)
    samples = tmvn_rvs([0.0], [[1.0]], lower=[-1.0], upper=[1.0], size=5000)
    # analytical std of a standard normal truncated at +-1 is ~0.5397
    assert abs(np.mean(samples)) < 0.05
    assert abs(np.std(samples) - 0.5397) < 0.05


def test_tmvn_rvs_half_normal_lower_bound_only():
    np.random.seed(5)
    samples = tmvn_rvs([0.0], [[1.0]], lower=[0.0], size=5000)
    assert np.all(samples > 0.0)
    # mean of a half-normal distribution is sqrt(2/pi)
    assert abs(np.mean(samples) - np.sqrt(2.0 / np.pi)) < 0.05


def test_tmvn_rvs_multivariate_box_limits():
    np.random.seed(6)
    mu = [0.0, 0.0]
    COV = np.eye(2)
    lower = [-1.0, -0.5]
    upper = [1.0, 2.0]
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=1000)
    assert samples.shape == (1000, 2)
    assert np.all(samples[:, 0] > lower[0])
    assert np.all(samples[:, 0] < upper[0])
    assert np.all(samples[:, 1] > lower[1])
    assert np.all(samples[:, 1] < upper[1])


def test_tmvn_rvs_upper_bound_only_multivariate():
    np.random.seed(7)
    samples = tmvn_rvs([0.0, 0.0], np.eye(2), upper=[0.5, 0.5], size=500)
    assert samples.shape == (500, 2)
    assert np.all(samples < 0.5)
    # truncation from above shifts the mean downward
    assert np.all(np.mean(samples, axis=0) < 0.0)


def test_tmvn_rvs_mixed_infinite_bounds():
    np.random.seed(8)
    lower = [-np.inf, 0.0]
    upper = [np.inf, np.inf]
    samples = tmvn_rvs([0.0, 0.0], np.eye(2), lower=lower, upper=upper,
                       size=800)
    assert samples.shape == (800, 2)
    assert np.all(samples[:, 1] > 0.0)
    # first dimension is untruncated and should have values on both sides
    assert samples[:, 0].min() < 0.0 < samples[:, 0].max()


@pytest.mark.parametrize("size", [1, 7, 100])
def test_tmvn_rvs_returns_exact_sample_count(size):
    np.random.seed(9)
    samples = tmvn_rvs([0.0, 0.0], np.eye(2), lower=[-1.0, -1.0],
                       upper=[1.0, 1.0], size=size)
    assert samples.shape == (size, 2)


def test_tmvn_rvs_reproducible_with_seed():
    kwargs = dict(mu=[0.0, 0.0], COV=np.eye(2), lower=[-1.0, -1.0],
                  upper=[1.0, 1.0], size=50)
    np.random.seed(10)
    s1 = tmvn_rvs(**kwargs)
    np.random.seed(10)
    s2 = tmvn_rvs(**kwargs)
    assert np.array_equal(s1, s2)


def test_tmvn_rvs_raises_when_no_probability_mass_in_limits():
    with pytest.raises(ValueError):
        tmvn_rvs([0.0], [[1.0]], lower=[40.0], upper=[50.0], size=10)


def test_tmvn_rvs_truncation_keeps_positive_correlation():
    np.random.seed(11)
    COV = np.array([[1.0, 0.7], [0.7, 1.0]])
    samples = tmvn_rvs([0.0, 0.0], COV, lower=[-2.0, -2.0],
                       upper=[2.0, 2.0], size=4000)
    assert np.all(np.abs(samples) < 2.0)
    assert np.corrcoef(samples.T)[0, 1] > 0.5


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_tmvn_mle_univariate_no_truncation():
    np.random.seed(20)
    data = np.random.normal(3.0, 2.0, size=500)
    mu, COV = tmvn_MLE(data)
    assert abs(float(mu) - np.mean(data)) < 0.1 * np.std(data)
    assert abs(float(COV) - np.var(data)) < 0.25 * np.var(data)


def test_tmvn_mle_univariate_shifted_scaled_data_and_list_input():
    np.random.seed(21)
    data = 100.0 + 5.0 * np.random.normal(size=300)
    mu, COV = tmvn_MLE(list(data))
    assert abs(float(mu) - 100.0) < 2.0
    assert 10.0 < float(COV) < 45.0


def test_tmvn_mle_multivariate_no_truncation():
    np.random.seed(22)
    true_mu = np.array([10.0, -5.0])
    true_COV = np.array([[4.0, 1.5], [1.5, 1.0]])
    data = np.random.multivariate_normal(true_mu, true_COV, size=800).T
    mu, COV = tmvn_MLE(data)
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    sample_std = np.std(data, axis=1)
    assert np.all(np.abs(mu - np.mean(data, axis=1)) < 0.15 * sample_std)
    assert np.allclose(np.diag(COV), np.var(data, axis=1), rtol=0.3)


def test_tmvn_mle_preserves_sample_correlation():
    np.random.seed(23)
    COV_true = np.array([[1.0, 0.6], [0.6, 2.0]])
    data = np.random.multivariate_normal([0.0, 0.0], COV_true, size=400).T
    mu, COV = tmvn_MLE(data)
    sig = np.sqrt(np.diag(COV))
    corr_fit = COV / np.outer(sig, sig)
    corr_sample = np.corrcoef(data)
    assert np.allclose(corr_fit, corr_sample, atol=1e-6)


def test_tmvn_mle_three_dimensions_shapes_and_symmetry():
    np.random.seed(24)
    A = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.2, 0.3, 1.0]])
    data = A @ np.random.normal(size=(3, 600))
    mu, COV = tmvn_MLE(data)
    assert mu.shape == (3,)
    assert COV.shape == (3, 3)
    assert np.allclose(COV, COV.T)
    assert np.all(np.linalg.eigvalsh(COV) > 0.0)


def test_tmvn_mle_does_not_modify_input_samples():
    np.random.seed(25)
    data = np.random.normal(size=(2, 200))
    data_copy = data.copy()
    tmvn_MLE(data)
    assert np.array_equal(data, data_copy)


def test_tmvn_mle_truncated_bivariate_inflates_variance():
    np.random.seed(26)
    samples = tmvn_rvs([0.0, 0.0], np.eye(2), lower=[-1.0, -1.0],
                       upper=[1.0, 1.0], size=1000).T
    mu, COV = tmvn_MLE(samples,
                       tr_lower=np.array([-1.0, -1.0]),
                       tr_upper=np.array([1.0, 1.0]))
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.all(np.abs(mu) < 0.2)
    var_fit = np.diag(COV)
    # accounting for truncation must give a larger variance than the raw
    # sample variance, and it should be in the vicinity of the true value (1)
    assert np.all(var_fit > np.var(samples, axis=1))
    assert np.all(var_fit > 0.5)
    assert np.all(var_fit < 2.0)


def test_tmvn_mle_censored_bivariate_inflates_variance():
    np.random.seed(27)
    raw = np.random.normal(size=(2, 1500))
    limit = 1.5
    detected = np.all(np.abs(raw) < limit, axis=0)
    samples = raw[:, detected]
    censored_count = int(np.sum(~detected))
    assert censored_count > 0

    _, COV_plain = tmvn_MLE(samples)
    mu, COV_cen = tmvn_MLE(samples,
                           censored_count=censored_count,
                           det_lower=np.array([-limit, -limit]),
                           det_upper=np.array([limit, limit]))
    assert mu.shape == (2,)
    assert np.all(np.isfinite(COV_cen))
    assert np.all(np.diag(COV_cen) > np.diag(COV_plain))
    assert np.all(np.abs(mu) < 0.3)
