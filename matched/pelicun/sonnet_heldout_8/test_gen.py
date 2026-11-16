import warnings

import numpy as np
import pytest

from pelicun.uq import tmvn_rvs, tmvn_MLE


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(12345)
    yield


# ---------------------------------------------------------------------------
# tmvn_rvs
# ---------------------------------------------------------------------------

def test_tmvn_rvs_untruncated_univariate_moments():
    samples = tmvn_rvs(mu=[2.0], COV=[[4.0]], size=20000)
    assert np.size(samples) == 20000
    assert np.mean(samples) == pytest.approx(2.0, abs=0.1)
    assert np.std(samples) == pytest.approx(2.0, rel=0.05)


def test_tmvn_rvs_scalar_inputs_untruncated():
    samples = tmvn_rvs(mu=1.0, COV=0.25, size=10000)
    assert np.size(samples) == 10000
    assert np.mean(samples) == pytest.approx(1.0, abs=0.05)
    assert np.std(samples) == pytest.approx(0.5, rel=0.05)


def test_tmvn_rvs_untruncated_multivariate_shape_and_cov():
    mu = np.array([0.0, 5.0])
    COV = np.array([[1.0, 0.6], [0.6, 2.0]])
    samples = tmvn_rvs(mu, COV, size=20000)
    assert samples.shape == (20000, 2)
    assert np.mean(samples, axis=0) == pytest.approx(mu, abs=0.1)
    assert np.cov(samples.T) == pytest.approx(COV, abs=0.1)


def test_tmvn_rvs_univariate_samples_within_bounds():
    samples = tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=-1.0, upper=1.0, size=500)
    assert np.size(samples) == 500
    assert np.all(samples > -1.0)
    assert np.all(samples < 1.0)


def test_tmvn_rvs_one_sided_lower_truncation_mean():
    # mean of a standard normal truncated below at zero is sqrt(2/pi)
    samples = tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=0.0, upper=np.inf,
                       size=20000)
    assert np.all(samples > 0.0)
    assert np.mean(samples) == pytest.approx(np.sqrt(2.0 / np.pi), abs=0.03)


def test_tmvn_rvs_only_upper_bound_given():
    samples = tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=None, upper=0.5, size=300)
    assert np.size(samples) == 300
    assert np.all(samples < 0.5)


def test_tmvn_rvs_multivariate_samples_within_bounds():
    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.5], [0.5, 1.0]])
    lower = np.array([-1.0, -1.5])
    upper = np.array([1.0, 2.0])
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=500)
    assert samples.shape == (500, 2)
    assert np.all(samples[:, 0] > lower[0]) and np.all(samples[:, 0] < upper[0])
    assert np.all(samples[:, 1] > lower[1]) and np.all(samples[:, 1] < upper[1])
    # the positive correlation should survive the truncation
    assert np.corrcoef(samples.T)[0, 1] > 0.2


def test_tmvn_rvs_multivariate_only_upper_bounds():
    mu = np.array([0.0, 0.0])
    COV = np.eye(2)
    samples = tmvn_rvs(mu, COV, lower=None, upper=np.array([0.0, 0.0]),
                       size=200)
    assert samples.shape == (200, 2)
    assert np.all(samples < 0.0)


def test_tmvn_rvs_multivariate_infinite_bound_in_one_dimension():
    mu = np.array([0.0, 0.0])
    COV = np.eye(2)
    lower = np.array([0.0, -np.inf])
    upper = np.array([np.inf, np.inf])
    samples = tmvn_rvs(mu, COV, lower=lower, upper=upper, size=400)
    assert samples.shape == (400, 2)
    assert np.all(samples[:, 0] > 0.0)
    # the second dimension is untruncated and should have values on both sides
    assert samples[:, 1].min() < 0.0 < samples[:, 1].max()


@pytest.mark.parametrize("size", [1, 7, 100])
def test_tmvn_rvs_returns_requested_number_of_samples(size):
    mu = np.array([0.0, 0.0])
    COV = np.eye(2)
    samples = tmvn_rvs(mu, COV, lower=np.array([-1.0, -1.0]),
                       upper=np.array([1.0, 1.0]), size=size)
    assert np.reshape(samples, (-1, 2)).shape[0] == size


def test_tmvn_rvs_is_reproducible_with_seed():
    args = dict(mu=np.array([0.0, 1.0]), COV=np.array([[1.0, 0.2], [0.2, 1.0]]),
                lower=np.array([-1.0, 0.0]), upper=np.array([1.0, 3.0]),
                size=50)
    np.random.seed(7)
    first = tmvn_rvs(**args)
    np.random.seed(7)
    second = tmvn_rvs(**args)
    assert np.array_equal(first, second)


def test_tmvn_rvs_raises_when_truncation_region_has_no_density():
    with pytest.raises(ValueError):
        tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=50.0, upper=60.0, size=10)


def test_tmvn_rvs_warns_for_low_density_region():
    with pytest.warns(UserWarning):
        samples = tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=3.5, upper=np.inf,
                           size=5)
    assert np.size(samples) == 5
    assert np.all(samples > 3.5)


# ---------------------------------------------------------------------------
# tmvn_MLE
# ---------------------------------------------------------------------------

def test_tmvn_MLE_univariate_untruncated_recovers_parameters():
    data = np.random.normal(5.0, 2.0, size=1000)
    mu, COV = tmvn_MLE(data)
    assert mu == pytest.approx(np.mean(data), abs=0.1)
    assert COV == pytest.approx(np.var(data), rel=0.1)


def test_tmvn_MLE_accepts_list_input():
    data = np.random.normal(-3.0, 0.5, size=300)
    mu, COV = tmvn_MLE(list(data))
    assert np.isfinite(mu)
    assert COV > 0.0
    assert mu == pytest.approx(-3.0, abs=0.15)


def test_tmvn_MLE_multivariate_untruncated():
    mu_true = np.array([1.0, 10.0])
    COV_true = np.array([[1.0, 0.5], [0.5, 4.0]])
    data = np.random.multivariate_normal(mu_true, COV_true, size=1000).T
    mu, COV = tmvn_MLE(data)

    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.allclose(COV, COV.T)
    assert np.all(np.diag(COV) > 0.0)

    sample_cov = np.cov(data)
    assert mu == pytest.approx(np.mean(data, axis=1), abs=0.15)
    assert np.diag(COV) == pytest.approx(np.diag(sample_cov), rel=0.15)
    corr = COV[0, 1] / np.sqrt(COV[0, 0] * COV[1, 1])
    sample_corr = np.corrcoef(data)[0, 1]
    assert corr == pytest.approx(sample_corr, abs=0.05)


def test_tmvn_MLE_truncated_univariate_inflates_variance():
    # data from N(0, 1) truncated to [-1, 1]; the truncated-MLE should
    # estimate a variance well above the variance of the truncated data
    data = tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=-1.0, upper=1.0, size=2000)
    data = np.asarray(data).flatten()
    mu, COV = tmvn_MLE(data, tr_lower=-1.0, tr_upper=1.0)
    assert abs(mu) < 0.3
    assert COV > np.var(data)
    assert 0.5 < COV < 2.0


def test_tmvn_MLE_censored_univariate_corrects_bias():
    full = np.random.normal(0.0, 1.0, size=1500)
    det_upper = 1.0
    observed = full[full <= det_upper]
    censored_count = int(np.sum(full > det_upper))
    assert censored_count > 0

    mu, COV = tmvn_MLE(observed, censored_count=censored_count,
                       det_lower=-np.inf, det_upper=det_upper)
    # the naive estimates from observed data are strongly biased low
    assert np.mean(observed) < -0.15
    assert abs(mu) < 0.3
    assert 0.6 < COV < 1.6


def test_tmvn_MLE_multivariate_truncated_returns_valid_output():
    data = tmvn_rvs(np.array([0.0, 0.0]), np.eye(2),
                    lower=np.array([-2.0, -2.0]),
                    upper=np.array([2.0, 2.0]), size=400).T
    mu, COV = tmvn_MLE(data, tr_lower=np.array([-2.0, -2.0]),
                       tr_upper=np.array([2.0, 2.0]))
    assert mu.shape == (2,)
    assert COV.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.diag(COV) > 0.0)
    assert np.allclose(COV, COV.T)
    assert np.all(np.abs(mu) < 0.4)


def test_tmvn_MLE_constant_samples_give_negligible_variance():
    data = np.full(20, 3.0)
    mu, COV = tmvn_MLE(data)
    assert mu == pytest.approx(3.0, abs=1e-4)
    assert 0.0 <= COV < 1e-6


def test_tmvn_MLE_single_sample_does_not_optimize():
    mu, COV = tmvn_MLE(np.array([2.0]))
    assert mu == pytest.approx(2.0, abs=1e-4)
    assert np.isfinite(COV)
    assert COV >= 0.0
