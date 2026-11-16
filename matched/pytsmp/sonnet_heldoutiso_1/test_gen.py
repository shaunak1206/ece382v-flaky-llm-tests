import numpy as np
import pytest

from pytsmp import utils


def naive_sliding_dot(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def naive_rolling(t, w):
    means = np.array([np.mean(t[i:i + w]) for i in range(len(t) - w + 1)])
    sds = np.array([np.std(t[i:i + w]) for i in range(len(t) - w + 1)])
    return means, sds


def naive_distance_profile(Q, T):
    m = len(Q)
    zq = (Q - np.mean(Q)) / np.std(Q)
    out = []
    for i in range(len(T) - m + 1):
        sub = T[i:i + m]
        zs = (sub - np.mean(sub)) / np.std(sub)
        out.append(np.linalg.norm(zq - zs))
    return np.array(out)


# ---------------- sliding_dot_product ----------------

@pytest.mark.parametrize("n,m", [(10, 3), (11, 3), (16, 4), (17, 5), (30, 29)])
def test_sliding_dot_product_matches_naive(n, m):
    rng = np.random.RandomState(0)
    T = rng.rand(n)
    Q = rng.rand(m)
    result = utils.sliding_dot_product(Q, T)
    assert result.shape == (n - m + 1,)
    np.testing.assert_allclose(result, naive_sliding_dot(Q, T), atol=1e-8)


def test_sliding_dot_product_equal_length():
    rng = np.random.RandomState(1)
    for n in (8, 9):
        T = rng.rand(n)
        Q = rng.rand(n)
        result = utils.sliding_dot_product(Q, T)
        assert result.shape == (1,)
        np.testing.assert_allclose(result, [np.dot(Q, T)], atol=1e-8)


def test_sliding_dot_product_query_length_one():
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Q = np.array([2.0])
    result = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, 2.0 * T, atol=1e-8)


def test_sliding_dot_product_known_values():
    T = np.array([1.0, 2.0, 3.0, 4.0])
    Q = np.array([1.0, 1.0])
    np.testing.assert_allclose(utils.sliding_dot_product(Q, T), [3.0, 5.0, 7.0], atol=1e-8)


def test_sliding_dot_product_query_longer_raises():
    with pytest.raises(ValueError):
        utils.sliding_dot_product(np.arange(5.0), np.arange(4.0))


# ---------------- rolling_avg_sd ----------------

@pytest.mark.parametrize("n,w", [(10, 3), (15, 4), (20, 7)])
def test_rolling_avg_sd_matches_naive(n, w):
    rng = np.random.RandomState(2)
    t = rng.rand(n)
    mean, sd = utils.rolling_avg_sd(t, w)
    exp_mean, exp_sd = naive_rolling(t, w)
    assert mean.shape == (n - w + 1,)
    assert sd.shape == (n - w + 1,)
    np.testing.assert_allclose(mean, exp_mean, atol=1e-8)
    np.testing.assert_allclose(sd, exp_sd, atol=1e-6)


def test_rolling_avg_sd_window_equals_length():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = utils.rolling_avg_sd(t, 4)
    assert mean.shape == (1,)
    np.testing.assert_allclose(mean, [2.5])
    np.testing.assert_allclose(sd, [np.std(t)], atol=1e-8)


def test_rolling_avg_sd_window_one():
    t = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    mean, sd = utils.rolling_avg_sd(t, 1)
    np.testing.assert_allclose(mean, t)
    np.testing.assert_allclose(sd, np.zeros(len(t)), atol=1e-6)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(10, 5.0)
    mean, sd = utils.rolling_avg_sd(t, 4)
    np.testing.assert_allclose(mean, np.full(7, 5.0))
    np.testing.assert_allclose(sd, np.zeros(7), atol=1e-6)
    assert np.all(sd >= 0)


def test_rolling_avg_sd_window_too_large_raises():
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(np.arange(3.0), 4)


# ---------------- calculate_distance_profile ----------------

def _profile_inputs(Q, T):
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    return QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T


def test_distance_profile_matches_naive():
    rng = np.random.RandomState(3)
    T = rng.rand(40)
    Q = rng.rand(6)
    D = utils.calculate_distance_profile(*_profile_inputs(Q, T))
    assert D.shape == (35,)
    np.testing.assert_allclose(D, naive_distance_profile(Q, T), atol=1e-5)


def test_distance_profile_self_match_is_zero_and_nonnegative():
    rng = np.random.RandomState(4)
    T = rng.rand(30)
    Q = T[10:18].copy()
    D = utils.calculate_distance_profile(*_profile_inputs(Q, T))
    assert D[10] == pytest.approx(0, abs=1e-5)
    assert np.all(D >= 0)
    assert not np.any(np.isnan(D))


def test_distance_profile_array_query_stats():
    rng = np.random.RandomState(5)
    T = rng.rand(20)
    Q = rng.rand(5)
    QT, m, mean_Q, sigma_Q, mean_T, sigma_T = _profile_inputs(Q, T)
    k = len(QT)
    D_scalar = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    D_array = utils.calculate_distance_profile(QT, m, np.full(k, mean_Q), np.full(k, sigma_Q),
                                               mean_T, sigma_T)
    np.testing.assert_allclose(D_scalar, D_array, atol=1e-10)


def test_distance_profile_constant_subsequences():
    m = 4
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 1.0, 1.0])
    sigma_T = np.array([0.0, 0.5, 0.0])
    # non-constant query: constant subsequences get sqrt(m)
    D = utils.calculate_distance_profile(QT, m, 1.0, 1.0, mean_T, sigma_T)
    assert D[0] == pytest.approx(np.sqrt(m))
    assert D[2] == pytest.approx(np.sqrt(m))
    assert not np.isnan(D[1])


def test_distance_profile_constant_query():
    m = 4
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 1.0, 1.0])
    sigma_T = np.array([0.0, 0.5, 2.0])
    D = utils.calculate_distance_profile(QT, m, 1.0, 0.0, mean_T, sigma_T)
    # constant vs constant -> 0; constant vs non-constant -> sqrt(m)
    np.testing.assert_allclose(D, [0.0, 2.0, 2.0])


def test_distance_profile_custom_epsilon():
    m = 9
    QT = np.array([1.0])
    D = utils.calculate_distance_profile(QT, m, 0.0, 1.0, np.array([0.0]), np.array([0.5]),
                                         epsilon=1.0)
    assert D[0] == pytest.approx(3.0)


def test_distance_profile_dimension_mismatch_raises():
    QT = np.ones(3)
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, 0.0, 1.0, np.ones(2), np.ones(3))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, 0.0, 1.0, np.ones(3), np.ones(2))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, np.ones(2), np.ones(2), np.ones(3), np.ones(3))
