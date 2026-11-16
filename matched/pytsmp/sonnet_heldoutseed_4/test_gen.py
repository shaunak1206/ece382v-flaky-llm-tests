import numpy as np
import pytest

from pytsmp import utils


def _naive_sliding_dot(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _naive_rolling(t, w):
    means = np.array([np.mean(t[i:i + w]) for i in range(len(t) - w + 1)])
    sds = np.array([np.std(t[i:i + w]) for i in range(len(t) - w + 1)])
    return means, sds


def _naive_distance_profile(Q, T):
    m = len(Q)
    zq = (Q - np.mean(Q)) / np.std(Q)
    out = []
    for i in range(len(T) - m + 1):
        sub = T[i:i + m]
        zs = (sub - np.mean(sub)) / np.std(sub)
        out.append(np.linalg.norm(zq - zs))
    return np.array(out)


# ---------------- sliding_dot_product ----------------

def test_sliding_dot_product_matches_naive_even_length():
    rs = np.random.RandomState(0)
    T = rs.randn(50)
    Q = rs.randn(8)
    result = utils.sliding_dot_product(Q, T)
    assert np.allclose(result, _naive_sliding_dot(Q, T))


def test_sliding_dot_product_matches_naive_odd_length():
    rs = np.random.RandomState(1)
    T = rs.randn(51)
    Q = rs.randn(7)
    result = utils.sliding_dot_product(Q, T)
    assert np.allclose(result, _naive_sliding_dot(Q, T))


def test_sliding_dot_product_output_length():
    T = np.arange(20, dtype=float)
    Q = np.arange(5, dtype=float)
    assert utils.sliding_dot_product(Q, T).shape == (16,)


def test_sliding_dot_product_known_values():
    T = np.array([1.0, 2.0, 3.0, 4.0])
    Q = np.array([1.0, 1.0])
    assert np.allclose(utils.sliding_dot_product(Q, T), [3.0, 5.0, 7.0])


def test_sliding_dot_product_equal_length():
    rs = np.random.RandomState(2)
    T = rs.randn(10)
    Q = rs.randn(10)
    result = utils.sliding_dot_product(Q, T)
    assert result.shape == (1,)
    assert np.allclose(result, [np.dot(Q, T)])


def test_sliding_dot_product_query_longer_raises():
    with pytest.raises(ValueError):
        utils.sliding_dot_product(np.arange(6, dtype=float), np.arange(5, dtype=float))


def test_sliding_dot_product_length_one_query():
    T = np.array([1.0, -2.0, 3.0, 4.0, 5.0])
    Q = np.array([2.0])
    assert np.allclose(utils.sliding_dot_product(Q, T), 2 * T)


# ---------------- rolling_avg_sd ----------------

def test_rolling_avg_sd_matches_naive():
    rs = np.random.RandomState(3)
    t = rs.randn(60)
    mean, sd = utils.rolling_avg_sd(t, 10)
    exp_mean, exp_sd = _naive_rolling(t, 10)
    assert np.allclose(mean, exp_mean)
    assert np.allclose(sd, exp_sd, atol=1e-6)


def test_rolling_avg_sd_output_shapes():
    t = np.arange(30, dtype=float)
    mean, sd = utils.rolling_avg_sd(t, 7)
    assert mean.shape == (24,)
    assert sd.shape == (24,)


def test_rolling_avg_sd_window_larger_than_series_raises():
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(np.arange(5, dtype=float), 6)


def test_rolling_avg_sd_window_equals_length():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, sd = utils.rolling_avg_sd(t, 5)
    assert mean.shape == (1,)
    assert np.allclose(mean, [3.0])
    assert np.allclose(sd, [np.std(t)])


def test_rolling_avg_sd_window_one():
    t = np.array([3.0, -1.0, 4.0, 1.0])
    mean, sd = utils.rolling_avg_sd(t, 1)
    assert np.allclose(mean, t)
    assert np.allclose(sd, 0, atol=1e-6)


def test_rolling_avg_sd_constant_series_zero_sd():
    t = np.full(12, 5.0)
    mean, sd = utils.rolling_avg_sd(t, 4)
    assert np.allclose(mean, 5.0)
    assert np.allclose(sd, 0.0)
    assert np.all(sd >= 0)


# ---------------- calculate_distance_profile ----------------

def test_calculate_distance_profile_matches_naive():
    rs = np.random.RandomState(4)
    T = rs.randn(40)
    Q = rs.randn(8)
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sd_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sd_T)
    assert D.shape == (len(T) - m + 1,)
    assert np.allclose(D, _naive_distance_profile(Q, T), atol=1e-5)


def test_calculate_distance_profile_self_match_is_zero():
    rs = np.random.RandomState(5)
    T = rs.randn(30)
    m = 6
    Q = T[10:10 + m].copy()
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sd_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sd_T)
    assert D[10] == pytest.approx(0, abs=1e-5)
    assert np.all(D >= 0)


def test_calculate_distance_profile_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(5), 4, 0.0, 1.0, np.zeros(4), np.ones(4))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(4), 4, 0.0, 1.0, np.zeros(4), np.ones(3))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(4), 4, np.zeros(3), np.ones(3), np.zeros(4), np.ones(4))


def test_calculate_distance_profile_constant_query():
    m = 4
    QT = np.zeros(3)
    mean_T = np.array([0.0, 1.0, 2.0])
    sd_T = np.array([1.0, 0.0, 2.0])
    D = utils.calculate_distance_profile(QT, m, 5.0, 0.0, mean_T, sd_T)
    # constant vs constant -> 0; constant vs non-constant -> sqrt(m)
    assert np.allclose(D, [2.0, 0.0, 2.0])


def test_calculate_distance_profile_constant_subsequence_nonconstant_query():
    m = 9
    QT = np.array([1.0, 2.0])
    mean_T = np.array([1.0, 1.0])
    sd_T = np.array([0.0, 0.0])
    D = utils.calculate_distance_profile(QT, m, 0.5, 1.5, mean_T, sd_T)
    assert np.allclose(D, [3.0, 3.0])


def test_calculate_distance_profile_custom_epsilon_and_clamping():
    m = 4
    # custom epsilon: sigma_T = 0.05 treated as constant when epsilon = 0.1
    D = utils.calculate_distance_profile(np.zeros(2), m, 0.0, 1.0,
                                         np.zeros(2), np.array([0.05, 1.0]), epsilon=0.1)
    assert D[0] == pytest.approx(2.0)
    assert D[1] == pytest.approx(np.sqrt(8))
    # negative values inside the sqrt are clamped to zero (no NaN)
    D2 = utils.calculate_distance_profile(np.array([10.0]), m, 0.0, 1.0,
                                          np.zeros(1), np.ones(1))
    assert D2[0] == 0.0
    assert not np.any(np.isnan(D2))
