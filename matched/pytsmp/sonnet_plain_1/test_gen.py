import numpy as np
import pytest

from pytsmp.utils import (
    calculate_distance_profile,
    rolling_avg_sd,
    sliding_dot_product,
)


def _naive_dot(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _naive_znorm_dist(Q, T):
    m = len(Q)
    zq = (Q - np.mean(Q)) / np.std(Q)
    out = []
    for i in range(len(T) - m + 1):
        sub = T[i:i + m]
        zs = (sub - np.mean(sub)) / np.std(sub)
        out.append(np.linalg.norm(zq - zs))
    return np.array(out)


# ---------------------------------------------------------------- sliding_dot_product

def test_sliding_dot_product_matches_naive_even_length():
    rng = np.random.RandomState(0)
    T = rng.randn(20)
    Q = rng.randn(5)
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, _naive_dot(Q, T), atol=1e-8)


def test_sliding_dot_product_matches_naive_odd_length():
    rng = np.random.RandomState(1)
    T = rng.randn(21)
    Q = rng.randn(6)
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, _naive_dot(Q, T), atol=1e-8)


def test_sliding_dot_product_output_shape():
    T = np.arange(15, dtype=float)
    Q = np.arange(4, dtype=float)
    result = sliding_dot_product(Q, T)
    assert result.shape == (len(T) - len(Q) + 1,)


def test_sliding_dot_product_known_values():
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Q = np.array([1.0, 1.0])
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, [3.0, 5.0, 7.0, 9.0, 11.0], atol=1e-8)


def test_sliding_dot_product_query_same_length_as_series():
    rng = np.random.RandomState(2)
    T = rng.randn(8)
    Q = rng.randn(8)
    result = sliding_dot_product(Q, T)
    assert result.shape == (1,)
    np.testing.assert_allclose(result[0], np.dot(Q, T), atol=1e-8)


def test_sliding_dot_product_query_same_length_odd():
    rng = np.random.RandomState(3)
    T = rng.randn(9)
    Q = rng.randn(9)
    result = sliding_dot_product(Q, T)
    assert result.shape == (1,)
    np.testing.assert_allclose(result[0], np.dot(Q, T), atol=1e-8)


def test_sliding_dot_product_query_longer_than_series_raises():
    with pytest.raises(ValueError):
        sliding_dot_product(np.ones(5), np.ones(4))


def test_sliding_dot_product_length_one_query():
    T = np.array([2.0, -1.0, 4.0, 0.5, 3.0])
    Q = np.array([3.0])
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, 3.0 * T, atol=1e-8)


def test_sliding_dot_product_does_not_modify_inputs():
    rng = np.random.RandomState(4)
    T = rng.randn(11)
    Q = rng.randn(3)
    T_copy = T.copy()
    Q_copy = Q.copy()
    sliding_dot_product(Q, T)
    np.testing.assert_array_equal(T, T_copy)
    np.testing.assert_array_equal(Q, Q_copy)


# ---------------------------------------------------------------- rolling_avg_sd

def test_rolling_avg_sd_matches_numpy():
    rng = np.random.RandomState(5)
    t = rng.randn(30)
    w = 7
    mean, sd = rolling_avg_sd(t, w)
    expected_mean = np.array([np.mean(t[i:i + w]) for i in range(len(t) - w + 1)])
    expected_sd = np.array([np.std(t[i:i + w]) for i in range(len(t) - w + 1)])
    np.testing.assert_allclose(mean, expected_mean, atol=1e-8)
    np.testing.assert_allclose(sd, expected_sd, atol=1e-6)


def test_rolling_avg_sd_output_shapes():
    t = np.arange(12, dtype=float)
    mean, sd = rolling_avg_sd(t, 5)
    assert mean.shape == (8,)
    assert sd.shape == (8,)


def test_rolling_avg_sd_window_larger_than_series_raises():
    with pytest.raises(ValueError):
        rolling_avg_sd(np.arange(4, dtype=float), 5)


def test_rolling_avg_sd_window_equals_length():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = rolling_avg_sd(t, 4)
    assert mean.shape == (1,)
    np.testing.assert_allclose(mean[0], 2.5, atol=1e-10)
    np.testing.assert_allclose(sd[0], np.std(t), atol=1e-10)


def test_rolling_avg_sd_window_one_gives_series_and_zero_sd():
    t = np.array([3.0, -1.0, 4.0, 1.5])
    mean, sd = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(mean, t, atol=1e-10)
    np.testing.assert_allclose(sd, np.zeros(4), atol=1e-6)


def test_rolling_avg_sd_constant_series_has_zero_sd_and_nonnegative():
    t = np.full(10, 2.5)
    mean, sd = rolling_avg_sd(t, 4)
    np.testing.assert_allclose(mean, np.full(7, 2.5), atol=1e-10)
    assert np.all(sd >= 0)
    np.testing.assert_allclose(sd, np.zeros(7), atol=1e-6)


# ---------------------------------------------------------------- calculate_distance_profile

def test_distance_profile_matches_naive_znorm_distance():
    rng = np.random.RandomState(6)
    T = rng.randn(40)
    Q = rng.randn(8)
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sd_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sd_T)
    np.testing.assert_allclose(D, _naive_znorm_dist(Q, T), atol=1e-5)


def test_distance_profile_self_match_is_zero():
    rng = np.random.RandomState(7)
    T = rng.randn(30)
    m = 6
    Q = T[10:10 + m].copy()
    QT = sliding_dot_product(Q, T)
    mean_T, sd_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sd_T)
    assert D.shape == (len(T) - m + 1,)
    assert D[10] == pytest.approx(0.0, abs=1e-3)
    assert np.all(D >= 0)


def test_distance_profile_constant_query_gives_sqrt_m_for_nonconstant_windows():
    m = 4
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([0.0, 1.0, 2.0])
    sigma_T = np.array([1.0, 2.0, 0.5])
    D = calculate_distance_profile(QT, m, 3.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.full(3, np.sqrt(m)))


def test_distance_profile_both_constant_gives_zero():
    m = 5
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 1.0, 1.0])
    sigma_T = np.zeros(3)
    D = calculate_distance_profile(QT, m, 2.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.zeros(3))


def test_distance_profile_constant_window_in_series_gives_sqrt_m():
    m = 4
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([0.0, 1.0, 2.0])
    sigma_T = np.array([1.0, 0.0, 1.0])
    D = calculate_distance_profile(QT, m, 0.5, 1.0, mean_T, sigma_T)
    assert D[1] == pytest.approx(np.sqrt(m))
    assert np.all(np.isfinite(D))


def test_distance_profile_epsilon_treats_small_sigma_as_constant():
    m = 9
    QT = np.array([1.0, 2.0])
    mean_T = np.array([0.0, 0.0])
    sigma_T = np.array([1e-9, 1.0])
    D = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert D[0] == pytest.approx(3.0)


def test_distance_profile_qt_length_mismatch_raises():
    with pytest.raises(ValueError):
        calculate_distance_profile(np.ones(3), 4, 0.0, 1.0, np.ones(4), np.ones(4))


def test_distance_profile_mean_sigma_length_mismatch_raises():
    with pytest.raises(ValueError):
        calculate_distance_profile(np.ones(3), 4, 0.0, 1.0, np.ones(3), np.ones(2))


def test_distance_profile_array_query_stats_length_mismatch_raises():
    with pytest.raises(ValueError):
        calculate_distance_profile(np.ones(3), 4, np.zeros(2), np.ones(2), np.ones(3), np.ones(3))
