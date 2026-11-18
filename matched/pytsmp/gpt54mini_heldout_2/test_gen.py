import numpy as np
import pytest

from pytsmp.utils import (
    calculate_distance_profile,
    rolling_avg_sd,
    sliding_dot_product,
)


def test_sliding_dot_product_basic_matches_manual():
    Q = np.array([1.0, 2.0])
    T = np.array([3.0, 4.0, 5.0, 6.0])

    result = sliding_dot_product(Q, T)

    expected = np.array([
        np.dot(Q, T[0:2]),
        np.dot(Q, T[1:3]),
        np.dot(Q, T[2:4]),
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert result.shape == (3,)


def test_sliding_dot_product_with_negative_values():
    Q = np.array([2.0, -1.0, 3.0])
    T = np.array([0.0, 1.0, -2.0, 4.0, 5.0])

    result = sliding_dot_product(Q, T)

    expected = np.array([
        np.dot(Q, T[0:3]),
        np.dot(Q, T[1:4]),
        np.dot(Q, T[2:5]),
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_with_odd_length_time_series():
    Q = np.array([1.5, -2.0])
    T = np.array([2.0, 0.0, 1.0, -1.0, 3.0])

    result = sliding_dot_product(Q, T)

    expected = np.array([np.dot(Q, T[i:i + len(Q)]) for i in range(len(T) - len(Q) + 1)])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_raises_when_query_longer_than_series():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([4.0, 5.0])

    with pytest.raises(ValueError, match="at least as long as Q"):
        sliding_dot_product(Q, T)


def test_sliding_dot_product_single_element_query():
    Q = np.array([2.5])
    T = np.array([-1.0, 0.0, 4.0])

    result = sliding_dot_product(Q, T)

    expected = Q[0] * T
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_rolling_avg_sd_basic_matches_manual():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    window = 2

    mean, sd = rolling_avg_sd(t, window)

    expected_mean = np.array([1.5, 2.5, 3.5])
    expected_sd = np.array([0.5, 0.5, 0.5])

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sd, expected_sd, rtol=1e-12, atol=1e-12)


def test_rolling_avg_sd_window_one_returns_values_and_zero_sd():
    t = np.array([3.0, -2.0, 7.5])
    mean, sd = rolling_avg_sd(t, 1)

    np.testing.assert_allclose(mean, t, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sd, np.zeros_like(t), rtol=1e-12, atol=1e-12)


def test_rolling_avg_sd_window_equals_length():
    t = np.array([2.0, 4.0, 6.0, 8.0])
    mean, sd = rolling_avg_sd(t, len(t))

    np.testing.assert_allclose(mean, np.array([5.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sd, np.array([np.std(t)]), rtol=1e-12, atol=1e-12)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.array([5.0, 5.0, 5.0, 5.0])
    mean, sd = rolling_avg_sd(t, 3)

    np.testing.assert_allclose(mean, np.array([5.0, 5.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sd, np.array([0.0, 0.0]), rtol=1e-12, atol=1e-12)


def test_rolling_avg_sd_raises_when_window_too_large():
    t = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="smaller than the length of time series"):
        rolling_avg_sd(t, 3)


def test_calculate_distance_profile_identical_nonconstant_series_zero_distance():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    np.testing.assert_allclose(D[0], 0.0, rtol=1e-12, atol=1e-12)
    assert D.shape == (3,)


def test_calculate_distance_profile_matches_manual_z_normalized_distance():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([2.0, 4.0, 6.0, 1.0, 2.0, 3.0])

    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

    expected = []
    for i in range(len(T) - m + 1):
        subseq = T[i:i + m]
        qz = (Q - mean_Q) / sigma_Q
        tz = (subseq - np.mean(subseq)) / np.std(subseq)
        expected.append(np.linalg.norm(qz - tz))
    expected = np.array(expected)

    np.testing.assert_allclose(D, expected, rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_constant_query_against_nonconstant_series_is_sqrt_m():
    Q = np.array([2.0, 2.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])

    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)

    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)

    np.testing.assert_allclose(D, np.full(len(T) - m + 1, np.sqrt(m)), rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_constant_query_and_constant_series_is_zero():
    Q = np.array([5.0, 5.0, 5.0])
    T = np.array([1.0, 1.0, 1.0, 1.0])

    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)

    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)

    np.testing.assert_allclose(D, np.zeros(len(T) - m + 1), rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_nonconstant_query_against_constant_series_is_sqrt_m():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([7.0, 7.0, 7.0, 7.0, 7.0])

    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)

    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)

    np.testing.assert_allclose(D, np.full(len(T) - m + 1, np.sqrt(m)), rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_dimension_mismatch_raises():
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([0.1, 0.2])
    mean_Q = 1.0
    sigma_Q = 1.0

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_dimension_mismatch_with_array_query_stats_raises():
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 2.0, 3.0])
    sigma_T = np.array([0.1, 0.2, 0.3])
    mean_Q = np.array([1.0, 2.0])
    sigma_Q = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)
