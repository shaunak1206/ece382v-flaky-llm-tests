import numpy as np
import pytest

from pytsmp.utils import (
    calculate_distance_profile,
    rolling_avg_sd,
    sliding_dot_product,
)


def test_sliding_dot_product_matches_direct_computation_even_length():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

    result = sliding_dot_product(Q, T)
    expected = np.array([
        np.dot(Q, T[i : i + len(Q)])
        for i in range(len(T) - len(Q) + 1)
    ])

    assert np.allclose(result, expected)


def test_sliding_dot_product_matches_direct_computation_odd_length():
    Q = np.array([2.0, -1.0])
    T = np.array([3.0, 0.0, 4.0, -2.0, 1.0])

    result = sliding_dot_product(Q, T)
    expected = np.array([
        np.dot(Q, T[i : i + len(Q)])
        for i in range(len(T) - len(Q) + 1)
    ])

    assert np.allclose(result, expected)


def test_sliding_dot_product_with_single_element_query():
    Q = np.array([2.5])
    T = np.array([1.0, -2.0, 3.0])

    result = sliding_dot_product(Q, T)

    assert np.allclose(result, Q[0] * T)


def test_sliding_dot_product_raises_when_query_longer_than_series():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([4.0, 5.0])

    with pytest.raises(ValueError, match="T should be a series at least as long as Q"):
        sliding_dot_product(Q, T)


def test_rolling_avg_sd_basic_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = rolling_avg_sd(t, 2)

    assert np.allclose(mean, np.array([1.5, 2.5, 3.5]))
    assert np.allclose(sd, np.array([0.5, 0.5, 0.5]))


def test_rolling_avg_sd_window_equals_length():
    t = np.array([1.0, 2.0, 3.0])

    mean, sd = rolling_avg_sd(t, 3)

    assert mean.shape == (1,)
    assert sd.shape == (1,)
    assert np.allclose(mean, np.array([2.0]))
    assert np.allclose(sd, np.array([np.std(t)]))


def test_rolling_avg_sd_constant_series_returns_zero_sd():
    t = np.array([5.0, 5.0, 5.0, 5.0])

    mean, sd = rolling_avg_sd(t, 2)

    assert np.allclose(mean, np.array([5.0, 5.0, 5.0]))
    assert np.allclose(sd, np.zeros(3))


def test_rolling_avg_sd_raises_when_window_too_large():
    t = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Window size should be smaller than the length of time series."):
        rolling_avg_sd(t, 3)


def test_rolling_avg_sd_handles_large_numbers_without_negative_variance():
    t = np.array([1e12, 1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0])

    mean, sd = rolling_avg_sd(t, 2)

    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(sd))
    assert np.all(sd >= 0)


def test_calculate_distance_profile_identical_nonconstant_series_is_zero():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([1.0, 2.0, 3.0])

    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.zeros(1))


def test_calculate_distance_profile_constant_query_and_constant_series_is_zero():
    Q = np.array([2.0, 2.0])
    T = np.array([2.0, 2.0, 2.0])

    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.zeros(len(T) - len(Q) + 1))


def test_calculate_distance_profile_constant_query_nonconstant_series_is_sqrt_m():
    Q = np.array([2.0, 2.0])
    T = np.array([1.0, 2.0, 3.0])

    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.full(len(T) - len(Q) + 1, np.sqrt(len(Q))))


def test_calculate_distance_profile_nonconstant_query_constant_series_is_sqrt_m():
    Q = np.array([1.0, 2.0])
    T = np.array([5.0, 5.0, 5.0])

    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.full(len(T) - len(Q) + 1, np.sqrt(len(Q))))


def test_calculate_distance_profile_dimension_mismatch_raises():
    QT = np.array([1.0, 2.0])
    mean_T = np.array([1.0, 2.0, 3.0])
    sigma_T = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Input dimension mismatch."):
        calculate_distance_profile(QT, 2, 0.0, 1.0, mean_T, sigma_T)


def test_calculate_distance_profile_mismatched_mean_and_sigma_lengths_raise():
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 2.0, 3.0])
    sigma_T = np.array([1.0, 2.0, 3.0])
    mean_Q = np.array([0.0, 1.0])
    sigma_Q = np.array([1.0])

    with pytest.raises(ValueError, match="Input dimension mismatch."):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)
