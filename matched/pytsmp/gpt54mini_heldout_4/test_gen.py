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
    assert np.allclose(result, expected)


def test_sliding_dot_product_query_length_one():
    Q = np.array([2.5])
    T = np.array([1.0, -2.0, 3.0])

    result = sliding_dot_product(Q, T)

    assert np.allclose(result, 2.5 * T)


def test_sliding_dot_product_identical_query_and_series():
    Q = np.array([1.0, -1.0, 2.0])
    T = Q.copy()

    result = sliding_dot_product(Q, T)

    assert result.shape == (1,)
    assert np.allclose(result, np.array([np.dot(Q, T)]))


def test_sliding_dot_product_raises_when_query_longer_than_series():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="T should be a series at least as long as Q"):
        sliding_dot_product(Q, T)


def test_sliding_dot_product_handles_odd_length_series():
    Q = np.array([1.0, 0.0, -1.0])
    T = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    result = sliding_dot_product(Q, T)

    expected = np.array([
        np.dot(Q, T[0:3]),
        np.dot(Q, T[1:4]),
        np.dot(Q, T[2:5]),
    ])
    assert np.allclose(result, expected)


def test_rolling_avg_sd_basic_matches_manual():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    window = 2

    mean, sd = rolling_avg_sd(t, window)

    assert np.allclose(mean, np.array([1.5, 2.5, 3.5]))
    assert np.allclose(sd, np.array([0.5, 0.5, 0.5]))


def test_rolling_avg_sd_window_equals_length():
    t = np.array([2.0, 4.0, 6.0])
    window = 3

    mean, sd = rolling_avg_sd(t, window)

    assert mean.shape == (1,)
    assert sd.shape == (1,)
    assert np.allclose(mean, np.array([4.0]))
    assert np.allclose(sd, np.array([np.std(t)]))


def test_rolling_avg_sd_window_one_returns_original_and_zeros():
    t = np.array([5.0, -1.0, 7.0])
    window = 1

    mean, sd = rolling_avg_sd(t, window)

    assert np.allclose(mean, t)
    assert np.allclose(sd, np.zeros_like(t))


def test_rolling_avg_sd_raises_when_window_too_large():
    t = np.array([1.0, 2.0])
    window = 3

    with pytest.raises(ValueError, match="Window size should be smaller than the length of time series"):
        rolling_avg_sd(t, window)


def test_rolling_avg_sd_constant_series_has_zero_std():
    t = np.array([3.0, 3.0, 3.0, 3.0])
    window = 2

    mean, sd = rolling_avg_sd(t, window)

    assert np.allclose(mean, np.array([3.0, 3.0, 3.0]))
    assert np.allclose(sd, np.zeros(3))


def test_calculate_distance_profile_basic_non_constant_series():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    expected = np.array([
        0.0,
        np.sqrt(2.0),
        2.0,
    ])
    assert np.allclose(D, expected)


def test_calculate_distance_profile_identical_series_gives_zero():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([1.0, 2.0, 3.0])
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert D.shape == (1,)
    assert np.allclose(D, np.array([0.0]))


def test_calculate_distance_profile_constant_query_and_constant_target_returns_zero():
    QT = np.array([6.0, 6.0, 6.0])
    m = 3
    mean_Q = np.array([2.0, 2.0, 2.0])
    sigma_Q = np.array([0.0, 0.0, 0.0])
    mean_T = np.array([2.0, 2.0, 2.0])
    sigma_T = np.array([0.0, 0.0, 0.0])

    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.zeros(3))


def test_calculate_distance_profile_constant_query_nonconstant_target_returns_sqrt_m():
    QT = np.array([1.0, 2.0, 3.0])
    m = 4
    mean_Q = np.array([1.0, 1.0, 1.0])
    sigma_Q = np.array([0.0, 0.0, 0.0])
    mean_T = np.array([0.5, 1.0, 1.5])
    sigma_T = np.array([1.0, 1.0, 1.0])

    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.full(3, np.sqrt(m)))


def test_calculate_distance_profile_nonconstant_query_constant_target_returns_sqrt_m():
    QT = np.array([1.0, 2.0])
    m = 5
    mean_Q = 1.0
    sigma_Q = 2.0
    mean_T = np.array([0.0, 1.0])
    sigma_T = np.array([0.0, 0.0])

    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.full(2, np.sqrt(m)))


def test_calculate_distance_profile_raises_on_dimension_mismatch():
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([1.0, 1.0])
    mean_Q = 1.0
    sigma_Q = 1.0

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)
