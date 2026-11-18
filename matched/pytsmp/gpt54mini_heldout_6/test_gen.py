import numpy as np
import pytest

from pytsmp.utils import (
    calculate_distance_profile,
    rolling_avg_sd,
    sliding_dot_product,
)


def test_sliding_dot_product_matches_manual_result_small_sequences():
    Q = np.array([1.0, 2.0])
    T = np.array([3.0, 4.0, 5.0, 6.0])

    result = sliding_dot_product(Q, T)

    expected = np.array([
        np.dot(Q, T[0:2]),
        np.dot(Q, T[1:3]),
        np.dot(Q, T[2:4]),
    ])
    assert np.allclose(result, expected)


def test_sliding_dot_product_with_negative_and_float_values():
    Q = np.array([0.5, -1.0, 2.0])
    T = np.array([1.0, 3.0, -2.0, 4.0, 0.0])

    result = sliding_dot_product(Q, T)

    expected = np.array([np.dot(Q, T[i:i + len(Q)]) for i in range(len(T) - len(Q) + 1)])
    assert np.allclose(result, expected)


def test_sliding_dot_product_accepts_equal_length_query_and_series():
    Q = np.array([1.0, -2.0, 3.0])
    T = np.array([4.0, 5.0, 6.0])

    result = sliding_dot_product(Q, T)

    assert result.shape == (1,)
    assert np.allclose(result, np.array([np.dot(Q, T)]))


def test_sliding_dot_product_raises_value_error_when_query_longer_than_series():
    Q = np.array([1.0, 2.0, 3.0, 4.0])
    T = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="T should be a series at least as long as Q"):
        sliding_dot_product(Q, T)


def test_sliding_dot_product_matches_direct_computation_for_odd_length_series():
    Q = np.array([2.0, 1.0])
    T = np.array([1.0, 0.0, -1.0, 2.0, 3.0])

    result = sliding_dot_product(Q, T)
    expected = np.array([np.dot(Q, T[i:i + len(Q)]) for i in range(len(T) - len(Q) + 1)])

    assert np.allclose(result, expected)


def test_rolling_avg_sd_matches_manual_mean_and_std():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    window_size = 2

    mean, sd = rolling_avg_sd(t, window_size)

    expected_mean = np.array([1.5, 2.5, 3.5])
    expected_sd = np.array([0.5, 0.5, 0.5])

    assert np.allclose(mean, expected_mean)
    assert np.allclose(sd, expected_sd)


def test_rolling_avg_sd_window_size_one_returns_original_and_zeros():
    t = np.array([3.0, -1.0, 7.5])
    mean, sd = rolling_avg_sd(t, 1)

    assert np.allclose(mean, t)
    assert np.allclose(sd, np.zeros_like(t, dtype=float))


def test_rolling_avg_sd_full_length_window_returns_single_value():
    t = np.array([2.0, 4.0, 6.0, 8.0])
    mean, sd = rolling_avg_sd(t, len(t))

    assert mean.shape == (1,)
    assert sd.shape == (1,)
    assert np.allclose(mean, np.array([5.0]))
    assert np.allclose(sd, np.array([np.std(t)]))


def test_rolling_avg_sd_raises_value_error_when_window_too_large():
    t = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Window size should be smaller than the length of time series."):
        rolling_avg_sd(t, 4)


def test_rolling_avg_sd_handles_constant_sequence_with_zero_std():
    t = np.array([5.0, 5.0, 5.0, 5.0])
    mean, sd = rolling_avg_sd(t, 2)

    assert np.allclose(mean, np.array([5.0, 5.0, 5.0]))
    assert np.allclose(sd, np.zeros(3))


def test_calculate_distance_profile_returns_zero_for_identical_nonconstant_series():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert D.shape == (len(T) - len(Q) + 1,)
    assert np.allclose(D[0], 0.0)


def test_calculate_distance_profile_matches_manual_formula_for_nonconstant_case():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 4.0])

    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    expected = []
    m = len(Q)
    for i in range(len(T) - len(Q) + 1):
        dist = np.sqrt(
            max(
                2 * m * (1 - (QT[i] - m * mean_Q * mean_T[i]) / (m * sigma_Q * sigma_T[i])),
                0,
            )
        )
        expected.append(dist)

    assert np.allclose(D, np.array(expected))


def test_calculate_distance_profile_constant_query_and_nonconstant_target_gives_sqrt_m():
    Q = np.array([5.0, 5.0])
    T = np.array([1.0, 2.0, 3.0])

    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.full(len(T) - len(Q) + 1, np.sqrt(len(Q))))


def test_calculate_distance_profile_constant_query_and_constant_target_gives_zero():
    Q = np.array([2.0, 2.0])
    T = np.array([3.0, 3.0, 3.0])

    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.zeros(len(T) - len(Q) + 1))


def test_calculate_distance_profile_constant_target_and_nonconstant_query_gives_sqrt_m():
    Q = np.array([1.0, 3.0])
    T = np.array([4.0, 4.0, 4.0])

    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.full(len(T) - len(Q) + 1, np.sqrt(len(Q))))


def test_calculate_distance_profile_raises_value_error_on_dimension_mismatch():
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([1.0, 2.0])
    mean_Q = 1.0
    sigma_Q = 1.0

    with pytest.raises(ValueError, match="Input dimension mismatch."):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_raises_value_error_when_mean_and_sigma_arrays_have_different_lengths():
    QT = np.array([1.0, 2.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([1.0, 2.0])
    mean_Q = np.array([1.0, 2.0])
    sigma_Q = np.array([1.0])

    with pytest.raises(ValueError, match="Input dimension mismatch."):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)
