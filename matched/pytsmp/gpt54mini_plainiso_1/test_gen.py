import numpy as np
import pytest

from pytsmp.utils import (
    calculate_distance_profile,
    rolling_avg_sd,
    sliding_dot_product,
)


def test_sliding_dot_product_matches_naive_example():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

    result = sliding_dot_product(Q, T)
    expected = np.array([
        np.dot(Q, T[i : i + len(Q)]) for i in range(len(T) - len(Q) + 1)
    ])

    assert np.allclose(result, expected)


def test_sliding_dot_product_with_negative_values():
    Q = np.array([-1.0, 2.0])
    T = np.array([3.0, -4.0, 5.0, -6.0])

    result = sliding_dot_product(Q, T)
    expected = np.array([
        np.dot(Q, T[i : i + len(Q)]) for i in range(len(T) - len(Q) + 1)
    ])

    assert np.allclose(result, expected)


def test_sliding_dot_product_raises_when_query_longer_than_series():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="T should be a series at least as long as Q"):
        sliding_dot_product(Q, T)


def test_sliding_dot_product_single_element_query():
    Q = np.array([2.5])
    T = np.array([1.0, -1.0, 4.0, 0.5])

    result = sliding_dot_product(Q, T)
    expected = 2.5 * T

    assert np.allclose(result, expected)


def test_rolling_avg_sd_matches_manual_computation():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window_size = 3

    mean, sd = rolling_avg_sd(t, window_size)

    expected_mean = np.array([2.0, 3.0, 4.0])
    expected_sd = np.array([
        np.std(t[i : i + window_size]) for i in range(len(t) - window_size + 1)
    ])

    assert np.allclose(mean, expected_mean)
    assert np.allclose(sd, expected_sd)


def test_rolling_avg_sd_window_size_one():
    t = np.array([10.0, -2.0, 7.5])
    mean, sd = rolling_avg_sd(t, 1)

    assert np.allclose(mean, t)
    assert np.allclose(sd, np.zeros_like(t))


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.array([3.0, 3.0, 3.0, 3.0])
    mean, sd = rolling_avg_sd(t, 2)

    assert np.allclose(mean, np.array([3.0, 3.0, 3.0]))
    assert np.allclose(sd, np.zeros(3))


def test_rolling_avg_sd_raises_when_window_too_large():
    t = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Window size should be smaller than the length of time series."):
        rolling_avg_sd(t, 3)


def test_rolling_avg_sd_with_integer_input_returns_float_results():
    t = np.array([1, 2, 3, 4], dtype=int)
    mean, sd = rolling_avg_sd(t, 2)

    assert mean.dtype.kind == "f"
    assert sd.dtype.kind == "f"
    assert np.allclose(mean, np.array([1.5, 2.5, 3.5]))


def test_calculate_distance_profile_matches_manual_formula():
    QT = np.array([5.0, 8.0, 11.0])
    m = 2
    mean_Q = 1.5
    sigma_Q = np.sqrt(0.25)
    mean_T = np.array([2.0, 3.0, 4.0])
    sigma_T = np.array([0.5, 1.0, 1.5])

    result = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

    expected = np.sqrt(
        np.maximum(
            2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)),
            0,
        )
    )

    assert np.allclose(result, expected)


def test_calculate_distance_profile_constant_query_and_constant_series_is_zero():
    QT = np.array([4.0, 4.0])
    m = 2
    mean_Q = 2.0
    sigma_Q = 0.0
    mean_T = np.array([2.0, 2.0])
    sigma_T = np.array([0.0, 0.0])

    result = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(result, np.zeros_like(QT))


def test_calculate_distance_profile_constant_query_and_nonconstant_series_is_sqrt_m():
    QT = np.array([1.0, 2.0, 3.0])
    m = 3
    mean_Q = 1.0
    sigma_Q = 0.0
    mean_T = np.array([0.0, 0.0, 0.0])
    sigma_T = np.array([1.0, 2.0, 3.0])

    result = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(result, np.full(3, np.sqrt(m)))


def test_calculate_distance_profile_nonconstant_query_and_constant_series_is_sqrt_m():
    QT = np.array([1.0, 2.0, 3.0])
    m = 3
    mean_Q = 1.0
    sigma_Q = 2.0
    mean_T = np.array([0.0, 0.0, 0.0])
    sigma_T = np.array([0.0, 0.0, 0.0])

    result = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(result, np.full(3, np.sqrt(m)))


def test_calculate_distance_profile_raises_on_dimension_mismatch_qt_mean_sigma_lengths():
    QT = np.array([1.0, 2.0])
    m = 2
    mean_Q = 0.0
    sigma_Q = 1.0
    mean_T = np.array([0.0, 1.0, 2.0])
    sigma_T = np.array([1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="Input dimension mismatch."):
        calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_raises_on_dimension_mismatch_when_mean_q_and_sigma_q_arrays():
    QT = np.array([1.0, 2.0, 3.0])
    m = 2
    mean_Q = np.array([1.0, 2.0])
    sigma_Q = np.array([1.0, 2.0, 3.0])  # mismatched length
    mean_T = np.array([0.0, 0.0, 0.0])
    sigma_T = np.array([1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="Input dimension mismatch."):
        calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
