import numpy as np
import pytest

from pytsmp.utils import (
    calculate_distance_profile,
    rolling_avg_sd,
    sliding_dot_product,
)


def brute_force_sliding_dot_product(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i : i + m]) for i in range(len(T) - m + 1)])


def brute_force_distance_profile(Q, T):
    m = len(Q)
    out = []
    for i in range(len(T) - m + 1):
        sub = T[i : i + m]
        out.append(np.linalg.norm(Q - sub))
    return np.array(out)


def test_sliding_dot_product_matches_bruteforce_simple_even_lengths():
    Q = np.array([1.0, 2.0])
    T = np.array([3.0, 4.0, 5.0, 6.0])
    expected = brute_force_sliding_dot_product(Q, T)
    actual = sliding_dot_product(Q, T)
    assert np.allclose(actual, expected)


def test_sliding_dot_product_matches_bruteforce_odd_lengths():
    Q = np.array([2.0, -1.0, 3.0])
    T = np.array([0.0, 1.0, 2.0, -2.0, 4.0])
    expected = brute_force_sliding_dot_product(Q, T)
    actual = sliding_dot_product(Q, T)
    assert np.allclose(actual, expected)


def test_sliding_dot_product_with_numpy_integer_arrays():
    Q = np.array([1, 0, -1], dtype=np.int64)
    T = np.array([2, 3, 4, 5, 6], dtype=np.int64)
    expected = brute_force_sliding_dot_product(Q, T)
    actual = sliding_dot_product(Q, T)
    assert actual.dtype.kind == "f"
    assert np.allclose(actual, expected)


def test_sliding_dot_product_raises_when_query_longer_than_series():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([4.0, 5.0])
    with pytest.raises(ValueError, match="T should be a series at least as long as Q"):
        sliding_dot_product(Q, T)


def test_sliding_dot_product_length_is_correct():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    actual = sliding_dot_product(Q, T)
    assert len(actual) == len(T) - len(Q) + 1


def test_rolling_avg_sd_basic_values():
    T = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = rolling_avg_sd(T, 2)
    assert np.allclose(mean, np.array([1.5, 2.5, 3.5]))
    assert np.allclose(sd, np.array([0.5, 0.5, 0.5]))


def test_rolling_avg_sd_window_size_one():
    T = np.array([5.0, -1.5, 7.25])
    mean, sd = rolling_avg_sd(T, 1)
    assert np.allclose(mean, T)
    assert np.allclose(sd, np.zeros_like(T))


def test_rolling_avg_sd_constant_series_has_zero_sd():
    T = np.array([4.2, 4.2, 4.2, 4.2])
    mean, sd = rolling_avg_sd(T, 3)
    assert np.allclose(mean, np.array([4.2, 4.2]))
    assert np.allclose(sd, np.zeros(2))


def test_rolling_avg_sd_raises_when_window_longer_than_series():
    T = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Window size should be smaller than the length of time series."):
        rolling_avg_sd(T, 3)


def test_calculate_distance_profile_matches_bruteforce_for_general_case():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    actual = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)
    expected = brute_force_distance_profile(Q, T)
    assert np.allclose(actual, expected, atol=1e-7)


def test_calculate_distance_profile_returns_zero_for_identical_series():
    Q = np.array([1.0, 2.0, 3.0, 4.0])
    T = Q.copy()
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    actual = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)
    assert np.allclose(actual, np.array([0.0]))


def test_calculate_distance_profile_constant_query_and_nonconstant_target_yields_sqrt_m():
    Q = np.array([5.0, 5.0, 5.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    actual = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)
    assert np.allclose(actual, np.full(len(T) - len(Q) + 1, np.sqrt(len(Q))))


def test_calculate_distance_profile_nonconstant_query_and_constant_target_yields_sqrt_m():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([7.0, 7.0, 7.0, 7.0, 7.0])
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    actual = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)
    assert np.allclose(actual, np.full(len(T) - len(Q) + 1, np.sqrt(len(Q))))


def test_calculate_distance_profile_both_constant_returns_zero():
    Q = np.array([2.0, 2.0, 2.0])
    T = np.array([9.0, 9.0, 9.0, 9.0])
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    actual = calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)
    assert np.allclose(actual, np.zeros(len(T) - len(Q) + 1))


def test_calculate_distance_profile_raises_on_length_mismatch():
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="Input dimension mismatch."):
        calculate_distance_profile(QT, 3, 1.0, 1.0, mean_T, sigma_T)


def test_calculate_distance_profile_raises_when_mean_q_and_sigma_q_arrays_have_mismatched_lengths():
    QT = np.array([1.0, 2.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([0.1, 0.2])
    mean_Q = np.array([1.0, 2.0])
    sigma_Q = np.array([0.5])
    with pytest.raises(ValueError, match="Input dimension mismatch."):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)
