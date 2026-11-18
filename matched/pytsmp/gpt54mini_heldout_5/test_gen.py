import numpy as np
import pytest

from pytsmp.utils import (
    calculate_distance_profile,
    rolling_avg_sd,
    sliding_dot_product,
)


def test_sliding_dot_product_matches_manual_for_simple_case():
    q = np.array([1.0, 2.0])
    t = np.array([3.0, 4.0, 5.0, 6.0])

    result = sliding_dot_product(q, t)

    expected = np.array([
        np.dot(q, t[0:2]),
        np.dot(q, t[1:3]),
        np.dot(q, t[2:4]),
    ])
    np.testing.assert_allclose(result, expected)


def test_sliding_dot_product_handles_odd_length_time_series():
    q = np.array([2.0, -1.0, 3.0])
    t = np.array([1.0, 4.0, -2.0, 0.5, 6.0])

    result = sliding_dot_product(q, t)

    expected = np.array([
        np.dot(q, t[0:3]),
        np.dot(q, t[1:4]),
        np.dot(q, t[2:5]),
    ])
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_sliding_dot_product_with_single_element_query():
    q = np.array([2.5])
    t = np.array([-1.0, 0.0, 3.0])

    result = sliding_dot_product(q, t)

    np.testing.assert_allclose(result, 2.5 * t)


def test_sliding_dot_product_raises_when_query_longer_than_series():
    q = np.array([1.0, 2.0, 3.0])
    t = np.array([4.0, 5.0])

    with pytest.raises(ValueError, match="T should be a series at least as long as Q"):
        sliding_dot_product(q, t)


def test_sliding_dot_product_length_matches_expected():
    q = np.array([1.0, 0.0, -1.0])
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = sliding_dot_product(q, t)

    assert len(result) == len(t) - len(q) + 1


def test_rolling_avg_sd_matches_manual_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = rolling_avg_sd(t, 2)

    np.testing.assert_allclose(mean, np.array([1.5, 2.5, 3.5]))
    np.testing.assert_allclose(sd, np.array([0.5, 0.5, 0.5]))


def test_rolling_avg_sd_window_size_one_returns_original_and_zeros():
    t = np.array([5.0, -2.0, 7.5])
    mean, sd = rolling_avg_sd(t, 1)

    np.testing.assert_allclose(mean, t)
    np.testing.assert_allclose(sd, np.zeros_like(t))


def test_rolling_avg_sd_constant_series_has_zero_std():
    t = np.array([3.0, 3.0, 3.0, 3.0])
    mean, sd = rolling_avg_sd(t, 3)

    np.testing.assert_allclose(mean, np.array([3.0, 3.0]))
    np.testing.assert_allclose(sd, np.array([0.0, 0.0]))


def test_rolling_avg_sd_raises_when_window_too_large():
    t = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Window size should be smaller than the length of time series"):
        rolling_avg_sd(t, 3)


def test_calculate_distance_profile_identical_non_constant_series_is_zero():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    q = np.array([1.0, 2.0])

    QT = sliding_dot_product(q, t)
    mean_T, sigma_T = rolling_avg_sd(t, len(q))
    mean_Q = np.mean(q)
    sigma_Q = np.std(q)

    D = calculate_distance_profile(QT, len(q), mean_Q, sigma_Q, mean_T, sigma_T)

    np.testing.assert_allclose(D, np.zeros_like(D), atol=1e-10)


def test_calculate_distance_profile_constant_query_and_nonconstant_series_gives_sqrt_m():
    q = np.array([5.0, 5.0])
    t = np.array([1.0, 2.0, 3.0, 4.0])

    QT = sliding_dot_product(q, t)
    mean_T, sigma_T = rolling_avg_sd(t, len(q))
    mean_Q = np.mean(q)
    sigma_Q = np.std(q)

    D = calculate_distance_profile(QT, len(q), mean_Q, sigma_Q, mean_T, sigma_T)

    np.testing.assert_allclose(D, np.full(len(t) - len(q) + 1, np.sqrt(len(q))))


def test_calculate_distance_profile_constant_subsequence_and_nonconstant_query_gives_sqrt_m():
    q = np.array([1.0, 2.0])
    QT = np.array([3.0, 3.0, 3.0])
    mean_T = np.array([1.0, 1.0, 1.0])
    sigma_T = np.array([0.0, 0.0, 0.0])
    mean_Q = np.mean(q)
    sigma_Q = np.std(q)

    D = calculate_distance_profile(QT, len(q), mean_Q, sigma_Q, mean_T, sigma_T)

    np.testing.assert_allclose(D, np.full(3, np.sqrt(len(q))))


def test_calculate_distance_profile_both_constant_returns_zero():
    QT = np.array([10.0, 10.0])
    mean_T = np.array([4.0, 4.0])
    sigma_T = np.array([0.0, 0.0])
    mean_Q = 4.0
    sigma_Q = 0.0

    D = calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)

    np.testing.assert_allclose(D, np.zeros(2))


def test_calculate_distance_profile_dimension_mismatch_raises():
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([0.1, 0.2])
    mean_Q = 1.5
    sigma_Q = 0.5

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_array_mean_q_sigma_q_mismatch_raises():
    QT = np.array([1.0, 2.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([0.1, 0.2])
    mean_Q = np.array([1.0, 2.0])
    sigma_Q = np.array([0.5])

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)
