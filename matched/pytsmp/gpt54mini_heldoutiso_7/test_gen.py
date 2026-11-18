import numpy as np
import pytest

from pytsmp.utils import (
    sliding_dot_product,
    rolling_avg_sd,
    calculate_distance_profile,
)


def test_sliding_dot_product_basic_matches_manual():
    q = np.array([1.0, 2.0])
    t = np.array([3.0, 4.0, 5.0, 6.0])

    result = sliding_dot_product(q, t)
    expected = np.array([
        np.dot(q, t[0:2]),
        np.dot(q, t[1:3]),
        np.dot(q, t[2:4]),
    ])

    assert np.allclose(result, expected)


def test_sliding_dot_product_with_odd_length_time_series():
    q = np.array([1.0, -1.0, 2.0])
    t = np.array([2.0, 0.0, 1.0, 3.0, -1.0])

    result = sliding_dot_product(q, t)
    expected = np.array([
        np.dot(q, t[0:3]),
        np.dot(q, t[1:4]),
        np.dot(q, t[2:5]),
    ])

    assert result.shape == (len(t) - len(q) + 1,)
    assert np.allclose(result, expected)


def test_sliding_dot_product_equal_length_returns_single_value():
    q = np.array([2.0, 1.0, -1.0])
    t = np.array([4.0, 0.0, 3.0])

    result = sliding_dot_product(q, t)

    assert result.shape == (1,)
    assert np.allclose(result, np.array([np.dot(q, t)]))


def test_sliding_dot_product_raises_when_query_longer_than_time_series():
    q = np.array([1.0, 2.0, 3.0])
    t = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="T should be a series at least as long as Q"):
        sliding_dot_product(q, t)


def test_sliding_dot_product_handles_negative_values():
    q = np.array([-1.0, 2.0])
    t = np.array([3.0, -4.0, 5.0])

    result = sliding_dot_product(q, t)
    expected = np.array([
        np.dot(q, t[0:2]),
        np.dot(q, t[1:3]),
    ])

    assert np.allclose(result, expected)


def test_rolling_avg_sd_basic_matches_manual():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    window = 2

    mean, sd = rolling_avg_sd(t, window)

    assert np.allclose(mean, np.array([1.5, 2.5, 3.5]))
    assert np.allclose(sd, np.array([0.5, 0.5, 0.5]))


def test_rolling_avg_sd_window_size_one_returns_original_and_zero_sd():
    t = np.array([5.0, -1.0, 7.5])
    mean, sd = rolling_avg_sd(t, 1)

    assert np.allclose(mean, t)
    assert np.allclose(sd, np.zeros_like(t))


def test_rolling_avg_sd_full_length_window_returns_single_window_stats():
    t = np.array([2.0, 4.0, 6.0, 8.0])
    mean, sd = rolling_avg_sd(t, len(t))

    assert mean.shape == (1,)
    assert sd.shape == (1,)
    assert np.allclose(mean, np.array([5.0]))
    assert np.allclose(sd, np.array([np.std(t)]))


def test_rolling_avg_sd_raises_when_window_too_large():
    t = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Window size should be smaller than the length of time series"):
        rolling_avg_sd(t, 4)


def test_rolling_avg_sd_for_constant_series_has_zero_sd():
    t = np.array([3.0, 3.0, 3.0, 3.0])
    mean, sd = rolling_avg_sd(t, 2)

    assert np.allclose(mean, np.array([3.0, 3.0, 3.0]))
    assert np.allclose(sd, np.zeros(3))


def test_calculate_distance_profile_matches_manual_nonconstant_case():
    q = np.array([1.0, 2.0])
    t = np.array([1.0, 2.0, 3.0, 4.0])

    qt = sliding_dot_product(q, t)
    mean_t, sigma_t = rolling_avg_sd(t, len(q))
    mean_q = np.mean(q)
    sigma_q = np.std(q)

    result = calculate_distance_profile(qt, len(q), mean_q, sigma_q, mean_t, sigma_t)

    expected = np.array([
        np.linalg.norm((q - mean_q) / sigma_q - (t[i:i + len(q)] - mean_t[i]) / sigma_t[i])
        for i in range(len(t) - len(q) + 1)
    ])

    assert np.allclose(result, expected)


def test_calculate_distance_profile_constant_query_against_nonconstant_series_returns_sqrt_m():
    m = 3
    qt = np.array([0.0, 0.0, 0.0])
    mean_t = np.array([1.0, 2.0, 3.0])
    sigma_t = np.array([1.0, 1.0, 1.0])
    mean_q = 5.0
    sigma_q = 0.0

    result = calculate_distance_profile(qt, m, mean_q, sigma_q, mean_t, sigma_t)

    assert np.allclose(result, np.full(3, np.sqrt(m)))


def test_calculate_distance_profile_constant_query_and_constant_series_returns_zero():
    m = 4
    qt = np.array([0.0, 0.0])
    mean_t = np.array([3.0, 3.0])
    sigma_t = np.array([0.0, 0.0])
    mean_q = 3.0
    sigma_q = 0.0

    result = calculate_distance_profile(qt, m, mean_q, sigma_q, mean_t, sigma_t)

    assert np.allclose(result, np.zeros(2))


def test_calculate_distance_profile_constant_subsequence_returns_sqrt_m():
    m = 2
    qt = np.array([1.0, 2.0])
    mean_t = np.array([1.0, 1.5])
    sigma_t = np.array([0.0, 0.0])
    mean_q = 0.0
    sigma_q = 1.0

    result = calculate_distance_profile(qt, m, mean_q, sigma_q, mean_t, sigma_t)

    assert np.allclose(result, np.full(2, np.sqrt(m)))


def test_calculate_distance_profile_raises_on_dimension_mismatch_between_inputs():
    qt = np.array([1.0, 2.0, 3.0])
    mean_t = np.array([1.0, 2.0])
    sigma_t = np.array([1.0, 2.0])
    mean_q = 0.0
    sigma_q = 1.0

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        calculate_distance_profile(qt, 2, mean_q, sigma_q, mean_t, sigma_t)


def test_calculate_distance_profile_raises_on_dimension_mismatch_for_array_query_stats():
    qt = np.array([1.0, 2.0, 3.0])
    mean_t = np.array([1.0, 2.0, 3.0])
    sigma_t = np.array([1.0, 2.0, 3.0])
    mean_q = np.array([0.0, 0.0])
    sigma_q = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        calculate_distance_profile(qt, 2, mean_q, sigma_q, mean_t, sigma_t)
