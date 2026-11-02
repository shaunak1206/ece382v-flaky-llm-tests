import numpy as np
import pytest

from pytsmp.utils import sliding_dot_product, rolling_avg_sd, calculate_distance_profile


def test_sliding_dot_product_basic_matches_manual_convolution():
    q = np.array([1.0, 2.0, 3.0])
    t = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

    result = sliding_dot_product(q, t)
    expected = np.array([
        np.dot(q, t[0:3]),
        np.dot(q, t[1:4]),
        np.dot(q, t[2:5]),
    ])

    assert result.shape == (3,)
    assert np.allclose(result, expected)


def test_sliding_dot_product_with_query_length_one_returns_original_series():
    q = np.array([2.5])
    t = np.array([1.0, -3.0, 4.0, 0.5])

    result = sliding_dot_product(q, t)

    assert np.allclose(result, q[0] * t)
    assert result.shape == (len(t),)


def test_sliding_dot_product_raises_when_query_longer_than_series():
    q = np.array([1.0, 2.0, 3.0])
    t = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="at least as long as Q"):
        sliding_dot_product(q, t)


def test_sliding_dot_product_with_odd_length_time_series():
    q = np.array([1.0, -1.0])
    t = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    result = sliding_dot_product(q, t)
    expected = np.array([
        np.dot(q, t[0:2]),
        np.dot(q, t[1:3]),
        np.dot(q, t[2:4]),
        np.dot(q, t[3:5]),
    ])

    assert np.allclose(result, expected)


def test_rolling_avg_sd_basic_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = rolling_avg_sd(t, 2)

    assert np.allclose(mean, np.array([1.5, 2.5, 3.5]))
    assert np.allclose(sd, np.array([0.5, 0.5, 0.5]))


def test_rolling_avg_sd_window_size_one_has_zero_sd_and_same_mean():
    t = np.array([3.0, -2.0, 7.5])

    mean, sd = rolling_avg_sd(t, 1)

    assert np.allclose(mean, t)
    assert np.allclose(sd, np.zeros_like(t))


def test_rolling_avg_sd_window_size_equals_length_returns_single_value():
    t = np.array([1.0, 2.0, 3.0, 4.0])

    mean, sd = rolling_avg_sd(t, len(t))

    assert mean.shape == (1,)
    assert sd.shape == (1,)
    assert np.allclose(mean, np.array([2.5]))
    assert np.allclose(sd, np.array([np.std(t)]))


def test_rolling_avg_sd_raises_when_window_too_large():
    t = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="smaller than the length of time series"):
        rolling_avg_sd(t, 4)


def test_rolling_avg_sd_returns_non_negative_sd_for_constant_series():
    t = np.array([5.0, 5.0, 5.0, 5.0])

    mean, sd = rolling_avg_sd(t, 3)

    assert np.allclose(mean, np.array([5.0, 5.0]))
    assert np.allclose(sd, np.array([0.0, 0.0]))


def test_calculate_distance_profile_dimension_mismatch_raises():
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([0.0, 0.0, 0.0])
    sigma_T = np.array([1.0, 1.0])  # mismatch

    with pytest.raises(ValueError, match="dimension mismatch"):
        calculate_distance_profile(QT, 2, 0.0, 1.0, mean_T, sigma_T)


def test_calculate_distance_profile_identical_normalized_windows_gives_zero_distance():
    q = np.array([1.0, 2.0, 3.0])
    t = np.array([1.0, 2.0, 3.0])

    QT = sliding_dot_product(q, t)
    mean_T, sigma_T = rolling_avg_sd(t, len(q))
    mean_Q = np.mean(q)
    sigma_Q = np.std(q)

    D = calculate_distance_profile(QT, len(q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert D.shape == (1,)
    assert np.allclose(D, np.array([0.0]))


def test_calculate_distance_profile_constant_query_against_nonconstant_series_is_sqrt_m():
    q = np.array([2.0, 2.0, 2.0])
    t = np.array([1.0, 2.0, 3.0, 4.0])

    QT = sliding_dot_product(q, t)
    mean_T, sigma_T = rolling_avg_sd(t, len(q))
    mean_Q = np.mean(q)
    sigma_Q = np.std(q)

    D = calculate_distance_profile(QT, len(q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.full(len(t) - len(q) + 1, np.sqrt(len(q))))


def test_calculate_distance_profile_constant_query_and_constant_series_is_zero():
    q = np.array([2.0, 2.0, 2.0])
    t = np.array([5.0, 5.0, 5.0, 5.0])

    QT = sliding_dot_product(q, t)
    mean_T, sigma_T = rolling_avg_sd(t, len(q))
    mean_Q = np.mean(q)
    sigma_Q = np.std(q)

    D = calculate_distance_profile(QT, len(q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.zeros(len(t) - len(q) + 1))


def test_calculate_distance_profile_constant_subsequence_in_time_series_gets_sqrt_m():
    q = np.array([1.0, 0.0, -1.0])
    t = np.array([1.0, 1.0, 1.0, 2.0, 3.0])

    QT = sliding_dot_product(q, t)
    mean_T, sigma_T = rolling_avg_sd(t, len(q))
    mean_Q = np.mean(q)
    sigma_Q = np.std(q)

    D = calculate_distance_profile(QT, len(q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert D.shape == (len(t) - len(q) + 1,)
    assert np.isclose(D[0], np.sqrt(len(q)))
    assert np.all(D[1:] >= 0)


def test_calculate_distance_profile_matches_manual_formula_for_nonconstant_data():
    q = np.array([1.0, 3.0])
    t = np.array([2.0, 4.0, 6.0])

    QT = sliding_dot_product(q, t)
    mean_T, sigma_T = rolling_avg_sd(t, len(q))
    mean_Q = np.mean(q)
    sigma_Q = np.std(q)

    D = calculate_distance_profile(QT, len(q), mean_Q, sigma_Q, mean_T, sigma_T)

    expected = []
    for i in range(len(t) - len(q) + 1):
        subseq = t[i:i + len(q)]
        expected.append(np.linalg.norm(
            (subseq - np.mean(subseq)) / np.std(subseq) - (q - mean_Q) / sigma_Q
        ))
    expected = np.array(expected)

    assert np.allclose(D, expected)


def test_calculate_distance_profile_respects_epsilon_threshold_for_near_constant_query():
    q = np.array([1.0, 1.0, 1.0 + 1e-10])
    t = np.array([1.0, 2.0, 3.0, 4.0])

    QT = sliding_dot_product(q, t)
    mean_T, sigma_T = rolling_avg_sd(t, len(q))
    mean_Q = np.mean(q)
    sigma_Q = np.std(q)

    D = calculate_distance_profile(QT, len(q), mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)

    assert np.allclose(D, np.full(len(t) - len(q) + 1, np.sqrt(len(q))))
