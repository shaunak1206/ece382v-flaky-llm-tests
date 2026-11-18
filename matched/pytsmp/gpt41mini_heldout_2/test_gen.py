import numpy as np
import pytest
from pytsmp.utils import sliding_dot_product, rolling_avg_sd, calculate_distance_profile


def test_sliding_dot_product_basic():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    result = sliding_dot_product(Q, T)
    expected = np.array([
        np.dot(Q, T[0:3]),
        np.dot(Q, T[1:4]),
        np.dot(Q, T[2:5]),
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-8)


def test_sliding_dot_product_query_larger_error():
    Q = np.array([1, 2, 3, 4])
    T = np.array([1, 2])
    with pytest.raises(ValueError):
        sliding_dot_product(Q, T)


def test_sliding_dot_product_identical_series():
    T = np.arange(10).astype(float)
    Q = T[:5]
    result = sliding_dot_product(Q, T)
    expected = np.correlate(T, Q[::-1], mode="valid")
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-8)


def test_rolling_avg_sd_basic():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window_size = 3
    avg, sd = rolling_avg_sd(t, window_size)
    expected_avg = np.array([2.0, 3.0, 4.0])
    expected_sd = np.array([
        np.std(t[0:3], ddof=0),
        np.std(t[1:4], ddof=0),
        np.std(t[2:5], ddof=0),
    ])
    np.testing.assert_allclose(avg, expected_avg, rtol=1e-7)
    np.testing.assert_allclose(sd, expected_sd, rtol=1e-7)


def test_rolling_avg_sd_window_larger_than_series():
    t = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        rolling_avg_sd(t, window_size=5)


def test_rolling_avg_sd_constant_series():
    t = np.ones(5)
    avg, sd = rolling_avg_sd(t, window_size=3)
    expected_avg = np.ones(3)
    expected_sd = np.zeros(3)
    np.testing.assert_allclose(avg, expected_avg)
    np.testing.assert_allclose(sd, expected_sd)


def test_calculate_distance_profile_basic():
    # Prepare inputs
    m = 3
    QT = np.array([10, 20, 30], dtype=float)
    mean_Q = 2.0
    sigma_Q = 1.0
    mean_T = np.array([1.5, 2.0, 2.5], dtype=float)
    sigma_T = np.array([1.0, 2.0, 3.0], dtype=float)
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Manually compute expected distances according to formula
    expected = np.sqrt(
        np.maximum(
            2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)),
            0,
        )
    )
    np.testing.assert_allclose(D, expected, rtol=1e-7)


def test_calculate_distance_profile_length_mismatch_QT_meanT():
    QT = np.array([1, 2])
    m = 2
    mean_Q = 1.0
    sigma_Q = 1.0
    mean_T = np.array([1.0])  # length mismatch
    sigma_T = np.array([1.0])
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_length_mismatch_meanQ_sigmaQ():
    QT = np.array([1, 2])
    m = 2
    mean_Q = np.array([1.0, 2.0])
    sigma_Q = np.array([1.0])  # mismatched length
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_constant_Q_variance():
    m = 4
    length = 3
    QT = np.array([0.0, 0.0, 0.0], dtype=float)
    mean_Q = 1.0
    sigma_Q = 0.0  # constant Q sequence
    mean_T = np.array([1.0, 2.0, 3.0])
    sigma_T = np.array([1.0, 2.0, 3.0])
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Distance should be sqrt(m) for sigma_Q < epsilon (except where sigma_T < epsilon)
    expected = np.sqrt(m) * np.ones_like(QT)
    np.testing.assert_allclose(D, expected, rtol=1e-7)


def test_calculate_distance_profile_constant_T_variance():
    m = 4
    length = 3
    QT = np.array([0.0, 0.0, 0.0], dtype=float)
    mean_Q = 1.0
    sigma_Q = 2.0
    mean_T = np.array([1.0, 1.0, 1.0])
    sigma_T = np.array([0.0, 0.0, 0.0])  # constant subsequences in T
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Distance should be sqrt(m) for sigma_T < epsilon (except where sigma_Q < epsilon)
    expected = np.sqrt(m) * np.ones_like(QT)
    np.testing.assert_allclose(D, expected, rtol=1e-7)


def test_calculate_distance_profile_both_constant_variance_zero():
    m = 3
    QT = np.array([5.0, 5.0], dtype=float)
    mean_Q = 10.0
    sigma_Q = 0.0
    mean_T = np.array([8.0, 8.0])
    sigma_T = np.array([0.0, 0.0])
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Both std zero => distance zero for all
    expected = np.zeros_like(QT)
    np.testing.assert_allclose(D, expected, rtol=1e-7)


def test_sliding_dot_product_with_zeros_and_negatives():
    Q = np.array([-1.0, 0.0, 1.0])
    T = np.array([0.0, -1.0, -2.0, 1.0, 2.0])
    result = sliding_dot_product(Q, T)
    expected = np.array([
        np.dot(Q, T[0:3]),
        np.dot(Q, T[1:4]),
        np.dot(Q, T[2:5]),
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_rolling_avg_sd_with_negatives():
    t = np.array([-1, -2, -3, -4, -5], dtype=float)
    window_size = 2
    avg, sd = rolling_avg_sd(t, window_size)
    # Manual calculation
    expected_avg = np.array([-1.5, -2.5, -3.5, -4.5])
    expected_sd = np.array([0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(avg, expected_avg, rtol=1e-7)
    np.testing.assert_allclose(sd, expected_sd, rtol=1e-7)


def test_calculate_distance_profile_with_arrays_for_mean_sigma_Q():
    m = 3
    QT = np.array([3.0, 3.5, 4.0])
    # mean_Q and sigma_Q as arrays (same length as QT)
    mean_Q = np.array([1.0, 1.5, 2.0])
    sigma_Q = np.array([0.5, 0.5, 0.5])
    mean_T = np.array([1.1, 1.4, 1.8])
    sigma_T = np.array([0.6, 0.7, 0.8])
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.sqrt(
        np.maximum(
            2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)),
            0,
        )
    )
    np.testing.assert_allclose(D, expected, rtol=1e-7)


def test_sliding_dot_product_length_output():
    Q = np.arange(4)
    T = np.arange(10)
    result = sliding_dot_product(Q, T)
    # Length should be len(T)-len(Q)+1
    expected_length = len(T) - len(Q) + 1
    assert len(result) == expected_length


def test_rolling_avg_sd_output_length():
    t = np.arange(10, dtype=float)
    window_size = 4
    avg, sd = rolling_avg_sd(t, window_size)
    expected_length = len(t) - window_size + 1
    assert len(avg) == expected_length
    assert len(sd) == expected_length


def test_calculate_distance_profile_output_length():
    length = 5
    QT = np.arange(length, dtype=float)
    m = 3
    mean_Q = 1.0
    sigma_Q = 0.5
    mean_T = np.arange(length, dtype=float) + 1
    sigma_T = np.full(length, 0.5)
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert len(D) == length


def test_calculate_distance_profile_no_negative_sqrt():
    # Test calculation protects against negative sqrt input
    m = 3
    length = 4
    # Construct QT, mean_Q, sigma_Q, mean_T, sigma_T to force negative term inside sqrt
    QT = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    mean_Q = 1.0
    sigma_Q = 1.0
    mean_T = np.array([1.0, 1.0, 1.0, 1.0])
    sigma_T = np.array([0.1, 0.1, 0.1, 0.1])  # very small sigma_T
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Distance should be finite and no nan
    assert np.all(np.isfinite(D))
    assert np.all(D >= 0)


def test_calculate_distance_profile_epsilon_tolerance():
    m = 3
    length = 3
    QT = np.array([5.0, 6.0, 7.0])
    mean_Q = 1.0
    sigma_Q = 1e-10
    mean_T = np.array([2.0, 2.0, 2.0])
    sigma_T = np.array([1e-10, 1e-10, 1e-10])
    # Because sigma_Q and sigma_T < epsilon, result entries are either 0 or sqrt(m)
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    expected = np.sqrt(m) * np.ones_like(QT)
    np.testing.assert_allclose(D, expected, rtol=1e-7)
