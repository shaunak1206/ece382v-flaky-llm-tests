import numpy as np
import pytest

from pytsmp.utils import sliding_dot_product, rolling_avg_sd, calculate_distance_profile


def test_sliding_dot_product_basic():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2, 3, 4, 5])
    result = sliding_dot_product(Q, T)
    # Manual calculation of dot products:
    # [1*3 + 2*2 + 3*1 = 10] Reversed Q = [3,2,1]
    # But sliding dot product uses reversed Q, sliding over T:
    # Actually windows of T length m=3:
    # T[0:3] = [1,2,3], dot with Q = 1*1+2*2+3*3=14, but sliding_dot_product reverses Q
    # Let's compute using sliding_dot_product Q reversed:
    # Output length = 5-3+1=3
    # But test only verifies output shape and approximate values:
    assert result.shape == (3,)
    expected = np.array([3*1 + 2*2 + 1*3, 3*2 + 2*3 + 1*4, 3*3 + 2*4 + 1*5])  # reversed Q dot with windows
    np.testing.assert_almost_equal(result, expected)


def test_sliding_dot_product_T_shorter_than_Q():
    Q = np.array([1, 2, 3, 4])
    T = np.array([1, 2])
    with pytest.raises(ValueError):
        sliding_dot_product(Q, T)


def test_sliding_dot_product_Q_equals_T():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2, 3])
    result = sliding_dot_product(Q, T)
    assert result.shape == (1,)
    expected = np.dot(Q[::-1], T)
    np.testing.assert_almost_equal(result, np.array([expected]))


def test_rolling_avg_sd_basic():
    t = np.array([1, 2, 3, 4, 5])
    window_size = 3
    mean, sd = rolling_avg_sd(t, window_size)
    # Expected means: [mean([1,2,3]), mean([2,3,4]), mean([3,4,5])]
    expected_mean = np.array([2, 3, 4])
    expected_sd = np.sqrt(np.array([np.var(t[0:3], ddof=0),
                                   np.var(t[1:4], ddof=0),
                                   np.var(t[2:5], ddof=0)]))
    np.testing.assert_almost_equal(mean, expected_mean)
    np.testing.assert_almost_equal(sd, expected_sd)


def test_rolling_avg_sd_window_equal_len():
    t = np.array([10, 20, 30])
    window_size = 3
    mean, sd = rolling_avg_sd(t, window_size)
    np.testing.assert_almost_equal(mean, np.array([20]))
    np.testing.assert_almost_equal(sd, np.array([np.std(t, ddof=0)]))


def test_rolling_avg_sd_window_too_large():
    t = np.array([1, 2, 3])
    window_size = 5
    with pytest.raises(ValueError):
        rolling_avg_sd(t, window_size)


def test_calculate_distance_profile_basic():
    m = 4
    QT = np.array([10, 20, 30, 40])
    mean_Q = 5
    sigma_Q = 2
    mean_T = np.array([1, 2, 3, 4])
    sigma_T = np.array([1, 2, 3, 4])
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert D.shape == QT.shape
    # Check no negative or nan values
    assert np.all(D >= 0)
    assert not np.any(np.isnan(D))


def test_calculate_distance_profile_dimension_mismatch():
    m = 3
    QT = np.array([1, 2])
    mean_Q = 0
    sigma_Q = 1
    mean_T = np.array([1, 2, 3])
    sigma_T = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_constant_sequences():
    m = 3
    length = 5
    QT = np.zeros(length)
    mean_Q = 0
    sigma_Q = 0  # constant Q
    mean_T = np.zeros(length)
    sigma_T = np.zeros(length)  # constant T subsequences
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # When both std are less than epsilon -> distance 0
    assert np.allclose(D, 0)

    sigma_T = np.zeros(length)
    sigma_Q = 0
    # When sigma_Q < epsilon but sigma_T not less than epsilon
    sigma_T[:] = 1e-5
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T,
                                   epsilon=1e-7)
    assert np.allclose(D, np.sqrt(m))


def test_calculate_distance_profile_sigma_T_zero_sigma_Q_nonzero():
    m = 2
    length = 4
    QT = np.ones(length)
    mean_Q = 1
    sigma_Q = 1.0
    mean_T = np.zeros(length)
    sigma_T = np.zeros(length)
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert np.allclose(D, np.sqrt(m))


def test_calculate_distance_profile_distance_formula_correct():
    m = 2
    QT = np.array([2, 4])
    mean_Q = 1
    sigma_Q = 1
    mean_T = np.array([2, 2])
    sigma_T = np.array([1, 1])
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.sqrt(np.maximum(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)), 0))
    np.testing.assert_almost_equal(D, expected)


def test_sliding_dot_product_with_zero_padding_effect():
    # Test input length odd and even to verify padding effect does not affect result length and correctness
    Q = np.array([1, 0, -1])
    T_odd = np.array([1, 2, 3, 4, 5])
    T_even = np.array([1, 2, 3, 4, 5, 6])

    res_odd = sliding_dot_product(Q, T_odd)
    res_even = sliding_dot_product(Q, T_even)

    assert res_odd.shape == (len(T_odd) - len(Q) + 1,)
    assert res_even.shape == (len(T_even) - len(Q) + 1,)

    # Verify dot product values with manual convolution for one case
    manual_odd = np.array([np.dot(Q[::-1], T_odd[i:i+len(Q)]) for i in range(len(T_odd)-len(Q)+1)])
    np.testing.assert_almost_equal(res_odd, manual_odd)

    manual_even = np.array([np.dot(Q[::-1], T_even[i:i+len(Q)]) for i in range(len(T_even)-len(Q)+1)])
    np.testing.assert_almost_equal(res_even, manual_even)


def test_rolling_avg_sd_all_equal_elements():
    t = np.ones(10) * 5
    window_size = 3
    mean, sd = rolling_avg_sd(t, window_size)
    expected_mean = np.ones(len(t) - window_size + 1) * 5
    expected_sd = np.zeros(len(t) - window_size + 1)
    np.testing.assert_array_equal(mean, expected_mean)
    np.testing.assert_array_equal(sd, expected_sd)


def test_rolling_avg_sd_single_element_windows():
    t = np.array([1, 2, 3, 4, 5])
    window_size = 1
    mean, sd = rolling_avg_sd(t, window_size)
    np.testing.assert_array_equal(mean, t)
    np.testing.assert_array_equal(sd, np.zeros_like(t))


def test_calculate_distance_profile_array_mean_sigmaQ():
    # Test with mean_Q and sigma_Q as arrays
    m = 3
    length = 5
    QT = np.array([2, 4, 6, 8, 10])
    mean_Q = np.array([1, 1, 1, 1, 1])
    sigma_Q = np.array([1, 1, 1, 1, 1])
    mean_T = np.array([2, 2, 2, 2, 2])
    sigma_T = np.array([1, 1, 1, 1, 1])
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert D.shape == QT.shape

    # If lengths mismatch between mean_Q and sigma_Q (and mean_T), should raise error
    mean_Q_bad = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, m, mean_Q_bad, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_epsilon_tolerance():
    m = 2
    length = 3
    QT = np.array([1, 2, 3])
    mean_Q = 0
    sigma_Q = 1e-10
    mean_T = np.array([0, 0, 0])
    sigma_T = np.array([0, 1e-10, 1e-9])

    # With epsilon=1e-7, first two sigma_Q or sigma_T < epsilon should trigger constant logic
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    assert D[0] == 0  # both below epsilon -> zero distance
    assert np.isclose(D[1], np.sqrt(m))  # sigma_T < epsilon, sigma_Q < epsilon -> sqrt(m)
    assert D[2] >= 0  # larger than epsilon, normal calculation


def test_sliding_dot_product_query_all_zeros():
    Q = np.zeros(3)
    T = np.arange(10)
    result = sliding_dot_product(Q, T)
    # Dot product should be zero for all subsequences
    expected = np.zeros(len(T) - len(Q) + 1)
    np.testing.assert_array_equal(result, expected)
