import numpy as np
import pytest
from pytsmp import utils


def test_sliding_dot_product_basic():
    Q = np.array([1, 2, 3])
    T = np.array([0, 1, 2, 3, 4])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([
        np.dot(Q, T[0:3]),
        np.dot(Q, T[1:4]),
        np.dot(Q, T[2:5])
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_non_overlapping_lengths():
    # T shorter than Q -> ValueError
    Q = np.array([1, 2, 3, 4])
    T = np.array([1, 2])
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_sliding_dot_product_all_zeros():
    Q = np.zeros(3)
    T = np.zeros(5)
    result = utils.sliding_dot_product(Q, T)
    expected = np.zeros(len(T) - len(Q) + 1)
    np.testing.assert_array_equal(result, expected)


def test_sliding_dot_product_known_values():
    Q = np.array([1, -1])
    T = np.array([2, 2, 2])
    result = utils.sliding_dot_product(Q, T)
    # Expected dot products:
    # dot([1, -1], [2,2]) = 1*2 + (-1)*2 = 0
    # dot([1, -1], [2,2]) = 0 (same window shifted)
    expected = np.array([0, 0])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_rolling_avg_sd_basic():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = 3
    avg, sd = utils.rolling_avg_sd(t, w)
    # Manually computed:
    # windows: [1 2 3], [2 3 4], [3 4 5]
    expected_avg = np.array([2.0, 3.0, 4.0])
    expected_sd = np.array([
        np.std([1, 2, 3], ddof=0),
        np.std([2, 3, 4], ddof=0),
        np.std([3, 4, 5], ddof=0),
    ])
    np.testing.assert_allclose(avg, expected_avg, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sd, expected_sd, rtol=1e-12, atol=1e-12)


def test_rolling_avg_sd_short_series_error():
    t = np.array([1.0, 2.0])
    w = 3
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(t, w)


def test_rolling_avg_sd_constant_series_sd_zero():
    t = np.ones(10)
    w = 5
    avg, sd = utils.rolling_avg_sd(t, w)
    expected_avg = np.ones(len(t) - w + 1)
    expected_sd = np.zeros(len(t) - w + 1)
    np.testing.assert_allclose(avg, expected_avg, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sd, expected_sd, rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_basic_usage():
    # Preparations:
    m = 3
    QT = np.array([3.0, 4.0, 5.0])
    mean_Q = 2.0
    sigma_Q = 1.0
    mean_T = np.array([1.0, 2.0, 3.0])
    sigma_T = np.array([1.0, 2.0, 3.0])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # All sigma_Q and sigma_T > epsilon, so use sqrt(2*m*(1 - (...)))
    expected = np.sqrt(np.maximum(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)), 0))
    np.testing.assert_allclose(D, expected, rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_sigmaQ_zero_sigmT_zero():
    m = 4
    QT = np.array([1.0, 1.0])
    mean_Q = 5.0
    sigma_Q = 0.0
    mean_T = np.array([2.0, 3.0])
    sigma_T = np.array([0.0, 1.0])
    # sigma_Q < epsilon and sigma_T first is < epsilon, second > epsilon
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    expected = np.array([0, np.sqrt(m)])
    np.testing.assert_allclose(D, expected, rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_sigmaT_zero_sigmaQ_nonzero():
    m = 2
    QT = np.array([1.0])
    mean_Q = 1.0
    sigma_Q = 1.0
    mean_T = np.array([2.0])
    sigma_T = np.array([0.0])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    expected = np.array([np.sqrt(m)])
    np.testing.assert_allclose(D, expected, rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_dimension_mismatch():
    m = 3
    QT = np.array([1.0, 2.0])
    mean_Q = 1.0
    sigma_Q = 1.0
    mean_T = np.array([2.0])
    sigma_T = np.array([2.0])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_numpy_array_mean_sigmaQ():
    # mean_Q and sigma_Q are numpy arrays, must check length matching mean_T length
    m = 3
    QT = np.array([1.0, 2.0])
    mean_Q = np.array([1.0, 2.0])
    sigma_Q = np.array([0.5, 0.5])
    mean_T = np.array([2.0, 3.0])
    sigma_T = np.array([1.0, 1.0])
    # Should not raise
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert D.shape == (2,)


def test_calculate_distance_profile_numpy_array_length_mismatch():
    m = 3
    QT = np.array([1.0, 2.0])
    mean_Q = np.array([1.0])
    sigma_Q = np.array([0.5, 0.5])
    mean_T = np.array([2.0, 3.0])
    sigma_T = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_sliding_dot_product_q_length_equals_t_length():
    Q = np.array([1, 2, 3])
    T = np.array([4, 5, 6])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([np.dot(Q, T)])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_with_negative_values():
    Q = np.array([-1, -2, -3])
    T = np.array([3, 2, 1, 0])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([
        np.dot(Q, T[0:3]),
        np.dot(Q, T[1:4])
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_mass_basic_distance_profile():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    D = utils.mass(Q, T)
    assert D.shape == (len(T) - len(Q) + 1,)
    # Distances are always >= 0
    assert np.all(D >= 0)
    # Distance array finite
    assert np.all(np.isfinite(D))


def test_mass_length_error():
    Q = np.array([1.0, 2.0, 3.0, 4.0])
    T = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        utils.mass(Q, T)


def test_rolling_avg_sd_vs_manual_computation():
    t = np.array([2, 4, 6, 8, 10])
    w = 2
    rolling_avg, rolling_sd = utils.rolling_avg_sd(t, w)
    expected_avg = np.array([(2+4)/2, (4+6)/2, (6+8)/2, (8+10)/2])
    expected_sd = np.array([
        np.std(t[0:2], ddof=0),
        np.std(t[1:3], ddof=0),
        np.std(t[2:4], ddof=0),
        np.std(t[3:5], ddof=0),
    ])
    np.testing.assert_allclose(rolling_avg, expected_avg, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rolling_sd, expected_sd, rtol=1e-12, atol=1e-12)
