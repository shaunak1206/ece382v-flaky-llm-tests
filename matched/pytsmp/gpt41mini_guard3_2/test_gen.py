import numpy as np
import pytest
from pytsmp import utils


def test_sliding_dot_product_basic():
    np.random.seed(0)
    Q = np.array([1, 2, 3])
    T = np.array([0, 1, 2, 3, 4, 5])
    result = utils.sliding_dot_product(Q, T)
    expected_length = len(T) - len(Q) + 1
    assert result.shape == (expected_length,)
    # Manually computed dot products for the subsequences of T with Q reversed:
    # Q reversed = [3,2,1]
    # subsequences: [0,1,2], [1,2,3], [2,3,4], [3,4,5]
    expected = np.array([
        3*0 + 2*1 + 1*2,
        3*1 + 2*2 + 1*3,
        3*2 + 2*3 + 1*4,
        3*3 + 2*4 + 1*5,
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_sliding_dot_product_raises_for_shorter_T():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2])
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_sliding_dot_product_constant_series():
    np.random.seed(1)
    Q = np.ones(5)
    T = np.ones(10)
    result = utils.sliding_dot_product(Q, T)
    # Dot product of all subsequences of T with Q should equal len(Q) * 1*1 = 5
    expected = np.full(len(T) - len(Q) + 1, 5)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_rolling_avg_sd_basic():
    np.random.seed(2)
    t = np.array([1, 2, 3, 4, 5])
    window_size = 3
    mean_, sd_ = utils.rolling_avg_sd(t, window_size)
    expected_mean = np.array([
        (1 + 2 + 3) / 3,
        (2 + 3 + 4) / 3,
        (3 + 4 + 5) / 3,
    ])
    expected_sd = np.array([
        np.std([1, 2, 3], ddof=0),
        np.std([2, 3, 4], ddof=0),
        np.std([3, 4, 5], ddof=0),
    ])
    np.testing.assert_allclose(mean_, expected_mean, rtol=1e-12)
    np.testing.assert_allclose(sd_, expected_sd, rtol=1e-12)


def test_rolling_avg_sd_raises_for_large_window():
    t = np.array([1, 2, 3])
    window_size = 4
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(t, window_size)


def test_rolling_avg_sd_constant_series_zero_sd():
    t = np.ones(10)
    window_size = 3
    mean_, sd_ = utils.rolling_avg_sd(t, window_size)
    expected_mean = np.ones(8)
    expected_sd = np.zeros(8)
    np.testing.assert_allclose(mean_, expected_mean, rtol=1e-12)
    np.testing.assert_allclose(sd_, expected_sd, rtol=1e-12)


def test_calculate_distance_profile_basic():
    # Use small fixed data to verify output
    m = 3
    QT = np.array([3, 6, 9, 12])
    mean_Q = 2.0
    sigma_Q = 1.0
    mean_T = np.array([1, 2, 3, 4])
    sigma_T = np.array([1, 1, 1, 1])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.sqrt(np.maximum(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)), 0))
    np.testing.assert_allclose(D, expected, rtol=1e-12)


def test_calculate_distance_profile_raises_dimension_mismatch():
    QT = np.array([1, 2])
    mean_Q = 1.0
    sigma_Q = 1.0
    mean_T = np.array([1])
    sigma_T = np.array([1, 2])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 1, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_constant_query_and_subsequence():
    m = 4
    length = 5
    QT = np.ones(length)
    mean_Q = 1.0
    sigma_Q = 0.0  # constant query
    mean_T = np.ones(length)
    sigma_T = np.zeros(length)  # constant subsequences
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.zeros(length)  # distance is 0 for constant to constant
    np.testing.assert_allclose(D, expected, rtol=1e-12)


def test_calculate_distance_profile_constant_query_nonconstant_subsequence():
    m = 4
    length = 3
    QT = np.ones(length)
    mean_Q = 1.0
    sigma_Q = 0.0  # constant query
    mean_T = np.ones(length)
    sigma_T = np.ones(length)  # nonconstant subsequences
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.full(length, np.sqrt(m))  # distance is sqrt(m)
    np.testing.assert_allclose(D, expected, rtol=1e-12)


def test_calculate_distance_profile_nonconstant_query_constant_subsequence():
    m = 4
    length = 3
    QT = np.ones(length)
    mean_Q = 1.0
    sigma_Q = 1.0  # nonconstant query
    mean_T = np.ones(length)
    sigma_T = np.zeros(length)  # constant subsequences
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.full(length, np.sqrt(m))  # distance is sqrt(m)
    np.testing.assert_allclose(D, expected, rtol=1e-12)


def test_mass_basic_consistency():
    np.random.seed(42)
    T = np.array([1, 2, 3, 2, 1, 2, 3, 4, 5])
    Q = np.array([2, 3, 2])
    D = utils.mass(Q, T)
    expected_len = len(T) - len(Q) + 1
    assert D.shape == (expected_len,)
    # All distances should be non-negative
    assert np.all(D >= 0)


def test_mass_raises_for_shorter_T():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2])
    with pytest.raises(ValueError):
        utils.mass(Q, T)


def test_mass_constant_query_results():
    np.random.seed(3)
    Q = np.ones(4)
    T = np.arange(10)
    D = utils.mass(Q, T)
    # Since query is constant, distances should be sqrt(m) if subsequence is not constant, 0 if constant
    # The subsequences of T are not constant, except possibly if window=4 covers equal elements, which it doesn't here
    assert D.shape == (len(T) - len(Q) + 1,)
    # Distances should be >= 0 and <= sqrt(m)
    assert np.all(D >= 0)
    assert np.all(D <= np.sqrt(len(Q)))


def test_mass_constant_T_variable_query():
    np.random.seed(4)
    T = np.ones(10)
    Q = np.array([1, 2, 3])
    D = utils.mass(Q, T)
    # The subsequences in T are constant, so distances are either 0 or sqrt(m)
    # Because query is non-constant, subsequence constant → distance sqrt(m)
    expected = np.full(len(T) - len(Q) + 1, np.sqrt(len(Q)))
    np.testing.assert_allclose(D, expected, rtol=1e-12)
