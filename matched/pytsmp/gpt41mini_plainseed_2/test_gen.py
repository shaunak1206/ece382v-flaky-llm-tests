import numpy as np
import pytest
from pytsmp import utils


def test_sliding_dot_product_basic():
    Q = np.array([1, 2, 3])
    T = np.array([0, 1, 2, 3, 4, 5])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([
        1*0 + 2*1 + 3*2,   # Q reversed dot with T[0:3]
        1*1 + 2*2 + 3*3,
        1*2 + 2*3 + 3*4,
        1*3 + 2*4 + 3*5,
    ])
    np.testing.assert_allclose(result, expected)


def test_sliding_dot_product_T_shorter_than_Q_raises():
    Q = np.array([1, 2, 3, 4])
    T = np.array([1, 2])
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_sliding_dot_product_result_length():
    Q = np.arange(5)
    T = np.arange(10)
    result = utils.sliding_dot_product(Q, T)
    assert len(result) == len(T) - len(Q) + 1


def test_rolling_avg_sd_basic():
    t = np.array([1, 2, 3, 4, 5])
    window_size = 3
    mean, sd = utils.rolling_avg_sd(t, window_size)
    expected_mean = np.array([2., 3., 4.])
    expected_sd = np.array([np.std([1, 2, 3], ddof=0),
                            np.std([2, 3, 4], ddof=0),
                            np.std([3, 4, 5], ddof=0)])
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(sd, expected_sd)


def test_rolling_avg_sd_window_larger_than_series_raises():
    t = np.array([1, 2])
    window_size = 5
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(t, window_size)


def test_rolling_avg_sd_constant_series_zero_sd():
    t = np.ones(10)
    mean, sd = utils.rolling_avg_sd(t, 5)
    assert np.allclose(sd, 0)


def test_calculate_distance_profile_dimension_mismatch_QT_meanT_sigmaT():
    QT = np.array([1, 2, 3])
    mean_T = np.array([1, 2])
    sigma_T = np.array([1, 2, 3])
    m = 3
    mean_Q = 1
    sigma_Q = 1
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_dimension_mismatch_meanQ_sigmaQ_arrays():
    QT = np.array([1, 2, 3])
    mean_T = np.array([1, 2, 3])
    sigma_T = np.array([1, 2, 3])
    m = 3
    mean_Q = np.array([1, 2])
    sigma_Q = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_constant_sigmaQ_and_sigmaT_zero():
    # If both sigmas are less than epsilon, distance_profile should be all zeros.
    m = 5
    length = 4
    QT = np.zeros(length)
    mean_Q = 2
    sigma_Q = 0
    mean_T = np.ones(length)
    sigma_T = np.zeros(length)
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    np.testing.assert_array_equal(D, np.zeros(length))


def test_calculate_distance_profile_constant_sigmaQ_nonzero_sigmaT():
    m = 3
    length = 3
    QT = np.array([0.1, 0.2, 0.3])
    mean_Q = 5
    sigma_Q = 0
    mean_T = np.array([1, 2, 3])
    sigma_T = np.array([1, 1, 1])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    expected = np.sqrt(m) * np.ones(length)
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_nonzero_sigmaQ_sigmaT():
    m = 2
    QT = np.array([5.0, 10.0])
    mean_Q = 3
    sigma_Q = 1
    mean_T = np.array([2, 4])
    sigma_T = np.array([0.5, 1.5])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Manually calculate expected distances:
    expected = np.sqrt(np.maximum(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)), 0))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_sigmaT_zero_sigmaQ_nonzero():
    m = 4
    length = 3
    QT = np.array([1, 2, 3])
    mean_Q = 0
    sigma_Q = 1
    mean_T = np.array([1, 2, 3])
    sigma_T = np.array([0, 0, 0])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.sqrt(m) * np.ones(length)
    np.testing.assert_allclose(D, expected)


def test_sliding_dot_product_with_floats():
    Q = np.array([0.5, 1.5])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([
        0.5*1.0 + 1.5*2.0,
        0.5*2.0 + 1.5*3.0,
        0.5*3.0 + 1.5*4.0,
    ])
    np.testing.assert_allclose(result, expected)


def test_sliding_dot_product_identical_series():
    Q = np.array([1,2,3])
    T = np.array([1,2,3])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([1*1 + 2*2 + 3*3])
    np.testing.assert_allclose(result, expected)


def test_sliding_dot_product_zero_query():
    Q = np.array([0, 0, 0])
    T = np.array([1, 2, 3, 4])
    result = utils.sliding_dot_product(Q, T)
    expected = np.zeros(len(T) - len(Q) + 1)
    np.testing.assert_allclose(result, expected)


def test_calculate_distance_profile_with_arrays_meanQ_sigmaQ():
    QT = np.array([3, 3, 3])
    m = 3
    mean_Q = np.array([1, 1, 1])
    sigma_Q = np.array([1, 1, 1])
    mean_T = np.array([2, 2, 2])
    sigma_T = np.array([1, 1, 1])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.sqrt(np.maximum(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)), 0))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_with_negative_values():
    QT = np.array([-10, -20, -30])
    m = 3
    mean_Q = 0
    sigma_Q = 1
    mean_T = np.array([-1, -2, -3])
    sigma_T = np.array([1, 1, 1])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.sqrt(np.maximum(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)), 0))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_distance_nonnegative():
    # The distance must always be >= 0
    QT = np.array([100, 200, 300])
    m = 2
    mean_Q = 1
    sigma_Q = 1
    mean_T = np.array([1, 2, 3])
    sigma_T = np.array([10, 0, 5])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert np.all(D >= 0)


def test_sliding_dot_product_large_series():
    np.random.seed(0)
    Q = np.random.randn(50)
    T = np.random.randn(100)
    result = utils.sliding_dot_product(Q, T)
    assert len(result) == len(T) - len(Q) + 1
    assert not np.any(np.isnan(result))


def test_sliding_dot_product_vs_manual_dot():
    Q = np.array([1, 2, 3])
    T = np.array([4, 5, 6, 7, 8])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([
        np.dot(Q[::-1], T[0:3]),
        np.dot(Q[::-1], T[1:4]),
        np.dot(Q[::-1], T[2:5]),
    ])
    np.testing.assert_allclose(result, expected)
