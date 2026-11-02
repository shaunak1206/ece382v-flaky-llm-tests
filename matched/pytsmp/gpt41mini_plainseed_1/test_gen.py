import numpy as np
import pytest
from pytsmp import utils


def test_sliding_dot_product_basic():
    T = np.array([1, 2, 3, 4, 5])
    Q = np.array([1, 0])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([1*1 + 0*2, 1*2 + 0*3, 1*3 + 0*4, 1*4 + 0*5])
    np.testing.assert_allclose(result, expected)


def test_sliding_dot_product_full_dot():
    T = np.array([1, 2, 3, 4])
    Q = np.array([1, 2])
    result = utils.sliding_dot_product(Q, T)
    # Subsequences of length 2: [1,2], [2,3], [3,4]
    # Dot products:
    # Q reversed = [2,1]
    # Expected: dot([2,1], [1,2])=2*1+1*2=4, dot([2,1],[2,3])=2*2+1*3=7, dot([2,1],[3,4])=2*3+1*4=10
    expected = np.array([4, 7, 10])
    np.testing.assert_allclose(result, expected)


def test_sliding_dot_product_query_longer_than_T_raises():
    T = np.array([1, 2, 3])
    Q = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_rolling_avg_sd_simple():
    t = np.array([1, 2, 3, 4, 5])
    window = 3
    mean, sd = utils.rolling_avg_sd(t, window)
    expected_means = np.array([(1+2+3)/3, (2+3+4)/3, (3+4+5)/3])
    expected_sds = np.array([np.std([1, 2, 3], ddof=0),
                             np.std([2, 3, 4], ddof=0),
                             np.std([3, 4, 5], ddof=0)])
    np.testing.assert_allclose(mean, expected_means)
    np.testing.assert_allclose(sd, expected_sds)


def test_rolling_avg_sd_window_too_large():
    t = np.array([1, 2])
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(t, 3)


def test_rolling_avg_sd_constant_series():
    t = np.array([5, 5, 5, 5])
    mean, sd = utils.rolling_avg_sd(t, 2)
    expected_mean = np.array([5, 5, 5])
    expected_sd = np.array([0, 0, 0])
    np.testing.assert_array_equal(mean, expected_mean)
    np.testing.assert_array_equal(sd, expected_sd)


def test_calculate_distance_profile_basic():
    m = 3
    QT = np.array([3, 6, 9], dtype=float)
    mean_Q = 2.0
    sigma_Q = 1.0
    mean_T = np.array([1, 2, 3], dtype=float)
    sigma_T = np.array([1, 1, 1], dtype=float)
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # distance calculated by sqrt(2*m*(1 - (QT - m*mean_Q*mean_T)/(m*sigma_Q*sigma_T)))
    expected = np.sqrt(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_input_dimension_mismatch():
    QT = np.array([1, 2])
    m = 2
    mean_Q = 1.0
    sigma_Q = 1.0
    mean_T = np.array([1])   # length mismatch
    sigma_T = np.array([1])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_constant_Q_and_constant_T():
    m = 2
    length = 3
    QT = np.array([0, 0, 0], dtype=float)
    mean_Q = 1.0
    sigma_Q = 0.0  # constant Q
    mean_T = np.array([1, 2, 3], dtype=float)
    sigma_T = np.array([0.0, 0.0, 0.0], dtype=float)  # constant T subsequences
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    expected = np.zeros(len(QT))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_constant_Q_nonconstant_T():
    m = 2
    length = 3
    QT = np.array([0, 0, 0], dtype=float)
    mean_Q = 1.0
    sigma_Q = 0.0  # constant Q
    mean_T = np.array([1, 2, 3], dtype=float)
    sigma_T = np.array([1.0, 1.0, 1.0], dtype=float)
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    expected = np.full(len(QT), np.sqrt(m))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_nonconstant_Q_constant_T():
    m = 2
    length = 3
    QT = np.array([0, 0, 0], dtype=float)
    mean_Q = 1.0
    sigma_Q = 1.0
    mean_T = np.array([1, 1, 1], dtype=float)
    sigma_T = np.array([0.0, 0.0, 0.0], dtype=float)  # constant T
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    expected = np.full(len(QT), np.sqrt(m))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_negative_value_protection():
    # Construct inputs that could produce negative value inside sqrt
    m = 2
    QT = np.array([10.0, 10.0])
    mean_Q = 5.0
    sigma_Q = 2.0
    mean_T = np.array([5.0, 5.0])
    sigma_T = np.array([2.0, 2.0])
    # Force a negative inside sqrt by making (QT - m*mean_Q*mean_T)/(m*sigma_Q*sigma_T) > 1
    # e.g. (10 - 2*5*5) / (2*2*2) = (10 -50)/8 = -40/8 = -5, but that is less than 1 so no problem.
    # Reverse sign to test max(.,0)
    invalid_QT = np.array([50.0, 50.0])
    D = utils.calculate_distance_profile(invalid_QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    assert np.all(D >= 0)


def test_sliding_dot_product_with_zeros():
    T = np.zeros(5)
    Q = np.array([1, 2])
    result = utils.sliding_dot_product(Q, T)
    expected = np.zeros(len(T) - len(Q) + 1)
    np.testing.assert_array_equal(result, expected)


def test_sliding_dot_product_with_negative_values():
    T = np.array([1, -1, 1, -1, 1])
    Q = np.array([1, -1])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([1* -1 + -1 * 1, 1*1 + -1* -1, 1* -1 + -1*1, 1*1 + -1* -1])
    # careful: Actually expected is:
    # Sliding dot product is with Q reversed: Q[::-1] = [-1, 1]
    # So dot([-1,1],[1,-1]) = -1*1 + 1*(-1) = -1 -1 = -2
    # dot([-1,1],[-1,1]) = -1*-1 + 1*1 = 1 +1 =2
    # dot([-1,1],[1,-1]) = -2
    # dot([-1,1],[-1,1]) = 2
    expected = np.array([-2, 2, -2, 2])
    np.testing.assert_allclose(result, expected)


def test_sliding_dot_product_length_one_query():
    T = np.array([1, 3, 5, 7])
    Q = np.array([2])
    result = utils.sliding_dot_product(Q, T)
    expected = T * Q[0]
    np.testing.assert_allclose(result, expected)


def test_calculate_distance_profile_with_array_mean_sigma_Q():
    m = 3
    QT = np.array([3, 6, 9], dtype=float)
    mean_Q = np.array([1.0, 2.0, 3.0])
    sigma_Q = np.array([1.0, 1.0, 1.0])
    mean_T = np.array([1, 2, 3], dtype=float)
    sigma_T = np.array([1, 1, 1], dtype=float)
    with pytest.raises(ValueError):
        # mean_Q/sigma_Q length mismatch with mean_T length triggers ValueError
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_array_mean_sigma_Q_and_correct_length():
    m = 3
    length = 3
    QT = np.array([3, 6, 9], dtype=float)
    mean_Q = np.array([2.0, 2.0, 2.0])
    sigma_Q = np.array([1.0, 1.0, 1.0])
    mean_T = np.array([1, 2, 3], dtype=float)
    sigma_T = np.array([1, 1, 1], dtype=float)

    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.sqrt(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)))
    np.testing.assert_allclose(D, expected)


def test_sliding_dot_product_large_input_consistency():
    np.random.seed(0)
    T = np.random.rand(1000)
    Q = T[100:120]
    dot_products = utils.sliding_dot_product(Q, T)
    # confirm length correct
    assert dot_products.shape[0] == len(T) - len(Q) + 1
    # Confirm that dot_products[0] matches dot(Q[::-1], T[0:len(Q)])
    expected0 = np.dot(Q[::-1], T[0:len(Q)])
    np.testing.assert_allclose(dot_products[0], expected0)


def test_rolling_avg_sd_large_input_consistency():
    np.random.seed(1)
    t = np.random.rand(500)
    window = 50
    mean, sd = utils.rolling_avg_sd(t, window)
    assert mean.shape[0] == len(t) - window + 1
    assert sd.shape[0] == len(t) - window + 1
    # Check first window manually:
    expected_mean = np.mean(t[:window])
    expected_sd = np.std(t[:window])
    np.testing.assert_allclose(mean[0], expected_mean)
    np.testing.assert_allclose(sd[0], expected_sd)
