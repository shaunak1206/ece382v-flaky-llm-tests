import numpy as np
import pytest
from pytsmp import utils


def test_sliding_dot_product_basic():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2, 3, 4, 5])
    expected = np.array([
        1*3 + 2*2 + 3*1,  # subseq: [1,2,3]
        2*3 + 3*2 + 4*1,  # subseq: [2,3,4]
        3*3 + 4*2 + 5*1   # subseq: [3,4,5]
    ])
    # Note: sliding_dot_product reverses Q before dot product,
    # So actually corresponds to dot(Q[::-1], T subseq)
    # Sliding dot product here means for subsequence starting at i in T,
    # dot product with Q reversed:
    # For i=0: T[0]*Q[2] + T[1]*Q[1] + T[2]*Q[0] = 1*3 + 2*2 + 3*1 = 10
    # For i=1: 2*3+3*2+4*1=6+6+4=16
    # For i=2: 3*3+4*2+5*1=9+8+5=22
    np.testing.assert_allclose(utils.sliding_dot_product(Q, T), expected)


def test_sliding_dot_product_value_error():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2])
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_sliding_dot_product_identical_series():
    Q = np.ones(4)
    T = np.ones(10)
    result = utils.sliding_dot_product(Q, T)
    expected = np.full(len(T)-len(Q)+1, 4)  # dot product of all ones length 4
    np.testing.assert_allclose(result, expected)


def test_rolling_avg_sd_basic():
    t = np.array([1, 2, 3, 4, 5])
    window_size = 3
    avg, sd = utils.rolling_avg_sd(t, window_size)
    expected_avg = np.array([2, 3, 4])
    expected_sd = np.array([np.std([1,2,3], ddof=0),
                            np.std([2,3,4], ddof=0),
                            np.std([3,4,5], ddof=0)])
    np.testing.assert_allclose(avg, expected_avg)
    np.testing.assert_allclose(sd, expected_sd)


def test_rolling_avg_sd_window_too_large():
    t = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(t, window_size=5)


def test_rolling_avg_sd_constant_series():
    t = np.ones(10)
    avg, sd = utils.rolling_avg_sd(t, 5)
    np.testing.assert_array_equal(avg, np.ones(len(avg)))
    np.testing.assert_array_equal(sd, np.zeros(len(sd)))


def test_calculate_distance_profile_basic():
    m = 3
    QT = np.array([10, 20, 30], dtype=float)
    mean_Q = 2.0
    sigma_Q = 1.0
    mean_T = np.array([1, 2, 3], dtype=float)
    sigma_T = np.array([1, 1, 1], dtype=float)
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Manual check:
    # dist = sqrt(2*m*(1 - (QT - m*mean_Q*mean_T)/(m*sigma_Q*sigma_T)))
    expected = np.sqrt(np.maximum(2*m*(1 - (QT - m*mean_Q*mean_T)/(m*sigma_Q*sigma_T)), 0))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_dimension_mismatch():
    m = 3
    QT = np.array([1, 2])
    mean_Q = 1
    sigma_Q = 1
    mean_T = np.array([1, 2, 3])
    sigma_T = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

    QT = np.array([1, 2, 3])
    mean_Q = np.array([1, 2])
    sigma_Q = np.array([1, 2])
    mean_T = np.array([1, 2, 3])
    sigma_T = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_constant_query():
    m = 3
    length = 4
    QT = np.ones(length) * 5
    mean_Q = 2
    sigma_Q = 0    # constant query
    mean_T = np.ones(length)
    sigma_T = np.array([0.5, 0, 1, 0])  # some subsequences constant, some not

    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    # For sigma_Q < epsilon, 
    # if sigma_T < epsilon => distance = 0
    # else distance = sqrt(m)
    expected = np.array([
        np.sqrt(m),  # sigma_T=0.5>epsilon, but sigma_Q<epsilon → sqrt(m)
        0,           # sigma_T=0 < epsilon and sigma_Q<epsilon → 0
        np.sqrt(m),  # sigma_T=1 > epsilon → sqrt(m)
        0            # sigma_T=0 < epsilon → 0
    ])
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_constant_target_subsequences():
    m = 2
    length = 3
    QT = np.array([1, 2, 3])
    mean_Q = 1
    sigma_Q = 1
    mean_T = np.array([1, 2, 3])
    sigma_T = np.array([0.000001, 0, 1])  # threshold: epsilon=1e-7 
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    # For sigma_T < epsilon => distance = sqrt(m)
    expected = np.array([
        np.sqrt(m),
        np.sqrt(m),
        np.sqrt(np.maximum(2*m*(1 - (QT[2] - m*mean_Q*mean_T[2]) / (m*sigma_Q*sigma_T[2])), 0))
    ])
    np.testing.assert_allclose(D, expected)


def test_sliding_dot_product_output_length():
    Q = np.array([1, 2, 3])
    T = np.arange(10)
    result = utils.sliding_dot_product(Q, T)
    assert len(result) == len(T) - len(Q) + 1


def test_sliding_dot_product_single_element_Q():
    Q = np.array([5])
    T = np.array([1, 2, 3, 4])
    expected = T * 5
    np.testing.assert_allclose(utils.sliding_dot_product(Q, T), expected)


def test_sliding_dot_product_all_zeros():
    Q = np.zeros(3)
    T = np.array([1, 2, 3, 4, 5])
    expected = np.zeros(len(T) - len(Q) + 1)
    np.testing.assert_allclose(utils.sliding_dot_product(Q, T), expected)


def test_rolling_avg_sd_large_numbers_numerical_instability():
    t = np.full(100, 1e10)
    window_size = 10
    avg, sd = utils.rolling_avg_sd(t, window_size)
    np.testing.assert_allclose(avg, np.full(len(avg), 1e10))
    np.testing.assert_allclose(sd, np.zeros(len(sd)))


def test_sliding_dot_product_realistic_random_array():
    rng = np.random.default_rng(seed=42)
    T = rng.normal(size=100)
    Q = T[10:20]
    out = utils.sliding_dot_product(Q, T)
    assert out.shape == (len(T) - len(Q) + 1, )
    # dot product of subsequence at 10 should be dot(Q[::-1], Q)
    expected_center = np.dot(Q[::-1], Q)
    np.testing.assert_allclose(out[10], expected_center, rtol=1e-5)


def test_calculate_distance_profile_with_arrays_meanQ_sigmaQ():
    m = 2
    QT = np.array([1.5, 2.5])
    mean_Q = np.array([1, 2])
    sigma_Q = np.array([0.5, 1])
    mean_T = np.array([2, 3])
    sigma_T = np.array([1, 2])
    d = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Manual calculation for each element
    expected = np.sqrt(np.maximum(
        2*m*(1 - (QT - m*mean_Q*mean_T) / (m*sigma_Q*sigma_T)),
        0
    ))
    np.testing.assert_allclose(d, expected)


def test_calculate_distance_profile_negative_values_under_sqrt():
    # Construct inputs that cause negative value inside sqrt before max clamp
    m = 3
    QT = np.array([100, 200])
    mean_Q = 10
    sigma_Q = 5
    mean_T = np.array([3, 1])
    sigma_T = np.array([0.1, 0.1])
    d = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Should not throw error and all values >= 0
    assert np.all(d >= 0)


def test_calculate_distance_profile_constant_Q_and_T_zero_distance():
    m = 2
    QT = np.array([0, 0])
    mean_Q = 5
    sigma_Q = 0  # constant Q
    mean_T = np.array([10, 10])
    sigma_T = np.array([0, 0])  # constant subsequences in T
    d = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Distance should be zeros as per code for constant Q & constant subsequences in T
    np.testing.assert_array_equal(d, np.zeros_like(d))
