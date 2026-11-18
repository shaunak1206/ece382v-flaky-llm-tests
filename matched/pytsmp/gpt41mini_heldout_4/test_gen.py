import numpy as np
import pytest
from pytsmp import utils


def test_sliding_dot_product_basic():
    Q = np.array([1, 2])
    T = np.array([1, 2, 3, 4])
    result = utils.sliding_dot_product(Q, T)
    # Manual dot products:
    # [1*1 + 2*2, 2*1 + 3*2, 3*1 + 4*2] = [5,8,11]
    expected = np.array([5, 8, 11])
    np.testing.assert_allclose(result, expected)


def test_sliding_dot_product_query_longer_than_text():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2])
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_sliding_dot_product_with_zeros():
    Q = np.array([0, 0])
    T = np.array([1, 2, 3])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([0, 0])
    np.testing.assert_allclose(result, expected)


def test_sliding_dot_product_constant_query():
    Q = np.array([5, 5])
    T = np.array([1, 2, 3, 4, 5])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([5*1+5*2, 5*2+5*3, 5*3+5*4, 5*4+5*5])
    np.testing.assert_allclose(result, expected)


def test_rolling_avg_sd_basic():
    t = np.array([1, 2, 3, 4, 5])
    window = 3
    avg, sd = utils.rolling_avg_sd(t, window)
    expected_avg = np.array([(1+2+3)/3, (2+3+4)/3, (3+4+5)/3])
    expected_sd = np.array([np.std([1,2,3], ddof=0),
                            np.std([2,3,4], ddof=0),
                            np.std([3,4,5], ddof=0)])
    np.testing.assert_allclose(avg, expected_avg)
    np.testing.assert_allclose(sd, expected_sd)


def test_rolling_avg_sd_window_too_large():
    t = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(t, 5)


def test_rolling_avg_sd_constant_series():
    t = np.array([3, 3, 3, 3, 3])
    window = 2
    avg, sd = utils.rolling_avg_sd(t, window)
    expected_avg = np.array([3, 3, 3, 3])
    expected_sd = np.array([0, 0, 0, 0])
    np.testing.assert_allclose(avg, expected_avg)
    np.testing.assert_allclose(sd, expected_sd)


def test_calculate_distance_profile_shapes_mismatch():
    QT = np.array([1, 2])
    m = 2
    mean_Q = 1
    sigma_Q = 1
    mean_T = np.array([1])
    sigma_T = np.array([1])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_constant_sigmaQ_and_sigmaT():
    QT = np.array([3, 3, 3])
    m = 3
    mean_Q = 0
    sigma_Q = 0  # constant query
    mean_T = np.array([1, 1, 1])
    sigma_T = np.array([0, 0, 0])  # constant T subsequences
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # both constant => distance = 0
    expected = np.zeros_like(QT)
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_constant_sigmaQ_nonconstant_sigmaT():
    QT = np.array([3, 3])
    m = 2
    mean_Q = 0
    sigma_Q = 0  # constant Q
    mean_T = np.array([1, 1])
    sigma_T = np.array([0.1, 0.2])  # non-constant T subsequences
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.full(len(QT), np.sqrt(m))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_nonconstant_sigmaQ_constant_sigmaT():
    QT = np.array([4, 5, 6])
    m = 3
    mean_Q = 1
    sigma_Q = 2
    mean_T = np.array([1, 2, 3])
    sigma_T = np.array([0, 0, 0])  # sigma_T constant (zero)
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.full(len(QT), np.sqrt(m))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_nonconstant_sigmaQ_sigmaT():
    QT = np.array([14, 20])
    m = 3
    mean_Q = 3
    sigma_Q = 3
    mean_T = np.array([4, 5])
    sigma_T = np.array([2, 2])
    # Distance formula:
    # sqrt(2*m*(1 - (QT - m*mean_Q*mean_T) / (m*sigma_Q*sigma_T)))
    expected = np.sqrt(np.maximum(
        2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)),
        0
    ))
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_with_array_meanQ_sigmaQ_lengths():
    # mean_Q and sigma_Q are arrays and must have length == len(mean_T)
    QT = np.array([1, 2, 3])
    m = 3
    mean_Q = np.array([1, 1, 1])
    sigma_Q = np.array([1, 1, 1])
    mean_T = np.array([1, 1, 1])
    sigma_T = np.array([1, 1, 1])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert D.shape == (3,)


def test_calculate_distance_profile_with_array_meanQ_sigmaQ_length_mismatch():
    QT = np.array([1, 2])
    m = 2
    mean_Q = np.array([1, 1])  # length 2
    sigma_Q = np.array([1])  # length 1 => mismatch
    mean_T = np.array([1, 1])
    sigma_T = np.array([1, 1])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_sliding_dot_product_with_random_data_compare_brute_force():
    np.random.seed(0)
    Q = np.random.rand(5)
    T = np.random.rand(20)
    result = utils.sliding_dot_product(Q, T)

    # Brute force calculation
    expected = np.array([np.dot(Q, T[i:i+len(Q)]) for i in range(len(T)-len(Q)+1)])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_produces_nonnegative_distances():
    QT = np.array([10, 20, 30])
    m = 3
    mean_Q = 2
    sigma_Q = 1.5
    mean_T = np.array([3, 4, 5])
    sigma_T = np.array([1, 2, 0])  # last zero sigma_T triggers special case
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Distances must be non-negative:
    assert np.all(D >= 0)


def test_calculate_distance_profile_distance_for_constant_subsequence_approximate():
    m = 4
    QT = np.array([0, 0])
    mean_Q = 0
    sigma_Q = 0
    mean_T = np.array([0, 0])
    sigma_T = np.array([1e-9, 1e-9])  # less than epsilon=1e-7, so considered constant
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    expected = np.zeros_like(QT)
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_with_nonarray_meanQ_sigmaQ_scalars():
    QT = np.array([5, 6])
    m = 2
    mean_Q = 1
    sigma_Q = 1
    mean_T = np.array([1, 2])
    sigma_T = np.array([2, 3])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert D.shape == (2,)


def test_rolling_avg_sd_large_numbers_issue():
    # Confirm that it produces results but may be numerically unstable
    t = np.array([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3])
    window = 2
    avg, sd = utils.rolling_avg_sd(t, window)
    expected_avg = np.array([(1e10 + 1e10 + 1) / 2, (1e10 + 1 + 1e10 + 2) / 2, (1e10 + 2 + 1e10 + 3) / 2])
    expected_sd = np.array([0.5, 0.5, 0.5])  # exact manual for differences
    np.testing.assert_allclose(avg, expected_avg)
    np.testing.assert_allclose(sd, expected_sd, rtol=1e-10)


def test_sliding_dot_product_q_equals_t_length():
    Q = np.array([1, 2, 3])
    T = np.array([4, 5, 6])
    # Q and T same length, result length = 1
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([np.dot(Q, T)])
    np.testing.assert_allclose(result, expected)


def test_sliding_dot_product_with_empty_query():
    # Empty query should raise ValueError due to length check (len(T)<len(Q))
    Q = np.array([])
    T = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_sliding_dot_product_with_empty_text():
    Q = np.array([1])
    T = np.array([])
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_calculate_distance_profile_with_zero_length_inputs():
    # All zero-length arrays mismatch lengths and should raise
    QT = np.array([])
    m = 0
    mean_Q = np.array([])
    sigma_Q = np.array([])
    mean_T = np.array([])
    sigma_T = np.array([])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
