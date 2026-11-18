import pytest
import numpy as np
from pytsmp import utils


def test_sliding_dot_product_basic():
    Q = np.array([1, 2])
    T = np.array([3, 4, 5])
    expected = np.array([3*2 + 4*1, 4*2 + 5*1])  # manually compute for reversed Q: [2,1]
    result = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, expected)


def test_sliding_dot_product_value_error():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2])
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_sliding_dot_product_length():
    Q = np.array([1, 2, 3])
    T = np.arange(10)
    result = utils.sliding_dot_product(Q, T)
    assert len(result) == len(T) - len(Q) + 1


def test_sliding_dot_product_known_values():
    Q = np.array([1, 0, -1])
    T = np.array([1, 2, 3, 4, 5])
    # The sliding dot product Q[::-1] = [-1,0,1], compute dot products:
    # window1: [1,2,3] dot [-1,0,1] = 1*-1 + 2*0 + 3*1 = 2
    # window2: [2,3,4] dot [-1,0,1] = 2*-1 + 3*0 + 4*1 = 2
    # window3: [3,4,5] dot [-1,0,1] = 3*-1 + 4*0 + 5*1 = 2
    expected = np.array([2., 2., 2.])
    result = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, expected)


def test_rolling_avg_sd_basic():
    t = np.array([1, 2, 3, 4, 5])
    window_size = 3
    expected_means = np.array([2., 3., 4.])
    expected_sds = np.array([np.std([1,2,3]), np.std([2,3,4]), np.std([3,4,5])])
    means, sds = utils.rolling_avg_sd(t, window_size)
    np.testing.assert_allclose(means, expected_means)
    np.testing.assert_allclose(sds, expected_sds)


def test_rolling_avg_sd_value_error_too_large_window():
    t = np.array([1, 2])
    window_size = 3
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(t, window_size)


def test_rolling_avg_sd_constant_input():
    t = np.array([5, 5, 5, 5])
    window_size = 2
    means, sds = utils.rolling_avg_sd(t, window_size)
    expected_means = np.array([5, 5, 5])
    expected_sds = np.array([0, 0, 0])
    np.testing.assert_allclose(means, expected_means)
    np.testing.assert_allclose(sds, expected_sds)


def test_calculate_distance_profile_shapes_and_basic_behavior():
    m = 3
    QT = np.array([1., 2., 3.])
    mean_Q = 1.
    sigma_Q = 1.
    mean_T = np.array([1., 2., 3.])
    sigma_T = np.array([1., 1., 1.])
    dist = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert dist.shape == QT.shape
    assert np.all(dist >= 0)


def test_calculate_distance_profile_input_dimension_mismatch():
    m = 3
    QT = np.array([1., 2., 3.])
    mean_Q = np.array([1., 2.])
    sigma_Q = np.array([1., 2.])
    mean_T = np.array([1., 2., 3.])
    sigma_T = np.array([1., 1., 1.])
    # mean_Q length mismatch with sigma_Q or mean_T
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_sigma_q_zero_and_sigma_t_zero():
    m = 4
    length = 5
    QT = np.ones(length)
    mean_Q = 2
    sigma_Q = 0  # constant Q
    mean_T = np.ones(length)
    sigma_T = np.zeros(length)  # constant T subsequences
    dist = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Distance should be zero vector since both are constant sequences
    np.testing.assert_allclose(dist, np.zeros(length))


def test_calculate_distance_profile_sigma_q_zero_sigma_t_nonzero():
    m = 4
    length = 3
    QT = np.ones(length)
    mean_Q = 2
    sigma_Q = 0  # constant Q
    mean_T = np.array([1., 2., 3.])
    sigma_T = np.array([1., 0.1, 2.])  # not all zero
    dist = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Distance should be sqrt(m) for non-zero sigma_T and sigma_Q=0
    expected = np.full(length, np.sqrt(m))
    np.testing.assert_allclose(dist, expected)


def test_calculate_distance_profile_sigma_t_zero_sigma_q_nonzero():
    m = 5
    length = 4
    QT = np.ones(length)
    mean_Q = 3
    sigma_Q = 1
    mean_T = np.array([1.0, 2.0, 3.0, 4.0])
    sigma_T = np.array([0.0, 0.0, 0.0, 0.0])  # constant subsequences
    dist = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.full(length, np.sqrt(m))
    np.testing.assert_allclose(dist, expected)


def test_mass_basic_example_and_value_error():
    Q = np.array([1, 0])
    T = np.array([1, 0, 2, 1])
    D = utils.mass(Q, T)
    assert D.shape == (len(T) - len(Q) + 1,)
    # Test error raised when Q longer than T
    with pytest.raises(ValueError):
        utils.mass(np.arange(5), np.arange(4))


def test_mass_distance_nonnegative():
    Q = np.array([1, 2, 3])
    T = np.array([3, 2, 1, 2, 3, 4])
    D = utils.mass(Q, T)
    assert np.all(D >= 0), "Distance profile contains negative values"


def test_sliding_dot_product_padding_behavior():
    # Test with odd and even length T
    Q = np.array([1, 2])
    T_even = np.arange(6)  # length 6 even
    T_odd = np.arange(5)   # length 5 odd
    res_even = utils.sliding_dot_product(Q, T_even)
    res_odd = utils.sliding_dot_product(Q, T_odd)
    assert len(res_even) == len(T_even) - len(Q) + 1
    assert len(res_odd) == len(T_odd) - len(Q) + 1


def test_calculate_distance_profile_distance_formula_correctness():
    # Prepare inputs to test sqrt max(0, ..) part
    m = 2
    QT = np.array([5, 7])
    mean_Q = 2
    sigma_Q = 1
    mean_T = np.array([1.5, 3.5])
    sigma_T = np.array([1, 1.5])
    # distance calculation:
    # dist_i = sqrt(max(0, 2*m*(1 - (QT_i - m*mean_Q*mean_T_i) / (m*sigma_Q*sigma_T_i))))
    expected = np.sqrt(np.maximum(
        2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)), 0
    ))
    dist = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    np.testing.assert_allclose(dist, expected)


def test_calculate_distance_profile_with_array_mean_q_sigma_q():
    m = 3
    QT = np.array([1., 1., 1.])
    mean_Q = np.array([1., 1., 1.])
    sigma_Q = np.array([1., 1., 1.])
    mean_T = np.array([1., 1., 1.])
    sigma_T = np.array([1., 1., 1.])
    dist = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert dist.shape == QT.shape


def test_calculate_distance_profile_epsilon_tolerance():
    m = 3
    length = 2
    QT = np.array([1., 1.])
    mean_Q = 0
    sigma_Q = 1e-8  # smaller than epsilon parameter
    mean_T = np.array([0., 0.])
    sigma_T = np.array([0., 1e-9])
    dist = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    # Since sigma_Q < epsilon, conditional applies
    expected = np.where(np.abs(sigma_T) < 1e-7,
                        np.zeros_like(QT),
                        np.full_like(QT, np.sqrt(m)))
    np.testing.assert_allclose(dist, expected)


def test_rolling_avg_sd_with_float32_and_window():
    t = np.array([1., 2., 3., 4., 5.], dtype=np.float32)
    window_size = 3
    means, sds = utils.rolling_avg_sd(t, window_size)
    assert means.dtype == np.float64  # cumsum returns float64 by default
    assert sds.dtype == np.float64


def test_sliding_dot_product_output_real():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2, 3, 4, 5])
    out = utils.sliding_dot_product(Q, T)
    # Output should be real numbers (float)
    assert np.isrealobj(out)


def test_calculate_distance_profile_constant_mean_q_sigma_q_arrays():
    m = 2
    length = 3
    QT = np.array([2., 3., 4.])
    mean_Q = np.array([1., 1., 1.])
    sigma_Q = np.array([0., 0., 0.])  # constant Q windows
    mean_T = np.array([1., 1., 1.])
    sigma_T = np.array([0., 1., 0])  # some constant, some not
    dist = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.where(np.abs(sigma_Q) < 1e-7,
                        np.where(np.abs(sigma_T) < 1e-7, 0, np.sqrt(m)),
                        np.sqrt(np.maximum(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)), 0)))
    np.testing.assert_allclose(dist, expected)
