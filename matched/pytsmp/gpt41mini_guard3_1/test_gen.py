import numpy as np
import pytest
import random

from pytsmp.utils import sliding_dot_product, rolling_avg_sd, calculate_distance_profile


def test_sliding_dot_product_basic():
    random.seed(0)
    np.random.seed(0)
    Q = np.array([1, 2, 3])
    T = np.array([0, 1, 2, 3, 4])
    result = sliding_dot_product(Q, T)
    expected = np.array([8., 14., 20.])  # Manually computed: dot([1,2,3],[0,1,2])=8,...
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8)


def test_sliding_dot_product_length_check():
    Q = np.array([1, 2, 3, 4])
    T = np.array([1, 2])
    with pytest.raises(ValueError):
        sliding_dot_product(Q, T)


def test_sliding_dot_product_output_length():
    random.seed(1)
    np.random.seed(1)
    T = np.arange(10, dtype=float)
    Q = np.arange(4, dtype=float)
    res = sliding_dot_product(Q, T)
    expected_length = len(T) - len(Q) + 1
    assert len(res) == expected_length


def test_rolling_avg_sd_basic_values():
    random.seed(2)
    np.random.seed(2)
    t = np.array([1., 2., 3., 4., 5.])
    window_size = 3
    avg, sd = rolling_avg_sd(t, window_size)
    expected_avg = np.array([2., 3., 4.])
    expected_sd = np.array([np.std([1., 2., 3.], ddof=0),
                            np.std([2., 3., 4.], ddof=0),
                            np.std([3., 4., 5.], ddof=0)])
    np.testing.assert_allclose(avg, expected_avg, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(sd, expected_sd, rtol=1e-12, atol=1e-15)


def test_rolling_avg_sd_window_size_too_large():
    t = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        rolling_avg_sd(t, 4)


def test_rolling_avg_sd_constant_input_sd_zero():
    random.seed(3)
    np.random.seed(3)
    t = np.array([5., 5., 5., 5., 5.])
    window_size = 3
    avg, sd = rolling_avg_sd(t, window_size)
    expected_avg = np.array([5., 5., 5.])
    expected_sd = np.array([0., 0., 0.])
    np.testing.assert_allclose(avg, expected_avg)
    np.testing.assert_allclose(sd, expected_sd)


def helper_build_distance_params(m, size, const_sigma_Q=False, const_sigma_T_positions=()):
    """
    Helper function to build parameters to test calculate_distance_profile.
    const_sigma_Q: if True, sigma_Q set to 0
    const_sigma_T_positions: indices for which sigma_T is zero
    """
    np.random.seed(42)
    QT = np.random.rand(size)
    mean_Q = 2.0
    sigma_Q = 0.0 if const_sigma_Q else 1.5
    mean_T = np.random.rand(size) * 3
    sigma_T = np.random.rand(size)
    sigma_T[list(const_sigma_T_positions)] = 0.0
    return QT, mean_Q, sigma_Q, mean_T, sigma_T


def test_calculate_distance_profile_size_mismatch():
    QT = np.array([1, 2])
    mean_Q = 1.0
    sigma_Q = 1.0
    mean_T = np.array([1])
    sigma_T = np.array([1, 2])
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_all_nonzero_sigma():
    m = 3
    size = 5
    QT, mean_Q, sigma_Q, mean_T, sigma_T = helper_build_distance_params(m, size)
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert D.shape == (size,)
    assert np.all(D >= 0)  # distances are non-negative


def test_calculate_distance_profile_sigma_Q_zero_sigma_T_zero_pos():
    m = 4
    size = 6
    const_pos = [1, 3]
    QT, mean_Q, sigma_Q, mean_T, sigma_T = helper_build_distance_params(m, size, const_sigma_Q=True, const_sigma_T_positions=const_pos)
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # For indices where sigma_Q and sigma_T are both < epsilon, distance should be 0
    for i in range(size):
        if abs(sigma_Q) < 1e-7 and abs(sigma_T[i]) < 1e-7:
            assert np.isclose(D[i], 0)
        elif abs(sigma_Q) < 1e-7 or abs(sigma_T[i]) < 1e-7:
            assert np.isclose(D[i], np.sqrt(m))
        else:
            assert D[i] >= 0


def test_calculate_distance_profile_sigma_Q_zero_sigma_T_nonzero():
    m = 5
    size = 7
    QT, mean_Q, sigma_Q, mean_T, sigma_T = helper_build_distance_params(m, size, const_sigma_Q=True)
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Where sigma_Q < epsilon and sigma_T > epsilon -> distance sqrt(m)
    for i in range(size):
        if abs(sigma_T[i]) < 1e-7:
            # distance either 0 or sqrt(m) depending on sigma_T
            assert D[i] in (0, np.sqrt(m))
        else:
            assert np.isclose(D[i], np.sqrt(m))


def test_calculate_distance_profile_sigma_T_zero_sigma_Q_nonzero():
    m = 5
    size = 4
    const_pos = [0, 2]
    QT, mean_Q, sigma_Q, mean_T, sigma_T = helper_build_distance_params(m, size, const_sigma_Q=False, const_sigma_T_positions=const_pos)
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    for i in range(size):
        if abs(sigma_T[i]) < 1e-7:
            assert np.isclose(D[i], np.sqrt(m))
        else:
            assert D[i] >= 0


def test_calculate_distance_profile_vector_mean_sigmaQ_equal_lengths():
    m = 3
    size = 4
    np.random.seed(100)
    QT = np.random.rand(size)
    mean_Q = np.random.rand(size)
    sigma_Q = np.random.rand(size) + 0.1  # ensure non-zero
    mean_T = np.random.rand(size)
    sigma_T = np.random.rand(size) + 0.1
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert D.shape == (size,)
    assert np.all(D >= 0)


def test_calculate_distance_profile_vector_mean_sigmaQ_mismatch_raises():
    m = 2
    size = 3
    QT = np.ones(size)
    mean_Q = np.ones(size)
    sigma_Q = np.ones(size + 1)  # mismatch length
    mean_T = np.ones(size)
    sigma_T = np.ones(size)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_sliding_dot_product_full_sequence_reproducible():
    random.seed(42)
    np.random.seed(42)
    T = np.linspace(0, 1, 50)
    Q = np.sin(np.linspace(0, 2*np.pi, 10))
    res1 = sliding_dot_product(Q, T)
    np.random.seed(42)
    res2 = sliding_dot_product(Q, T)
    np.testing.assert_allclose(res1, res2)


def test_rolling_avg_sd_non_integer_window():
    # window size is int param, test float should cause failure (or converted). We expect TypeError or ValueError
    t = np.arange(10)
    with pytest.raises(TypeError):
        rolling_avg_sd(t, 3.5)


def test_rolling_avg_sd_single_element_window():
    # window size == 1 means avg == t and sd == 0
    t = np.array([5., 6., 7., 8.])
    avg, sd = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(avg, t)
    np.testing.assert_allclose(sd, np.zeros_like(t))


def test_sliding_dot_product_with_zeros():
    # test with zeros in Q and T
    Q = np.array([0., 0., 0.])
    T = np.array([0., 0., 0., 0., 0.])
    result = sliding_dot_product(Q, T)
    expected = np.zeros(len(T) - len(Q) + 1)
    np.testing.assert_allclose(result, expected)


def test_calculate_distance_profile_zero_length_inputs():
    # Empty inputs cause ValueError due to dimension check
    with pytest.raises(ValueError):
        calculate_distance_profile(np.array([]), 1, 0, 0, np.array([]), np.array([]))


def test_calculate_distance_profile_distance_range(monkeypatch):
    # Patch numpy.sqrt to track inputs to ensure sqrt arg non-negative
    # But since np.maximum(0, ...) is applied, result should never be nan
    m = 3
    size = 4
    QT, mean_Q, sigma_Q, mean_T, sigma_T = helper_build_distance_params(m, size)
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert np.all(np.isfinite(D))
    assert np.all(D >= 0)
