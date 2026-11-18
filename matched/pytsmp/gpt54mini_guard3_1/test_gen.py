import numpy as np
import pytest

from pytsmp import utils


def test_rolling_avg_sd_basic_window_2():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = utils.rolling_avg_sd(t, 2)
    np.testing.assert_allclose(mean, np.array([1.5, 2.5, 3.5]))
    np.testing.assert_allclose(sd, np.array([0.5, 0.5, 0.5]))


def test_rolling_avg_sd_window_equals_length():
    t = np.array([1.0, 5.0, 9.0])
    mean, sd = utils.rolling_avg_sd(t, 3)
    np.testing.assert_allclose(mean, np.array([5.0]))
    np.testing.assert_allclose(sd, np.array([np.std(t)]))


def test_rolling_avg_sd_single_element_windows_have_zero_sd():
    t = np.array([3.0, -2.0, 10.0])
    mean, sd = utils.rolling_avg_sd(t, 1)
    np.testing.assert_allclose(mean, t)
    np.testing.assert_allclose(sd, np.zeros_like(t))


def test_rolling_avg_sd_raises_when_window_too_large():
    t = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Window size should be smaller"):
        utils.rolling_avg_sd(t, 3)


def test_sliding_dot_product_basic_even_length():
    Q = np.array([1.0, 2.0])
    T = np.array([3.0, 4.0, 5.0, 6.0])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([11.0, 14.0, 17.0])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_basic_odd_length():
    Q = np.array([1.0, 0.0, -1.0])
    T = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([-2.0, -2.0, -2.0])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_matches_direct_computation():
    Q = np.array([2.0, -1.0, 3.0])
    T = np.array([0.0, 4.0, 1.0, -2.0, 5.0])
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([
        np.dot(Q, T[0:3]),
        np.dot(Q, T[1:4]),
        np.dot(Q, T[2:5]),
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_returns_length_n_minus_m_plus_1():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    result = utils.sliding_dot_product(Q, T)
    assert result.shape == (len(T) - len(Q) + 1,)


def test_sliding_dot_product_raises_when_query_longer_than_series():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([4.0, 5.0])
    with pytest.raises(ValueError, match="T should be a series at least as long as Q"):
        utils.sliding_dot_product(Q, T)


def test_calculate_distance_profile_all_constant_sequences_return_zero():
    QT = np.array([0.0, 0.0, 0.0])
    mean_Q = 1.0
    sigma_Q = 0.0
    mean_T = np.array([2.0, 2.0, 2.0])
    sigma_T = np.array([0.0, 0.0, 0.0])

    result = utils.calculate_distance_profile(QT, m=3, mean_Q=mean_Q, sigma_Q=sigma_Q,
                                              mean_T=mean_T, sigma_T=sigma_T)
    np.testing.assert_allclose(result, np.zeros(3))


def test_calculate_distance_profile_constant_query_against_nonconstant_series_returns_sqrt_m():
    QT = np.array([1.0, 2.0, 3.0])
    mean_Q = 2.0
    sigma_Q = 0.0
    mean_T = np.array([1.0, 2.0, 3.0])
    sigma_T = np.array([1.0, 1.0, 1.0])

    result = utils.calculate_distance_profile(QT, m=4, mean_Q=mean_Q, sigma_Q=sigma_Q,
                                              mean_T=mean_T, sigma_T=sigma_T)
    np.testing.assert_allclose(result, np.full(3, np.sqrt(4.0)))


def test_calculate_distance_profile_nonconstant_query_against_constant_series_returns_sqrt_m():
    QT = np.array([1.0, 2.0])
    mean_Q = 1.5
    sigma_Q = 0.5
    mean_T = np.array([3.0, 4.0])
    sigma_T = np.array([0.0, 0.0])

    result = utils.calculate_distance_profile(QT, m=9, mean_Q=mean_Q, sigma_Q=sigma_Q,
                                              mean_T=mean_T, sigma_T=sigma_T)
    np.testing.assert_allclose(result, np.full(2, np.sqrt(9.0)))


def test_calculate_distance_profile_matches_manual_formula():
    QT = np.array([10.0, 13.0])
    m = 2
    mean_Q = 1.5
    sigma_Q = 0.5
    mean_T = np.array([3.0, 4.0])
    sigma_T = np.array([1.0, 2.0])

    result = utils.calculate_distance_profile(QT, m=m, mean_Q=mean_Q, sigma_Q=sigma_Q,
                                              mean_T=mean_T, sigma_T=sigma_T)

    expected = np.sqrt(np.maximum(
        2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)),
        0,
    ))
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_raises_on_dimension_mismatch():
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        utils.calculate_distance_profile(
            QT, m=2, mean_Q=0.0, sigma_Q=1.0, mean_T=mean_T, sigma_T=sigma_T
        )


def test_calculate_distance_profile_accepts_scalar_query_statistics():
    QT = np.array([5.0, 6.0, 7.0])
    mean_T = np.array([1.0, 1.5, 2.0])
    sigma_T = np.array([1.0, 1.0, 1.0])

    result = utils.calculate_distance_profile(
        QT, m=2, mean_Q=1.0, sigma_Q=2.0, mean_T=mean_T, sigma_T=sigma_T
    )

    expected = np.sqrt(np.maximum(
        2 * 2 * (1 - (QT - 2 * 1.0 * mean_T) / (2 * 2.0 * sigma_T)),
        0,
    ))
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
