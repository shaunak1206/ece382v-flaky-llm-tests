import numpy as np
import pytest

from pytsmp import utils


def test_rolling_avg_sd_basic_small_series():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = utils.rolling_avg_sd(t, 2)

    expected_mean = np.array([1.5, 2.5, 3.5])
    expected_sd = np.array([0.5, 0.5, 0.5])

    assert np.allclose(mean, expected_mean)
    assert np.allclose(sd, expected_sd)


def test_rolling_avg_sd_window_equals_length():
    t = np.array([2.0, 4.0, 6.0])
    mean, sd = utils.rolling_avg_sd(t, 3)

    assert mean.shape == (1,)
    assert sd.shape == (1,)
    assert mean[0] == pytest.approx(4.0)
    assert sd[0] == pytest.approx(np.std(t))


def test_rolling_avg_sd_raises_for_too_large_window():
    with pytest.raises(ValueError, match="Window size should be smaller"):
        utils.rolling_avg_sd(np.array([1.0, 2.0]), 3)


def test_rolling_avg_sd_handles_constant_series():
    t = np.array([5.0, 5.0, 5.0, 5.0])
    mean, sd = utils.rolling_avg_sd(t, 2)

    assert np.allclose(mean, np.array([5.0, 5.0, 5.0]))
    assert np.allclose(sd, np.zeros(3))


def test_sliding_dot_product_basic_real_values():
    Q = np.array([1.0, 2.0])
    T = np.array([0.0, 1.0, 2.0, 3.0])

    result = utils.sliding_dot_product(Q, T)
    expected = np.array([2.0, 5.0, 8.0])

    assert np.allclose(result, expected)


def test_sliding_dot_product_matches_manual_convolution_style_result():
    Q = np.array([3.0, -1.0, 2.0])
    T = np.array([4.0, 0.5, -2.0, 1.0, 3.0])

    result = utils.sliding_dot_product(Q, T)
    expected = np.array([
        3.0 * 4.0 + (-1.0) * 0.5 + 2.0 * (-2.0),
        3.0 * 0.5 + (-1.0) * (-2.0) + 2.0 * 1.0,
        3.0 * (-2.0) + (-1.0) * 1.0 + 2.0 * 3.0,
    ])

    assert np.allclose(result, expected)


def test_sliding_dot_product_works_when_time_series_length_is_odd():
    Q = np.array([1.0, 0.0])
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = utils.sliding_dot_product(Q, T)
    expected = np.array([1.0, 2.0, 3.0, 4.0])

    assert np.allclose(result, expected)


def test_sliding_dot_product_raises_for_too_long_query():
    with pytest.raises(ValueError, match="T should be a series at least as long as Q"):
        utils.sliding_dot_product(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_calculate_distance_profile_identical_nonconstant_series_is_zero():
    T = np.array([1.0, 2.0, 3.0, 4.0])
    Q = np.array([1.0, 2.0])
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)

    D = utils.calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D, np.zeros_like(D))


def test_calculate_distance_profile_constant_query_and_constant_subsequences_are_zero():
    Q = np.array([2.0, 2.0])
    T = np.array([2.0, 2.0, 2.0, 2.0])
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, len(Q))

    D = utils.calculate_distance_profile(
        QT, len(Q), np.mean(Q), np.std(Q), mean_T, sigma_T
    )

    assert np.allclose(D, np.zeros_like(D))


def test_calculate_distance_profile_constant_query_and_nonconstant_subsequences_are_sqrt_m():
    Q = np.array([7.0, 7.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, len(Q))

    D = utils.calculate_distance_profile(
        QT, len(Q), np.mean(Q), np.std(Q), mean_T, sigma_T
    )

    assert np.allclose(D, np.full_like(D, np.sqrt(len(Q)), dtype=float))


def test_calculate_distance_profile_nonconstant_query_and_constant_subsequences_are_sqrt_m():
    Q = np.array([1.0, 2.0])
    T = np.array([3.0, 3.0, 3.0, 3.0])
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, len(Q))

    D = utils.calculate_distance_profile(
        QT, len(Q), np.mean(Q), np.std(Q), mean_T, sigma_T
    )

    assert np.allclose(D, np.full_like(D, np.sqrt(len(Q)), dtype=float))


def test_calculate_distance_profile_dimension_mismatch_raises():
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        utils.calculate_distance_profile(QT, 2, 0.0, 1.0, mean_T, sigma_T)


def test_calculate_distance_profile_mismatched_mean_sigma_arrays_raises():
    QT = np.array([1.0, 2.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        utils.calculate_distance_profile(
            QT,
            2,
            np.array([0.0, 0.0]),
            np.array([1.0]),
            mean_T,
            sigma_T,
        )
