import numpy as np
import pytest

from pytsmp import utils


def brute_sliding_dot_product(q, t):
    m = len(q)
    return np.array([np.dot(q, t[i : i + m]) for i in range(len(t) - m + 1)])


def test_sliding_dot_product_basic_matches_manual():
    q = np.array([1.0, 2.0, 3.0])
    t = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

    result = utils.sliding_dot_product(q, t)
    expected = brute_sliding_dot_product(q, t)

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert result.shape == (len(t) - len(q) + 1,)


def test_sliding_dot_product_with_negative_values():
    q = np.array([2.0, -1.0])
    t = np.array([3.0, 0.0, -2.0, 5.0])

    result = utils.sliding_dot_product(q, t)
    expected = brute_sliding_dot_product(q, t)

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_odd_length_series():
    q = np.array([1.0, 0.0, -1.0])
    t = np.array([2.0, 1.0, 0.0, -1.0, -2.0, 3.0, 4.0])

    result = utils.sliding_dot_product(q, t)
    expected = brute_sliding_dot_product(q, t)

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_raises_when_query_longer_than_time_series():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    t = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="at least as long as Q"):
        utils.sliding_dot_product(q, t)


def test_rolling_avg_sd_basic_matches_manual():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window = 3

    mean, sd = utils.rolling_avg_sd(t, window)

    expected_mean = np.array([2.0, 3.0, 4.0])
    expected_sd = np.array([np.std(t[i : i + window]) for i in range(len(t) - window + 1)])

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sd, expected_sd, rtol=1e-12, atol=1e-12)


def test_rolling_avg_sd_window_size_one():
    t = np.array([10.0, -2.5, 7.0])
    window = 1

    mean, sd = utils.rolling_avg_sd(t, window)

    np.testing.assert_allclose(mean, t)
    np.testing.assert_allclose(sd, np.zeros_like(t))


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.array([5.0, 5.0, 5.0, 5.0])
    window = 2

    mean, sd = utils.rolling_avg_sd(t, window)

    np.testing.assert_allclose(mean, np.array([5.0, 5.0, 5.0]))
    np.testing.assert_allclose(sd, np.zeros(3))


def test_rolling_avg_sd_raises_when_window_too_large():
    t = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="smaller than the length of time series"):
        utils.rolling_avg_sd(t, 3)


def test_calculate_distance_profile_identical_non_constant_series_near_zero():
    q = np.array([1.0, 2.0, 3.0])
    t = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])

    QT = utils.sliding_dot_product(q, t)
    mean_t, sigma_t = utils.rolling_avg_sd(t, len(q))
    mean_q = np.mean(q)
    sigma_q = np.std(q)

    d = utils.calculate_distance_profile(QT, len(q), mean_q, sigma_q, mean_t, sigma_t)

    assert d.shape == (len(t) - len(q) + 1,)
    assert d[0] < 1e-10


def test_calculate_distance_profile_constant_query_and_constant_time_series_returns_zero():
    q = np.array([3.0, 3.0, 3.0])
    t = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

    QT = utils.sliding_dot_product(q, t)
    mean_t, sigma_t = utils.rolling_avg_sd(t, len(q))
    mean_q = np.mean(q)
    sigma_q = np.std(q)

    d = utils.calculate_distance_profile(QT, len(q), mean_q, sigma_q, mean_t, sigma_t)

    np.testing.assert_allclose(d, np.zeros(len(t) - len(q) + 1))


def test_calculate_distance_profile_constant_query_against_nonconstant_series_returns_sqrt_m():
    q = np.array([2.0, 2.0, 2.0])
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    QT = utils.sliding_dot_product(q, t)
    mean_t, sigma_t = utils.rolling_avg_sd(t, len(q))
    mean_q = np.mean(q)
    sigma_q = np.std(q)

    d = utils.calculate_distance_profile(QT, len(q), mean_q, sigma_q, mean_t, sigma_t)

    np.testing.assert_allclose(d, np.full(len(t) - len(q) + 1, np.sqrt(len(q))))


def test_calculate_distance_profile_nonconstant_query_against_constant_subsequence_returns_sqrt_m():
    q = np.array([1.0, 2.0, 3.0])
    t = np.array([5.0, 5.0, 5.0, 1.0, 2.0, 3.0])

    QT = utils.sliding_dot_product(q, t)
    mean_t, sigma_t = utils.rolling_avg_sd(t, len(q))
    mean_q = np.mean(q)
    sigma_q = np.std(q)

    d = utils.calculate_distance_profile(QT, len(q), mean_q, sigma_q, mean_t, sigma_t)

    np.testing.assert_allclose(d[0], np.sqrt(len(q)))


def test_calculate_distance_profile_raises_on_dimension_mismatch():
    QT = np.array([1.0, 2.0, 3.0])
    mean_t = np.array([1.0, 2.0])
    sigma_t = np.array([1.0, 2.0])
    mean_q = 1.0
    sigma_q = 1.0

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        utils.calculate_distance_profile(QT, 2, mean_q, sigma_q, mean_t, sigma_t)


def test_calculate_distance_profile_raises_when_mean_q_and_sigma_q_array_lengths_disagree():
    QT = np.array([1.0, 2.0, 3.0])
    mean_t = np.array([1.0, 2.0, 3.0])
    sigma_t = np.array([1.0, 2.0, 3.0])
    mean_q = np.array([1.0, 2.0, 3.0])
    sigma_q = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        utils.calculate_distance_profile(QT, 3, mean_q, sigma_q, mean_t, sigma_t)
