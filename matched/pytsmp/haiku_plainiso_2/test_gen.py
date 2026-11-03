import numpy as np
import pytest

from pytsmp.utils import sliding_dot_product, rolling_avg_sd, calculate_distance_profile


def _naive_sdp(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _znorm_distance(a, b):
    za = (a - a.mean()) / a.std()
    zb = (b - b.mean()) / b.std()
    return np.sqrt(np.sum((za - zb) ** 2))


def _distance_profile(Q, T):
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    return calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)


# ---------------------------------------------------------------- sliding_dot_product

def test_sdp_matches_naive_even_length():
    rng = np.random.RandomState(0)
    T = rng.randn(16)
    Q = rng.randn(5)
    np.testing.assert_allclose(sliding_dot_product(Q, T), _naive_sdp(Q, T), atol=1e-9)


def test_sdp_matches_naive_odd_length():
    rng = np.random.RandomState(7)
    T = rng.randn(17)
    Q = rng.randn(4)
    np.testing.assert_allclose(sliding_dot_product(Q, T), _naive_sdp(Q, T), atol=1e-9)


def test_sdp_output_length_is_n_minus_m_plus_1():
    rng = np.random.RandomState(1)
    for n in (5, 8, 13):
        T = rng.randn(n)
        for m in range(1, n + 1):
            Q = rng.randn(m)
            assert len(sliding_dot_product(Q, T)) == n - m + 1


def test_sdp_known_small_example():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(sliding_dot_product(Q, T), [5.0, 8.0, 11.0], atol=1e-9)


def test_sdp_query_same_length_as_series_gives_single_dot():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(sliding_dot_product(Q, T), [14.0], atol=1e-9)


def test_sdp_zero_query_returns_zeros():
    T = np.random.RandomState(2).randn(10)
    np.testing.assert_allclose(sliding_dot_product(np.zeros(3), T), np.zeros(8), atol=1e-9)


def test_sdp_raises_when_query_longer_than_series():
    with pytest.raises(ValueError):
        sliding_dot_product(np.ones(5), np.ones(3))


# ---------------------------------------------------------------- rolling_avg_sd

def test_rolling_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, sd = rolling_avg_sd(t, 2)
    np.testing.assert_allclose(mean, [1.5, 2.5, 3.5, 4.5], atol=1e-12)
    np.testing.assert_allclose(sd, [0.5, 0.5, 0.5, 0.5], atol=1e-12)


def test_rolling_matches_numpy_per_window():
    t = np.random.RandomState(3).randn(23)
    w = 6
    mean, sd = rolling_avg_sd(t, w)
    expected_mean = np.array([np.mean(t[i:i + w]) for i in range(len(t) - w + 1)])
    expected_sd = np.array([np.std(t[i:i + w]) for i in range(len(t) - w + 1)])
    np.testing.assert_allclose(mean, expected_mean, atol=1e-9)
    np.testing.assert_allclose(sd, expected_sd, atol=1e-9)


def test_rolling_returns_pair_of_correct_length():
    t = np.random.RandomState(9).randn(12)
    for w in (1, 3, 12):
        result = rolling_avg_sd(t, w)
        assert len(result) == 2
        mean, sd = result
        assert len(mean) == len(sd) == len(t) - w + 1


def test_rolling_constant_series_has_zero_sd():
    t = np.full(10, 3.0)
    mean, sd = rolling_avg_sd(t, 4)
    np.testing.assert_allclose(mean, np.full(7, 3.0), atol=1e-12)
    np.testing.assert_allclose(sd, np.zeros(7), atol=1e-12)


def test_rolling_window_equal_to_length():
    t = np.array([2.0, 4.0, 6.0, 8.0])
    mean, sd = rolling_avg_sd(t, 4)
    np.testing.assert_allclose(mean, [5.0], atol=1e-12)
    np.testing.assert_allclose(sd, [np.sqrt(5.0)], atol=1e-9)


def test_rolling_window_of_one():
    t = np.random.RandomState(4).randn(7)
    mean, sd = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(mean, t, atol=1e-12)
    np.testing.assert_allclose(sd, np.zeros(7), atol=1e-6)


def test_rolling_sd_is_never_negative_with_large_offset():
    t = np.random.RandomState(5).randn(200) + 50.0
    mean, sd = rolling_avg_sd(t, 10)
    assert np.all(np.isfinite(sd))
    assert np.all(sd >= 0)


def test_rolling_raises_when_window_too_large():
    with pytest.raises(ValueError):
        rolling_avg_sd(np.ones(3), 4)


# ---------------------------------------------------------- calculate_distance_profile

def test_distance_profile_matches_bruteforce_znorm_distance():
    rng = np.random.RandomState(11)
    T = rng.randn(37)
    Q = rng.randn(7)
    D = _distance_profile(Q, T)
    expected = np.array([_znorm_distance(Q, T[i:i + 7]) for i in range(len(T) - 7 + 1)])
    assert D.shape == (len(T) - len(Q) + 1,)
    np.testing.assert_allclose(D, expected, atol=1e-5)


def test_distance_profile_zero_at_exact_match():
    rng = np.random.RandomState(12)
    T = rng.randn(50)
    m = 8
    Q = T[5:5 + m].copy()
    D = _distance_profile(Q, T)
    assert abs(D[5]) < 1e-5
    assert np.argmin(D) == 5


def test_distance_profile_bounded_by_two_sqrt_m():
    rng = np.random.RandomState(13)
    T = rng.randn(60)
    m = 9
    Q = rng.randn(m)
    D = _distance_profile(Q, T)
    assert np.all(D >= 0)
    assert np.all(D <= 2 * np.sqrt(m) + 1e-5)


def test_distance_profile_constant_query_gives_sqrt_m():
    m = 9
    QT = np.zeros(3)
    mean_T = np.zeros(3)
    sigma_T = np.array([1.0, 2.0, 0.5])
    D = calculate_distance_profile(QT, m, 1.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.full(3, 3.0), atol=1e-12)


def test_distance_profile_constant_window_gives_sqrt_m():
    m = 9
    QT = np.zeros(3)
    mean_T = np.zeros(3)
    sigma_T = np.array([1.0, 0.0, 2.0])
    D = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    expected = np.array([np.sqrt(18.0), 3.0, np.sqrt(18.0)])
    np.testing.assert_allclose(D, expected, atol=1e-12)


def test_distance_profile_constant_pair_is_zero():
    m = 5
    QT = np.zeros(2)
    mean_T = np.array([5.0, 5.0])
    sigma_T = np.array([0.0, 1e-9])
    D = calculate_distance_profile(QT, m, 5.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.zeros(2), atol=1e-12)


def test_distance_profile_epsilon_controls_constant_threshold():
    m = 4
    QT = np.array([0.0])
    mean_T = np.array([0.0])
    sigma_T = np.array([1.0])
    # sigma_Q = 1e-3 is above the default epsilon, so the normal formula applies.
    D_default = calculate_distance_profile(QT, m, 0.0, 1e-3, mean_T, sigma_T)
    np.testing.assert_allclose(D_default, [np.sqrt(8.0)], atol=1e-9)
    # With epsilon = 1e-2 the same query counts as constant, giving sqrt(m).
    D_loose = calculate_distance_profile(QT, m, 0.0, 1e-3, mean_T, sigma_T, epsilon=1e-2)
    np.testing.assert_allclose(D_loose, [2.0], atol=1e-12)


def test_distance_profile_raises_on_series_length_mismatch():
    with pytest.raises(ValueError):
        calculate_distance_profile(np.zeros(5), 3, 0.0, 1.0, np.zeros(4), np.ones(4))
    with pytest.raises(ValueError):
        calculate_distance_profile(np.zeros(5), 3, 0.0, 1.0, np.zeros(5), np.ones(4))


def test_distance_profile_raises_on_query_array_mismatch():
    # mean_Q and sigma_Q arrays disagree in length
    with pytest.raises(ValueError):
        calculate_distance_profile(np.zeros(3), 3, np.zeros(3), np.ones(2), np.zeros(3), np.ones(3))
    # mean_Q array does not match the length of mean_T
    with pytest.raises(ValueError):
        calculate_distance_profile(np.zeros(3), 3, np.zeros(4), np.ones(4), np.zeros(3), np.ones(3))
