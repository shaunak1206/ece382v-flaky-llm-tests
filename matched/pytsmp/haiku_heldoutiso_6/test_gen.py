import numpy as np
import pytest

from pytsmp.utils import sliding_dot_product, rolling_avg_sd, calculate_distance_profile


# ---------------------------------------------------------------------------
# Helpers (brute-force reference implementations)
# ---------------------------------------------------------------------------

def _naive_sliding_dot_product(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _naive_rolling_avg_sd(t, w):
    windows = [t[i:i + w] for i in range(len(t) - w + 1)]
    means = np.array([np.mean(x) for x in windows])
    sds = np.array([np.std(x) for x in windows])  # population sd (ddof=0)
    return means, sds


def _znorm(x):
    return (x - np.mean(x)) / np.std(x)


def _naive_znorm_distance_profile(Q, T):
    m = len(Q)
    zq = _znorm(Q)
    return np.array([np.linalg.norm(zq - _znorm(T[i:i + m])) for i in range(len(T) - m + 1)])


# ---------------------------------------------------------------------------
# sliding_dot_product
# ---------------------------------------------------------------------------

def test_sdp_matches_naive_even_length_series():
    rng = np.random.RandomState(0)
    T = rng.randn(20)  # even length
    Q = rng.randn(5)
    np.testing.assert_allclose(sliding_dot_product(Q, T), _naive_sliding_dot_product(Q, T), atol=1e-8)


def test_sdp_matches_naive_odd_length_series():
    rng = np.random.RandomState(1)
    T = rng.randn(21)  # odd length exercises the padding branch
    Q = rng.randn(6)
    np.testing.assert_allclose(sliding_dot_product(Q, T), _naive_sliding_dot_product(Q, T), atol=1e-8)


def test_sdp_matches_naive_for_odd_query_length():
    rng = np.random.RandomState(2)
    T = rng.randn(30)
    Q = rng.randn(7)
    np.testing.assert_allclose(sliding_dot_product(Q, T), _naive_sliding_dot_product(Q, T), atol=1e-8)


def test_sdp_known_small_example():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(sliding_dot_product(Q, T), np.array([5.0, 8.0, 11.0]), atol=1e-10)


def test_sdp_single_element_query():
    Q = np.array([2.0])
    T = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(sliding_dot_product(Q, T), np.array([2.0, 4.0, 6.0]), atol=1e-10)


def test_sdp_output_length_is_n_minus_m_plus_one():
    rng = np.random.RandomState(3)
    T = rng.randn(17)
    Q = rng.randn(4)
    assert sliding_dot_product(Q, T).shape == (len(T) - len(Q) + 1,)


def test_sdp_equal_length_query_returns_single_value():
    rng = np.random.RandomState(4)
    T = rng.randn(8)
    Q = T.copy()
    result = sliding_dot_product(Q, T)
    assert result.shape == (1,)
    np.testing.assert_allclose(result[0], np.dot(Q, T), atol=1e-8)


def test_sdp_query_longer_than_series_raises():
    with pytest.raises(ValueError):
        sliding_dot_product(np.arange(6.0), np.arange(4.0))


# ---------------------------------------------------------------------------
# rolling_avg_sd
# ---------------------------------------------------------------------------

def test_ravg_matches_naive_mean_and_std():
    rng = np.random.RandomState(10)
    t = rng.randn(40)
    w = 7
    mean, sd = rolling_avg_sd(t, w)
    exp_mean, exp_sd = _naive_rolling_avg_sd(t, w)
    np.testing.assert_allclose(mean, exp_mean, atol=1e-8)
    np.testing.assert_allclose(sd, exp_sd, atol=1e-6)


def test_ravg_returns_tuple_of_arrays_with_expected_shape():
    t = np.arange(12.0)
    w = 5
    result = rolling_avg_sd(t, w)
    assert isinstance(result, tuple)
    assert len(result) == 2
    mean, sd = result
    assert isinstance(mean, np.ndarray) and isinstance(sd, np.ndarray)
    assert mean.shape == (len(t) - w + 1,)
    assert sd.shape == (len(t) - w + 1,)


def test_ravg_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = rolling_avg_sd(t, 2)
    np.testing.assert_allclose(mean, [1.5, 2.5, 3.5], atol=1e-10)
    np.testing.assert_allclose(sd, [0.5, 0.5, 0.5], atol=1e-10)


def test_ravg_window_equal_to_length_gives_single_window():
    t = np.array([2.0, 4.0, 6.0, 8.0])
    mean, sd = rolling_avg_sd(t, len(t))
    assert mean.shape == (1,) and sd.shape == (1,)
    np.testing.assert_allclose(mean[0], np.mean(t), atol=1e-10)
    np.testing.assert_allclose(sd[0], np.std(t), atol=1e-8)


def test_ravg_window_of_one_returns_series_and_zero_sd():
    t = np.array([3.0, -1.0, 2.5, 7.0])
    mean, sd = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(mean, t, atol=1e-10)
    np.testing.assert_allclose(sd, np.zeros_like(t), atol=1e-8)


def test_ravg_window_larger_than_series_raises():
    with pytest.raises(ValueError):
        rolling_avg_sd(np.arange(3.0), 4)


def test_ravg_constant_series_has_zero_sd():
    t = np.full(10, 3.0)
    mean, sd = rolling_avg_sd(t, 4)
    np.testing.assert_allclose(mean, np.full(7, 3.0), atol=1e-10)
    np.testing.assert_allclose(sd, np.zeros(7), atol=1e-8)


def test_ravg_sd_is_finite_and_non_negative():
    rng = np.random.RandomState(11)
    t = rng.randn(100)
    _, sd = rolling_avg_sd(t, 10)
    assert np.all(np.isfinite(sd))
    assert np.all(sd >= 0)


# ---------------------------------------------------------------------------
# calculate_distance_profile
# ---------------------------------------------------------------------------

def test_cdp_matches_znormalized_euclidean_distance():
    rng = np.random.RandomState(20)
    T = rng.randn(40)
    Q = rng.randn(8)
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    np.testing.assert_allclose(D, _naive_znorm_distance_profile(Q, T), atol=1e-6)


def test_cdp_output_shape_matches_qt():
    rng = np.random.RandomState(21)
    T = rng.randn(25)
    Q = rng.randn(5)
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert isinstance(D, np.ndarray)
    assert D.shape == (len(T) - m + 1,)


def test_cdp_exact_self_match_has_zero_distance():
    rng = np.random.RandomState(22)
    T = rng.randn(30)
    start = 5
    Q = T[start:start + 10].copy()
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[start] == pytest.approx(0.0, abs=1e-5)


def test_cdp_constant_query_and_constant_subsequence_gives_zero():
    T = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    Q = np.array([2.0, 2.0, 2.0])
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    # Windows 0 and 1 are constant ([3,3,3]) -> distance 0 to a constant query
    assert D[0] == pytest.approx(0.0, abs=1e-10)
    assert D[1] == pytest.approx(0.0, abs=1e-10)


def test_cdp_constant_query_nonconstant_subsequence_gives_sqrt_m():
    T = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    Q = np.array([2.0, 2.0, 2.0])
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    # Window 2 is [3,3,1], non-constant -> sqrt(m)
    assert D[2] == pytest.approx(np.sqrt(m), abs=1e-10)


def test_cdp_nonconstant_query_constant_subsequence_gives_sqrt_m():
    T = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    Q = np.array([1.0, 4.0, 2.0])
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[0] == pytest.approx(np.sqrt(m), abs=1e-10)
    assert D[1] == pytest.approx(np.sqrt(m), abs=1e-10)


def test_cdp_custom_epsilon_treats_small_sigma_as_constant():
    QT = np.array([1.0, 2.0])
    mean_T = np.array([0.0, 0.0])
    sigma_T = np.array([1.0, 1.0])
    m = 4
    # sigma_Q = 1e-3 is below epsilon=1e-2 -> constant branch -> sqrt(m)
    D = calculate_distance_profile(QT, m, 0.0, 1e-3, mean_T, sigma_T, epsilon=1e-2)
    np.testing.assert_allclose(D, np.full(2, np.sqrt(m)), atol=1e-10)


def test_cdp_scalar_mean_and_sigma_for_query_accepted():
    rng = np.random.RandomState(23)
    T = rng.randn(20)
    Q = rng.randn(4)
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, float(np.mean(Q)), float(np.std(Q)), mean_T, sigma_T)
    np.testing.assert_allclose(D, _naive_znorm_distance_profile(Q, T), atol=1e-6)


def test_cdp_output_is_finite_and_non_negative():
    rng = np.random.RandomState(24)
    T = rng.randn(60)
    Q = rng.randn(10)
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert np.all(np.isfinite(D))
    assert np.all(D >= 0)


def test_cdp_mismatched_qt_and_mean_T_lengths_raises():
    QT = np.zeros(3)
    mean_T = np.zeros(4)
    sigma_T = np.ones(4)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 2, 0.0, 1.0, mean_T, sigma_T)


def test_cdp_mismatched_mean_T_and_sigma_T_lengths_raises():
    QT = np.zeros(3)
    mean_T = np.zeros(3)
    sigma_T = np.ones(2)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 2, 0.0, 1.0, mean_T, sigma_T)


def test_cdp_mismatched_array_query_stats_raises():
    QT = np.zeros(3)
    mean_T = np.zeros(3)
    sigma_T = np.ones(3)
    mean_Q = np.zeros(2)
    sigma_Q = np.ones(3)  # length differs from mean_Q
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)


def test_cdp_array_query_stats_length_not_matching_mean_T_raises():
    QT = np.zeros(3)
    mean_T = np.zeros(3)
    sigma_T = np.ones(3)
    mean_Q = np.zeros(2)
    sigma_Q = np.ones(2)  # consistent with each other but not with mean_T
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)
