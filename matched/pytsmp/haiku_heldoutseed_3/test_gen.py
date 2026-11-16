import numpy as np
import pytest

from pytsmp.utils import sliding_dot_product, rolling_avg_sd, calculate_distance_profile


# ---------------------------------------------------------------------------
# Naive reference implementations
# ---------------------------------------------------------------------------

def _naive_sliding_dot_product(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)], dtype=float)


def _naive_rolling_avg_sd(t, w):
    windows = [np.asarray(t[i:i + w], dtype=float) for i in range(len(t) - w + 1)]
    means = np.array([x.mean() for x in windows])
    sds = np.array([x.std() for x in windows])  # population sd (ddof=0)
    return means, sds


def _naive_znorm_distance_profile(Q, T):
    m = len(Q)
    Q = np.asarray(Q, dtype=float)

    def znorm(x):
        return (x - x.mean()) / x.std()

    zq = znorm(Q)
    return np.array([np.linalg.norm(zq - znorm(np.asarray(T[i:i + m], dtype=float)))
                     for i in range(len(T) - m + 1)])


# ---------------------------------------------------------------------------
# sliding_dot_product
# ---------------------------------------------------------------------------

def test_sliding_dot_product_matches_naive_even_length():
    rng = np.random.RandomState(0)
    T = rng.randn(40)
    Q = rng.randn(7)
    np.testing.assert_allclose(sliding_dot_product(Q, T), _naive_sliding_dot_product(Q, T), atol=1e-9)


def test_sliding_dot_product_matches_naive_odd_length():
    rng = np.random.RandomState(1)
    T = rng.randn(37)
    Q = rng.randn(5)
    np.testing.assert_allclose(sliding_dot_product(Q, T), _naive_sliding_dot_product(Q, T), atol=1e-9)


def test_sliding_dot_product_output_length():
    rng = np.random.RandomState(2)
    T = rng.randn(50)
    Q = rng.randn(11)
    result = sliding_dot_product(Q, T)
    assert result.shape == (len(T) - len(Q) + 1,)


def test_sliding_dot_product_known_small_example():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(sliding_dot_product(Q, T), [5.0, 8.0, 11.0], atol=1e-10)


def test_sliding_dot_product_equal_length_returns_single_value():
    Q = np.array([1.0, -2.0, 3.0])
    T = np.array([4.0, 0.5, -1.0])
    result = sliding_dot_product(Q, T)
    assert result.shape == (1,)
    assert result[0] == pytest.approx(np.dot(Q, T))


def test_sliding_dot_product_raises_when_query_longer_than_series():
    with pytest.raises(ValueError):
        sliding_dot_product(np.arange(5, dtype=float), np.arange(3, dtype=float))


def test_sliding_dot_product_is_real_valued():
    rng = np.random.RandomState(3)
    result = sliding_dot_product(rng.randn(4), rng.randn(20))
    assert np.isrealobj(result)


# ---------------------------------------------------------------------------
# rolling_avg_sd
# ---------------------------------------------------------------------------

def test_rolling_avg_sd_matches_naive():
    rng = np.random.RandomState(4)
    t = rng.randn(60) * 3 + 2
    w = 9
    means, sds = rolling_avg_sd(t, w)
    exp_means, exp_sds = _naive_rolling_avg_sd(t, w)
    np.testing.assert_allclose(means, exp_means, atol=1e-9)
    np.testing.assert_allclose(sds, exp_sds, atol=1e-7)


def test_rolling_avg_sd_output_shapes():
    t = np.arange(30, dtype=float)
    w = 6
    means, sds = rolling_avg_sd(t, w)
    assert means.shape == (len(t) - w + 1,)
    assert sds.shape == (len(t) - w + 1,)


def test_rolling_avg_sd_window_equals_length():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    means, sds = rolling_avg_sd(t, len(t))
    assert means.shape == (1,)
    assert means[0] == pytest.approx(2.5)
    assert sds[0] == pytest.approx(np.std(t))


def test_rolling_avg_sd_window_one():
    t = np.array([3.0, -1.0, 7.0, 0.5])
    means, sds = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(means, t, atol=1e-12)
    np.testing.assert_allclose(sds, np.zeros_like(t), atol=1e-12)


def test_rolling_avg_sd_known_example():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    means, sds = rolling_avg_sd(t, 2)
    np.testing.assert_allclose(means, [1.5, 2.5, 3.5], atol=1e-12)
    np.testing.assert_allclose(sds, [0.5, 0.5, 0.5], atol=1e-12)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(12, 3.0)
    means, sds = rolling_avg_sd(t, 4)
    np.testing.assert_allclose(means, np.full(9, 3.0), atol=1e-12)
    np.testing.assert_allclose(sds, np.zeros(9), atol=1e-12)


def test_rolling_avg_sd_raises_when_window_larger_than_series():
    with pytest.raises(ValueError):
        rolling_avg_sd(np.arange(4, dtype=float), 5)


def test_rolling_avg_sd_sd_is_nonnegative():
    rng = np.random.RandomState(5)
    _, sds = rolling_avg_sd(rng.randn(100), 10)
    assert np.all(sds >= 0)


# ---------------------------------------------------------------------------
# calculate_distance_profile
# ---------------------------------------------------------------------------

def test_distance_profile_raises_on_qt_mean_length_mismatch():
    QT = np.zeros(3)
    mean_T = np.zeros(4)
    sigma_T = np.ones(4)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 2, 0.0, 1.0, mean_T, sigma_T)


def test_distance_profile_raises_on_mean_sigma_T_length_mismatch():
    QT = np.zeros(3)
    mean_T = np.zeros(3)
    sigma_T = np.ones(2)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 2, 0.0, 1.0, mean_T, sigma_T)


def test_distance_profile_raises_on_array_query_stats_length_mismatch():
    QT = np.zeros(2)
    mean_T = np.zeros(2)
    sigma_T = np.ones(2)
    mean_Q = np.zeros(3)   # length 3 does not match mean_T length 2
    sigma_Q = np.ones(3)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 3, mean_Q, sigma_Q, mean_T, sigma_T)


def test_distance_profile_self_match_is_zero():
    rng = np.random.RandomState(6)
    T = rng.randn(50)
    m = 8
    i = 17
    Q = T[i:i + m]
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, Q.mean(), Q.std(), mean_T, sigma_T)
    assert D[i] == pytest.approx(0.0, abs=1e-6)
    assert np.all(D >= 0)


def test_distance_profile_matches_znormalized_euclidean():
    rng = np.random.RandomState(7)
    T = rng.randn(60)
    Q = rng.randn(10)
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, Q.mean(), Q.std(), mean_T, sigma_T)
    np.testing.assert_allclose(D, _naive_znorm_distance_profile(Q, T), atol=1e-6)


def test_distance_profile_constant_query_gives_sqrt_m():
    m = 4
    QT = np.array([1.0, -3.0, 2.5])
    mean_T = np.array([0.0, 1.0, -2.0])
    sigma_T = np.array([1.0, 2.0, 0.5])
    D = calculate_distance_profile(QT, m, 5.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.full(3, np.sqrt(m)), atol=1e-12)


def test_distance_profile_constant_subsequences_mixed():
    # m=4, zero means, unit sds, zero dot products -> normal entry is sqrt(8)
    m = 4
    QT = np.zeros(3)
    mean_T = np.zeros(3)
    sigma_T = np.array([0.0, 1.0, 0.0])
    D = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, [np.sqrt(m), np.sqrt(2 * m), np.sqrt(m)], atol=1e-12)


def test_distance_profile_both_constant_is_zero():
    m = 4
    QT = np.zeros(2)
    mean_T = np.zeros(2)
    sigma_T = np.array([0.0, 1e-12])  # both below default epsilon
    D = calculate_distance_profile(QT, m, 1.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.zeros(2), atol=1e-12)


def test_distance_profile_known_value():
    m = 2
    QT = np.array([0.0])
    D = calculate_distance_profile(QT, m, 0.0, 1.0, np.array([0.0]), np.array([1.0]))
    assert D[0] == pytest.approx(2.0)  # sqrt(2 * 2 * (1 - 0))


def test_distance_profile_clips_negative_radicand_to_zero():
    m = 4
    QT = np.array([8.0])  # 1 - 8 / 4 = -1 -> clipped to 0
    D = calculate_distance_profile(QT, m, 0.0, 1.0, np.array([0.0]), np.array([1.0]))
    np.testing.assert_allclose(D, [0.0], atol=1e-12)


def test_distance_profile_custom_epsilon_treats_small_sd_as_constant():
    m = 5
    QT = np.array([1.0, 2.0])
    mean_T = np.array([0.0, 0.0])
    sigma_T = np.array([1.0, 1.0])
    # sigma_Q = 1e-3 is below epsilon=1e-2, so the query is treated as constant
    D = calculate_distance_profile(QT, m, 0.0, 1e-3, mean_T, sigma_T, epsilon=1e-2)
    np.testing.assert_allclose(D, np.full(2, np.sqrt(m)), atol=1e-12)
