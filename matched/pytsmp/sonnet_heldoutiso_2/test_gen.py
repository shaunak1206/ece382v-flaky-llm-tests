import numpy as np
import pytest

from pytsmp.utils import (
    sliding_dot_product,
    rolling_avg_sd,
    calculate_distance_profile,
)


def naive_sliding_dot_product(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def naive_distance_profile(Q, T):
    m = len(Q)
    zq = (Q - Q.mean()) / Q.std()
    out = []
    for i in range(len(T) - m + 1):
        sub = T[i:i + m]
        zs = (sub - sub.mean()) / sub.std()
        out.append(np.linalg.norm(zq - zs))
    return np.array(out)


def full_profile(Q, T):
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    return calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)


# ---------------------- sliding_dot_product ----------------------

@pytest.mark.parametrize("n,m", [(10, 3), (11, 3), (16, 5), (17, 4), (9, 9), (10, 10)])
def test_sliding_dot_product_matches_naive(n, m):
    rng = np.random.RandomState(n * 100 + m)
    T = rng.randn(n)
    Q = rng.randn(m)
    result = sliding_dot_product(Q, T)
    assert result.shape == (n - m + 1,)
    np.testing.assert_allclose(result, naive_sliding_dot_product(Q, T), atol=1e-8)


def test_sliding_dot_product_known_values():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, [5.0, 8.0, 11.0], atol=1e-9)


def test_sliding_dot_product_query_length_one():
    T = np.array([1.0, -2.0, 3.0, 4.0, 5.0])
    Q = np.array([2.0])
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, 2.0 * T, atol=1e-9)


def test_sliding_dot_product_query_longer_raises():
    with pytest.raises(ValueError):
        sliding_dot_product(np.arange(6, dtype=float), np.arange(4, dtype=float))


def test_sliding_dot_product_integer_input():
    Q = np.array([1, 0, -1])
    T = np.array([3, 5, 7, 2, 4, 6])
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, naive_sliding_dot_product(Q, T), atol=1e-8)


# ---------------------- rolling_avg_sd ----------------------

def test_rolling_avg_sd_matches_numpy():
    rng = np.random.RandomState(0)
    t = rng.randn(30)
    w = 7
    mean, sd = rolling_avg_sd(t, w)
    exp_mean = np.array([t[i:i + w].mean() for i in range(len(t) - w + 1)])
    exp_sd = np.array([t[i:i + w].std() for i in range(len(t) - w + 1)])
    assert mean.shape == (len(t) - w + 1,)
    assert sd.shape == (len(t) - w + 1,)
    np.testing.assert_allclose(mean, exp_mean, atol=1e-9)
    np.testing.assert_allclose(sd, exp_sd, atol=1e-7)


def test_rolling_avg_sd_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = rolling_avg_sd(t, 2)
    np.testing.assert_allclose(mean, [1.5, 2.5, 3.5])
    np.testing.assert_allclose(sd, [0.5, 0.5, 0.5], atol=1e-9)


def test_rolling_avg_sd_window_equals_length():
    t = np.array([2.0, 4.0, 6.0, 8.0])
    mean, sd = rolling_avg_sd(t, 4)
    assert len(mean) == 1 and len(sd) == 1
    np.testing.assert_allclose(mean, [5.0])
    np.testing.assert_allclose(sd, [np.std(t)], atol=1e-9)


def test_rolling_avg_sd_window_one():
    t = np.array([3.0, -1.0, 4.0, 1.0])
    mean, sd = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(mean, t)
    np.testing.assert_allclose(sd, np.zeros(4), atol=1e-7)


def test_rolling_avg_sd_constant_series_sd_zero_and_nonnegative():
    t = np.full(12, 5.0)
    mean, sd = rolling_avg_sd(t, 4)
    np.testing.assert_allclose(mean, np.full(9, 5.0))
    assert np.all(sd >= 0)
    np.testing.assert_allclose(sd, np.zeros(9), atol=1e-6)


def test_rolling_avg_sd_window_too_large_raises():
    with pytest.raises(ValueError):
        rolling_avg_sd(np.arange(3, dtype=float), 4)


# ---------------------- calculate_distance_profile ----------------------

def test_distance_profile_matches_naive_znorm_distance():
    rng = np.random.RandomState(42)
    T = rng.randn(40)
    Q = rng.randn(6)
    D = full_profile(Q, T)
    assert D.shape == (35,)
    np.testing.assert_allclose(D, naive_distance_profile(Q, T), atol=1e-5)


def test_distance_profile_exact_match_is_zero():
    rng = np.random.RandomState(7)
    T = rng.randn(25)
    Q = T[5:11].copy()
    D = full_profile(Q, T)
    assert D[5] == pytest.approx(0, abs=1e-5)
    assert np.all(D >= 0)


def test_distance_profile_scale_and_shift_invariant():
    rng = np.random.RandomState(3)
    T = rng.randn(20)
    Q = T[2:7].copy()
    D1 = full_profile(Q, T)
    D2 = full_profile(3.0 * Q + 10.0, T)
    np.testing.assert_allclose(D1, D2, atol=1e-5)


def test_distance_profile_constant_subsequence_gives_sqrt_m():
    T = np.array([1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 2.0, 1.0, 4.0])
    Q = np.array([1.0, 2.0, 4.0, 3.0])
    m = len(Q)
    D = full_profile(Q, T)
    assert D[3] == pytest.approx(np.sqrt(m))


def test_distance_profile_constant_query():
    T = np.array([1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 2.0, 1.0, 4.0])
    m = 4
    Q = np.full(m, 7.0)
    D = full_profile(Q, T)
    expected = np.full(len(T) - m + 1, np.sqrt(m))
    expected[3] = 0  # constant vs constant
    np.testing.assert_allclose(D, expected, atol=1e-9)


def test_distance_profile_dimension_mismatch_raises():
    QT = np.zeros(5)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 3, 0.0, 1.0, np.zeros(4), np.ones(4))
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 3, 0.0, 1.0, np.zeros(5), np.ones(4))


def test_distance_profile_array_query_stats_mismatch_raises():
    QT = np.zeros(5)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 3, np.zeros(4), np.ones(4), np.zeros(5), np.ones(5))
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 3, np.zeros(5), np.ones(4), np.zeros(5), np.ones(5))
