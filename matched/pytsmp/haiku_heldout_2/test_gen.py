import numpy as np
import pytest

from pytsmp.utils import (
    sliding_dot_product,
    rolling_avg_sd,
    calculate_distance_profile,
)


# ---------------------------------------------------------------------------
# Helpers (naive reference implementations)
# ---------------------------------------------------------------------------

def naive_sliding_dot_product(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def naive_rolling_mean_std(t, w):
    means = []
    stds = []
    for i in range(len(t) - w + 1):
        window = t[i:i + w]
        means.append(np.mean(window))
        stds.append(np.std(window))  # population std (ddof=0)
    return np.array(means), np.array(stds)


def naive_z_distance_profile(Q, T):
    m = len(Q)
    zq = (Q - np.mean(Q)) / np.std(Q)
    out = []
    for i in range(len(T) - m + 1):
        w = T[i:i + m]
        zw = (w - np.mean(w)) / np.std(w)
        out.append(np.sqrt(np.sum((zq - zw) ** 2)))
    return np.array(out)


@pytest.fixture
def rng():
    return np.random.RandomState(12345)


# ---------------------------------------------------------------------------
# sliding_dot_product
# ---------------------------------------------------------------------------

def test_sliding_dot_product_matches_naive_even_length(rng):
    T = rng.randn(40)
    Q = rng.randn(7)
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, naive_sliding_dot_product(Q, T), rtol=1e-9, atol=1e-9)


def test_sliding_dot_product_matches_naive_odd_length(rng):
    T = rng.randn(41)
    Q = rng.randn(6)
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, naive_sliding_dot_product(Q, T), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("n,m", [(10, 2), (11, 3), (20, 20), (21, 1), (33, 16)])
def test_sliding_dot_product_output_length(n, m):
    rng = np.random.RandomState(0)
    T = rng.randn(n)
    Q = rng.randn(m)
    result = sliding_dot_product(Q, T)
    assert result.shape == (n - m + 1,)


def test_sliding_dot_product_equal_lengths_returns_single_dot():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([4.0, -1.0, 0.5])
    result = sliding_dot_product(Q, T)
    assert result.shape == (1,)
    assert result[0] == pytest.approx(np.dot(Q, T))


def test_sliding_dot_product_query_length_one_is_elementwise_product():
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Q = np.array([2.5])
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, T * 2.5, atol=1e-9)


def test_sliding_dot_product_raises_when_query_longer_than_series():
    T = np.arange(5, dtype=float)
    Q = np.arange(6, dtype=float)
    with pytest.raises(ValueError):
        sliding_dot_product(Q, T)


# ---------------------------------------------------------------------------
# rolling_avg_sd
# ---------------------------------------------------------------------------

def test_rolling_avg_sd_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = rolling_avg_sd(t, 2)
    np.testing.assert_allclose(mean, [1.5, 2.5, 3.5], atol=1e-12)
    np.testing.assert_allclose(sd, [0.5, 0.5, 0.5], atol=1e-12)


def test_rolling_avg_sd_matches_naive(rng):
    t = rng.randn(60)
    w = 9
    mean, sd = rolling_avg_sd(t, w)
    naive_mean, naive_sd = naive_rolling_mean_std(t, w)
    np.testing.assert_allclose(mean, naive_mean, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(sd, naive_sd, rtol=1e-6, atol=1e-6)


def test_rolling_avg_sd_output_shapes(rng):
    t = rng.randn(25)
    w = 4
    mean, sd = rolling_avg_sd(t, w)
    assert mean.shape == (len(t) - w + 1,)
    assert sd.shape == (len(t) - w + 1,)


def test_rolling_avg_sd_window_equals_length(rng):
    t = rng.randn(12)
    mean, sd = rolling_avg_sd(t, len(t))
    assert mean.shape == (1,)
    assert mean[0] == pytest.approx(np.mean(t), abs=1e-9)
    assert sd[0] == pytest.approx(np.std(t), abs=1e-6)


def test_rolling_avg_sd_window_one(rng):
    t = rng.randn(15)
    mean, sd = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(mean, t, atol=1e-9)
    np.testing.assert_allclose(sd, np.zeros_like(t), atol=1e-6)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(30, 7.5)
    mean, sd = rolling_avg_sd(t, 5)
    np.testing.assert_allclose(mean, np.full(26, 7.5), atol=1e-9)
    np.testing.assert_allclose(sd, np.zeros(26), atol=1e-5)


def test_rolling_avg_sd_std_is_non_negative(rng):
    t = rng.randn(100) * 1000.0
    _, sd = rolling_avg_sd(t, 10)
    assert np.all(sd >= 0)


def test_rolling_avg_sd_raises_when_window_too_large():
    t = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        rolling_avg_sd(t, 6)


# ---------------------------------------------------------------------------
# calculate_distance_profile
# ---------------------------------------------------------------------------

def test_distance_profile_full_pipeline_matches_naive_znorm(rng):
    m = 8
    Q = rng.randn(m)
    T = rng.randn(50)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    np.testing.assert_allclose(D, naive_z_distance_profile(Q, T), rtol=1e-5, atol=1e-5)


def test_distance_profile_output_length_matches_qt(rng):
    m = 5
    Q = rng.randn(m)
    T = rng.randn(31)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D.shape == (len(T) - m + 1,)


def test_distance_profile_identical_subsequence_is_near_zero(rng):
    m = 10
    T = rng.randn(60)
    Q = T[20:20 + m].copy()
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[20] == pytest.approx(0.0, abs=1e-4)


def test_distance_profile_anticorrelated_is_maximum():
    m = 6
    rng = np.random.RandomState(7)
    Q = rng.randn(m)
    T = np.concatenate([Q, -Q])  # subsequence at index m equals -Q
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[m] == pytest.approx(2 * np.sqrt(m), rel=1e-5)


def test_distance_profile_scalar_and_array_stats_give_same_result(rng):
    m = 4
    Q = rng.randn(m)
    T = rng.randn(20)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    mean_Q_arr = np.full(len(mean_T), np.mean(Q))
    sigma_Q_arr = np.full(len(sigma_T), np.std(Q))
    D_scalar = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    D_array = calculate_distance_profile(QT, m, mean_Q_arr, sigma_Q_arr, mean_T, sigma_T)
    np.testing.assert_allclose(D_scalar, D_array, rtol=1e-9, atol=1e-9)


def test_distance_profile_constant_query_gives_sqrt_m_for_nonconstant_T(rng):
    m = 9
    T = rng.randn(30)
    QT = np.zeros(len(T) - m + 1)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, 0.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.full(len(QT), np.sqrt(m)), atol=1e-12)


def test_distance_profile_constant_query_and_constant_subsequence_is_zero():
    m = 4
    QT = np.array([0.0, 0.0, 0.0])
    mean_T = np.array([1.0, 2.0, 3.0])
    sigma_T = np.array([0.0, 0.0, 0.0])
    D = calculate_distance_profile(QT, m, 1.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.zeros(3), atol=1e-12)


def test_distance_profile_constant_subsequence_gives_sqrt_m():
    m = 4
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([0.0, 1.0, 2.0])
    sigma_T = np.array([0.0, 1.0, 0.0])
    D = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert D[0] == pytest.approx(np.sqrt(m))
    assert D[2] == pytest.approx(np.sqrt(m))
    assert np.isfinite(D[1])
    assert D[1] != pytest.approx(np.sqrt(m))


def test_distance_profile_epsilon_controls_constant_detection():
    m = 4
    QT = np.array([0.0])
    mean_T = np.array([0.0])
    sigma_T = np.array([1e-3])
    # With default epsilon, sigma_T is not treated as constant.
    D_default = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert D_default[0] == pytest.approx(np.sqrt(2 * m), rel=1e-9)
    # With a larger epsilon, sigma_T is treated as constant.
    D_loose = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T, epsilon=1e-2)
    assert D_loose[0] == pytest.approx(np.sqrt(m), rel=1e-9)


def test_distance_profile_raises_on_qt_length_mismatch():
    m = 3
    QT = np.zeros(5)
    mean_T = np.zeros(4)
    sigma_T = np.ones(4)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)


def test_distance_profile_raises_on_mean_sigma_T_length_mismatch():
    m = 3
    QT = np.zeros(4)
    mean_T = np.zeros(4)
    sigma_T = np.ones(3)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)


def test_distance_profile_raises_on_query_stat_array_mismatch():
    m = 3
    QT = np.zeros(4)
    mean_T = np.zeros(4)
    sigma_T = np.ones(4)
    mean_Q = np.zeros(4)
    sigma_Q = np.ones(3)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
