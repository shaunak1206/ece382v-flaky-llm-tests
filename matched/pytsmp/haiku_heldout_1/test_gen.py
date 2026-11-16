import numpy as np
import pytest

from pytsmp.utils import calculate_distance_profile, rolling_avg_sd, sliding_dot_product


# ---------------------------------------------------------------------------
# Naive reference helpers
# ---------------------------------------------------------------------------

def _naive_sliding_dot_product(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _naive_rolling_avg_sd(t, w):
    windows = np.array([t[i:i + w] for i in range(len(t) - w + 1)], dtype=float)
    return windows.mean(axis=1), windows.std(axis=1)


def _naive_distance_profile(Q, T):
    m = len(Q)
    zq = (Q - Q.mean()) / Q.std()
    dists = []
    for i in range(len(T) - m + 1):
        sub = T[i:i + m]
        zs = (sub - sub.mean()) / sub.std()
        dists.append(np.linalg.norm(zq - zs))
    return np.array(dists)


# ---------------------------------------------------------------------------
# sliding_dot_product
# ---------------------------------------------------------------------------

def test_sdp_known_values():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(sliding_dot_product(Q, T), [5.0, 8.0, 11.0], atol=1e-9)


@pytest.mark.parametrize("n, m", [(50, 7), (37, 5), (10, 10), (11, 1), (20, 3), (9, 4), (7, 7), (64, 16)])
def test_sdp_matches_naive(n, m):
    rng = np.random.RandomState(n * 100 + m)
    T = rng.randn(n)
    Q = rng.randn(m)
    np.testing.assert_allclose(
        sliding_dot_product(Q, T), _naive_sliding_dot_product(Q, T), rtol=1e-9, atol=1e-9
    )


def test_sdp_output_length():
    T = np.arange(23, dtype=float)
    Q = np.arange(6, dtype=float)
    assert sliding_dot_product(Q, T).shape == (len(T) - len(Q) + 1,)


def test_sdp_equal_length_returns_single_dot_product():
    rng = np.random.RandomState(0)
    Q = rng.randn(8)
    T = rng.randn(8)
    result = sliding_dot_product(Q, T)
    assert result.shape == (1,)
    np.testing.assert_allclose(result[0], np.dot(Q, T), rtol=1e-9, atol=1e-9)


def test_sdp_query_of_length_one_scales_series():
    T = np.array([2.0, -1.0, 4.0, 0.5, 3.0])
    Q = np.array([3.0])
    np.testing.assert_allclose(sliding_dot_product(Q, T), 3.0 * T, atol=1e-9)


def test_sdp_raises_when_query_longer_than_series():
    with pytest.raises(ValueError):
        sliding_dot_product(np.ones(5), np.ones(4))


def test_sdp_recovers_embedded_query_value():
    rng = np.random.RandomState(42)
    T = rng.randn(40)
    Q = T[12:19].copy()
    QT = sliding_dot_product(Q, T)
    np.testing.assert_allclose(QT[12], np.dot(Q, Q), rtol=1e-9, atol=1e-9)


def test_sdp_accepts_integer_arrays():
    Q = np.array([1, 0, -1, 2])
    T = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    np.testing.assert_allclose(
        sliding_dot_product(Q, T), _naive_sliding_dot_product(Q.astype(float), T.astype(float)), atol=1e-9
    )


def test_sdp_does_not_modify_inputs():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    Q_copy, T_copy = Q.copy(), T.copy()
    sliding_dot_product(Q, T)
    np.testing.assert_array_equal(Q, Q_copy)
    np.testing.assert_array_equal(T, T_copy)


# ---------------------------------------------------------------------------
# rolling_avg_sd
# ---------------------------------------------------------------------------

def test_rolling_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = rolling_avg_sd(t, 2)
    np.testing.assert_allclose(mean, [1.5, 2.5, 3.5], atol=1e-9)
    np.testing.assert_allclose(sd, [0.5, 0.5, 0.5], atol=1e-9)


@pytest.mark.parametrize("n, w", [(50, 5), (37, 4), (20, 1), (15, 15), (30, 7)])
def test_rolling_matches_naive(n, w):
    rng = np.random.RandomState(n + w)
    t = rng.randn(n)
    mean, sd = rolling_avg_sd(t, w)
    exp_mean, exp_sd = _naive_rolling_avg_sd(t, w)
    np.testing.assert_allclose(mean, exp_mean, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(sd, exp_sd, rtol=1e-6, atol=1e-6)


def test_rolling_output_shapes():
    t = np.arange(12, dtype=float)
    mean, sd = rolling_avg_sd(t, 5)
    assert mean.shape == (len(t) - 5 + 1,)
    assert sd.shape == (len(t) - 5 + 1,)


def test_rolling_window_equal_to_length():
    t = np.array([2.0, 4.0, 6.0, 8.0])
    mean, sd = rolling_avg_sd(t, len(t))
    assert mean.shape == (1,)
    np.testing.assert_allclose(mean[0], 5.0, atol=1e-9)
    np.testing.assert_allclose(sd[0], np.std(t), atol=1e-9)


def test_rolling_window_one_gives_series_and_zero_sd():
    t = np.array([3.0, -1.0, 7.5, 0.0])
    mean, sd = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(mean, t, atol=1e-9)
    np.testing.assert_allclose(sd, np.zeros_like(t), atol=1e-9)


def test_rolling_constant_series_has_zero_sd():
    t = np.full(10, 3.0)
    mean, sd = rolling_avg_sd(t, 4)
    np.testing.assert_array_equal(mean, np.full(7, 3.0))
    np.testing.assert_array_equal(sd, np.zeros(7))


def test_rolling_raises_when_window_larger_than_series():
    with pytest.raises(ValueError):
        rolling_avg_sd(np.arange(4, dtype=float), 5)


def test_rolling_sd_is_finite_and_nonnegative():
    rng = np.random.RandomState(7)
    t = rng.randn(200)
    _, sd = rolling_avg_sd(t, 16)
    assert np.all(np.isfinite(sd))
    assert np.all(sd >= 0)


def test_rolling_does_not_modify_input():
    t = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    t_copy = t.copy()
    rolling_avg_sd(t, 3)
    np.testing.assert_array_equal(t, t_copy)


# ---------------------------------------------------------------------------
# calculate_distance_profile
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", [(60, 8), (45, 5), (30, 12)])
def test_cdp_matches_naive_znorm_distance(n, m):
    rng = np.random.RandomState(n * 7 + m)
    T = rng.randn(n)
    Q = T[3:3 + m] + 0.5 * rng.randn(m)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    np.testing.assert_allclose(D, _naive_distance_profile(Q, T), rtol=1e-5, atol=1e-5)


def test_cdp_self_match_is_zero():
    rng = np.random.RandomState(3)
    T = rng.randn(80)
    m = 10
    i = 25
    Q = T[i:i + m].copy()
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    np.testing.assert_allclose(D[i], 0.0, atol=1e-5)


def test_cdp_output_shape_and_nonnegative_bounded():
    rng = np.random.RandomState(11)
    T = rng.randn(100)
    m = 9
    Q = rng.randn(m)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D.shape == (len(T) - m + 1,)
    assert np.all(D >= 0)
    assert np.all(D <= 2 * np.sqrt(m) + 1e-6)


def test_cdp_known_single_value():
    # corr = (QT - m*mean_Q*mean_T) / (m*sigma_Q*sigma_T) = (14 - 12) / 4 = 0.5
    # D = sqrt(2*4*(1 - 0.5)) = sqrt(4) = 2
    D = calculate_distance_profile(
        np.array([14.0]), 4, 1.0, 2.0, np.array([3.0]), np.array([0.5])
    )
    np.testing.assert_allclose(D, [2.0], atol=1e-9)


def test_cdp_both_constant_gives_zero():
    m = 4
    D = calculate_distance_profile(
        np.zeros(3), m, 0.0, 0.0, np.zeros(3), np.array([0.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(D, np.zeros(3), atol=1e-12)


def test_cdp_constant_query_against_nonconstant_gives_sqrt_m():
    m = 4
    D = calculate_distance_profile(
        np.zeros(2), m, 0.0, 0.0, np.zeros(2), np.array([0.0, 1.0])
    )
    np.testing.assert_allclose(D[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(D[1], np.sqrt(m), atol=1e-9)


def test_cdp_constant_subsequence_against_nonconstant_query_gives_sqrt_m():
    m = 4
    # sigma_T = [0, 1, 0]; index 1 is non-constant and uses the formula (QT=0 -> D = sqrt(8))
    D = calculate_distance_profile(
        np.zeros(3), m, 0.0, 1.0, np.zeros(3), np.array([0.0, 1.0, 0.0])
    )
    np.testing.assert_allclose(D[0], np.sqrt(m), atol=1e-9)
    np.testing.assert_allclose(D[1], np.sqrt(8.0), atol=1e-9)
    np.testing.assert_allclose(D[2], np.sqrt(m), atol=1e-9)


def test_cdp_custom_epsilon_treats_small_sigma_as_constant():
    m = 5
    sigma_T = np.array([1e-4, 1.0])
    D = calculate_distance_profile(
        np.zeros(2), m, 0.0, 1.0, np.zeros(2), sigma_T, epsilon=1e-3
    )
    np.testing.assert_allclose(D[0], np.sqrt(m), atol=1e-9)


def test_cdp_raises_on_length_mismatch_qt_vs_mean_t():
    with pytest.raises(ValueError):
        calculate_distance_profile(
            np.zeros(3), 4, 0.0, 1.0, np.zeros(2), np.ones(2)
        )


def test_cdp_raises_on_mean_sigma_array_length_mismatch():
    with pytest.raises(ValueError):
        calculate_distance_profile(
            np.zeros(3), 4, np.zeros(2), np.zeros(3), np.zeros(3), np.ones(3)
        )


def test_cdp_raises_when_mean_q_array_does_not_match_mean_t():
    with pytest.raises(ValueError):
        calculate_distance_profile(
            np.zeros(3), 4, np.zeros(2), np.zeros(2), np.zeros(3), np.ones(3)
        )
