import numpy as np
import pytest

from pytsmp.utils import sliding_dot_product, rolling_avg_sd, calculate_distance_profile


# ---------------------------------------------------------------------------
# Helpers (naive reference implementations)
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


def _rolling_stats(T, m):
    return rolling_avg_sd(T, m)


# ---------------------------------------------------------------------------
# sliding_dot_product
# ---------------------------------------------------------------------------

def test_sdp_known_values_even_length():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    # windows: [1,2]->5, [2,3]->8, [3,4]->11
    np.testing.assert_allclose(sliding_dot_product(Q, T), [5.0, 8.0, 11.0], atol=1e-10)


def test_sdp_known_values_odd_length():
    Q = np.array([1.0, 1.0, 1.0])
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # windows sums: 6, 9, 12
    np.testing.assert_allclose(sliding_dot_product(Q, T), [6.0, 9.0, 12.0], atol=1e-10)


def test_sdp_query_is_not_symmetric():
    # A naive implementation that forgets to reverse Q would fail here.
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([0.0, 1.0, 0.0, 0.0])
    # windows: [0,1,0]->2, [1,0,0]->1
    np.testing.assert_allclose(sliding_dot_product(Q, T), [2.0, 1.0], atol=1e-10)


@pytest.mark.parametrize("n,m", [(10, 3), (11, 3), (16, 5), (17, 4), (8, 8), (9, 1)])
def test_sdp_matches_naive_random(n, m):
    rng = np.random.RandomState(n * 100 + m)
    T = rng.randn(n)
    Q = rng.randn(m)
    np.testing.assert_allclose(
        sliding_dot_product(Q, T), _naive_sliding_dot_product(Q, T), rtol=1e-9, atol=1e-9
    )


def test_sdp_output_length():
    rng = np.random.RandomState(1)
    T = rng.randn(23)
    Q = rng.randn(6)
    assert len(sliding_dot_product(Q, T)) == len(T) - len(Q) + 1


def test_sdp_equal_length_gives_single_dot_product():
    Q = np.array([1.0, -2.0, 3.0])
    T = np.array([4.0, 0.5, 2.0])
    result = sliding_dot_product(Q, T)
    assert result.shape == (1,)
    np.testing.assert_allclose(result[0], np.dot(Q, T), atol=1e-10)


def test_sdp_query_length_one_scales_series():
    T = np.array([1.0, -2.0, 3.0, 0.5, 7.0])
    Q = np.array([2.5])
    np.testing.assert_allclose(sliding_dot_product(Q, T), 2.5 * T, atol=1e-10)


def test_sdp_raises_when_query_longer_than_series():
    with pytest.raises(ValueError):
        sliding_dot_product(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


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

def test_ras_known_values_window2():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    means, sds = rolling_avg_sd(t, 2)
    np.testing.assert_allclose(means, [1.5, 2.5, 3.5, 4.5], atol=1e-10)
    np.testing.assert_allclose(sds, [0.5, 0.5, 0.5, 0.5], atol=1e-10)


def test_ras_known_values_window3_population_sd():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    means, sds = rolling_avg_sd(t, 3)
    np.testing.assert_allclose(means, [2.0, 3.0, 4.0], atol=1e-10)
    # population sd of [1,2,3] is sqrt(2/3)
    np.testing.assert_allclose(sds, [np.sqrt(2.0 / 3.0)] * 3, atol=1e-10)


@pytest.mark.parametrize("n,w", [(20, 4), (21, 5), (30, 1), (15, 15)])
def test_ras_matches_naive_random(n, w):
    rng = np.random.RandomState(n + w)
    t = rng.randn(n)
    means, sds = rolling_avg_sd(t, w)
    ref_means, ref_sds = _naive_rolling_avg_sd(t, w)
    np.testing.assert_allclose(means, ref_means, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(sds, ref_sds, rtol=1e-7, atol=1e-7)


def test_ras_output_shapes():
    t = np.arange(12, dtype=float)
    w = 5
    means, sds = rolling_avg_sd(t, w)
    assert means.shape == (len(t) - w + 1,)
    assert sds.shape == (len(t) - w + 1,)


def test_ras_window_equal_to_length_gives_single_stats():
    t = np.array([2.0, 4.0, 6.0, 8.0])
    means, sds = rolling_avg_sd(t, len(t))
    assert means.shape == (1,)
    np.testing.assert_allclose(means[0], np.mean(t), atol=1e-10)
    np.testing.assert_allclose(sds[0], np.std(t), atol=1e-10)


def test_ras_window_one_gives_zero_sd():
    t = np.array([3.0, -1.0, 4.0, 1.5])
    means, sds = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(means, t, atol=1e-10)
    np.testing.assert_allclose(sds, np.zeros(len(t)), atol=1e-10)


def test_ras_constant_series_has_zero_sd():
    t = np.full(10, 5.0)
    means, sds = rolling_avg_sd(t, 4)
    np.testing.assert_allclose(means, np.full(7, 5.0), atol=1e-10)
    np.testing.assert_allclose(sds, np.zeros(7), atol=1e-10)


def test_ras_sd_is_nonnegative_and_finite():
    rng = np.random.RandomState(7)
    t = 1e4 + rng.randn(50) * 1e-3  # stress cancellation in E[x^2] - E[x]^2
    _, sds = rolling_avg_sd(t, 8)
    assert np.all(sds >= 0)
    assert np.all(np.isfinite(sds))


def test_ras_raises_when_window_larger_than_series():
    with pytest.raises(ValueError):
        rolling_avg_sd(np.array([1.0, 2.0, 3.0]), 4)


# ---------------------------------------------------------------------------
# calculate_distance_profile
# ---------------------------------------------------------------------------

def test_cdp_matches_znormalized_euclidean_distance():
    rng = np.random.RandomState(42)
    T = rng.randn(40)
    Q = rng.randn(6)
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = _rolling_stats(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    np.testing.assert_allclose(D, _naive_znorm_distance_profile(Q, T), rtol=1e-6, atol=1e-6)


def test_cdp_self_match_has_zero_distance():
    rng = np.random.RandomState(3)
    T = rng.randn(30)
    m = 5
    start = 4
    Q = T[start:start + m]
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = _rolling_stats(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[start] == pytest.approx(0.0, abs=1e-5)


def test_cdp_output_length_and_nonnegative():
    rng = np.random.RandomState(11)
    T = rng.randn(25)
    Q = rng.randn(4)
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = _rolling_stats(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D.shape == (len(T) - m + 1,)
    assert np.all(D >= 0)


def test_cdp_constant_query_and_constant_subsequences_give_zero():
    T = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 7.0, 1.0, 4.0, 3.0, 9.0])
    Q = np.array([3.0, 3.0, 3.0])
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = _rolling_stats(T, m)
    D = calculate_distance_profile(QT, m, 3.0, 0.0, mean_T, sigma_T)
    # windows 0,1,2 are constant subsequences of value 2
    np.testing.assert_allclose(D[:3], np.zeros(3), atol=1e-10)


def test_cdp_constant_query_nonconstant_subsequence_gives_sqrt_m():
    T = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 7.0, 1.0, 4.0, 3.0, 9.0])
    Q = np.array([3.0, 3.0, 3.0])
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = _rolling_stats(T, m)
    D = calculate_distance_profile(QT, m, 3.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D[3:], np.full(len(D) - 3, np.sqrt(m)), atol=1e-10)


def test_cdp_nonconstant_query_constant_subsequence_gives_sqrt_m():
    T = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 7.0, 1.0, 4.0, 3.0, 9.0])
    Q = np.array([1.0, 5.0, 2.0])
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = _rolling_stats(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    # windows 0,1,2 are constant in T
    np.testing.assert_allclose(D[:3], np.full(3, np.sqrt(m)), atol=1e-10)


def test_cdp_custom_epsilon_treats_small_query_sd_as_constant():
    rng = np.random.RandomState(5)
    T = rng.randn(20)
    m = 4
    QT = sliding_dot_product(rng.randn(m), T)
    mean_T, sigma_T = _rolling_stats(T, m)
    # sigma_Q is below the custom epsilon, so the query counts as constant
    D = calculate_distance_profile(QT, m, 0.0, 1e-3, mean_T, sigma_T, epsilon=1e-2)
    np.testing.assert_allclose(D, np.full(len(D), np.sqrt(m)), atol=1e-10)


def test_cdp_raises_when_qt_and_mean_t_lengths_differ():
    QT = np.zeros(5)
    mean_T = np.zeros(4)
    sigma_T = np.ones(4)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 2, 0.0, 1.0, mean_T, sigma_T)


def test_cdp_raises_when_mean_q_array_length_mismatches_sigma_q():
    QT = np.zeros(4)
    mean_T = np.zeros(4)
    sigma_T = np.ones(4)
    mean_Q = np.zeros(3)
    sigma_Q = np.ones(2)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 3, mean_Q, sigma_Q, mean_T, sigma_T)
