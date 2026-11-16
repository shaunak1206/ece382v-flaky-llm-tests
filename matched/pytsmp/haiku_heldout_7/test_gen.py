import numpy as np
import pytest

from pytsmp.utils import sliding_dot_product, rolling_avg_sd, calculate_distance_profile


# ---------------------------------------------------------------------------
# Naive reference helpers (independent of the implementation under test)
# ---------------------------------------------------------------------------

def _naive_sliding_dot_product(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _naive_rolling_avg_sd(t, w):
    subs = np.array([t[i:i + w] for i in range(len(t) - w + 1)])
    return subs.mean(axis=1), subs.std(axis=1)


def _naive_znorm_distance_profile(Q, T):
    m = len(Q)
    zq = (Q - Q.mean()) / Q.std()
    out = []
    for i in range(len(T) - m + 1):
        s = T[i:i + m]
        zs = (s - s.mean()) / s.std()
        out.append(np.sqrt(np.sum((zq - zs) ** 2)))
    return np.array(out)


# ---------------------------------------------------------------------------
# sliding_dot_product
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", [(8, 3), (11, 3), (10, 4), (7, 2), (12, 5)])
def test_sdp_matches_naive_computation(n, m):
    rng = np.random.RandomState(n * 100 + m)
    Q = rng.randn(m)
    T = rng.randn(n)
    result = sliding_dot_product(Q, T)
    expected = _naive_sliding_dot_product(Q, T)
    assert np.allclose(result, expected, atol=1e-10)


def test_sdp_output_length():
    rng = np.random.RandomState(1)
    Q = rng.randn(4)
    T = rng.randn(20)
    result = sliding_dot_product(Q, T)
    assert result.shape == (len(T) - len(Q) + 1,)


def test_sdp_known_values():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    result = sliding_dot_product(Q, T)
    assert np.allclose(result, [5.0, 8.0, 11.0])


def test_sdp_query_same_length_as_series():
    Q = np.array([1.0, -2.0, 3.0])
    T = np.array([4.0, 0.5, -1.0])
    result = sliding_dot_product(Q, T)
    assert result.shape == (1,)
    assert np.isclose(result[0], np.dot(Q, T))


def test_sdp_query_of_length_one_scales_series():
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Q = np.array([2.5])
    result = sliding_dot_product(Q, T)
    assert np.allclose(result, 2.5 * T)


def test_sdp_raises_when_query_longer_than_series():
    Q = np.arange(6, dtype=float)
    T = np.arange(4, dtype=float)
    with pytest.raises(ValueError):
        sliding_dot_product(Q, T)


def test_sdp_is_linear_in_query():
    rng = np.random.RandomState(7)
    Q = rng.randn(5)
    T = rng.randn(16)
    assert np.allclose(sliding_dot_product(2.0 * Q, T),
                       2.0 * sliding_dot_product(Q, T), atol=1e-10)


# ---------------------------------------------------------------------------
# rolling_avg_sd
# ---------------------------------------------------------------------------

def test_ras_matches_naive_window_statistics():
    rng = np.random.RandomState(3)
    t = rng.randn(50)
    w = 7
    mean, sd = rolling_avg_sd(t, w)
    exp_mean, exp_sd = _naive_rolling_avg_sd(t, w)
    assert np.allclose(mean, exp_mean, atol=1e-10)
    assert np.allclose(sd, exp_sd, atol=1e-8)


def test_ras_returns_two_arrays_of_expected_shape():
    t = np.linspace(0, 1, 30)
    w = 6
    result = rolling_avg_sd(t, w)
    assert isinstance(result, tuple)
    assert len(result) == 2
    mean, sd = result
    assert mean.shape == (len(t) - w + 1,)
    assert sd.shape == (len(t) - w + 1,)


def test_ras_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = rolling_avg_sd(t, 2)
    assert np.allclose(mean, [1.5, 2.5, 3.5])
    assert np.allclose(sd, [0.5, 0.5, 0.5])


def test_ras_window_size_one():
    t = np.array([3.0, -1.0, 7.0, 2.0])
    mean, sd = rolling_avg_sd(t, 1)
    assert np.allclose(mean, t)
    assert np.allclose(sd, 0.0)


def test_ras_window_equal_to_series_length():
    t = np.array([1.0, 2.0, 4.0, 8.0])
    mean, sd = rolling_avg_sd(t, len(t))
    assert mean.shape == (1,)
    assert np.isclose(mean[0], np.mean(t))
    assert np.isclose(sd[0], np.std(t))


def test_ras_raises_when_window_larger_than_series():
    t = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        rolling_avg_sd(t, 6)


def test_ras_constant_series_has_zero_sd():
    t = np.full(20, 5.0)
    mean, sd = rolling_avg_sd(t, 4)
    assert np.allclose(mean, 5.0)
    assert np.allclose(sd, 0.0, atol=1e-8)


def test_ras_sd_is_finite_and_nonnegative_with_large_offset():
    rng = np.random.RandomState(11)
    t = 1e6 + rng.rand(100)
    _, sd = rolling_avg_sd(t, 10)
    assert np.all(np.isfinite(sd))
    assert np.all(sd >= 0)


def test_ras_shift_and_scale_behaviour():
    rng = np.random.RandomState(5)
    t = rng.randn(40)
    w = 5
    mean, sd = rolling_avg_sd(t, w)
    mean_shift, sd_shift = rolling_avg_sd(t + 10.0, w)
    assert np.allclose(mean_shift, mean + 10.0, atol=1e-9)
    assert np.allclose(sd_shift, sd, atol=1e-8)

    mean_scale, sd_scale = rolling_avg_sd(3.0 * t, w)
    assert np.allclose(mean_scale, 3.0 * mean, atol=1e-9)
    assert np.allclose(sd_scale, 3.0 * sd, atol=1e-8)


# ---------------------------------------------------------------------------
# calculate_distance_profile
# ---------------------------------------------------------------------------

def test_cdp_matches_naive_znormalized_distance():
    rng = np.random.RandomState(21)
    m = 6
    Q = rng.randn(m)
    T = rng.randn(40)
    QT = _naive_sliding_dot_product(Q, T)
    mean_T, sigma_T = _naive_rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    expected = _naive_znorm_distance_profile(Q, T)
    assert np.allclose(D, expected, atol=1e-6)


def test_cdp_output_length_matches_inputs():
    rng = np.random.RandomState(4)
    m = 5
    Q = rng.randn(m)
    T = rng.randn(25)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D.shape == (len(T) - m + 1,)


def test_cdp_exact_match_gives_zero_distance():
    rng = np.random.RandomState(8)
    m = 8
    T = rng.randn(50)
    p = 20
    Q = T[p:p + m].copy()
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert np.isclose(D[p], 0.0, atol=1e-4)


def test_cdp_constant_query_and_constant_subsequence_is_zero():
    m = 8
    rng = np.random.RandomState(2)
    # Integer-valued series so that rolling sigma for the constant region is exactly 0.
    T = rng.randint(0, 10, size=60).astype(float)
    T[10:20] = 5.0
    Q = np.full(m, 5.0)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[10] == 0.0


def test_cdp_constant_query_gives_sqrt_m_for_nonconstant_subsequences():
    m = 8
    rng = np.random.RandomState(9)
    T = rng.randn(40)
    Q = np.full(m, 2.0)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert np.allclose(D, np.sqrt(m))


def test_cdp_constant_subsequence_gives_sqrt_m_for_nonconstant_query():
    m = 8
    rng = np.random.RandomState(13)
    T = rng.randint(0, 10, size=60).astype(float)
    T[10:20] = 5.0
    Q = rng.randn(m)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert np.allclose(D[10:13], np.sqrt(m))


def test_cdp_upper_bound_is_two_sqrt_m():
    rng = np.random.RandomState(31)
    m = 10
    Q = rng.randn(m)
    T = rng.randn(80)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert np.all(D >= 0)
    assert np.all(D <= 2 * np.sqrt(m) + 1e-6)


def test_cdp_scalar_query_statistics_match_array_statistics():
    rng = np.random.RandomState(17)
    m = 6
    Q = rng.randn(m)
    T = rng.randn(30)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D_array = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    D_scalar = calculate_distance_profile(QT, m, float(np.mean(Q)), float(np.std(Q)),
                                          mean_T, sigma_T)
    assert np.allclose(D_array, D_scalar)


def test_cdp_raises_on_length_mismatch_between_qt_and_mean_t():
    m = 3
    QT = np.zeros(5)
    mean_T = np.zeros(4)
    sigma_T = np.ones(5)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)


def test_cdp_raises_when_mean_q_and_sigma_q_arrays_differ_in_length():
    m = 3
    QT = np.zeros(5)
    mean_T = np.zeros(5)
    sigma_T = np.ones(5)
    mean_Q = np.zeros(3)
    sigma_Q = np.ones(4)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
