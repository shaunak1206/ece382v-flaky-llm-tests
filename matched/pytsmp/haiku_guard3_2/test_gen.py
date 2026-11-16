import numpy as np
import pytest

from pytsmp.utils import calculate_distance_profile, rolling_avg_sd, sliding_dot_product


# ---------------------------------------------------------------------------
# Helpers (pure functions, no shared state)
# ---------------------------------------------------------------------------

def _naive_sliding_dot_product(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _naive_rolling_avg_sd(t, w):
    windows = [t[i:i + w] for i in range(len(t) - w + 1)]
    means = np.array([np.mean(x) for x in windows])
    sds = np.array([np.std(x) for x in windows])  # population sd, ddof=0
    return means, sds


def _naive_znorm_distance_profile(Q, T):
    m = len(Q)
    zq = (Q - np.mean(Q)) / np.std(Q)
    out = []
    for i in range(len(T) - m + 1):
        sub = T[i:i + m]
        zt = (sub - np.mean(sub)) / np.std(sub)
        out.append(np.sqrt(np.sum((zq - zt) ** 2)))
    return np.array(out)


def _sine_series(n):
    idx = np.arange(n, dtype=float)
    return np.sin(idx * 0.3) + 0.1 * np.cos(idx * 0.7)


# ---------------------------------------------------------------------------
# sliding_dot_product
# ---------------------------------------------------------------------------

def test_sliding_dot_product_known_values_even_length():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, [5.0, 8.0, 11.0], rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_known_values_odd_length():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, [5.0, 8.0, 11.0, 14.0], rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_matches_naive_on_sine_series():
    T = _sine_series(37)  # odd length exercises the padding branch
    Q = _sine_series(5)[::-1].copy() + 0.5
    result = sliding_dot_product(Q, T)
    expected = _naive_sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("n, m", [(10, 3), (11, 4), (7, 7), (20, 1), (64, 9)])
def test_sliding_dot_product_output_length(n, m):
    T = _sine_series(n)
    Q = _sine_series(m)
    result = sliding_dot_product(Q, T)
    assert result.shape == (n - m + 1,)


def test_sliding_dot_product_equal_length_gives_single_value():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([1.0, 2.0, 3.0])
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, [14.0], rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_single_element_query_scales_series():
    Q = np.array([2.0])
    T = np.array([1.0, 2.0, 3.0])
    result = sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, [2.0, 4.0, 6.0], rtol=1e-12, atol=1e-12)


def test_sliding_dot_product_raises_when_query_longer_than_series():
    Q = np.array([1.0, 2.0, 3.0, 4.0])
    T = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        sliding_dot_product(Q, T)


# ---------------------------------------------------------------------------
# rolling_avg_sd
# ---------------------------------------------------------------------------

def test_rolling_avg_sd_known_values_window_two():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    means, sds = rolling_avg_sd(t, 2)
    np.testing.assert_allclose(means, [1.5, 2.5, 3.5, 4.5], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sds, [0.5, 0.5, 0.5, 0.5], rtol=1e-12, atol=1e-12)


def test_rolling_avg_sd_known_values_window_three():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    means, sds = rolling_avg_sd(t, 3)
    expected_sd = np.sqrt(2.0 / 3.0)
    np.testing.assert_allclose(means, [2.0, 3.0, 4.0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sds, [expected_sd] * 3, rtol=1e-9, atol=1e-9)


def test_rolling_avg_sd_window_equal_to_length():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    means, sds = rolling_avg_sd(t, 5)
    np.testing.assert_allclose(means, [3.0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sds, [np.sqrt(2.0)], rtol=1e-9, atol=1e-9)


def test_rolling_avg_sd_window_one_returns_series_and_zero_sd():
    t = _sine_series(12)
    means, sds = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(means, t, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sds, np.zeros(len(t)), rtol=0, atol=1e-7)


def test_rolling_avg_sd_matches_naive_on_sine_series():
    t = _sine_series(40)
    w = 6
    means, sds = rolling_avg_sd(t, w)
    exp_means, exp_sds = _naive_rolling_avg_sd(t, w)
    np.testing.assert_allclose(means, exp_means, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(sds, exp_sds, rtol=1e-7, atol=1e-7)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(10, 3.7)
    means, sds = rolling_avg_sd(t, 4)
    np.testing.assert_allclose(means, np.full(7, 3.7), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sds, np.zeros(7), rtol=0, atol=1e-6)


def test_rolling_avg_sd_sd_is_never_negative():
    t = _sine_series(50)
    _, sds = rolling_avg_sd(t, 7)
    assert np.all(sds >= 0)
    assert not np.any(np.isnan(sds))


def test_rolling_avg_sd_output_shapes():
    t = _sine_series(23)
    w = 5
    means, sds = rolling_avg_sd(t, w)
    assert means.shape == (len(t) - w + 1,)
    assert sds.shape == (len(t) - w + 1,)


def test_rolling_avg_sd_raises_when_window_longer_than_series():
    t = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        rolling_avg_sd(t, 4)


# ---------------------------------------------------------------------------
# calculate_distance_profile
# ---------------------------------------------------------------------------

def _profile_inputs(T, Q):
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)
    return QT, m, mean_Q, sigma_Q, mean_T, sigma_T


def test_calculate_distance_profile_matches_naive_znorm_distance():
    T = _sine_series(40)
    Q = np.cos(np.arange(6) * 1.1)
    args = _profile_inputs(T, Q)
    D = calculate_distance_profile(*args)
    expected = _naive_znorm_distance_profile(Q, T)
    np.testing.assert_allclose(D, expected, rtol=1e-6, atol=1e-6)


def test_calculate_distance_profile_identical_subsequence_is_zero():
    T = _sine_series(40)
    m = 6
    Q = T[7:7 + m].copy()
    args = _profile_inputs(T, Q)
    D = calculate_distance_profile(*args)
    assert D[7] == pytest.approx(0.0, abs=1e-6)


def test_calculate_distance_profile_output_length_and_nonnegative():
    T = _sine_series(40)
    Q = np.cos(np.arange(6) * 1.1)
    args = _profile_inputs(T, Q)
    D = calculate_distance_profile(*args)
    assert D.shape == (len(T) - len(Q) + 1,)
    assert not np.any(np.isnan(D))
    assert np.all(D >= 0)


def test_calculate_distance_profile_both_constant_gives_zero():
    QT = np.zeros(5)
    m = 4
    mean_T = np.zeros(5)
    sigma_T = np.zeros(5)
    D = calculate_distance_profile(QT, m, 0.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.zeros(5), rtol=0, atol=0)


def test_calculate_distance_profile_constant_query_gives_sqrt_m():
    QT = np.zeros(5)
    m = 4
    mean_T = np.zeros(5)
    sigma_T = np.ones(5)
    D = calculate_distance_profile(QT, m, 0.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.full(5, np.sqrt(m)), rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_constant_subsequence_gives_sqrt_m():
    QT = np.zeros(5)
    m = 4
    mean_T = np.zeros(5)
    sigma_T = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    D = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    np.testing.assert_allclose(D[[0, 2, 4]], np.full(3, np.sqrt(m)), rtol=1e-12, atol=1e-12)


def test_calculate_distance_profile_epsilon_threshold_is_respected():
    # sigma_Q is below the default epsilon (1e-7) but above 1e-9, so the
    # result depends on the epsilon argument.
    m = 4
    QT = np.array([0.0])
    mean_T = np.array([0.0])
    sigma_T = np.array([1.0])
    sigma_Q = 5e-8

    D_default = calculate_distance_profile(QT, m, 0.0, sigma_Q, mean_T, sigma_T)
    D_strict = calculate_distance_profile(QT, m, 0.0, sigma_Q, mean_T, sigma_T, epsilon=1e-9)

    np.testing.assert_allclose(D_default, [np.sqrt(m)], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(D_strict, [np.sqrt(2 * m)], rtol=1e-9, atol=1e-9)


def test_calculate_distance_profile_raises_on_length_mismatch_qt_vs_mean_t():
    QT = np.zeros(4)
    mean_T = np.zeros(5)
    sigma_T = np.ones(5)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 3, 0.0, 1.0, mean_T, sigma_T)


def test_calculate_distance_profile_raises_when_mean_q_array_mismatches():
    QT = np.zeros(5)
    mean_T = np.zeros(5)
    sigma_T = np.ones(5)
    mean_Q = np.zeros(3)
    sigma_Q = np.ones(3)
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 3, mean_Q, sigma_Q, mean_T, sigma_T)
