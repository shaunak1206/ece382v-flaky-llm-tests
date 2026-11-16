import warnings

import numpy as np
import pytest

from pytsmp.utils import (
    calculate_distance_profile,
    rolling_avg_sd,
    sliding_dot_product,
)


def _naive_dot(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _naive_rolling(T, m):
    subs = np.array([T[i:i + m] for i in range(len(T) - m + 1)])
    return subs.mean(axis=1), subs.std(axis=1)


def _naive_distance_profile(Q, T):
    m = len(Q)
    zq = (Q - Q.mean()) / Q.std()
    out = []
    for i in range(len(T) - m + 1):
        sub = T[i:i + m]
        zt = (sub - sub.mean()) / sub.std()
        out.append(np.sqrt(np.sum((zq - zt) ** 2)))
    return np.array(out)


# ---------------------------------------------------------------------------
# sliding_dot_product
# ---------------------------------------------------------------------------

def test_sdp_matches_naive_even_length_series():
    rng = np.random.default_rng(0)
    T = rng.standard_normal(16)
    Q = rng.standard_normal(5)
    np.testing.assert_allclose(sliding_dot_product(Q, T), _naive_dot(Q, T), atol=1e-9)


def test_sdp_matches_naive_odd_length_series_and_query():
    # Odd len(T) exercises the padding branch.
    rng = np.random.default_rng(1)
    T = rng.standard_normal(17)
    Q = rng.standard_normal(4)
    np.testing.assert_allclose(sliding_dot_product(Q, T), _naive_dot(Q, T), atol=1e-9)


def test_sdp_matches_numpy_correlate_valid():
    rng = np.random.default_rng(2)
    T = rng.standard_normal(23)
    Q = rng.standard_normal(7)
    expected = np.correlate(T, Q, mode="valid")
    np.testing.assert_allclose(sliding_dot_product(Q, T), expected, atol=1e-9)


def test_sdp_known_small_example():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(sliding_dot_product(Q, T), np.array([5.0, 8.0, 11.0]), atol=1e-9)


def test_sdp_equal_length_returns_single_dot_product():
    T = np.array([1.0, -2.0, 3.0, 0.5])
    result = sliding_dot_product(T.copy(), T)
    assert result.shape == (1,)
    assert result[0] == pytest.approx(np.dot(T, T), abs=1e-9)


def test_sdp_single_element_query_scales_series():
    T = np.array([1.0, 2.0, 3.0])
    Q = np.array([2.0])
    np.testing.assert_allclose(sliding_dot_product(Q, T), np.array([2.0, 4.0, 6.0]), atol=1e-9)


@pytest.mark.parametrize("n,m", [(10, 3), (11, 3), (8, 8), (9, 1), (20, 6)])
def test_sdp_output_length(n, m):
    rng = np.random.default_rng(n * 100 + m)
    T = rng.standard_normal(n)
    Q = rng.standard_normal(m)
    assert sliding_dot_product(Q, T).shape == (n - m + 1,)


def test_sdp_self_match_gives_sum_of_squares_at_offset():
    rng = np.random.default_rng(3)
    T = rng.standard_normal(19)
    m = 5
    k = 7
    Q = T[k:k + m].copy()
    QT = sliding_dot_product(Q, T)
    assert QT[k] == pytest.approx(np.sum(Q ** 2), abs=1e-9)


def test_sdp_raises_when_query_longer_than_series():
    with pytest.raises(ValueError):
        sliding_dot_product(np.arange(6.0), np.arange(5.0))


def test_sdp_does_not_modify_inputs():
    rng = np.random.default_rng(4)
    T = rng.standard_normal(12)
    Q = rng.standard_normal(4)
    T_copy = T.copy()
    Q_copy = Q.copy()
    sliding_dot_product(Q, T)
    np.testing.assert_array_equal(T, T_copy)
    np.testing.assert_array_equal(Q, Q_copy)


def test_sdp_integer_inputs_match_float_result():
    T = np.array([1, 2, 3, 4, 5], dtype=int)
    Q = np.array([1, 0, -1], dtype=int)
    expected = _naive_dot(Q.astype(float), T.astype(float))
    np.testing.assert_allclose(sliding_dot_product(Q, T), expected, atol=1e-9)


# ---------------------------------------------------------------------------
# rolling_avg_sd
# ---------------------------------------------------------------------------

def test_rolling_matches_naive_mean_and_std():
    rng = np.random.default_rng(10)
    T = rng.standard_normal(40)
    w = 6
    mean, sd = rolling_avg_sd(T, w)
    naive_mean, naive_sd = _naive_rolling(T, w)
    np.testing.assert_allclose(mean, naive_mean, atol=1e-9)
    np.testing.assert_allclose(sd, naive_sd, atol=1e-7)


def test_rolling_known_values():
    T = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = rolling_avg_sd(T, 2)
    np.testing.assert_allclose(mean, [1.5, 2.5, 3.5], atol=1e-12)
    np.testing.assert_allclose(sd, [0.5, 0.5, 0.5], atol=1e-12)


@pytest.mark.parametrize("n,w", [(10, 2), (15, 5), (9, 9), (30, 1)])
def test_rolling_output_shapes(n, w):
    rng = np.random.default_rng(n + w)
    T = rng.standard_normal(n)
    mean, sd = rolling_avg_sd(T, w)
    assert mean.shape == (n - w + 1,)
    assert sd.shape == (n - w + 1,)


def test_rolling_returns_tuple_of_two_arrays():
    result = rolling_avg_sd(np.arange(8.0), 3)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_rolling_window_equal_to_length_gives_single_value():
    T = np.array([2.0, 4.0, 6.0, 8.0])
    mean, sd = rolling_avg_sd(T, 4)
    assert mean.shape == (1,)
    assert mean[0] == pytest.approx(5.0)
    assert sd[0] == pytest.approx(np.std(T))


def test_rolling_window_one_returns_series_and_zero_sd():
    T = np.array([3.0, -1.0, 2.5, 7.0])
    mean, sd = rolling_avg_sd(T, 1)
    np.testing.assert_allclose(mean, T, atol=1e-12)
    np.testing.assert_allclose(sd, np.zeros_like(T), atol=1e-12)


def test_rolling_constant_series_has_exactly_zero_sd():
    T = np.full(10, 3.0)
    mean, sd = rolling_avg_sd(T, 4)
    np.testing.assert_array_equal(mean, np.full(7, 3.0))
    np.testing.assert_array_equal(sd, np.zeros(7))


def test_rolling_sd_is_finite_and_nonnegative():
    rng = np.random.default_rng(11)
    T = 1e6 + 1e-3 * rng.standard_normal(50)
    _, sd = rolling_avg_sd(T, 8)
    assert np.all(np.isfinite(sd))
    assert np.all(sd >= 0)


def test_rolling_raises_when_window_larger_than_series():
    with pytest.raises(ValueError):
        rolling_avg_sd(np.arange(4.0), 5)


def test_rolling_does_not_modify_input():
    T = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    T_copy = T.copy()
    rolling_avg_sd(T, 3)
    np.testing.assert_array_equal(T, T_copy)


# ---------------------------------------------------------------------------
# calculate_distance_profile
# ---------------------------------------------------------------------------

def _distance_inputs(T, Q):
    m = len(Q)
    QT = _naive_dot(Q, T)
    mean_T, sigma_T = _naive_rolling(T, m)
    mean_Q = Q.mean()
    sigma_Q = Q.std()
    return QT, m, mean_Q, sigma_Q, mean_T, sigma_T


def test_dp_matches_naive_z_normalized_distance():
    rng = np.random.default_rng(20)
    T = rng.standard_normal(30)
    Q = rng.standard_normal(6)
    QT, m, mq, sq, mt, st = _distance_inputs(T, Q)
    D = calculate_distance_profile(QT, m, mq, sq, mt, st)
    np.testing.assert_allclose(D, _naive_distance_profile(Q, T), atol=1e-6)


def test_dp_identical_subsequence_has_zero_distance():
    rng = np.random.default_rng(21)
    T = rng.standard_normal(25)
    m = 5
    k = 9
    Q = T[k:k + m].copy()
    QT, m, mq, sq, mt, st = _distance_inputs(T, Q)
    D = calculate_distance_profile(QT, m, mq, sq, mt, st)
    assert D[k] == pytest.approx(0.0, abs=1e-6)


def test_dp_output_shape_matches_qt():
    rng = np.random.default_rng(22)
    T = rng.standard_normal(20)
    Q = rng.standard_normal(4)
    QT, m, mq, sq, mt, st = _distance_inputs(T, Q)
    D = calculate_distance_profile(QT, m, mq, sq, mt, st)
    assert D.shape == (len(T) - len(Q) + 1,)


def test_dp_mismatched_qt_length_raises():
    mean_T = np.zeros(5)
    sigma_T = np.ones(5)
    with pytest.raises(ValueError):
        calculate_distance_profile(np.zeros(4), 2, 0.0, 1.0, mean_T, sigma_T)


def test_dp_mismatched_sigma_T_length_raises():
    with pytest.raises(ValueError):
        calculate_distance_profile(np.zeros(5), 2, 0.0, 1.0, np.zeros(5), np.ones(4))


def test_dp_mismatched_array_query_stats_raises():
    # mean_Q as an array whose length differs from mean_T should be rejected.
    with pytest.raises(ValueError):
        calculate_distance_profile(
            np.zeros(5), 2, np.zeros(3), np.ones(3), np.zeros(5), np.ones(5)
        )


def test_dp_scalar_query_stats_are_accepted():
    rng = np.random.default_rng(23)
    T = rng.standard_normal(15)
    m = 4
    QT = rng.standard_normal(len(T) - m + 1)
    mean_T, sigma_T = _naive_rolling(T, m)
    D = calculate_distance_profile(QT, m, 0.1, 1.3, mean_T, sigma_T)
    assert D.shape == QT.shape


def test_dp_constant_query_gives_sqrt_m_everywhere():
    m = 4
    mean_T = np.array([0.0, 1.0, 2.0])
    sigma_T = np.array([1.0, 2.0, 0.5])
    D = calculate_distance_profile(np.zeros(3), m, 5.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.full(3, np.sqrt(m)), atol=1e-12)


def test_dp_constant_subsequence_gives_sqrt_m_at_that_index():
    m = 4
    mean_T = np.array([0.0, 1.0, 2.0])
    sigma_T = np.array([1.0, 0.0, 0.5])
    QT = np.array([3.0, 3.0, 3.0])
    D = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert D[1] == pytest.approx(np.sqrt(m))


def test_dp_constant_query_and_subsequence_gives_zero():
    m = 4
    mean_T = np.array([0.0, 1.0])
    sigma_T = np.array([0.0, 2.0])
    D = calculate_distance_profile(np.zeros(2), m, 3.0, 0.0, mean_T, sigma_T)
    assert D[0] == pytest.approx(0.0)
    assert D[1] == pytest.approx(np.sqrt(m))


def test_dp_epsilon_treats_small_sigma_as_constant():
    m = 9
    mean_T = np.array([0.0, 1.0, 2.0])
    sigma_T = np.array([1.0, 2.0, 3.0])
    # sigma_Q = 0.5 is below the custom epsilon of 1.0, so the query counts as constant.
    D = calculate_distance_profile(np.zeros(3), m, 0.0, 0.5, mean_T, sigma_T, epsilon=1.0)
    np.testing.assert_allclose(D, np.full(3, np.sqrt(m)), atol=1e-12)


def test_dp_no_runtime_warnings_with_constant_inputs():
    m = 4
    mean_T = np.array([0.0, 1.0])
    sigma_T = np.array([0.0, 2.0])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        calculate_distance_profile(np.zeros(2), m, 3.0, 1.0, mean_T, sigma_T)
    runtime = [w for w in record if issubclass(w.category, RuntimeWarning)]
    assert runtime == []


def test_dp_values_are_nonnegative_and_finite():
    rng = np.random.default_rng(24)
    T = rng.standard_normal(40)
    Q = rng.standard_normal(7)
    QT, m, mq, sq, mt, st = _distance_inputs(T, Q)
    D = calculate_distance_profile(QT, m, mq, sq, mt, st)
    assert np.all(np.isfinite(D))
    assert np.all(D >= 0)
