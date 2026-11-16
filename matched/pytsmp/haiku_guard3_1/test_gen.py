import numpy as np
import pytest

from pytsmp import utils


def _naive_sliding_dot(Q, T):
    Q = np.asarray(Q, dtype=float)
    T = np.asarray(T, dtype=float)
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _naive_rolling_avg_sd(t, m):
    t = np.asarray(t, dtype=float)
    means = np.array([np.mean(t[i:i + m]) for i in range(len(t) - m + 1)])
    sds = np.array([np.std(t[i:i + m]) for i in range(len(t) - m + 1)])
    return means, sds


def _naive_distance(Q, T):
    Q = np.asarray(Q, dtype=float)
    T = np.asarray(T, dtype=float)
    m = len(Q)
    zq = (Q - Q.mean()) / Q.std()
    out = []
    for i in range(len(T) - m + 1):
        w = T[i:i + m]
        zt = (w - w.mean()) / w.std()
        out.append(np.sqrt(np.sum((zq - zt) ** 2)))
    return np.array(out)


def _series(n):
    idx = np.arange(n, dtype=float)
    return np.sin(idx * 0.7) + 0.3 * np.cos(idx * 1.9)


# ---------------------------- sliding_dot_product ----------------------------

def test_sliding_dot_product_matches_naive_odd_length():
    T = _series(37)
    Q = _series(37)[5:12]
    result = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, _naive_sliding_dot(Q, T), rtol=1e-9, atol=1e-9)


def test_sliding_dot_product_matches_naive_even_length():
    T = _series(40)
    Q = _series(40)[3:10]
    result = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, _naive_sliding_dot(Q, T), rtol=1e-9, atol=1e-9)


def test_sliding_dot_product_output_length():
    T = _series(50)
    Q = _series(50)[:9]
    result = utils.sliding_dot_product(Q, T)
    assert result.shape == (len(T) - len(Q) + 1,)


def test_sliding_dot_product_query_equal_to_series_length():
    T = _series(21)
    result = utils.sliding_dot_product(T, T)
    assert result.shape == (1,)
    np.testing.assert_allclose(result[0], np.dot(T, T), rtol=1e-9)


def test_sliding_dot_product_query_length_one():
    T = np.array([2.0, -1.0, 4.0, 0.5, 3.0])
    Q = np.array([3.0])
    result = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, 3.0 * T, rtol=1e-9, atol=1e-12)


def test_sliding_dot_product_integer_inputs():
    T = np.array([1, 0, 2, 3, 1, 4, 2])
    Q = np.array([1, 2, 3])
    result = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, _naive_sliding_dot(Q, T), rtol=1e-9, atol=1e-9)


def test_sliding_dot_product_raises_when_query_longer():
    T = _series(5)
    Q = _series(6)
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


# ---------------------------- rolling_avg_sd ----------------------------

def test_rolling_avg_sd_matches_naive_odd_length():
    t = _series(33)
    m = 6
    mean, sd = utils.rolling_avg_sd(t, m)
    ref_mean, ref_sd = _naive_rolling_avg_sd(t, m)
    np.testing.assert_allclose(mean, ref_mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(sd, ref_sd, rtol=1e-6, atol=1e-6)


def test_rolling_avg_sd_matches_naive_even_length():
    t = _series(40)
    m = 9
    mean, sd = utils.rolling_avg_sd(t, m)
    ref_mean, ref_sd = _naive_rolling_avg_sd(t, m)
    np.testing.assert_allclose(mean, ref_mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(sd, ref_sd, rtol=1e-6, atol=1e-6)


def test_rolling_avg_sd_output_shapes():
    t = _series(25)
    mean, sd = utils.rolling_avg_sd(t, 4)
    assert mean.shape == (22,)
    assert sd.shape == (22,)


def test_rolling_avg_sd_window_equals_length():
    t = _series(12)
    mean, sd = utils.rolling_avg_sd(t, len(t))
    assert mean.shape == (1,)
    np.testing.assert_allclose(mean[0], np.mean(t), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(sd[0], np.std(t), rtol=1e-6, atol=1e-6)


def test_rolling_avg_sd_window_one():
    t = np.array([1.5, -2.0, 3.25, 0.0])
    mean, sd = utils.rolling_avg_sd(t, 1)
    np.testing.assert_allclose(mean, t, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(sd, np.zeros_like(t), atol=1e-6)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(20, 4.0)
    mean, sd = utils.rolling_avg_sd(t, 5)
    np.testing.assert_allclose(mean, np.full(16, 4.0), rtol=1e-9)
    np.testing.assert_allclose(sd, np.zeros(16), atol=1e-6)


def test_rolling_avg_sd_sd_is_nonnegative():
    t = _series(30) * 1000.0
    _, sd = utils.rolling_avg_sd(t, 7)
    assert np.all(sd >= 0)


def test_rolling_avg_sd_raises_when_window_too_large():
    t = _series(4)
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(t, 5)


# ---------------------------- calculate_distance_profile ----------------------------

def test_distance_profile_matches_naive_znorm_distance():
    T = _series(60)
    Q = T[10:22].copy()
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    np.testing.assert_allclose(D, _naive_distance(Q, T), rtol=1e-5, atol=1e-5)


def test_distance_profile_output_shape():
    T = _series(45)
    Q = _series(45)[:8]
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D.shape == (len(T) - m + 1,)


def test_distance_profile_identical_subsequence_is_zero():
    T = _series(50)
    m = 10
    Q = T[20:30].copy()
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[20] == pytest.approx(0.0, abs=1e-6)


def test_distance_profile_negated_subsequence_is_twice_sqrt_m():
    m = 8
    base = _series(m)
    T = np.concatenate([np.linspace(0, 1, 5), -base])
    Q = base.copy()
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[5] == pytest.approx(2 * np.sqrt(m), rel=1e-5)


def test_distance_profile_constant_query_nonconstant_T_gives_sqrt_m():
    T = _series(30)
    m = 6
    QT = utils.sliding_dot_product(np.ones(m), T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, 1.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.full(len(T) - m + 1, np.sqrt(m)))


def test_distance_profile_both_constant_gives_zero():
    m = 4
    T = np.array([2.0] * 10)
    Q = np.array([2.0] * m)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, 2.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.zeros(len(T) - m + 1))


def test_distance_profile_constant_window_in_T_gives_sqrt_m():
    m = 5
    T = np.concatenate([np.full(5, 3.0), _series(10)])
    Q = _series(m)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[0] == pytest.approx(np.sqrt(m), rel=1e-9)


def test_distance_profile_epsilon_treats_tiny_sd_as_constant():
    m = 4
    T = _series(12)
    Q = _series(m)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    sigma_T = sigma_T.copy()
    sigma_T[3] = 1e-9
    D = utils.calculate_distance_profile(
        QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T, epsilon=1e-7)
    assert D[3] == pytest.approx(np.sqrt(m), rel=1e-9)


def test_distance_profile_raises_on_length_mismatch():
    m = 3
    QT = np.zeros(5)
    mean_T = np.zeros(5)
    sigma_T = np.ones(4)
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
