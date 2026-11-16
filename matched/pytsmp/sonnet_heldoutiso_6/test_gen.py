import numpy as np
import pytest

from pytsmp import utils
from pytsmp.utils import (
    sliding_dot_product,
    rolling_avg_sd,
    calculate_distance_profile,
)


def _naive_rolling(t, w):
    means = np.array([np.mean(t[i:i + w]) for i in range(len(t) - w + 1)])
    sds = np.array([np.std(t[i:i + w]) for i in range(len(t) - w + 1)])
    return means, sds


def _naive_znorm_dist(Q, T):
    m = len(Q)
    q = (Q - np.mean(Q)) / np.std(Q)
    out = []
    for i in range(len(T) - m + 1):
        s = T[i:i + m]
        s = (s - np.mean(s)) / np.std(s)
        out.append(np.linalg.norm(q - s))
    return np.array(out)


# ---------------- sliding_dot_product ----------------

def test_sliding_dot_product_matches_naive_even_length():
    rng = np.random.RandomState(0)
    T = rng.rand(20)
    Q = rng.rand(5)
    expected = np.array([np.dot(Q, T[i:i + 5]) for i in range(16)])
    result = sliding_dot_product(Q, T)
    assert np.allclose(result, expected)


def test_sliding_dot_product_matches_naive_odd_length():
    rng = np.random.RandomState(1)
    T = rng.rand(21)
    Q = rng.rand(6)
    expected = np.correlate(T, Q, mode='valid')
    result = sliding_dot_product(Q, T)
    assert np.allclose(result, expected)


def test_sliding_dot_product_output_length():
    T = np.arange(15, dtype=float)
    Q = np.arange(4, dtype=float)
    assert sliding_dot_product(Q, T).shape == (12,)


def test_sliding_dot_product_known_values():
    T = np.array([1., 2., 3., 4., 5.])
    Q = np.array([1., 1.])
    assert np.allclose(sliding_dot_product(Q, T), [3., 5., 7., 9.])


def test_sliding_dot_product_query_same_length_as_series():
    T = np.array([1., 2., 3., 4., 5.])
    Q = np.array([2., 0., 1., 3., -1.])
    result = sliding_dot_product(Q, T)
    assert result.shape == (1,)
    assert np.allclose(result, [np.dot(Q, T)])


def test_sliding_dot_product_query_length_one():
    T = np.array([3., -1., 4., 1., 5., 9., 2.])
    Q = np.array([2.])
    assert np.allclose(sliding_dot_product(Q, T), 2 * T)


def test_sliding_dot_product_query_longer_raises():
    with pytest.raises(ValueError):
        sliding_dot_product(np.arange(6, dtype=float), np.arange(5, dtype=float))


# ---------------- rolling_avg_sd ----------------

def test_rolling_avg_sd_matches_naive():
    rng = np.random.RandomState(2)
    t = rng.rand(30)
    mean, sd = rolling_avg_sd(t, 7)
    exp_mean, exp_sd = _naive_rolling(t, 7)
    assert np.allclose(mean, exp_mean)
    assert np.allclose(sd, exp_sd, atol=1e-6)


def test_rolling_avg_sd_output_shapes():
    t = np.arange(10, dtype=float)
    mean, sd = rolling_avg_sd(t, 4)
    assert mean.shape == (7,)
    assert sd.shape == (7,)


def test_rolling_avg_sd_known_values():
    t = np.array([1., 2., 3., 4.])
    mean, sd = rolling_avg_sd(t, 2)
    assert np.allclose(mean, [1.5, 2.5, 3.5])
    assert np.allclose(sd, [0.5, 0.5, 0.5])


def test_rolling_avg_sd_window_equals_length():
    t = np.array([2., 4., 4., 4., 5., 5., 7., 9.])
    mean, sd = rolling_avg_sd(t, len(t))
    assert np.allclose(mean, [5.0])
    assert np.allclose(sd, [2.0])


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(12, 3.0)
    mean, sd = rolling_avg_sd(t, 5)
    assert np.allclose(mean, 3.0)
    assert np.allclose(sd, 0.0, atol=1e-6)
    assert np.all(sd >= 0)


def test_rolling_avg_sd_window_too_large_raises():
    with pytest.raises(ValueError):
        rolling_avg_sd(np.arange(4, dtype=float), 5)


# ---------------- calculate_distance_profile ----------------

def test_calculate_distance_profile_matches_naive_znorm_distance():
    rng = np.random.RandomState(3)
    T = rng.rand(40)
    Q = rng.rand(8)
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D.shape == (len(T) - m + 1,)
    assert np.allclose(D, _naive_znorm_dist(Q, T), atol=1e-5)


def test_calculate_distance_profile_self_match_is_zero():
    rng = np.random.RandomState(4)
    T = rng.rand(25)
    m = 6
    Q = T[5:5 + m].copy()
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[5] == pytest.approx(0.0, abs=1e-5)
    assert np.all(D >= 0)
    assert not np.any(np.isnan(D))


def test_calculate_distance_profile_anticorrelated_gives_max_distance():
    T = np.array([4., 3., 2., 1.])
    Q = np.array([1., 2., 3., 4.])
    m = 4
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert np.allclose(D, [2 * np.sqrt(m)])


def test_calculate_distance_profile_constant_subsequence_gives_sqrt_m():
    m = 4
    QT = np.array([1., 2., 3.])
    mean_T = np.array([1., 1., 1.])
    sigma_T = np.array([1., 0., 1.])
    D = calculate_distance_profile(QT, m, 0.5, 1.0, mean_T, sigma_T)
    assert D[1] == pytest.approx(np.sqrt(m))
    assert not np.any(np.isnan(D))


def test_calculate_distance_profile_constant_query():
    m = 4
    QT = np.array([1., 2., 3.])
    mean_T = np.array([1., 1., 1.])
    sigma_T = np.array([1., 0., 2.])
    D = calculate_distance_profile(QT, m, 2.0, 0.0, mean_T, sigma_T)
    assert np.allclose(D, [2.0, 0.0, 2.0])


def test_calculate_distance_profile_epsilon_controls_constant_threshold():
    m = 9
    QT = np.array([5., 5.])
    mean_T = np.array([1., 1.])
    sigma_T = np.array([1e-3, 1e-3])
    D = calculate_distance_profile(QT, m, 1.0, 1.0, mean_T, sigma_T, epsilon=1e-2)
    assert np.allclose(D, [3.0, 3.0])


def test_calculate_distance_profile_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        calculate_distance_profile(np.ones(3), 4, 0.0, 1.0, np.ones(3), np.ones(2))
    with pytest.raises(ValueError):
        calculate_distance_profile(np.ones(3), 4, 0.0, 1.0, np.ones(2), np.ones(2))


def test_calculate_distance_profile_array_query_stats_mismatch_raises():
    with pytest.raises(ValueError):
        calculate_distance_profile(np.ones(3), 4, np.zeros(2), np.ones(2),
                                   np.ones(3), np.ones(3))
