import numpy as np
import pytest

from pytsmp import utils


def _brute_dot(Q, T):
    m = len(Q)
    n = len(T)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(n - m + 1)])


def _brute_avg_sd(t, w):
    means = np.array([np.mean(t[i:i + w]) for i in range(len(t) - w + 1)])
    sds = np.array([np.std(t[i:i + w]) for i in range(len(t) - w + 1)])
    return means, sds


def _znorm(x):
    return (x - np.mean(x)) / np.std(x)


# ---------------------------------------------------------------- sliding_dot_product

def test_sliding_dot_product_matches_bruteforce_even_length():
    rng = np.random.RandomState(0)
    T = rng.randn(32)
    Q = rng.randn(5)
    result = utils.sliding_dot_product(Q, T)
    assert np.allclose(result, _brute_dot(Q, T), atol=1e-8)


def test_sliding_dot_product_matches_bruteforce_odd_length():
    rng = np.random.RandomState(1)
    T = rng.randn(33)
    Q = rng.randn(6)
    result = utils.sliding_dot_product(Q, T)
    assert np.allclose(result, _brute_dot(Q, T), atol=1e-8)


def test_sliding_dot_product_output_shape():
    rng = np.random.RandomState(2)
    for n, m in [(10, 3), (11, 3), (20, 20), (21, 21), (50, 1)]:
        result = utils.sliding_dot_product(rng.randn(m), rng.randn(n))
        assert result.shape == (n - m + 1,)


def test_sliding_dot_product_query_same_length_as_series():
    Q = np.array([1.0, 2.0, 3.0, 4.0])
    T = np.array([2.0, -1.0, 0.5, 3.0])
    result = utils.sliding_dot_product(Q, T)
    assert result.shape == (1,)
    assert result[0] == pytest.approx(np.dot(Q, T))


def test_sliding_dot_product_known_values():
    Q = np.array([1.0, 1.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    result = utils.sliding_dot_product(Q, T)
    assert np.allclose(result, [3.0, 5.0, 7.0], atol=1e-9)


def test_sliding_dot_product_length_one_query():
    T = np.array([1.0, -2.0, 3.0, 4.0, 5.0])
    Q = np.array([2.0])
    result = utils.sliding_dot_product(Q, T)
    assert np.allclose(result, 2.0 * T, atol=1e-9)


def test_sliding_dot_product_raises_when_query_longer():
    with pytest.raises(ValueError):
        utils.sliding_dot_product(np.arange(6.0), np.arange(5.0))


# ---------------------------------------------------------------- rolling_avg_sd

def test_rolling_avg_sd_matches_bruteforce():
    rng = np.random.RandomState(4)
    t = rng.randn(40)
    for w in (1, 2, 5, 17, 40):
        mean, sd = utils.rolling_avg_sd(t, w)
        exp_mean, exp_sd = _brute_avg_sd(t, w)
        assert np.allclose(mean, exp_mean, atol=1e-6)
        assert np.allclose(sd, exp_sd, atol=1e-6)


def test_rolling_avg_sd_output_shapes():
    t = np.arange(12.0)
    mean, sd = utils.rolling_avg_sd(t, 5)
    assert mean.shape == (8,)
    assert sd.shape == (8,)


def test_rolling_avg_sd_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = utils.rolling_avg_sd(t, 2)
    assert np.allclose(mean, [1.5, 2.5, 3.5])
    assert np.allclose(sd, [0.5, 0.5, 0.5], atol=1e-9)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(10, 3.0)
    mean, sd = utils.rolling_avg_sd(t, 4)
    assert np.allclose(mean, 3.0)
    assert np.allclose(sd, 0.0, atol=1e-6)


def test_rolling_avg_sd_window_one_returns_series_and_zero_sd():
    t = np.array([5.0, -1.0, 2.5, 7.0])
    mean, sd = utils.rolling_avg_sd(t, 1)
    assert np.allclose(mean, t)
    assert np.allclose(sd, 0.0, atol=1e-6)


def test_rolling_avg_sd_sd_never_negative_or_nan():
    rng = np.random.RandomState(5)
    t = rng.randn(60) * 1e-3 + 100.0
    _, sd = utils.rolling_avg_sd(t, 8)
    assert not np.any(np.isnan(sd))
    assert np.all(sd >= 0)


def test_rolling_avg_sd_integer_input():
    t = np.array([1, 2, 3, 4, 5, 6])
    mean, sd = utils.rolling_avg_sd(t, 3)
    assert np.allclose(mean, [2.0, 3.0, 4.0, 5.0])
    assert np.allclose(sd, np.sqrt(2.0 / 3.0), atol=1e-9)


def test_rolling_avg_sd_raises_when_window_too_large():
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(np.arange(5.0), 6)


# ---------------------------------------------------------------- calculate_distance_profile

def _profile_inputs(Q, T):
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    return QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T


def test_calculate_distance_profile_matches_znorm_euclidean():
    rng = np.random.RandomState(6)
    T = rng.randn(30)
    Q = rng.randn(6)
    QT, m, mq, sq, mt, st = _profile_inputs(Q, T)
    D = utils.calculate_distance_profile(QT, m, mq, sq, mt, st)
    expected = np.array([np.linalg.norm(_znorm(Q) - _znorm(T[i:i + m]))
                         for i in range(len(T) - m + 1)])
    assert D.shape == expected.shape
    assert np.allclose(D, expected, atol=1e-4)


def test_calculate_distance_profile_exact_match_is_near_zero():
    rng = np.random.RandomState(7)
    T = rng.randn(25)
    Q = T[4:10].copy()
    QT, m, mq, sq, mt, st = _profile_inputs(Q, T)
    D = utils.calculate_distance_profile(QT, m, mq, sq, mt, st)
    assert D[4] == pytest.approx(0.0, abs=1e-3)
    assert np.all(D >= 0)


def test_calculate_distance_profile_scale_and_shift_invariant():
    rng = np.random.RandomState(8)
    T = rng.randn(25)
    Q = T[2:8].copy()
    QT, m, mq, sq, mt, st = _profile_inputs(3.0 * Q + 10.0, T)
    D = utils.calculate_distance_profile(QT, m, mq, sq, mt, st)
    assert D[2] == pytest.approx(0.0, abs=1e-3)


def test_calculate_distance_profile_constant_query_and_constant_subsequence():
    QT = np.zeros(4)
    m = 5
    mean_T = np.zeros(4)
    sigma_T = np.zeros(4)
    D = utils.calculate_distance_profile(QT, m, 0.0, 0.0, mean_T, sigma_T)
    assert np.allclose(D, 0.0)


def test_calculate_distance_profile_constant_query_nonconstant_subsequence():
    QT = np.ones(3)
    m = 9
    D = utils.calculate_distance_profile(QT, m, 0.0, 0.0, np.zeros(3), np.ones(3))
    assert np.allclose(D, np.sqrt(m))


def test_calculate_distance_profile_nonconstant_query_constant_subsequence():
    QT = np.ones(3)
    m = 16
    sigma_T = np.array([1.0, 0.0, 1.0])
    mean_T = np.zeros(3)
    D = utils.calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert D[1] == pytest.approx(np.sqrt(m))
    assert not np.any(np.isnan(D))


def test_calculate_distance_profile_epsilon_controls_constant_threshold():
    QT = np.array([0.0, 0.0])
    m = 4
    mean_T = np.zeros(2)
    sigma_T = np.array([1e-3, 1e-3])
    # With a large epsilon the subsequences count as constant, and so does the query.
    D_const = utils.calculate_distance_profile(QT, m, 0.0, 1e-3, mean_T, sigma_T, epsilon=1e-2)
    assert np.allclose(D_const, 0.0)
    # With a tiny epsilon they are treated as regular series.
    D_reg = utils.calculate_distance_profile(QT, m, 0.0, 1e-3, mean_T, sigma_T, epsilon=1e-9)
    assert not np.allclose(D_reg, 0.0)


def test_calculate_distance_profile_clamps_negative_to_zero():
    # Correlation above 1 would make the sqrt argument negative; result must be 0, not nan.
    m = 4
    QT = np.array([m * 1.0 * 1.0 * 1.5])
    D = utils.calculate_distance_profile(QT, m, 0.0, 1.0, np.zeros(1), np.ones(1))
    assert D[0] == 0.0


def test_calculate_distance_profile_length_mismatch_raises():
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(4), 3, 0.0, 1.0, np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(3), 3, 0.0, 1.0, np.zeros(3), np.ones(4))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(3), 3, 0.0, 1.0, np.zeros(4), np.ones(4))


def test_calculate_distance_profile_array_query_stats_mismatch_raises():
    QT = np.zeros(3)
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 3, np.zeros(2), np.ones(3), np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 3, np.zeros(3), np.ones(2), np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 3, np.zeros(4), np.ones(4), np.zeros(3), np.ones(3))
