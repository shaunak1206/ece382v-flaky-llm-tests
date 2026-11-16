import numpy as np
import pytest

from pytsmp import utils


def _brute_sliding_dot(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _brute_znorm_distance(Q, T):
    m = len(Q)
    q = (Q - np.mean(Q)) / np.std(Q)
    out = []
    for i in range(len(T) - m + 1):
        w = T[i:i + m]
        w = (w - np.mean(w)) / np.std(w)
        out.append(np.linalg.norm(q - w))
    return np.array(out)


def _profile_inputs(Q, T):
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    return QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T


# ---------------- sliding_dot_product ----------------

def test_sliding_dot_product_matches_bruteforce_even_length():
    rng = np.random.RandomState(0)
    T = rng.randn(40)
    Q = rng.randn(7)
    res = utils.sliding_dot_product(Q, T)
    assert np.allclose(res, _brute_sliding_dot(Q, T))


def test_sliding_dot_product_matches_bruteforce_odd_length():
    rng = np.random.RandomState(1)
    T = rng.randn(41)
    Q = rng.randn(6)
    res = utils.sliding_dot_product(Q, T)
    assert np.allclose(res, _brute_sliding_dot(Q, T))


def test_sliding_dot_product_output_shape():
    T = np.arange(25, dtype=float)
    Q = np.arange(4, dtype=float)
    res = utils.sliding_dot_product(Q, T)
    assert res.shape == (len(T) - len(Q) + 1,)


def test_sliding_dot_product_query_same_length_as_series():
    for n in (8, 9):
        rng = np.random.RandomState(n)
        T = rng.randn(n)
        Q = rng.randn(n)
        res = utils.sliding_dot_product(Q, T)
        assert res.shape == (1,)
        assert np.allclose(res, [np.dot(Q, T)])


def test_sliding_dot_product_query_length_one():
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Q = np.array([3.0])
    res = utils.sliding_dot_product(Q, T)
    assert np.allclose(res, 3.0 * T)


def test_sliding_dot_product_integer_input():
    T = np.array([1, 2, 3, 4, 5, 6])
    Q = np.array([1, 0, -1])
    res = utils.sliding_dot_product(Q, T)
    assert np.allclose(res, [-2, -2, -2, -2])


def test_sliding_dot_product_query_longer_raises():
    with pytest.raises(ValueError):
        utils.sliding_dot_product(np.arange(10.0), np.arange(5.0))


# ---------------- rolling_avg_sd ----------------

def test_rolling_avg_sd_matches_bruteforce():
    rng = np.random.RandomState(2)
    t = rng.randn(50)
    w = 8
    mean, sd = utils.rolling_avg_sd(t, w)
    exp_mean = np.array([np.mean(t[i:i + w]) for i in range(len(t) - w + 1)])
    exp_sd = np.array([np.std(t[i:i + w]) for i in range(len(t) - w + 1)])
    assert np.allclose(mean, exp_mean)
    assert np.allclose(sd, exp_sd, atol=1e-6)


def test_rolling_avg_sd_output_shapes():
    t = np.arange(20, dtype=float)
    mean, sd = utils.rolling_avg_sd(t, 5)
    assert mean.shape == (16,)
    assert sd.shape == (16,)


def test_rolling_avg_sd_window_equals_length():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = utils.rolling_avg_sd(t, 4)
    assert mean.shape == (1,)
    assert np.allclose(mean, [2.5])
    assert np.allclose(sd, [np.std(t)])


def test_rolling_avg_sd_window_one():
    t = np.array([3.0, -1.0, 4.0, 1.0])
    mean, sd = utils.rolling_avg_sd(t, 1)
    assert np.allclose(mean, t)
    assert np.allclose(sd, 0, atol=1e-6)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(12, 5.0)
    mean, sd = utils.rolling_avg_sd(t, 4)
    assert np.allclose(mean, 5.0)
    assert np.all(sd >= 0)
    assert np.allclose(sd, 0, atol=1e-6)


def test_rolling_avg_sd_integer_input():
    t = np.array([1, 2, 3, 4, 5])
    mean, sd = utils.rolling_avg_sd(t, 2)
    assert np.allclose(mean, [1.5, 2.5, 3.5, 4.5])
    assert np.allclose(sd, 0.5)


def test_rolling_avg_sd_window_too_large_raises():
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(np.arange(5.0), 6)


# ---------------- calculate_distance_profile ----------------

def test_distance_profile_matches_bruteforce_znorm_distance():
    rng = np.random.RandomState(3)
    T = rng.randn(60)
    Q = rng.randn(9)
    D = utils.calculate_distance_profile(*_profile_inputs(Q, T))
    assert D.shape == (len(T) - len(Q) + 1,)
    assert np.allclose(D, _brute_znorm_distance(Q, T), atol=1e-5)


def test_distance_profile_zero_for_embedded_query_and_affine_copy():
    rng = np.random.RandomState(4)
    T = rng.randn(40)
    Q = rng.randn(8)
    T[10:18] = 3.0 * Q + 5.0
    D = utils.calculate_distance_profile(*_profile_inputs(Q, T))
    assert D[10] == pytest.approx(0, abs=1e-5)
    assert np.all(D >= 0)


def test_distance_profile_negated_subsequence_has_max_distance():
    rng = np.random.RandomState(5)
    T = rng.randn(30)
    Q = rng.randn(6)
    T[5:11] = -Q
    D = utils.calculate_distance_profile(*_profile_inputs(Q, T))
    assert D[5] == pytest.approx(2 * np.sqrt(len(Q)), abs=1e-5)


def test_distance_profile_constant_subsequence_gives_sqrt_m():
    m = 4
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([0.0, 1.0, 2.0])
    sigma_T = np.array([0.0, 1.0, 0.0])
    D = utils.calculate_distance_profile(QT, m, 0.5, 1.0, mean_T, sigma_T)
    assert D[0] == pytest.approx(np.sqrt(m))
    assert D[2] == pytest.approx(np.sqrt(m))
    assert np.isfinite(D).all()


def test_distance_profile_constant_query_and_constant_subsequence():
    m = 4
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([0.0, 1.0, 2.0])
    sigma_T = np.array([0.0, 1.0, 0.0])
    D = utils.calculate_distance_profile(QT, m, 1.0, 0.0, mean_T, sigma_T)
    assert np.allclose(D, [0.0, np.sqrt(m), 0.0])


def test_distance_profile_dimension_mismatch_raises():
    QT = np.zeros(5)
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 3, 0.0, 1.0, np.zeros(4), np.ones(5))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 3, 0.0, 1.0, np.zeros(5), np.ones(4))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 3, np.zeros(3), np.ones(3), np.zeros(5), np.ones(5))
