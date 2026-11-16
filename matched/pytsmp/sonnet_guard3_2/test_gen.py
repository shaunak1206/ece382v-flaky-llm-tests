import numpy as np
import pytest

from pytsmp import utils


def _naive_sliding_dot(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _naive_znorm_dist(Q, T):
    m = len(Q)
    zq = (Q - Q.mean()) / Q.std()
    out = []
    for i in range(len(T) - m + 1):
        s = T[i:i + m]
        zs = (s - s.mean()) / s.std()
        out.append(np.linalg.norm(zq - zs))
    return np.array(out)


# ---------------------------- sliding_dot_product ----------------------------

def test_sliding_dot_product_known_values():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    res = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(res, [5.0, 8.0, 11.0], atol=1e-8)


def test_sliding_dot_product_matches_naive_even_length():
    np.random.seed(0)
    T = np.random.randn(32)
    Q = np.random.randn(5)
    res = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(res, _naive_sliding_dot(Q, T), atol=1e-8)


def test_sliding_dot_product_matches_naive_odd_length():
    np.random.seed(1)
    T = np.random.randn(31)
    Q = np.random.randn(6)
    res = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(res, _naive_sliding_dot(Q, T), atol=1e-8)


def test_sliding_dot_product_output_length():
    np.random.seed(2)
    for n, m in [(10, 3), (11, 3), (20, 20), (21, 21), (9, 1)]:
        T = np.random.randn(n)
        Q = np.random.randn(m)
        assert len(utils.sliding_dot_product(Q, T)) == n - m + 1


def test_sliding_dot_product_equal_length_is_single_dot():
    np.random.seed(3)
    for n in (8, 9):
        T = np.random.randn(n)
        Q = np.random.randn(n)
        res = utils.sliding_dot_product(Q, T)
        assert res.shape == (1,)
        np.testing.assert_allclose(res[0], np.dot(Q, T), atol=1e-8)


def test_sliding_dot_product_query_of_length_one():
    np.random.seed(4)
    T = np.random.randn(15)
    Q = np.array([2.5])
    res = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(res, 2.5 * T, atol=1e-8)


def test_sliding_dot_product_query_longer_raises():
    with pytest.raises(ValueError):
        utils.sliding_dot_product(np.arange(5.0), np.arange(4.0))


def test_sliding_dot_product_does_not_modify_inputs():
    np.random.seed(5)
    T = np.random.randn(12)
    Q = np.random.randn(4)
    T_copy, Q_copy = T.copy(), Q.copy()
    utils.sliding_dot_product(Q, T)
    np.testing.assert_array_equal(T, T_copy)
    np.testing.assert_array_equal(Q, Q_copy)


# ------------------------------- rolling_avg_sd -------------------------------

def test_rolling_avg_sd_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = utils.rolling_avg_sd(t, 2)
    np.testing.assert_allclose(mean, [1.5, 2.5, 3.5], atol=1e-10)
    np.testing.assert_allclose(sd, [0.5, 0.5, 0.5], atol=1e-6)


def test_rolling_avg_sd_matches_numpy():
    np.random.seed(6)
    t = np.random.randn(40)
    w = 7
    mean, sd = utils.rolling_avg_sd(t, w)
    exp_mean = np.array([t[i:i + w].mean() for i in range(len(t) - w + 1)])
    exp_sd = np.array([t[i:i + w].std() for i in range(len(t) - w + 1)])
    assert mean.shape == (len(t) - w + 1,)
    assert sd.shape == (len(t) - w + 1,)
    np.testing.assert_allclose(mean, exp_mean, atol=1e-8)
    np.testing.assert_allclose(sd, exp_sd, atol=1e-6)


def test_rolling_avg_sd_window_one():
    t = np.array([3.0, -1.0, 4.0, 1.5])
    mean, sd = utils.rolling_avg_sd(t, 1)
    np.testing.assert_allclose(mean, t, atol=1e-10)
    np.testing.assert_allclose(sd, np.zeros(4), atol=1e-6)


def test_rolling_avg_sd_window_equals_length():
    np.random.seed(7)
    t = np.random.randn(10)
    mean, sd = utils.rolling_avg_sd(t, 10)
    assert mean.shape == (1,)
    np.testing.assert_allclose(mean[0], t.mean(), atol=1e-8)
    np.testing.assert_allclose(sd[0], t.std(), atol=1e-6)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(12, 5.0)
    mean, sd = utils.rolling_avg_sd(t, 4)
    np.testing.assert_allclose(mean, np.full(9, 5.0), atol=1e-10)
    np.testing.assert_allclose(sd, np.zeros(9), atol=1e-6)


def test_rolling_avg_sd_sd_nonnegative():
    np.random.seed(8)
    t = np.random.randn(50) * 10
    _, sd = utils.rolling_avg_sd(t, 5)
    assert np.all(sd >= 0)
    assert not np.any(np.isnan(sd))


def test_rolling_avg_sd_window_too_large_raises():
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(np.arange(3.0), 4)


# -------------------------- calculate_distance_profile --------------------------

def test_distance_profile_matches_naive():
    np.random.seed(9)
    T = np.random.randn(30)
    Q = np.random.randn(6)
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, Q.mean(), Q.std(), mean_T, sigma_T)
    assert D.shape == (len(T) - m + 1,)
    np.testing.assert_allclose(D, _naive_znorm_dist(Q, T), atol=1e-5)


def test_distance_profile_zero_for_scaled_shifted_match():
    np.random.seed(10)
    T = np.random.randn(25)
    m = 5
    Q = 2.0 * T[7:7 + m] + 3.0
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, Q.mean(), Q.std(), mean_T, sigma_T)
    assert D[7] == pytest.approx(0.0, abs=1e-5)
    assert np.all(D >= 0)


def test_distance_profile_anticorrelated_gives_two_sqrt_m():
    np.random.seed(11)
    T = np.random.randn(20)
    m = 4
    Q = -T[3:3 + m]
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, Q.mean(), Q.std(), mean_T, sigma_T)
    assert D[3] == pytest.approx(2 * np.sqrt(m), abs=1e-5)


def test_distance_profile_constant_query_and_constant_subsequence():
    m = 4
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([0.0, 1.0, 2.0])
    sigma_T = np.array([0.0, 0.0, 1.0])
    D = utils.calculate_distance_profile(QT, m, 1.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, [0.0, 0.0, np.sqrt(m)], atol=1e-12)


def test_distance_profile_constant_subsequence_with_nonconstant_query():
    m = 9
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([0.0, 1.0, 2.0])
    sigma_T = np.array([0.0, 1.0, 1e-9])
    D = utils.calculate_distance_profile(QT, m, 0.5, 1.0, mean_T, sigma_T)
    assert D[0] == pytest.approx(3.0, abs=1e-12)
    assert D[2] == pytest.approx(3.0, abs=1e-12)
    assert np.isfinite(D[1])


def test_distance_profile_custom_epsilon():
    m = 4
    QT = np.array([1.0, 2.0])
    mean_T = np.array([0.0, 0.0])
    sigma_T = np.array([0.05, 0.05])
    # With a large epsilon, both are treated as constant -> distance 0 for constant query.
    D = utils.calculate_distance_profile(QT, m, 0.0, 0.01, mean_T, sigma_T, epsilon=0.1)
    np.testing.assert_allclose(D, [0.0, 0.0], atol=1e-12)


def test_distance_profile_dimension_mismatch_raises():
    QT = np.zeros(3)
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, 0.0, 1.0, np.zeros(2), np.ones(3))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, 0.0, 1.0, np.zeros(3), np.ones(2))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, np.zeros(2), np.ones(2), np.zeros(3), np.ones(3))


def test_distance_profile_array_query_stats():
    np.random.seed(12)
    T = np.random.randn(20)
    m = 4
    Q = T[5:5 + m].copy()
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    n_sub = len(T) - m + 1
    mean_Q = np.full(n_sub, Q.mean())
    sigma_Q = np.full(n_sub, Q.std())
    D_arr = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    D_scalar = utils.calculate_distance_profile(QT, m, Q.mean(), Q.std(), mean_T, sigma_T)
    np.testing.assert_allclose(D_arr, D_scalar, atol=1e-10)
