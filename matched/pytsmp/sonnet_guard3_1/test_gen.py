import numpy as np
import pytest

from pytsmp import utils


def _naive_sliding_dot(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _naive_rolling(t, w):
    means = np.array([np.mean(t[i:i + w]) for i in range(len(t) - w + 1)])
    sds = np.array([np.std(t[i:i + w]) for i in range(len(t) - w + 1)])
    return means, sds


def _naive_znorm_dist(Q, T):
    m = len(Q)
    zq = (Q - np.mean(Q)) / np.std(Q)
    out = []
    for i in range(len(T) - m + 1):
        sub = T[i:i + m]
        zs = (sub - np.mean(sub)) / np.std(sub)
        out.append(np.linalg.norm(zq - zs))
    return np.array(out)


# ---------------- sliding_dot_product ----------------

@pytest.mark.parametrize("n,m", [(10, 3), (11, 4), (16, 16), (15, 15), (9, 1), (20, 7)])
def test_sliding_dot_product_matches_naive(n, m):
    np.random.seed(0)
    T = np.random.randn(n)
    Q = np.random.randn(m)
    res = utils.sliding_dot_product(Q, T)
    assert res.shape == (n - m + 1,)
    assert np.allclose(res, _naive_sliding_dot(Q, T), atol=1e-8)


def test_sliding_dot_product_known_values():
    np.random.seed(0)
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 0.0, 3.0, 1.0])
    # windows: [1,0] -> 1, [0,3] -> 6, [3,1] -> 5
    res = utils.sliding_dot_product(Q, T)
    assert np.allclose(res, [1.0, 6.0, 5.0])


def test_sliding_dot_product_query_longer_raises():
    np.random.seed(0)
    with pytest.raises(ValueError):
        utils.sliding_dot_product(np.arange(5.0), np.arange(4.0))


def test_sliding_dot_product_integer_input():
    np.random.seed(0)
    T = np.array([1, 2, 3, 4, 5, 6, 7])
    Q = np.array([1, 0, -1])
    res = utils.sliding_dot_product(Q, T)
    assert np.allclose(res, [-2.0] * 5)


def test_sliding_dot_product_does_not_modify_inputs():
    np.random.seed(0)
    T = np.random.randn(9)
    Q = np.random.randn(4)
    T0, Q0 = T.copy(), Q.copy()
    utils.sliding_dot_product(Q, T)
    assert np.array_equal(T, T0)
    assert np.array_equal(Q, Q0)


# ---------------- rolling_avg_sd ----------------

@pytest.mark.parametrize("n,w", [(10, 3), (11, 5), (8, 8), (12, 1)])
def test_rolling_avg_sd_matches_naive(n, w):
    np.random.seed(0)
    t = np.random.randn(n)
    mean, sd = utils.rolling_avg_sd(t, w)
    exp_mean, exp_sd = _naive_rolling(t, w)
    assert mean.shape == (n - w + 1,)
    assert sd.shape == (n - w + 1,)
    assert np.allclose(mean, exp_mean, atol=1e-8)
    assert np.allclose(sd, exp_sd, atol=1e-6)


def test_rolling_avg_sd_known_values():
    np.random.seed(0)
    t = np.array([1.0, 3.0, 5.0, 7.0])
    mean, sd = utils.rolling_avg_sd(t, 2)
    assert np.allclose(mean, [2.0, 4.0, 6.0])
    assert np.allclose(sd, [1.0, 1.0, 1.0])


def test_rolling_avg_sd_constant_series_has_zero_sd():
    np.random.seed(0)
    t = np.full(10, 4.5)
    mean, sd = utils.rolling_avg_sd(t, 4)
    assert np.allclose(mean, 4.5)
    assert np.allclose(sd, 0.0, atol=1e-6)
    assert np.all(sd >= 0)


def test_rolling_avg_sd_window_too_large_raises():
    np.random.seed(0)
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(np.arange(3.0), 4)


# ---------------- calculate_distance_profile ----------------

def test_distance_profile_known_values():
    np.random.seed(0)
    m = 4
    QT = np.array([4.0, 0.0, -4.0])
    mean_T = np.zeros(3)
    sigma_T = np.ones(3)
    D = utils.calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert np.allclose(D, [0.0, np.sqrt(8.0), 4.0])


def test_distance_profile_negative_clamped_to_zero():
    np.random.seed(0)
    m = 4
    QT = np.array([8.0])  # correlation > 1, would give negative under sqrt
    D = utils.calculate_distance_profile(QT, m, 0.0, 1.0, np.zeros(1), np.ones(1))
    assert np.allclose(D, [0.0])
    assert not np.any(np.isnan(D))


def test_distance_profile_matches_naive_pipeline():
    np.random.seed(0)
    T = np.random.randn(40)
    Q = np.random.randn(8)
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D.shape == (len(T) - m + 1,)
    assert np.allclose(D, _naive_znorm_dist(Q, T), atol=1e-5)


def test_distance_profile_self_match_is_near_zero():
    np.random.seed(0)
    T = np.random.randn(30)
    m = 6
    Q = T[10:10 + m].copy()
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[10] == pytest.approx(0.0, abs=1e-4)
    assert np.all(D >= 0)


def test_distance_profile_constant_query():
    np.random.seed(0)
    m = 5
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([0.5, 1.0, 2.0])
    sigma_T = np.array([1.0, 0.0, 2.0])
    D = utils.calculate_distance_profile(QT, m, 3.0, 0.0, mean_T, sigma_T)
    # constant vs non-constant -> sqrt(m); constant vs constant -> 0
    assert np.allclose(D, [np.sqrt(m), 0.0, np.sqrt(m)])


def test_distance_profile_constant_subsequence_in_T():
    np.random.seed(0)
    m = 9
    QT = np.array([1.0, 2.0, 3.0])
    mean_T = np.array([0.5, 1.0, 2.0])
    sigma_T = np.array([0.0, 1e-9, 1.0])
    D = utils.calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert D[0] == pytest.approx(3.0)
    assert D[1] == pytest.approx(3.0)
    assert np.isfinite(D[2])


def test_distance_profile_custom_epsilon():
    np.random.seed(0)
    m = 4
    QT = np.array([0.0])
    mean_T = np.zeros(1)
    sigma_T = np.array([1e-3])
    D_default = utils.calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    D_eps = utils.calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T, epsilon=1e-2)
    assert D_eps[0] == pytest.approx(2.0)  # sqrt(4), treated as constant
    assert D_default[0] == pytest.approx(np.sqrt(8.0))


def test_distance_profile_array_query_stats():
    np.random.seed(0)
    m = 4
    QT = np.array([4.0, 0.0])
    mean_Q = np.zeros(2)
    sigma_Q = np.ones(2)
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, np.zeros(2), np.ones(2))
    assert np.allclose(D, [0.0, np.sqrt(8.0)])


def test_distance_profile_dimension_mismatch_raises():
    np.random.seed(0)
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(3), 4, 0.0, 1.0, np.zeros(3), np.ones(2))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(3), 4, 0.0, 1.0, np.zeros(2), np.ones(2))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(3), 4, np.zeros(2), np.ones(2),
                                         np.zeros(3), np.ones(3))
