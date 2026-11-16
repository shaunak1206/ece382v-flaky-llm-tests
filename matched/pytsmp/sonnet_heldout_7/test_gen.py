import numpy as np
import pytest

from pytsmp import utils


def naive_dot(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def naive_znorm_dist(Q, T):
    m = len(Q)
    zq = (Q - Q.mean()) / Q.std()
    out = []
    for i in range(len(T) - m + 1):
        s = T[i:i + m]
        zs = (s - s.mean()) / s.std()
        out.append(np.linalg.norm(zq - zs))
    return np.array(out)


# ---------------- sliding_dot_product ----------------

def test_sliding_dot_product_matches_naive_even_length():
    rng = np.random.RandomState(0)
    T = rng.randn(40)
    Q = rng.randn(8)
    res = utils.sliding_dot_product(Q, T)
    assert res.shape == (len(T) - len(Q) + 1,)
    assert np.allclose(res, naive_dot(Q, T))


def test_sliding_dot_product_matches_naive_odd_length():
    rng = np.random.RandomState(1)
    T = rng.randn(41)
    Q = rng.randn(7)
    res = utils.sliding_dot_product(Q, T)
    assert res.shape == (len(T) - len(Q) + 1,)
    assert np.allclose(res, naive_dot(Q, T))


def test_sliding_dot_product_known_values():
    T = np.array([1., 2., 3., 4., 5.])
    Q = np.array([1., 1.])
    res = utils.sliding_dot_product(Q, T)
    assert np.allclose(res, [3., 5., 7., 9.])


def test_sliding_dot_product_query_same_length_as_series():
    rng = np.random.RandomState(2)
    for n in (10, 11):
        T = rng.randn(n)
        Q = rng.randn(n)
        res = utils.sliding_dot_product(Q, T)
        assert res.shape == (1,)
        assert np.allclose(res, np.dot(Q, T))


def test_sliding_dot_product_query_longer_raises():
    with pytest.raises(ValueError):
        utils.sliding_dot_product(np.arange(6, dtype=float), np.arange(5, dtype=float))


def test_sliding_dot_product_length_one_query():
    T = np.array([1., -2., 3., 4.])
    Q = np.array([2.])
    res = utils.sliding_dot_product(Q, T)
    assert np.allclose(res, 2 * T)


# ---------------- rolling_avg_sd ----------------

def test_rolling_avg_sd_matches_numpy():
    rng = np.random.RandomState(3)
    t = rng.randn(50)
    w = 7
    mean, sd = utils.rolling_avg_sd(t, w)
    exp_mean = np.array([t[i:i + w].mean() for i in range(len(t) - w + 1)])
    exp_sd = np.array([t[i:i + w].std() for i in range(len(t) - w + 1)])
    assert mean.shape == (len(t) - w + 1,)
    assert sd.shape == (len(t) - w + 1,)
    assert np.allclose(mean, exp_mean)
    assert np.allclose(sd, exp_sd, atol=1e-6)


def test_rolling_avg_sd_known_values():
    t = np.array([1., 2., 3., 4.])
    mean, sd = utils.rolling_avg_sd(t, 2)
    assert np.allclose(mean, [1.5, 2.5, 3.5])
    assert np.allclose(sd, [0.5, 0.5, 0.5])


def test_rolling_avg_sd_window_equals_length():
    t = np.array([2., 4., 6., 8.])
    mean, sd = utils.rolling_avg_sd(t, 4)
    assert mean.shape == (1,)
    assert np.allclose(mean, [5.])
    assert np.allclose(sd, [np.std(t)])


def test_rolling_avg_sd_window_too_large_raises():
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(np.array([1., 2., 3.]), 4)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(10, 5.0)
    mean, sd = utils.rolling_avg_sd(t, 4)
    assert np.allclose(mean, 5.0)
    assert np.allclose(sd, 0.0, atol=1e-6)
    assert np.all(sd >= 0)


def test_rolling_avg_sd_window_one():
    t = np.array([3., -1., 4., 1.])
    mean, sd = utils.rolling_avg_sd(t, 1)
    assert np.allclose(mean, t)
    assert np.allclose(sd, 0.0, atol=1e-6)


# ---------------- calculate_distance_profile ----------------

def test_calculate_distance_profile_matches_naive():
    rng = np.random.RandomState(4)
    T = rng.randn(60)
    Q = rng.randn(9)
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D.shape == (len(T) - m + 1,)
    assert np.allclose(D, naive_znorm_dist(Q, T), atol=1e-5)


def test_calculate_distance_profile_self_match_is_zero():
    rng = np.random.RandomState(5)
    T = rng.randn(30)
    m = 6
    Q = T[10:10 + m].copy()
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[10] == pytest.approx(0, abs=1e-5)
    assert np.all(D >= 0)
    assert not np.any(np.isnan(D))


def test_calculate_distance_profile_dimension_mismatch_raises():
    QT = np.array([1., 2., 3.])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, 0.0, 1.0, np.zeros(2), np.ones(3))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, 0.0, 1.0, np.zeros(3), np.ones(2))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, np.zeros(2), np.ones(2), np.zeros(3), np.ones(3))


def test_calculate_distance_profile_constant_query():
    # Constant query: distance is sqrt(m) to non-constant subsequences, 0 to constant ones.
    T = np.array([1., 2., 4., 3., 5., 5., 5., 5., 1., 7.])
    m = 4
    Q = np.full(m, 3.0)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    expected = np.where(sigma_T < 1e-7, 0.0, np.sqrt(m))
    assert np.allclose(D, expected)
    assert D[4] == pytest.approx(0.0)  # window [5,5,5,5]
    assert D[0] == pytest.approx(np.sqrt(m))


def test_calculate_distance_profile_constant_subsequence_in_series():
    T = np.array([1., 3., 2., 4., 6., 6., 6., 6., 1., 5., 2.])
    Q = np.array([1., 2., 4., 3.])
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    const_idx = 4  # window [6,6,6,6]
    assert D[const_idx] == pytest.approx(np.sqrt(m))
    assert not np.any(np.isnan(D))


def test_calculate_distance_profile_epsilon_controls_constant_threshold():
    QT = np.array([1., 3.])
    mean_T = np.array([0., 0.])
    sigma_T = np.array([1e-3, 1.0])
    D_default = utils.calculate_distance_profile(QT, 4, 0.0, 1.0, mean_T, sigma_T)
    D_eps = utils.calculate_distance_profile(QT, 4, 0.0, 1.0, mean_T, sigma_T, epsilon=1e-2)
    assert D_default[0] == pytest.approx(0.0)
    assert D_eps[0] == pytest.approx(2.0)  # sqrt(m)
    assert D_default[1] == pytest.approx(np.sqrt(2))
    assert D_eps[1] == pytest.approx(np.sqrt(2))


def test_calculate_distance_profile_negative_values_clipped_to_zero():
    # QT large enough to make the correlation exceed 1 numerically; result must be 0, not NaN.
    QT = np.array([100.])
    D = utils.calculate_distance_profile(QT, 4, 0.0, 1.0, np.array([0.]), np.array([1.]))
    assert D[0] == 0.0
    assert not np.isnan(D[0])
