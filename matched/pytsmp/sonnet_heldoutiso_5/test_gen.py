import numpy as np
import pytest

from pytsmp import utils


def _naive_rolling(t, w):
    means = np.array([np.mean(t[i:i + w]) for i in range(len(t) - w + 1)])
    sds = np.array([np.std(t[i:i + w]) for i in range(len(t) - w + 1)])
    return means, sds


def _naive_distance_profile(Q, T):
    m = len(Q)
    Qn = (Q - np.mean(Q)) / np.std(Q)
    out = []
    for i in range(len(T) - m + 1):
        sub = T[i:i + m]
        subn = (sub - np.mean(sub)) / np.std(sub)
        out.append(np.linalg.norm(Qn - subn))
    return np.array(out)


# ---------------- sliding_dot_product ----------------

@pytest.mark.parametrize("n,m", [(10, 3), (11, 3), (16, 16), (15, 15), (20, 1), (9, 8)])
def test_sliding_dot_product_matches_naive(n, m):
    rng = np.random.RandomState(n * 100 + m)
    T = rng.randn(n)
    Q = rng.randn(m)
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([np.dot(Q, T[i:i + m]) for i in range(n - m + 1)])
    assert result.shape == (n - m + 1,)
    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_sliding_dot_product_known_values():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 1.0, 1.0, 3.0])
    result = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, [3.0, 3.0, 7.0], atol=1e-9)


def test_sliding_dot_product_query_longer_raises():
    with pytest.raises(ValueError):
        utils.sliding_dot_product(np.arange(5.0), np.arange(4.0))


def test_sliding_dot_product_equal_length_single_value():
    Q = np.array([1.0, -2.0, 3.0])
    T = np.array([4.0, 5.0, 6.0])
    result = utils.sliding_dot_product(Q, T)
    assert len(result) == 1
    assert result[0] == pytest.approx(12.0)


# ---------------- rolling_avg_sd ----------------

def test_rolling_avg_sd_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    mean, sd = utils.rolling_avg_sd(t, 2)
    np.testing.assert_allclose(mean, [1.5, 2.5, 3.5])
    np.testing.assert_allclose(sd, [0.5, 0.5, 0.5])


@pytest.mark.parametrize("n,w", [(20, 5), (21, 4), (10, 1), (7, 7)])
def test_rolling_avg_sd_matches_naive(n, w):
    rng = np.random.RandomState(n + w)
    t = rng.randn(n)
    mean, sd = utils.rolling_avg_sd(t, w)
    exp_mean, exp_sd = _naive_rolling(t, w)
    assert mean.shape == (n - w + 1,)
    assert sd.shape == (n - w + 1,)
    np.testing.assert_allclose(mean, exp_mean, atol=1e-8)
    np.testing.assert_allclose(sd, exp_sd, atol=1e-6)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(10, 3.0)
    mean, sd = utils.rolling_avg_sd(t, 4)
    np.testing.assert_allclose(mean, 3.0)
    np.testing.assert_array_equal(sd, np.zeros(7))


def test_rolling_avg_sd_window_too_large_raises():
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(np.arange(3.0), 4)


def test_rolling_avg_sd_sd_never_negative_or_nan():
    t = np.array([1e3 + 0.1] * 5 + [1e3 + 0.2] * 5)
    _, sd = utils.rolling_avg_sd(t, 3)
    assert not np.any(np.isnan(sd))
    assert np.all(sd >= 0)


# ---------------- calculate_distance_profile ----------------

def test_distance_profile_matches_naive_znorm_distance():
    rng = np.random.RandomState(1)
    T = rng.randn(40)
    Q = rng.randn(6)
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    np.testing.assert_allclose(D, _naive_distance_profile(Q, T), atol=1e-5)


def test_distance_profile_self_match_is_zero():
    rng = np.random.RandomState(2)
    T = rng.randn(30)
    m = 5
    Q = T[10:10 + m].copy()
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[10] == pytest.approx(0, abs=1e-5)
    assert np.all(D >= 0)


def test_distance_profile_anticorrelated_is_two_sqrt_m():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([3.0, 2.0, 1.0])
    m = 3
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[0] == pytest.approx(2 * np.sqrt(3), abs=1e-6)


def test_distance_profile_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(3), 4, 0.0, 1.0, np.zeros(2), np.ones(2))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(3), 4, 0.0, 1.0, np.zeros(3), np.ones(2))


def test_distance_profile_array_query_stats_mismatch_raises():
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(3), 4, np.zeros(2), np.ones(2),
                                         np.zeros(3), np.ones(3))


def test_distance_profile_constant_subsequences():
    m = 4
    QT = np.array([0.0, 0.0, 0.0])
    mean_T = np.zeros(3)
    sigma_T = np.array([0.0, 1.0, 1e-9])
    # Non-constant query: constant subsequences are at distance sqrt(m).
    D = utils.calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert D[0] == pytest.approx(np.sqrt(m))
    assert D[2] == pytest.approx(np.sqrt(m))
    assert D[1] == pytest.approx(np.sqrt(2 * m))
    # Constant query: distance 0 to constant subsequences, sqrt(m) to others.
    D = utils.calculate_distance_profile(QT, m, 0.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, [0.0, np.sqrt(m), 0.0])


def test_distance_profile_custom_epsilon():
    m = 4
    QT = np.zeros(2)
    mean_T = np.zeros(2)
    sigma_T = np.array([0.01, 1.0])
    D_default = utils.calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert D_default[0] == pytest.approx(np.sqrt(2 * m))
    D_eps = utils.calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T, epsilon=0.1)
    assert D_eps[0] == pytest.approx(np.sqrt(m))


def test_distance_profile_clamps_negative_to_zero():
    D = utils.calculate_distance_profile(np.array([100.0]), 4, 0.0, 1.0,
                                         np.array([0.0]), np.array([1.0]))
    assert D[0] == 0
    assert not np.isnan(D[0])
