import numpy as np
import pytest

from pytsmp import utils


def naive_dot(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def naive_distance_profile(Q, T):
    m = len(Q)
    zq = (Q - np.mean(Q)) / np.std(Q)
    out = []
    for i in range(len(T) - m + 1):
        sub = T[i:i + m]
        zs = (sub - np.mean(sub)) / np.std(sub)
        out.append(np.linalg.norm(zq - zs))
    return np.array(out)


# ---------------------------- sliding_dot_product ----------------------------

@pytest.mark.parametrize("n,m", [(10, 3), (11, 3), (16, 4), (15, 6), (7, 1)])
def test_sliding_dot_product_matches_naive(n, m):
    rng = np.random.RandomState(n * 100 + m)
    T = rng.randn(n)
    Q = rng.randn(m)
    result = utils.sliding_dot_product(Q, T)
    assert np.allclose(result, naive_dot(Q, T))


def test_sliding_dot_product_output_length():
    T = np.arange(20, dtype=float)
    Q = np.arange(5, dtype=float)
    assert utils.sliding_dot_product(Q, T).shape == (16,)


def test_sliding_dot_product_known_values():
    T = np.array([1.0, 2.0, 3.0, 4.0])
    Q = np.array([1.0, 1.0])
    result = utils.sliding_dot_product(Q, T)
    assert np.allclose(result, [3.0, 5.0, 7.0])


def test_sliding_dot_product_equal_length_gives_single_value():
    rng = np.random.RandomState(1)
    for n in (6, 7):
        T = rng.randn(n)
        Q = rng.randn(n)
        result = utils.sliding_dot_product(Q, T)
        assert result.shape == (1,)
        assert np.allclose(result, np.dot(Q, T))


def test_sliding_dot_product_query_longer_raises():
    with pytest.raises(ValueError):
        utils.sliding_dot_product(np.arange(5.0), np.arange(4.0))


def test_sliding_dot_product_single_element_query_scales_series():
    T = np.array([1.0, -2.0, 3.0, 0.5, 4.0])
    result = utils.sliding_dot_product(np.array([2.0]), T)
    assert np.allclose(result, 2.0 * T)


# ------------------------------ rolling_avg_sd ------------------------------

def test_rolling_avg_sd_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, sd = utils.rolling_avg_sd(t, 3)
    assert np.allclose(mean, [2.0, 3.0, 4.0])
    assert np.allclose(sd, np.sqrt(2.0 / 3.0))


@pytest.mark.parametrize("n,w", [(12, 4), (13, 5), (9, 9), (10, 2)])
def test_rolling_avg_sd_matches_naive(n, w):
    rng = np.random.RandomState(n + w)
    t = rng.randn(n)
    mean, sd = utils.rolling_avg_sd(t, w)
    exp_mean = np.array([np.mean(t[i:i + w]) for i in range(n - w + 1)])
    exp_sd = np.array([np.std(t[i:i + w]) for i in range(n - w + 1)])
    assert mean.shape == (n - w + 1,)
    assert sd.shape == (n - w + 1,)
    assert np.allclose(mean, exp_mean)
    assert np.allclose(sd, exp_sd, atol=1e-6)


def test_rolling_avg_sd_window_one():
    t = np.array([3.0, -1.0, 4.0, 1.5])
    mean, sd = utils.rolling_avg_sd(t, 1)
    assert np.allclose(mean, t)
    assert np.allclose(sd, 0)


def test_rolling_avg_sd_constant_series_has_zero_sd_and_no_nan():
    t = np.full(10, 2.5)
    mean, sd = utils.rolling_avg_sd(t, 4)
    assert np.allclose(mean, 2.5)
    assert np.all(sd >= 0)
    assert not np.any(np.isnan(sd))
    assert np.allclose(sd, 0, atol=1e-6)


def test_rolling_avg_sd_window_larger_than_series_raises():
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(np.arange(3.0), 4)


# ------------------------ calculate_distance_profile ------------------------

def test_distance_profile_known_values():
    Q = np.array([-1.0, 1.0])
    T = np.array([-1.0, 1.0, -1.0])
    m = 2
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert np.allclose(D, [0.0, np.sqrt(8)], atol=1e-6)


def test_distance_profile_matches_naive_z_normalized_distance():
    rng = np.random.RandomState(42)
    T = rng.randn(40)
    Q = rng.randn(6)
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D.shape == (len(T) - m + 1,)
    assert np.allclose(D, naive_distance_profile(Q, T), atol=1e-5)


def test_distance_profile_subsequence_has_zero_distance():
    rng = np.random.RandomState(7)
    T = rng.randn(30)
    Q = 3.0 * T[10:16] + 5.0  # scaled and shifted copy; z-normalization removes this
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert D[10] == pytest.approx(0, abs=1e-4)
    assert np.all(D >= 0)


def test_distance_profile_constant_subsequences_get_sqrt_m():
    m = 4
    QT = np.zeros(2)
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([0.0, 0.0])
    D = utils.calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert np.allclose(D, np.sqrt(m))


def test_distance_profile_constant_query_cases():
    m = 9
    QT = np.zeros(2)
    mean_T = np.array([0.0, 0.0])
    sigma_T = np.array([0.0, 1.0])
    D = utils.calculate_distance_profile(QT, m, 1.0, 0.0, mean_T, sigma_T)
    # both constant -> 0; only query constant -> sqrt(m)
    assert np.allclose(D, [0.0, 3.0])


def test_distance_profile_epsilon_threshold():
    m = 4
    QT = np.zeros(1)
    mean_T = np.zeros(1)
    sigma_T = np.array([1e-9])
    # default epsilon 1e-7: T subsequence treated as constant, query not -> sqrt(m)
    D = utils.calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert np.allclose(D, 2.0)
    # both treated as constant with a larger epsilon -> 0
    D2 = utils.calculate_distance_profile(QT, m, 0.0, 1e-9, mean_T, sigma_T, epsilon=1e-3)
    assert np.allclose(D2, 0.0)


def test_distance_profile_dimension_mismatch_raises():
    QT = np.zeros(3)
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, 0.0, 1.0, np.zeros(2), np.ones(3))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, 0.0, 1.0, np.zeros(3), np.ones(2))
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(np.zeros(2), 4, 0.0, 1.0, np.zeros(3), np.ones(3))


def test_distance_profile_array_query_stats_mismatch_raises():
    QT = np.zeros(3)
    mean_T = np.zeros(3)
    sigma_T = np.ones(3)
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, 4, np.zeros(2), np.ones(2), mean_T, sigma_T)


def test_distance_profile_never_negative_or_nan():
    rng = np.random.RandomState(3)
    T = rng.randn(50)
    Q = T[5:13].copy()
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    D = utils.calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)
    assert not np.any(np.isnan(D))
    assert np.all(D >= 0)
