import numpy as np
import pytest

from pytsmp.utils import (
    calculate_distance_profile,
    rolling_avg_sd,
    sliding_dot_product,
)


def _naive_sliding_dot(Q, T):
    m = len(Q)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(len(T) - m + 1)])


def _naive_znorm_distance_profile(Q, T):
    m = len(Q)
    q = (Q - Q.mean()) / Q.std()
    out = []
    for i in range(len(T) - m + 1):
        s = T[i:i + m]
        z = (s - s.mean()) / s.std()
        out.append(np.sqrt(np.sum((q - z) ** 2)))
    return np.array(out)


def _distance_profile(Q, T):
    m = len(Q)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    return calculate_distance_profile(QT, m, np.mean(Q), np.std(Q), mean_T, sigma_T)


# sliding_dot_product

def test_sliding_dot_product_known_values():
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(sliding_dot_product(Q, T), [14.0, 20.0, 26.0], atol=1e-9)


def test_sliding_dot_product_reverses_query():
    # An impulse in T exposes the order in which Q is applied.
    Q = np.array([1.0, 2.0, 3.0])
    T = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(sliding_dot_product(Q, T), [0.0, 3.0, 2.0, 1.0], atol=1e-9)


def test_sliding_dot_product_matches_naive_even_length():
    rng = np.random.RandomState(0)
    T = rng.randn(40)
    Q = rng.randn(7)
    np.testing.assert_allclose(sliding_dot_product(Q, T), _naive_sliding_dot(Q, T), atol=1e-9)


def test_sliding_dot_product_matches_naive_odd_length():
    rng = np.random.RandomState(1)
    T = rng.randn(41)
    Q = rng.randn(6)
    np.testing.assert_allclose(sliding_dot_product(Q, T), _naive_sliding_dot(Q, T), atol=1e-9)


def test_sliding_dot_product_output_length_and_type():
    rng = np.random.RandomState(2)
    for n, m in [(10, 3), (11, 3), (12, 12), (13, 1), (64, 5)]:
        out = sliding_dot_product(rng.randn(m), rng.randn(n))
        assert isinstance(out, np.ndarray)
        assert out.shape == (n - m + 1,)


def test_sliding_dot_product_equal_length_returns_single_dot():
    Q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    T = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    out = sliding_dot_product(Q, T)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(35.0)


def test_sliding_dot_product_raises_when_query_longer():
    with pytest.raises(ValueError):
        sliding_dot_product(np.ones(5), np.ones(4))


# rolling_avg_sd

def test_rolling_avg_sd_known_values():
    t = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    mean, sd = rolling_avg_sd(t, 8)
    np.testing.assert_allclose(mean, [5.0], atol=1e-9)
    np.testing.assert_allclose(sd, [2.0], atol=1e-9)


def test_rolling_avg_sd_matches_numpy_per_window():
    rng = np.random.RandomState(3)
    t = rng.randn(60)
    w = 9
    n_windows = len(t) - w + 1
    mean, sd = rolling_avg_sd(t, w)
    assert mean.shape == (n_windows,)
    assert sd.shape == (n_windows,)
    expected_mean = np.array([t[i:i + w].mean() for i in range(n_windows)])
    expected_sd = np.array([t[i:i + w].std() for i in range(n_windows)])
    np.testing.assert_allclose(mean, expected_mean, atol=1e-9)
    np.testing.assert_allclose(sd, expected_sd, atol=1e-9)


def test_rolling_avg_sd_window_one_returns_series_and_zero_sd():
    t = np.array([1, 5, 2, 8, 3])
    mean, sd = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(mean, t.astype(float))
    np.testing.assert_allclose(sd, np.zeros(5), atol=1e-12)


def test_rolling_avg_sd_constant_series_has_zero_sd():
    t = np.full(12, 3.0)
    mean, sd = rolling_avg_sd(t, 4)
    np.testing.assert_allclose(mean, np.full(9, 3.0))
    np.testing.assert_allclose(sd, np.zeros(9), atol=1e-12)


def test_rolling_avg_sd_raises_when_window_too_large():
    with pytest.raises(ValueError):
        rolling_avg_sd(np.arange(5.0), 6)


def test_rolling_avg_sd_finite_nonnegative_with_large_offset():
    rng = np.random.RandomState(4)
    t = 1e6 + rng.rand(200)
    mean, sd = rolling_avg_sd(t, 10)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(sd))
    assert np.all(sd >= 0.0)


# calculate_distance_profile

def test_distance_profile_identical_subsequence_is_zero():
    rng = np.random.RandomState(5)
    T = rng.randn(100)
    start = 30
    Q = T[start:start + 10].copy()
    D = _distance_profile(Q, T)
    assert D[start] == pytest.approx(0.0, abs=1e-5)
    assert np.argmin(D) == start


def test_distance_profile_matches_bruteforce_znormalized_distance():
    rng = np.random.RandomState(6)
    T = rng.randn(80)
    Q = rng.randn(8)
    D = _distance_profile(Q, T)
    expected = _naive_znorm_distance_profile(Q, T)
    assert D.shape == expected.shape
    np.testing.assert_allclose(D, expected, atol=1e-5)


def test_distance_profile_perfect_correlation_is_zero():
    m = 4
    mean_Q, sigma_Q = 1.0, 2.0
    mean_T, sigma_T = np.array([3.0]), np.array([0.5])
    # Pearson correlation of 1 means QT = m*mu_Q*mu_T + m*sigma_Q*sigma_T = 12 + 4
    QT = np.array([16.0])
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    np.testing.assert_allclose(D, [0.0], atol=1e-12)


def test_distance_profile_perfect_anticorrelation_is_two_sqrt_m():
    m = 4
    mean_Q, sigma_Q = 1.0, 2.0
    mean_T, sigma_T = np.array([3.0]), np.array([0.5])
    # Correlation of -1 means QT = 12 - 4 = 8
    QT = np.array([8.0])
    D = calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    np.testing.assert_allclose(D, [4.0], atol=1e-9)


def test_distance_profile_constant_query_gives_sqrt_m():
    m = 9
    QT = np.array([1.0, -2.0, 5.0])
    mean_T = np.array([0.0, 1.0, -1.0])
    sigma_T = np.array([1.0, 2.0, 0.5])
    D = calculate_distance_profile(QT, m, 4.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.full(3, 3.0), atol=1e-12)


def test_distance_profile_constant_subsequence_gives_sqrt_m_at_that_index():
    m = 4
    QT = np.array([1000.0, 0.0])
    mean_T = np.array([0.0, 0.0])
    sigma_T = np.array([0.0, 1.0])
    D = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    # Index 0 has a constant subsequence, so QT is ignored and the distance is sqrt(m).
    # Index 1 has correlation 0, so the distance is sqrt(2m).
    np.testing.assert_allclose(D, [2.0, np.sqrt(8.0)], atol=1e-12)


def test_distance_profile_both_constant_gives_zero():
    m = 5
    D = calculate_distance_profile(
        np.array([7.0, 7.0]), m, 2.0, 0.0, np.array([1.0, 1.0]), np.array([0.0, 0.0])
    )
    np.testing.assert_allclose(D, [0.0, 0.0], atol=1e-12)


def test_distance_profile_custom_epsilon_treats_small_sd_as_constant():
    m = 4
    QT = np.array([0.0])
    mean_T = np.array([0.0])
    sigma_T = np.array([0.3])
    D_default = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    D_custom = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T, epsilon=0.5)
    assert D_default[0] == pytest.approx(np.sqrt(8.0))
    assert D_custom[0] == pytest.approx(2.0)


def test_distance_profile_clips_negative_radicand_to_zero():
    m = 4
    # Slightly above the perfect-correlation value of 16, so the radicand is negative.
    QT = np.array([16.000001])
    D = calculate_distance_profile(QT, m, 1.0, 2.0, np.array([3.0]), np.array([0.5]))
    assert np.all(np.isfinite(D))
    assert D[0] == 0.0


def test_distance_profile_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        calculate_distance_profile(np.zeros(3), 4, 0.0, 1.0, np.zeros(4), np.ones(4))
    with pytest.raises(ValueError):
        calculate_distance_profile(np.zeros(3), 4, 0.0, 1.0, np.zeros(3), np.ones(2))


def test_distance_profile_raises_when_query_arrays_mismatch():
    QT = np.zeros(3)
    mean_T = np.zeros(3)
    sigma_T = np.ones(3)
    # mean_Q and sigma_Q arrays have different lengths.
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 4, np.zeros(3), np.ones(2), mean_T, sigma_T)
    # mean_Q and sigma_Q agree with each other but not with len(mean_T).
    with pytest.raises(ValueError):
        calculate_distance_profile(QT, 4, np.zeros(2), np.ones(2), mean_T, sigma_T)
