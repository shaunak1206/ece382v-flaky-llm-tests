import numpy as np
import pytest

from pytsmp.utils import sliding_dot_product, rolling_avg_sd, calculate_distance_profile


def _naive_sliding_dot_product(Q, T):
    m = len(Q)
    n = len(T)
    return np.array([np.dot(Q, T[i:i + m]) for i in range(n - m + 1)], dtype=float)


def _naive_znorm_distance_profile(Q, T):
    m = len(Q)
    Qz = (Q - Q.mean()) / Q.std()
    out = []
    for i in range(len(T) - m + 1):
        sub = T[i:i + m]
        Sz = (sub - sub.mean()) / sub.std()
        out.append(np.linalg.norm(Qz - Sz))
    return np.array(out)


# ---------------------------------------------------------------------------
# sliding_dot_product
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, m", [(10, 3), (11, 4), (7, 7), (20, 1), (33, 8)])
def test_sliding_dot_product_matches_naive(n, m):
    rng = np.random.RandomState(n * 100 + m)
    T = rng.randn(n)
    Q = rng.randn(m)
    np.testing.assert_allclose(
        sliding_dot_product(Q, T), _naive_sliding_dot_product(Q, T), rtol=1e-7, atol=1e-9
    )


def test_sliding_dot_product_known_values():
    Q = np.array([1.0, 2.0])
    T = np.array([1.0, 2.0, 3.0, 4.0])
    # windows: [1,2]->5, [2,3]->8, [3,4]->11
    np.testing.assert_allclose(sliding_dot_product(Q, T), [5.0, 8.0, 11.0], atol=1e-9)


def test_sliding_dot_product_integer_inputs():
    Q = np.array([1, 2, 3])
    T = np.array([2, 0, 1, 4, 5])
    # windows: [2,0,1]->5, [0,1,4]->14, [1,4,5]->24
    np.testing.assert_allclose(sliding_dot_product(Q, T), [5.0, 14.0, 24.0], atol=1e-9)


def test_sliding_dot_product_equal_length_returns_single_dot():
    rng = np.random.RandomState(0)
    Q = rng.randn(9)
    result = sliding_dot_product(Q, Q.copy())
    assert result.shape == (1,)
    assert result[0] == pytest.approx(np.dot(Q, Q), abs=1e-9)


def test_sliding_dot_product_output_length():
    result = sliding_dot_product(np.ones(4), np.ones(13))
    assert result.shape == (10,)


def test_sliding_dot_product_raises_when_query_longer():
    with pytest.raises(ValueError):
        sliding_dot_product(np.ones(5), np.ones(4))


# ---------------------------------------------------------------------------
# rolling_avg_sd
# ---------------------------------------------------------------------------

def test_rolling_avg_sd_matches_numpy_over_windows():
    rng = np.random.RandomState(1)
    t = rng.randn(40)
    w = 7
    means, sds = rolling_avg_sd(t, w)
    expected_means = np.array([np.mean(t[i:i + w]) for i in range(len(t) - w + 1)])
    expected_sds = np.array([np.std(t[i:i + w]) for i in range(len(t) - w + 1)])
    np.testing.assert_allclose(means, expected_means, atol=1e-9)
    np.testing.assert_allclose(sds, expected_sds, atol=1e-7)


def test_rolling_avg_sd_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    means, sds = rolling_avg_sd(t, 2)
    np.testing.assert_allclose(means, [1.5, 2.5, 3.5], atol=1e-9)
    np.testing.assert_allclose(sds, [0.5, 0.5, 0.5], atol=1e-9)


def test_rolling_avg_sd_constant_series():
    t = np.full(10, 3.0)
    means, sds = rolling_avg_sd(t, 4)
    np.testing.assert_allclose(means, np.full(7, 3.0), atol=1e-12)
    np.testing.assert_allclose(sds, np.zeros(7), atol=1e-12)


def test_rolling_avg_sd_window_equals_length():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    means, sds = rolling_avg_sd(t, 5)
    assert means.shape == (1,) and sds.shape == (1,)
    assert means[0] == pytest.approx(3.0, abs=1e-9)
    assert sds[0] == pytest.approx(np.sqrt(2.0), abs=1e-9)


def test_rolling_avg_sd_window_one_gives_series_and_zero_sd():
    rng = np.random.RandomState(2)
    t = rng.randn(12)
    means, sds = rolling_avg_sd(t, 1)
    np.testing.assert_allclose(means, t, atol=1e-9)
    np.testing.assert_allclose(sds, np.zeros_like(t), atol=1e-6)


def test_rolling_avg_sd_output_shape_and_nonnegative_sd():
    rng = np.random.RandomState(3)
    t = rng.randn(25)
    w = 6
    means, sds = rolling_avg_sd(t, w)
    assert means.shape == (len(t) - w + 1,)
    assert sds.shape == (len(t) - w + 1,)
    assert np.all(sds >= 0)


def test_rolling_avg_sd_raises_when_window_too_large():
    with pytest.raises(ValueError):
        rolling_avg_sd(np.ones(3), 4)


# ---------------------------------------------------------------------------
# calculate_distance_profile
# ---------------------------------------------------------------------------

def test_distance_profile_matches_naive_znorm_distance():
    rng = np.random.RandomState(4)
    n, m = 30, 6
    T = rng.randn(n)
    Q = rng.randn(m)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, Q.mean(), Q.std(), mean_T, sigma_T)
    assert D.shape == (n - m + 1,)
    np.testing.assert_allclose(D, _naive_znorm_distance_profile(Q, T), atol=1e-6)


def test_distance_profile_identical_subsequence_is_zero():
    rng = np.random.RandomState(5)
    m = 5
    T = rng.randn(25)
    Q = T[5:5 + m].copy()
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, Q.mean(), Q.std(), mean_T, sigma_T)
    assert D[5] == pytest.approx(0.0, abs=1e-6)


def test_distance_profile_both_constant_is_zero():
    m = 4
    Q = np.full(m, 2.0)
    T = np.full(12, 5.0)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, Q.mean(), Q.std(), mean_T, sigma_T)
    np.testing.assert_allclose(D, np.zeros(len(T) - m + 1), atol=1e-12)


def test_distance_profile_constant_query_gives_sqrt_m():
    rng = np.random.RandomState(6)
    m = 4
    Q = np.full(m, 2.0)
    T = rng.randn(15)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, 2.0, 0.0, mean_T, sigma_T)
    np.testing.assert_allclose(D, np.full(len(T) - m + 1, np.sqrt(m)), atol=1e-12)


def test_distance_profile_constant_subsequence_gives_sqrt_m():
    # Integer-valued series so the rolling statistics of the constant window are exact.
    m = 4
    T = np.array([3.0, -1.0, 4.0, 2.0, 2.0, 2.0, 2.0, 5.0, 0.0, -2.0, 6.0, 1.0])
    Q = np.array([1.0, 0.0, -1.0, 2.0])
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, Q.mean(), Q.std(), mean_T, sigma_T)
    assert D[3] == pytest.approx(np.sqrt(m), abs=1e-12)


def test_distance_profile_epsilon_threshold():
    m = 4
    QT = np.zeros(3)
    mean_T = np.zeros(3)
    sigma_T = np.array([1e-8, 1.0, 1.0])
    # Default epsilon (1e-7): sigma_T[0] is treated as constant -> sqrt(m)
    D_default = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T)
    assert D_default[0] == pytest.approx(np.sqrt(m), abs=1e-12)
    # Smaller epsilon: sigma_T[0] is not constant -> formula gives sqrt(2m)
    D_small = calculate_distance_profile(QT, m, 0.0, 1.0, mean_T, sigma_T, epsilon=1e-9)
    assert D_small[0] == pytest.approx(np.sqrt(2 * m), abs=1e-12)


def test_distance_profile_raises_on_qt_length_mismatch():
    with pytest.raises(ValueError):
        calculate_distance_profile(
            np.zeros(4), 3, 0.0, 1.0, np.zeros(5), np.ones(5)
        )


def test_distance_profile_raises_on_query_stats_array_mismatch():
    with pytest.raises(ValueError):
        calculate_distance_profile(
            np.zeros(5), 4, np.zeros(4), np.ones(3), np.zeros(5), np.ones(5)
        )


def test_distance_profile_output_is_finite_and_nonnegative():
    rng = np.random.RandomState(7)
    m = 8
    T = rng.randn(60)
    Q = rng.randn(m)
    QT = sliding_dot_product(Q, T)
    mean_T, sigma_T = rolling_avg_sd(T, m)
    D = calculate_distance_profile(QT, m, Q.mean(), Q.std(), mean_T, sigma_T)
    assert np.all(np.isfinite(D))
    assert np.all(D >= 0)
