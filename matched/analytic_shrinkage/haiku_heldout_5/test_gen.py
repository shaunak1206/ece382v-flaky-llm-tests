import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _make_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p)


def test_output_shape_is_square_p_by_p():
    data = _make_data(n=100, p=6, seed=1)
    result = shrink_cov(data)
    assert result.shape == (6, 6)


def test_output_is_numpy_array():
    data = _make_data(n=50, p=4, seed=2)
    result = shrink_cov(data)
    assert isinstance(result, np.ndarray)


def test_output_is_symmetric():
    data = _make_data(n=120, p=5, seed=3)
    result = shrink_cov(data)
    np.testing.assert_allclose(result, result.T, rtol=1e-10, atol=1e-12)


def test_output_is_positive_definite_when_n_greater_than_p():
    data = _make_data(n=200, p=10, seed=4)
    result = shrink_cov(data)
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals > 0)


def test_default_demeans_data():
    data = _make_data(n=80, p=4, seed=5) + 10.0
    centered = data - data.mean(axis=0)
    np.testing.assert_allclose(
        shrink_cov(data), shrink_cov(centered, k=1), rtol=1e-10, atol=1e-12
    )


def test_default_equals_explicit_k1_on_already_centered_data():
    data = _make_data(n=80, p=4, seed=6)
    centered = data - data.mean(axis=0)
    np.testing.assert_allclose(
        shrink_cov(centered), shrink_cov(centered, k=1), rtol=1e-10, atol=1e-12
    )


def test_k_zero_does_not_demean():
    data = _make_data(n=100, p=3, seed=7) + 5.0
    default_result = shrink_cov(data)
    no_demean_result = shrink_cov(data, k=0)
    assert not np.allclose(default_result, no_demean_result)


def test_k_affects_effective_sample_size():
    data = _make_data(n=100, p=4, seed=8)
    centered = data - data.mean(axis=0)
    result_k1 = shrink_cov(centered, k=1)
    result_k2 = shrink_cov(centered, k=2)
    assert not np.allclose(result_k1, result_k2)


def test_scale_equivariance():
    data = _make_data(n=150, p=5, seed=9)
    c = 3.0
    scaled = shrink_cov(c * data)
    original = shrink_cov(data)
    np.testing.assert_allclose(scaled, (c ** 2) * original, rtol=1e-6, atol=1e-10)


def test_row_permutation_invariance():
    rng = np.random.RandomState(10)
    data = _make_data(n=100, p=4, seed=10)
    perm = rng.permutation(data.shape[0])
    np.testing.assert_allclose(
        shrink_cov(data), shrink_cov(data[perm]), rtol=1e-8, atol=1e-10
    )


def test_column_permutation_equivariance():
    data = _make_data(n=100, p=4, seed=11)
    perm = np.array([2, 0, 3, 1])
    original = shrink_cov(data)
    permuted = shrink_cov(data[:, perm])
    np.testing.assert_allclose(
        permuted, original[np.ix_(perm, perm)], rtol=1e-8, atol=1e-10
    )


def test_consistent_estimate_for_large_sample():
    true_cov = np.array([
        [2.0, 0.5, 0.0],
        [0.5, 1.0, 0.3],
        [0.0, 0.3, 1.5],
    ])
    rng = np.random.RandomState(12)
    data = rng.multivariate_normal(np.zeros(3), true_cov, size=5000)
    estimate = shrink_cov(data)
    np.testing.assert_allclose(estimate, true_cov, atol=0.25)


def test_high_dimensional_case_p_greater_than_n():
    data = _make_data(n=30, p=50, seed=13)
    result = shrink_cov(data)
    assert result.shape == (50, 50)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, result.T, rtol=1e-8, atol=1e-10)


def test_single_variable_returns_1x1_matrix():
    data = _make_data(n=50, p=1, seed=14)
    result = shrink_cov(data)
    assert result.shape == (1, 1)
    assert np.isfinite(result[0, 0])
    assert result[0, 0] > 0


def test_rejects_one_dimensional_input():
    data = np.arange(30, dtype=float)
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_rejects_three_dimensional_input():
    data = np.zeros((20, 3, 2))
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_rejects_too_small_effective_sample_size_default():
    # With default k=None, effective n = 12 - 1 = 11 < 12
    data = _make_data(n=12, p=3, seed=15)
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_accepts_minimum_effective_sample_size_default():
    # With default k=None, effective n = 13 - 1 = 12, which is allowed
    data = _make_data(n=13, p=3, seed=16)
    result = shrink_cov(data)
    assert result.shape == (3, 3)


def test_k_zero_allows_n_equal_12():
    # With k=0 the effective sample size stays at n = 12
    data = _make_data(n=12, p=3, seed=17)
    result = shrink_cov(data, k=0)
    assert result.shape == (3, 3)


def test_singular_matrix_raises_value_error():
    data = _make_data(n=30, p=3, seed=18)
    data[:, 2] = data[:, 0]  # duplicate column makes the covariance singular
    with pytest.raises(ValueError, match="singular"):
        shrink_cov(data)


def test_input_array_is_not_modified():
    data = _make_data(n=60, p=4, seed=19) + 2.0
    original_copy = data.copy()
    shrink_cov(data)
    np.testing.assert_array_equal(data, original_copy)


def test_repeated_calls_are_deterministic():
    data = _make_data(n=90, p=5, seed=20)
    first = shrink_cov(data)
    second = shrink_cov(data)
    np.testing.assert_array_equal(first, second)
