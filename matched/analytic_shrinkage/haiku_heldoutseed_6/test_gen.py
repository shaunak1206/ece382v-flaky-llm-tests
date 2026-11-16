import os
import sys

import numpy as np
import pytest

# Make the project root importable when pytest is run from the project root.
_ROOT = os.path.abspath(os.getcwd())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from nonlinshrink.nonlinshrink import shrink_cov  # noqa: E402


def _gaussian(n, p, seed=0):
    return np.random.RandomState(seed).randn(n, p)


def _orthonormal_scaled(n, p, seed=0):
    """Return X (n x p) with X^T X / n == identity."""
    q, _ = np.linalg.qr(np.random.RandomState(seed).randn(n, p))
    return np.sqrt(n) * q


def test_output_shape_is_p_by_p():
    data = _gaussian(60, 5, seed=1)
    result = shrink_cov(data)
    assert result.shape == (5, 5)


def test_output_is_symmetric():
    data = _gaussian(80, 6, seed=2)
    result = shrink_cov(data)
    np.testing.assert_allclose(result, result.T, atol=1e-10)


def test_output_is_positive_definite_when_n_greater_than_p():
    data = _gaussian(100, 5, seed=3)
    result = shrink_cov(data)
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals > 0)


def test_default_demeans_data_so_constant_offset_is_ignored():
    data = _gaussian(70, 4, seed=4)
    np.testing.assert_allclose(shrink_cov(data + 5.0), shrink_cov(data), rtol=1e-8, atol=1e-10)


def test_default_equivalent_to_explicit_k1_on_centered_data():
    data = _gaussian(70, 4, seed=5)
    centered = data - data.mean(axis=0)
    np.testing.assert_allclose(shrink_cov(data), shrink_cov(centered, k=1), rtol=1e-10, atol=1e-12)


def test_k_zero_uses_full_sample_size_and_differs_from_default():
    data = _gaussian(70, 4, seed=6)
    centered = data - data.mean(axis=0)
    default_result = shrink_cov(centered)
    k0_result = shrink_cov(centered, k=0)
    assert not np.allclose(default_result, k0_result, rtol=1e-4, atol=1e-6)


def test_does_not_modify_input_array():
    data = _gaussian(50, 4, seed=7)
    original = data.copy()
    shrink_cov(data)
    shrink_cov(data, k=0)
    np.testing.assert_array_equal(data, original)


def test_row_permutation_invariance():
    data = _gaussian(60, 4, seed=8)
    perm = np.random.RandomState(81).permutation(data.shape[0])
    np.testing.assert_allclose(shrink_cov(data[perm]), shrink_cov(data), rtol=1e-8, atol=1e-10)


def test_column_permutation_equivariance():
    data = _gaussian(60, 4, seed=9)
    perm = np.array([2, 0, 3, 1])
    permuted_result = shrink_cov(data[:, perm])
    expected = shrink_cov(data)[np.ix_(perm, perm)]
    np.testing.assert_allclose(permuted_result, expected, rtol=1e-7, atol=1e-9)


def test_scale_equivariance_quadratic_in_scale_factor():
    data = _gaussian(60, 4, seed=10)
    c = 3.0
    np.testing.assert_allclose(shrink_cov(c * data), c ** 2 * shrink_cov(data), rtol=1e-6, atol=1e-9)


def test_identity_sample_covariance_gives_scalar_multiple_of_identity():
    data = _orthonormal_scaled(50, 4, seed=11)
    result = shrink_cov(data, k=0)
    diag = np.diag(result)
    np.testing.assert_allclose(diag, diag[0] * np.ones(4), rtol=1e-6)
    off_diag = result - np.diag(diag)
    np.testing.assert_allclose(off_diag, np.zeros((4, 4)), atol=1e-6)


def test_large_sample_result_close_to_sample_covariance():
    data = _gaussian(5000, 3, seed=12)
    centered = data - data.mean(axis=0)
    sample_cov = centered.T @ centered / (data.shape[0] - 1)
    result = shrink_cov(data)
    np.testing.assert_allclose(result, sample_cov, atol=0.05)


def test_rejects_one_dimensional_input():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(30, dtype=float))


def test_rejects_three_dimensional_input():
    with pytest.raises(AssertionError):
        shrink_cov(np.zeros((20, 3, 2)))


def test_rejects_effective_sample_size_below_twelve_with_default_demeaning():
    # With default k=None, effective n is 12 - 1 = 11 < 12.
    data = _gaussian(12, 3, seed=13)
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_accepts_minimum_effective_sample_size():
    # Default: effective n = 13 - 1 = 12 is allowed.
    shrink_cov(_gaussian(13, 3, seed=14))
    # k=0: effective n = 12 is allowed.
    shrink_cov(_gaussian(12, 3, seed=15), k=0)


def test_large_k_reduces_effective_sample_size_below_minimum():
    # Effective n = 13 - 2 = 11 < 12.
    data = _gaussian(13, 3, seed=16)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=2)


def test_constant_column_raises_singular_value_error():
    data = _gaussian(40, 3, seed=17)
    data[:, 1] = 7.0  # becomes zero after demeaning -> singular sample covariance
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_duplicate_columns_raise_singular_value_error():
    base = _gaussian(40, 2, seed=18)
    data = np.column_stack([base[:, 0], base[:, 1], base[:, 0]])
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_high_dimensional_case_p_greater_than_n_shape_and_symmetry():
    data = _gaussian(20, 30, seed=19)
    result = shrink_cov(data, k=0)
    assert result.shape == (30, 30)
    np.testing.assert_allclose(result, result.T, atol=1e-8)


def test_high_dimensional_case_output_finite_and_nonnegative_spectrum():
    data = _gaussian(20, 30, seed=20)
    result = shrink_cov(data, k=0)
    assert np.all(np.isfinite(result))
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals > -1e-8 * max(1.0, np.abs(eigvals).max()))


def test_integer_input_is_accepted_and_returns_finite_float_matrix():
    data = np.random.RandomState(21).randint(-10, 10, size=(60, 4))
    result = shrink_cov(data)
    assert result.shape == (4, 4)
    assert np.all(np.isfinite(result))
