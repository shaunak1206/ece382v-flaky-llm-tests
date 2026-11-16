import os
import sys

import numpy as np
import pytest

# Make the project importable when pytest is invoked from the project root
# or from a tests directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (os.getcwd(), _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nonlinshrink.nonlinshrink import shrink_cov  # noqa: E402


def _data(n, p, seed=0):
    """Return a reproducible n-by-p Gaussian data matrix."""
    return np.random.RandomState(seed).randn(n, p)


def _demean(X):
    return X - X.mean(axis=0)


def test_output_shape_is_p_by_p():
    X = _data(50, 5)
    assert shrink_cov(X).shape == (5, 5)


def test_returns_numpy_ndarray():
    X = _data(50, 4)
    assert isinstance(shrink_cov(X), np.ndarray)


def test_output_is_symmetric():
    X = _data(60, 6, seed=1)
    out = shrink_cov(X)
    np.testing.assert_allclose(out, out.T, rtol=1e-10, atol=1e-12)


def test_output_is_positive_definite_when_p_less_than_n():
    X = _data(100, 5, seed=2)
    out = shrink_cov(X)
    assert np.linalg.eigvalsh(out).min() > 0


def test_does_not_modify_input_array():
    X = _data(40, 4, seed=3)
    X_copy = X.copy()
    shrink_cov(X)
    np.testing.assert_array_equal(X, X_copy)


def test_repeated_calls_are_deterministic():
    X = _data(80, 4, seed=4)
    np.testing.assert_array_equal(shrink_cov(X), shrink_cov(X))


def test_default_demeaning_makes_result_shift_invariant():
    X = _data(70, 4, seed=5)
    np.testing.assert_allclose(shrink_cov(X), shrink_cov(X + 7.5), rtol=1e-8, atol=1e-10)


def test_k_none_matches_k_one_on_already_demeaned_data():
    X = _demean(_data(70, 4, seed=6))
    np.testing.assert_allclose(shrink_cov(X), shrink_cov(X, k=1), rtol=1e-10, atol=1e-12)


def test_k_zero_uses_full_sample_size():
    X = _demean(_data(70, 4, seed=7))
    out_k0 = shrink_cov(X, k=0)
    out_k1 = shrink_cov(X, k=1)
    assert out_k0.shape == out_k1.shape
    assert not np.allclose(out_k0, out_k1)


def test_scale_equivariance():
    X = _data(90, 5, seed=8)
    c = 3.0
    np.testing.assert_allclose(
        shrink_cov(c * X), (c ** 2) * shrink_cov(X), rtol=1e-6, atol=1e-10
    )


def test_column_permutation_equivariance():
    X = _data(90, 5, seed=9)
    perm = [2, 0, 4, 1, 3]
    out = shrink_cov(X)
    out_perm = shrink_cov(X[:, perm])
    np.testing.assert_allclose(out_perm, out[np.ix_(perm, perm)], rtol=1e-8, atol=1e-10)


def test_row_permutation_invariance():
    X = _data(90, 4, seed=10)
    order = np.random.RandomState(11).permutation(X.shape[0])
    np.testing.assert_allclose(shrink_cov(X), shrink_cov(X[order]), rtol=1e-8, atol=1e-10)


def test_preserves_sample_eigenvectors():
    X = _data(120, 5, seed=12)
    Xc = _demean(X)
    n = X.shape[0]
    sample = Xc.T @ Xc / (n - 1)
    _, u = np.linalg.eigh(sample)
    out = shrink_cov(X)
    M = u.T @ out @ u
    off_diag = M - np.diag(np.diag(M))
    scale = max(1.0, np.max(np.abs(M)))
    assert np.max(np.abs(off_diag)) < 1e-8 * scale


def test_high_dimensional_case_p_greater_than_n_shape():
    X = _data(20, 50, seed=13)
    out = shrink_cov(X)
    assert out.shape == (50, 50)


def test_high_dimensional_case_is_symmetric_and_positive_definite():
    X = _data(20, 50, seed=14)
    out = shrink_cov(X)
    np.testing.assert_allclose(out, out.T, rtol=1e-8, atol=1e-10)
    assert np.linalg.eigvalsh(out).min() > 0


def test_one_dimensional_input_raises_assertion_error():
    with pytest.raises(AssertionError):
        shrink_cov(np.random.RandomState(15).randn(50))


def test_three_dimensional_input_raises_assertion_error():
    with pytest.raises(AssertionError):
        shrink_cov(np.random.RandomState(16).randn(20, 3, 2))


def test_minimum_effective_sample_size_default_is_accepted():
    # n = 13 with default demeaning gives effective sample size 12.
    X = _data(13, 3, seed=17)
    assert shrink_cov(X).shape == (3, 3)


def test_too_few_observations_default_raises_assertion_error():
    # n = 12 with default demeaning gives effective sample size 11.
    X = _data(12, 3, seed=18)
    with pytest.raises(AssertionError):
        shrink_cov(X)


def test_k_zero_minimum_sample_size_boundary():
    X_ok = _demean(_data(12, 3, seed=19))
    assert shrink_cov(X_ok, k=0).shape == (3, 3)

    X_bad = _demean(_data(11, 3, seed=20))
    with pytest.raises(AssertionError):
        shrink_cov(X_bad, k=0)


def test_large_k_reduces_effective_sample_size_below_minimum():
    # n = 13 with k = 2 gives effective sample size 11, which must fail.
    X = _demean(_data(13, 3, seed=21))
    with pytest.raises(AssertionError):
        shrink_cov(X, k=2)


def test_constant_column_raises_value_error():
    X = _data(60, 4, seed=22)
    X[:, 1] = 3.0  # demeaned to zero, making the sample covariance singular
    with pytest.raises(ValueError):
        shrink_cov(X)


def test_duplicated_columns_raise_value_error():
    X = _data(60, 4, seed=23)
    X[:, 3] = X[:, 0]  # perfectly collinear columns give a singular covariance
    with pytest.raises(ValueError):
        shrink_cov(X)
