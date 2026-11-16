import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _random_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p)


def _sample_cov(data):
    centered = data - data.mean(axis=0)
    return centered.T @ centered / (data.shape[0] - 1)


def test_returns_square_ndarray_of_expected_shape():
    X = _random_data(50, 5, seed=1)
    out = shrink_cov(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 5)


def test_output_is_symmetric():
    X = _random_data(60, 6, seed=2)
    out = shrink_cov(X)
    np.testing.assert_allclose(out, out.T, rtol=1e-10, atol=1e-12)


def test_output_is_positive_definite():
    X = _random_data(80, 7, seed=3)
    out = shrink_cov(X)
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(eigvals > 0)


def test_output_shares_eigenvectors_with_sample_covariance():
    # The estimator is u diag(d) u^T with u the sample eigenvectors,
    # so it must commute with the (demeaned, n-1 normalised) sample covariance.
    X = _random_data(70, 4, seed=4)
    S = _sample_cov(X)
    out = shrink_cov(X)
    scale = np.abs(out).max()
    np.testing.assert_allclose(out @ S, S @ out, atol=1e-8 * scale, rtol=0)


def test_rejects_1d_input():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(20, dtype=float))


def test_rejects_3d_input():
    with pytest.raises(AssertionError):
        shrink_cov(np.zeros((20, 3, 2)))


def test_rejects_sample_size_below_minimum_with_default_k():
    # n = 12 leaves effective sample size 11 after demeaning, which must fail.
    X = _random_data(12, 3, seed=5)
    with pytest.raises(AssertionError):
        shrink_cov(X)


def test_accepts_minimum_sample_size_with_default_k():
    # n = 13 leaves effective sample size 12, which is the minimum allowed.
    X = _random_data(13, 3, seed=6)
    out = shrink_cov(X)
    assert out.shape == (3, 3)


def test_large_k_reduces_effective_sample_size_below_minimum():
    # n = 13 with k = 2 gives effective sample size 11.
    X = _random_data(13, 3, seed=7)
    with pytest.raises(AssertionError):
        shrink_cov(X, k=2)


def test_k_zero_allows_sample_size_of_twelve():
    # With k = 0 the effective sample size stays at n = 12.
    X = _random_data(12, 3, seed=8)
    out = shrink_cov(X, k=0)
    assert out.shape == (3, 3)


def test_centered_data_default_k_matches_explicit_k_one():
    X = _random_data(40, 4, seed=9)
    Xc = X - X.mean(axis=0)
    out_default = shrink_cov(Xc)
    out_k1 = shrink_cov(Xc, k=1)
    np.testing.assert_allclose(out_default, out_k1, rtol=1e-10, atol=1e-12)


def test_k_zero_differs_from_k_one_on_centered_data():
    # Same centered data, but k=0 uses n effective observations while k=1 uses n-1.
    X = _random_data(40, 4, seed=10)
    Xc = X - X.mean(axis=0)
    out_k0 = shrink_cov(Xc, k=0)
    out_k1 = shrink_cov(Xc, k=1)
    assert not np.allclose(out_k0, out_k1)


def test_invariant_to_mean_shift_with_default_k():
    X = _random_data(50, 4, seed=11)
    shift = np.array([3.0, -2.0, 0.5, 10.0])
    out_original = shrink_cov(X)
    out_shifted = shrink_cov(X + shift)
    np.testing.assert_allclose(out_original, out_shifted, rtol=1e-8, atol=1e-10)


def test_scale_equivariance_quadratic_in_scalar():
    X = _random_data(60, 5, seed=12)
    a = 2.5
    out = shrink_cov(X)
    out_scaled = shrink_cov(a * X)
    np.testing.assert_allclose(out_scaled, a ** 2 * out, rtol=1e-6, atol=1e-10)


def test_invariant_to_row_permutation():
    X = _random_data(55, 4, seed=13)
    perm = np.random.RandomState(99).permutation(X.shape[0])
    out = shrink_cov(X)
    out_perm = shrink_cov(X[perm])
    np.testing.assert_allclose(out, out_perm, rtol=1e-8, atol=1e-10)


def test_column_permutation_permutes_output_consistently():
    X = _random_data(65, 5, seed=14)
    perm = np.array([2, 0, 4, 1, 3])
    out = shrink_cov(X)
    out_perm = shrink_cov(X[:, perm])
    np.testing.assert_allclose(out_perm, out[np.ix_(perm, perm)], rtol=1e-7, atol=1e-9)


def test_does_not_modify_input_array():
    X = _random_data(30, 3, seed=15)
    X_copy = X.copy()
    shrink_cov(X)
    np.testing.assert_array_equal(X, X_copy)


def test_large_sample_close_to_true_covariance():
    rng = np.random.RandomState(16)
    true_var = np.array([1.0, 2.0, 3.0, 4.0])
    X = rng.randn(20000, 4) * np.sqrt(true_var)
    out = shrink_cov(X)
    np.testing.assert_allclose(np.diag(out), true_var, atol=0.15)
    off_diag = out - np.diag(np.diag(out))
    assert np.abs(off_diag).max() < 0.15


def test_rank_deficient_data_raises_singular_error():
    # Third column is an exact linear combination of the first two.
    rng = np.random.RandomState(17)
    base = rng.randn(50, 2)
    X = np.column_stack([base[:, 0], base[:, 1], base[:, 0] + base[:, 1]])
    with pytest.raises(ValueError, match="singular"):
        shrink_cov(X)


def test_p_greater_than_n_with_k_zero_returns_finite_symmetric_matrix():
    # With k=0 the sample covariance has rank n, so the kept eigenvalues are nonzero.
    X = _random_data(15, 30, seed=18)
    out = shrink_cov(X, k=0)
    assert out.shape == (30, 30)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, out.T, rtol=1e-8, atol=1e-10)


def test_single_variable_returns_positive_one_by_one_matrix():
    X = _random_data(30, 1, seed=19)
    out = shrink_cov(X)
    assert out.shape == (1, 1)
    assert out[0, 0] > 0
