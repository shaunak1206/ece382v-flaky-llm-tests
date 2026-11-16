import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _gaussian(n, p, seed=0):
    return np.random.RandomState(seed).randn(n, p)


def test_output_shape_square_case():
    X = _gaussian(60, 5, seed=1)
    out = shrink_cov(X)
    assert out.shape == (5, 5)


def test_output_shape_high_dimensional():
    # p > n: the estimator must still return a p-by-p matrix
    X = _gaussian(20, 40, seed=2)
    out = shrink_cov(X)
    assert out.shape == (40, 40)


def test_output_is_symmetric():
    X = _gaussian(80, 6, seed=3)
    out = shrink_cov(X)
    np.testing.assert_allclose(out, out.T, rtol=1e-10, atol=1e-12)


def test_output_is_positive_definite_well_conditioned():
    X = _gaussian(100, 6, seed=4)
    out = shrink_cov(X)
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(eigvals > 0)


def test_high_dimensional_output_is_positive_definite():
    # With p > n the kernel supplies strictly positive values for the
    # null-space eigenvalues, so the output should be full rank.
    X = _gaussian(25, 50, seed=5)
    out = shrink_cov(X)
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(eigvals > 0)


def test_default_demeans_data_translation_invariance():
    X = _gaussian(70, 4, seed=6)
    shift = np.array([10.0, -3.0, 0.5, 7.0])
    out_raw = shrink_cov(X)
    out_shifted = shrink_cov(X + shift)
    np.testing.assert_allclose(out_shifted, out_raw, rtol=1e-8, atol=1e-10)


def test_k_zero_does_not_demean():
    X = _gaussian(70, 4, seed=7)
    shift = np.array([10.0, -3.0, 0.5, 7.0])
    out_raw = shrink_cov(X, k=0)
    out_shifted = shrink_cov(X + shift, k=0)
    assert not np.allclose(out_raw, out_shifted)


def test_k_none_matches_explicit_demeaning_with_k_one():
    X = _gaussian(90, 5, seed=8) + 4.0
    X_centered = X - X.mean(axis=0)
    np.testing.assert_allclose(
        shrink_cov(X), shrink_cov(X_centered, k=1), rtol=1e-10, atol=1e-12
    )


def test_larger_k_reduces_effective_sample_size():
    X = _gaussian(90, 5, seed=9)
    X_centered = X - X.mean(axis=0)
    out_k1 = shrink_cov(X_centered, k=1)
    out_k2 = shrink_cov(X_centered, k=2)
    assert out_k1.shape == out_k2.shape
    assert not np.allclose(out_k1, out_k2)


def test_scale_equivariance():
    X = _gaussian(80, 5, seed=10)
    c = 3.0
    out = shrink_cov(X)
    out_scaled = shrink_cov(c * X)
    np.testing.assert_allclose(out_scaled, c ** 2 * out, rtol=1e-6, atol=1e-10)


def test_row_permutation_invariance():
    X = _gaussian(75, 4, seed=11)
    perm = np.random.RandomState(12).permutation(X.shape[0])
    np.testing.assert_allclose(
        shrink_cov(X[perm]), shrink_cov(X), rtol=1e-8, atol=1e-10
    )


def test_column_permutation_equivariance():
    X = _gaussian(75, 5, seed=13)
    perm = np.array([3, 0, 4, 1, 2])
    P = np.eye(5)[:, perm]
    out = shrink_cov(X)
    out_perm = shrink_cov(X[:, perm])
    np.testing.assert_allclose(out_perm, P.T @ out @ P, rtol=1e-7, atol=1e-10)


def test_orthogonal_rotation_equivariance():
    rng = np.random.RandomState(14)
    p = 4
    X = rng.randn(90, p)
    Q, _ = np.linalg.qr(rng.randn(p, p))
    out = shrink_cov(X)
    out_rot = shrink_cov(X @ Q)
    np.testing.assert_allclose(out_rot, Q.T @ out @ Q, rtol=1e-7, atol=1e-10)


def test_deterministic_repeated_calls():
    X = _gaussian(60, 6, seed=15)
    first = shrink_cov(X)
    second = shrink_cov(X)
    np.testing.assert_array_equal(first, second)


def test_input_not_modified():
    X = _gaussian(60, 5, seed=16) + 2.0
    X_copy = X.copy()
    shrink_cov(X)
    shrink_cov(X, k=0)
    np.testing.assert_array_equal(X, X_copy)


def test_rejects_non_2d_input():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(30, dtype=float))
    with pytest.raises(AssertionError):
        shrink_cov(np.ones((4, 5, 6)))


def test_rejects_effective_sample_size_below_twelve_default_k():
    # n=12 with default k=None gives effective n=11, which must be rejected
    X = _gaussian(12, 3, seed=17)
    with pytest.raises(AssertionError):
        shrink_cov(X)


def test_rejects_effective_sample_size_below_twelve_explicit_k():
    # n=13 with k=2 gives effective n=11
    X = _gaussian(13, 3, seed=18)
    with pytest.raises(AssertionError):
        shrink_cov(X, k=2)


def test_accepts_minimum_effective_sample_size():
    # n=13 with default k gives effective n=12, the smallest allowed
    X = _gaussian(13, 3, seed=19)
    out = shrink_cov(X)
    assert out.shape == (3, 3)
    assert np.all(np.isfinite(out))


def test_raises_on_singular_covariance():
    X = _gaussian(50, 3, seed=20)
    X[:, 2] = X[:, 0]  # exact duplicate column makes the sample covariance singular
    with pytest.raises(ValueError):
        shrink_cov(X)


def test_close_to_identity_for_large_sample_with_identity_truth():
    X = _gaussian(5000, 4, seed=21)
    out = shrink_cov(X)
    np.testing.assert_allclose(out, np.eye(4), atol=0.1)
