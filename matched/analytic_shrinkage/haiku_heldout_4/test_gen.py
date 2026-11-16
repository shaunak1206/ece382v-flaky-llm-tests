import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def make_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p)


def test_output_shape_is_square_p_by_p():
    X = make_data(50, 4)
    out = shrink_cov(X)
    assert out.shape == (4, 4)


def test_output_is_symmetric():
    X = make_data(60, 5, seed=1)
    out = shrink_cov(X)
    np.testing.assert_allclose(out, out.T, rtol=1e-10, atol=1e-12)


def test_output_is_positive_definite_when_n_exceeds_p():
    X = make_data(80, 6, seed=2)
    out = shrink_cov(X)
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(eigvals > 0)


def test_integer_input_is_accepted_and_returns_floats():
    rng = np.random.RandomState(3)
    X = (rng.randn(50, 3) * 10).astype(int)
    out = shrink_cov(X)
    assert out.shape == (3, 3)
    assert np.issubdtype(out.dtype, np.floating)
    assert np.all(np.isfinite(out))


def test_1d_input_raises_assertion_error():
    x = np.arange(20, dtype=float)
    with pytest.raises(AssertionError):
        shrink_cov(x)


def test_3d_input_raises_assertion_error():
    X = np.zeros((20, 3, 2))
    with pytest.raises(AssertionError):
        shrink_cov(X)


def test_effective_sample_size_below_12_raises_with_default_k():
    # k=None subtracts 1 from n, so n=12 gives effective size 11
    X = make_data(12, 3, seed=4)
    with pytest.raises(AssertionError):
        shrink_cov(X)


def test_effective_sample_size_boundary_with_default_k_is_ok():
    # n=13 gives effective size 12, which is allowed
    X = make_data(13, 3, seed=5)
    out = shrink_cov(X)
    assert out.shape == (3, 3)


def test_effective_sample_size_depends_on_k():
    # k=0 keeps n: n=11 fails, n=12 succeeds
    X_small = make_data(11, 3, seed=6)
    with pytest.raises(AssertionError):
        shrink_cov(X_small, k=0)

    X_ok = make_data(12, 3, seed=6)
    out = shrink_cov(X_ok, k=0)
    assert out.shape == (3, 3)

    # k=2 reduces effective size: n=13 -> 11 fails
    X_k2 = make_data(13, 3, seed=6)
    with pytest.raises(AssertionError):
        shrink_cov(X_k2, k=2)


def test_default_mode_is_invariant_to_column_offsets():
    X = make_data(70, 4, seed=7)
    offset = np.array([3.0, -2.0, 10.0, 0.5])
    out_a = shrink_cov(X)
    out_b = shrink_cov(X + offset)
    np.testing.assert_allclose(out_a, out_b, rtol=1e-8, atol=1e-10)


def test_k_none_matches_k_one_on_already_centered_data():
    X = make_data(60, 4, seed=8)
    Xc = X - X.mean(axis=0)
    out_default = shrink_cov(Xc)
    out_k1 = shrink_cov(Xc, k=1)
    np.testing.assert_allclose(out_default, out_k1, rtol=1e-8, atol=1e-10)


def test_k_zero_differs_from_default_demeaning():
    X = make_data(60, 4, seed=9) + 5.0
    out_default = shrink_cov(X)
    out_k0 = shrink_cov(X, k=0)
    assert not np.allclose(out_default, out_k0)


def test_scale_equivariance():
    X = make_data(64, 4, seed=10)
    c = 3.7
    out = shrink_cov(X)
    out_scaled = shrink_cov(c * X)
    np.testing.assert_allclose(out_scaled, (c ** 2) * out, rtol=1e-6, atol=1e-10)


def test_column_permutation_equivariance():
    n, p = 70, 5
    X = make_data(n, p, seed=11)
    rng = np.random.RandomState(12)
    perm = rng.permutation(p)
    out = shrink_cov(X)
    out_perm = shrink_cov(X[:, perm])
    np.testing.assert_allclose(out_perm, out[np.ix_(perm, perm)], rtol=1e-8, atol=1e-10)


def test_row_permutation_invariance():
    n, p = 60, 4
    X = make_data(n, p, seed=13)
    rng = np.random.RandomState(14)
    out = shrink_cov(X)
    out_rows = shrink_cov(X[rng.permutation(n)])
    np.testing.assert_allclose(out_rows, out, rtol=1e-8, atol=1e-10)


def test_output_commutes_with_sample_covariance():
    # The estimator shares eigenvectors with the sample covariance
    n, p = 80, 5
    X = make_data(n, p, seed=15)
    Xc = X - X.mean(axis=0)
    S = np.dot(Xc.T, Xc) / (n - 1)
    out = shrink_cov(X)
    np.testing.assert_allclose(out @ S, S @ out, rtol=1e-6, atol=1e-8)


def test_large_sample_limit_is_close_to_sample_covariance():
    # With p/n tiny, shrinkage is negligible
    n, p = 20000, 3
    X = make_data(n, p, seed=16)
    out = shrink_cov(X)
    S = np.cov(X, rowvar=False)
    np.testing.assert_allclose(out, S, rtol=1e-2, atol=1e-3)


def test_singular_sample_covariance_raises_value_error():
    X = make_data(30, 4, seed=17)
    X[:, 3] = X[:, 0]  # duplicate column makes covariance singular
    with pytest.raises(ValueError):
        shrink_cov(X)


def test_constant_column_raises_value_error():
    X = make_data(30, 3, seed=18)
    X[:, 1] = 5.0  # zero variance after demeaning
    with pytest.raises(ValueError):
        shrink_cov(X)


def test_high_dimensional_case_p_greater_than_n_is_finite_and_pd():
    # n=13 with k=None gives effective n=12 < p=20
    X = make_data(13, 20, seed=19)
    out = shrink_cov(X)
    assert out.shape == (20, 20)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, out.T, rtol=1e-8, atol=1e-10)
    assert np.linalg.eigvalsh(out).min() > 0


def test_input_array_is_not_modified():
    X = make_data(50, 4, seed=20)
    X_copy = X.copy()
    shrink_cov(X)
    shrink_cov(X, k=0)
    np.testing.assert_array_equal(X, X_copy)


def test_repeated_calls_are_deterministic():
    X = make_data(55, 4, seed=21)
    out1 = shrink_cov(X)
    out2 = shrink_cov(X)
    np.testing.assert_array_equal(out1, out2)
