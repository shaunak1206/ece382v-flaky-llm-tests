import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p)


def test_output_shape_low_dimensional():
    X = _data(100, 10)
    out = shrink_cov(X)
    assert out.shape == (10, 10)


def test_output_is_symmetric():
    X = _data(80, 15, seed=1)
    out = shrink_cov(X)
    assert np.allclose(out, out.T, atol=1e-10)


def test_output_is_finite_and_positive_definite():
    X = _data(120, 20, seed=2)
    out = shrink_cov(X)
    assert np.all(np.isfinite(out))
    eigvals = np.linalg.eigvalsh((out + out.T) / 2)
    assert eigvals.min() > 0


def test_high_dimensional_p_greater_than_n():
    X = _data(20, 40, seed=3)
    out = shrink_cov(X)
    assert out.shape == (40, 40)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out.T, atol=1e-8)


def test_one_dimensional_input_raises_assertion():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(50, dtype=float))


def test_three_dimensional_input_raises_assertion():
    with pytest.raises(AssertionError):
        shrink_cov(np.zeros((20, 3, 2)))


def test_too_few_samples_default_k_raises():
    # n=12 with default demeaning gives effective sample size 11
    X = _data(12, 3, seed=4)
    with pytest.raises(AssertionError):
        shrink_cov(X)


def test_minimum_sample_size_accepted():
    # n=13, k=None -> effective n = 12, which is allowed
    out = shrink_cov(_data(13, 3, seed=5))
    assert out.shape == (3, 3)
    # n=12, k=0 -> effective n = 12
    out = shrink_cov(_data(12, 3, seed=6), k=0)
    assert out.shape == (3, 3)


def test_k_reduces_effective_sample_size():
    X = _data(16, 3, seed=7)
    assert shrink_cov(X, k=4).shape == (3, 3)
    with pytest.raises(AssertionError):
        shrink_cov(X, k=5)


def test_singular_matrix_raises_value_error():
    X = _data(50, 4, seed=8)
    X[:, 3] = 0.0
    with pytest.raises(ValueError):
        shrink_cov(X, k=0)


def test_duplicate_column_raises_value_error():
    X = _data(60, 5, seed=9)
    X[:, 4] = X[:, 0]
    with pytest.raises(ValueError):
        shrink_cov(X)


def test_default_k_equals_manual_demeaning_with_k1():
    X = _data(60, 8, seed=10) + 5.0
    out_default = shrink_cov(X)
    out_manual = shrink_cov(X - X.mean(axis=0), k=1)
    assert np.allclose(out_default, out_manual, atol=1e-10)


def test_input_not_modified():
    X = _data(50, 6, seed=11)
    X_copy = X.copy()
    shrink_cov(X)
    assert np.array_equal(X, X_copy)


def test_scale_equivariance():
    X = _data(70, 8, seed=12)
    c = 3.0
    out1 = shrink_cov(X)
    out2 = shrink_cov(c * X)
    assert np.allclose(out2, c ** 2 * out1, rtol=1e-6, atol=1e-8)


def test_row_permutation_invariance():
    X = _data(60, 7, seed=13)
    perm = np.random.RandomState(99).permutation(60)
    out1 = shrink_cov(X)
    out2 = shrink_cov(X[perm])
    assert np.allclose(out1, out2, atol=1e-8)


def test_shares_eigenvectors_with_sample_covariance():
    X = _data(100, 6, seed=14)
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / (100 - 1)
    out = shrink_cov(X)
    assert np.allclose(S @ out, out @ S, atol=1e-8)


def test_shrinkage_improves_conditioning_for_identity_covariance():
    X = _data(200, 50, seed=15)
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / (200 - 1)
    out = shrink_cov(X)
    assert np.linalg.cond(out) < np.linalg.cond(S)


def test_shrunk_estimate_close_to_identity_for_identity_truth():
    X = _data(500, 10, seed=16)
    out = shrink_cov(X)
    assert np.linalg.norm(out - np.eye(10)) / np.sqrt(10) < 0.3


def test_list_input_fails():
    with pytest.raises(AttributeError):
        shrink_cov([[1.0, 2.0], [3.0, 4.0]])
