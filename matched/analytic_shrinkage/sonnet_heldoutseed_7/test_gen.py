import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _make_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p)


def _make_corr_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(p, p)
    cov = a.dot(a.T) / p + np.eye(p)
    return rng.multivariate_normal(np.zeros(p), cov, size=n), cov


def test_output_shape_p_less_than_n():
    X = _make_data(50, 5)
    out = shrink_cov(X)
    assert out.shape == (5, 5)


def test_output_shape_p_greater_than_n():
    X = _make_data(20, 30, seed=1)
    out = shrink_cov(X)
    assert out.shape == (30, 30)


def test_output_is_symmetric():
    X, _ = _make_corr_data(100, 10, seed=2)
    out = shrink_cov(X)
    assert np.allclose(out, out.T, atol=1e-10)


def test_output_is_positive_definite():
    X, _ = _make_corr_data(100, 10, seed=3)
    out = shrink_cov(X)
    eig = np.linalg.eigvalsh(out)
    assert np.all(eig > 0)


def test_output_finite_high_dimensional():
    X = _make_data(20, 30, seed=4)
    out = shrink_cov(X)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out.T, atol=1e-10)


def test_one_dimensional_input_raises():
    with pytest.raises(AssertionError):
        shrink_cov(np.random.RandomState(0).randn(50))


def test_three_dimensional_input_raises():
    with pytest.raises(AssertionError):
        shrink_cov(np.random.RandomState(0).randn(20, 3, 2))


def test_too_few_samples_default_k_raises():
    # n=12 with default demeaning gives effective sample size 11 < 12
    X = _make_data(12, 3)
    with pytest.raises(AssertionError):
        shrink_cov(X)


def test_minimum_sample_size_accepted():
    # n=13 with k=None -> effective 12, allowed
    X = _make_data(13, 3, seed=5)
    out = shrink_cov(X)
    assert out.shape == (3, 3)
    # n=12 with k=0 -> effective 12, allowed
    X = _make_data(12, 3, seed=6)
    out = shrink_cov(X, k=0)
    assert out.shape == (3, 3)


def test_too_few_samples_with_large_k_raises():
    X = _make_data(20, 3)
    with pytest.raises(AssertionError):
        shrink_cov(X, k=9)  # effective n = 11


def test_singular_matrix_raises_value_error():
    X = _make_data(50, 4, seed=7)
    X[:, 0] = 0.0
    with pytest.raises(ValueError):
        shrink_cov(X)


def test_default_k_equals_manual_demeaning_with_k1():
    X = _make_data(60, 6, seed=8) + 3.0
    out_default = shrink_cov(X)
    out_manual = shrink_cov(X - X.mean(axis=0), k=1)
    assert np.allclose(out_default, out_manual, atol=1e-10)


def test_k_zero_differs_from_k_one_on_centered_data():
    X = _make_data(60, 6, seed=9)
    X = X - X.mean(axis=0)
    out0 = shrink_cov(X, k=0)
    out1 = shrink_cov(X, k=1)
    assert not np.allclose(out0, out1)


def test_input_not_modified():
    X = _make_data(40, 5, seed=10) + 1.0
    X_copy = X.copy()
    shrink_cov(X)
    assert np.array_equal(X, X_copy)


def test_deterministic():
    X = _make_data(40, 5, seed=11)
    assert np.array_equal(shrink_cov(X), shrink_cov(X))


def test_scale_equivariance():
    X = _make_data(80, 8, seed=12)
    c = 3.5
    out = shrink_cov(X)
    out_scaled = shrink_cov(c * X)
    assert np.allclose(out_scaled, c ** 2 * out, rtol=1e-5, atol=1e-8)


def test_row_permutation_invariance():
    X, _ = _make_corr_data(80, 8, seed=13)
    perm = np.random.RandomState(99).permutation(X.shape[0])
    out = shrink_cov(X)
    out_perm = shrink_cov(X[perm])
    assert np.allclose(out, out_perm, atol=1e-8)


def test_identity_covariance_recovered_for_large_n():
    X = _make_data(2000, 5, seed=14)
    out = shrink_cov(X)
    assert np.allclose(out, np.eye(5), atol=0.3)


def test_shrinkage_reduces_condition_number():
    X = _make_data(100, 50, seed=15)
    sample = np.cov(X, rowvar=False)
    shrunk = shrink_cov(X)
    cond_sample = np.linalg.cond(sample)
    cond_shrunk = np.linalg.cond(shrunk)
    assert cond_shrunk < cond_sample


def test_shrinkage_closer_to_truth_than_sample_covariance():
    X = _make_data(100, 50, seed=16)
    truth = np.eye(50)
    sample = np.cov(X, rowvar=False)
    shrunk = shrink_cov(X)
    err_sample = np.linalg.norm(sample - truth)
    err_shrunk = np.linalg.norm(shrunk - truth)
    assert err_shrunk < err_sample
