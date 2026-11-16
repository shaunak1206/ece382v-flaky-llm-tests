import numpy as np
import pytest

import nonlinshrink.nonlinshrink as nls_module
from nonlinshrink.nonlinshrink import shrink_cov


def test_output_shape_matches_number_of_variables():
    np.random.seed(1)
    X = np.random.randn(60, 4)
    out = shrink_cov(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 4)


def test_output_is_symmetric():
    np.random.seed(2)
    X = np.random.randn(80, 5)
    out = shrink_cov(X)
    assert np.allclose(out, out.T, rtol=1e-10, atol=1e-12)


def test_output_is_positive_definite_when_p_less_than_n():
    np.random.seed(3)
    X = np.random.randn(200, 5)
    out = shrink_cov(X)
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(eigvals > 0)


def test_high_dimensional_case_returns_finite_symmetric_matrix():
    # p > n: the code takes the second branch of the formula
    np.random.seed(4)
    X = np.random.randn(20, 40)
    out = shrink_cov(X)
    assert out.shape == (40, 40)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out.T, rtol=1e-8, atol=1e-10)


def test_none_k_matches_explicit_k_one_on_centered_data():
    np.random.seed(5)
    X = np.random.randn(50, 3) + 2.0
    centered = X - X.mean(axis=0)
    out_none = shrink_cov(X)
    out_k1 = shrink_cov(centered, k=1)
    assert np.allclose(out_none, out_k1, rtol=1e-7, atol=1e-10)


def test_constant_shift_does_not_change_result_when_k_is_none():
    np.random.seed(6)
    X = np.random.randn(70, 4)
    out = shrink_cov(X)
    out_shifted = shrink_cov(X + 5.0)
    assert np.allclose(out, out_shifted, rtol=1e-7, atol=1e-10)


def test_scale_equivariance():
    # Scaling data by c scales the covariance estimate by c**2
    np.random.seed(7)
    X = np.random.randn(90, 4)
    c = 3.0
    out = shrink_cov(X)
    out_scaled = shrink_cov(c * X)
    assert np.allclose(out_scaled, c ** 2 * out, rtol=1e-6, atol=1e-10)


def test_row_order_does_not_change_result():
    np.random.seed(8)
    X = np.random.randn(60, 4)
    out = shrink_cov(X)
    out_reversed = shrink_cov(X[::-1].copy())
    assert np.allclose(out, out_reversed, rtol=1e-7, atol=1e-10)


def test_column_permutation_permutes_result():
    np.random.seed(9)
    X = np.random.randn(100, 4)
    perm = [2, 0, 3, 1]
    out = shrink_cov(X)
    out_perm = shrink_cov(X[:, perm])
    expected = out[np.ix_(perm, perm)]
    assert np.allclose(out_perm, expected, rtol=1e-6, atol=1e-8)


def test_repeated_calls_are_identical():
    np.random.seed(10)
    X = np.random.randn(64, 6)
    first = shrink_cov(X)
    second = shrink_cov(X)
    assert np.array_equal(first, second)


def test_input_array_is_not_modified():
    np.random.seed(11)
    X = np.random.randn(40, 3) + 1.0
    original = X.copy()
    shrink_cov(X)
    assert np.array_equal(X, original)


def test_rejects_non_2d_input():
    np.random.seed(12)
    flat = np.random.randn(30)
    cube = np.random.randn(20, 3, 2)
    with pytest.raises(AssertionError):
        shrink_cov(flat)
    with pytest.raises(AssertionError):
        shrink_cov(cube)


def test_rejects_sample_size_below_twelve_after_adjustment():
    np.random.seed(13)
    # With k=None one observation is used for demeaning, so n_eff = rows - 1.
    X_11 = np.random.randn(11, 2)  # n_eff = 10
    X_12 = np.random.randn(12, 2)  # n_eff = 11
    with pytest.raises(AssertionError):
        shrink_cov(X_11)
    with pytest.raises(AssertionError):
        shrink_cov(X_12)


def test_accepts_minimum_effective_sample_size():
    np.random.seed(14)
    X = np.random.randn(13, 2)  # n_eff = 12 with k=None
    out = shrink_cov(X)
    assert out.shape == (2, 2)
    assert np.all(np.isfinite(out))


def test_explicit_k_reduces_effective_sample_size():
    np.random.seed(15)
    X = np.random.randn(13, 2)
    # k=2 gives n_eff = 11 < 12, so the assertion must fire
    with pytest.raises(AssertionError):
        shrink_cov(X, k=2)


def test_k_zero_uses_full_sample_size_and_differs_from_default():
    np.random.seed(16)
    X = np.random.randn(60, 3)
    X = X - X.mean(axis=0)
    out_k0 = shrink_cov(X, k=0)
    out_default = shrink_cov(X)
    assert out_k0.shape == (3, 3)
    assert np.allclose(out_k0, out_k0.T, rtol=1e-10, atol=1e-12)
    assert not np.allclose(out_k0, out_default, rtol=1e-6, atol=1e-10)


def test_rejects_singular_sample_covariance():
    np.random.seed(17)
    X = np.random.randn(30, 3)
    X[:, 2] = X[:, 0]  # duplicate column makes sample covariance singular
    with pytest.raises(ValueError):
        shrink_cov(X)


def test_eps_threshold_patched_via_monkeypatch(monkeypatch):
    np.random.seed(18)
    X = np.random.randn(50, 3)
    # With EPS = 1.0 every normalized eigenvalue ratio is below the threshold,
    # so the singularity check must raise even for well-conditioned data.
    monkeypatch.setattr(nls_module, "EPS", 1.0)
    with pytest.raises(ValueError):
        shrink_cov(X)


def test_large_sample_approximates_sample_covariance():
    # When p/n is tiny, the shrunk estimate is close to the sample covariance
    np.random.seed(19)
    A = np.array([[1.0, 0.5, 0.0],
                  [0.0, 1.0, 0.3],
                  [0.0, 0.0, 0.7]])
    X = np.random.randn(5000, 3) @ A
    out = shrink_cov(X)
    sample = np.cov(X, rowvar=False)
    assert np.allclose(out, sample, rtol=0.02, atol=0.02)
