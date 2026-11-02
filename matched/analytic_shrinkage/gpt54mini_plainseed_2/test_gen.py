import numpy as np
import pytest

from nonlinshrink.nonlinshrink import EPS, shrink_cov


def _make_data(n=20, p=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p))


def test_shrink_cov_returns_square_matrix_same_shape():
    data = _make_data(n=20, p=5, seed=1)
    out = shrink_cov(data)
    assert out.shape == (5, 5)


def test_shrink_cov_output_is_symmetric():
    data = _make_data(n=20, p=6, seed=2)
    out = shrink_cov(data)
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_dtype_is_float():
    data = _make_data(n=18, p=4, seed=3).astype(np.float64)
    out = shrink_cov(data)
    assert np.issubdtype(out.dtype, np.floating)


def test_shrink_cov_is_positive_on_diagonal():
    data = _make_data(n=22, p=7, seed=4)
    out = shrink_cov(data)
    assert np.all(np.diag(out) > 0)


def test_shrink_cov_with_k_none_matches_explicit_centered_k1():
    data = _make_data(n=19, p=5, seed=5)
    centered = data - np.mean(data, axis=0)
    out_none = shrink_cov(data, k=None)
    out_k1 = shrink_cov(centered, k=1)
    assert np.allclose(out_none, out_k1, atol=1e-10, rtol=1e-10)


def test_shrink_cov_invariant_to_additive_constant_when_k_none():
    data = _make_data(n=20, p=4, seed=6)
    shifted = data + 123.456
    out1 = shrink_cov(data, k=None)
    out2 = shrink_cov(shifted, k=None)
    assert np.allclose(out1, out2, atol=1e-10, rtol=1e-10)


def test_shrink_cov_k0_differs_from_k_none_for_shifted_data():
    data = _make_data(n=20, p=4, seed=7)
    shifted = data + 10.0
    out_none = shrink_cov(shifted, k=None)
    out_k0 = shrink_cov(shifted, k=0)
    assert not np.allclose(out_none, out_k0)


def test_shrink_cov_accepts_precentered_data_with_k1():
    data = _make_data(n=16, p=6, seed=8)
    centered = data - np.mean(data, axis=0)
    out = shrink_cov(centered, k=1)
    assert out.shape == (6, 6)
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_high_dimensional_case_p_greater_than_n():
    data = _make_data(n=14, p=18, seed=9)
    out = shrink_cov(data)
    assert out.shape == (18, 18)
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_raises_for_non_2d_input():
    data = np.arange(20.0)
    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(data)


def test_shrink_cov_raises_when_effective_sample_size_too_small_due_to_k():
    data = _make_data(n=13, p=3, seed=10)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data, k=2)


def test_shrink_cov_raises_when_raw_sample_size_too_small_after_default_demeaning():
    data = _make_data(n=11, p=3, seed=11)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data)


def test_shrink_cov_raises_on_singular_matrix():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(20, 1))
    data = np.hstack([x, x, x])  # perfectly collinear columns
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data)


def test_eps_constant_is_small_positive():
    assert EPS > 0
    assert EPS < 1e-6


def test_shrink_cov_output_is_finite_for_random_input():
    data = _make_data(n=24, p=8, seed=13)
    out = shrink_cov(data)
    assert np.isfinite(out).all()
