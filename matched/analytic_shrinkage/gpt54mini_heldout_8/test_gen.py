import numpy as np
import pytest

from nonlinshrink.nonlinshrink import EPS, shrink_cov


def _rand_data(n, p, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)) * scale


def _is_symmetric(a, tol=1e-10):
    return np.allclose(a, a.T, atol=tol, rtol=0)


def test_shrink_cov_returns_square_matrix_same_shape():
    data = _rand_data(20, 5, seed=1)
    out = shrink_cov(data)
    assert out.shape == (5, 5)


def test_shrink_cov_output_is_symmetric():
    data = _rand_data(25, 7, seed=2)
    out = shrink_cov(data)
    assert _is_symmetric(out)


def test_shrink_cov_output_is_finite_for_typical_input():
    data = _rand_data(30, 8, seed=3)
    out = shrink_cov(data)
    assert np.isfinite(out).all()


def test_shrink_cov_raises_on_non_2d_input_1d():
    data = np.arange(20.0)
    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(data)


def test_shrink_cov_raises_on_non_2d_input_3d():
    data = np.zeros((2, 3, 4))
    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(data)


def test_shrink_cov_raises_when_effective_sample_size_too_small_with_default_k():
    data = _rand_data(12, 5, seed=4)  # default k=1 => effective n = 11
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data)


def test_shrink_cov_raises_when_effective_sample_size_too_small_with_explicit_k():
    data = _rand_data(13, 5, seed=5)  # k=2 => effective n = 11
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data, k=2)


def test_shrink_cov_default_k_is_invariant_to_constant_shift():
    data = _rand_data(24, 6, seed=6)
    shifted = data + 123.456
    out1 = shrink_cov(data)
    out2 = shrink_cov(shifted)
    assert np.allclose(out1, out2, atol=1e-10, rtol=1e-10)


def test_shrink_cov_k_zero_is_not_invariant_to_constant_shift_due_to_no_demeaning():
    data = _rand_data(24, 6, seed=7)
    shifted = data + 10.0
    out1 = shrink_cov(data, k=0)
    out2 = shrink_cov(shifted, k=0)
    assert not np.allclose(out1, out2)


def test_shrink_cov_matches_default_behavior_on_already_centered_data_with_k1():
    data = _rand_data(22, 5, seed=8)
    centered = data - np.mean(data, axis=0)
    out_default = shrink_cov(centered)  # demeaning again should have no effect
    out_k1 = shrink_cov(centered, k=1)
    assert np.allclose(out_default, out_k1, atol=1e-10, rtol=1e-10)


def test_shrink_cov_keeps_shape_for_high_dimensional_case_p_greater_than_n():
    data = _rand_data(18, 30, seed=9)
    out = shrink_cov(data)
    assert out.shape == (30, 30)
    assert _is_symmetric(out)
    assert np.isfinite(out).all()


def test_shrink_cov_high_dimensional_case_handles_rank_deficient_sample_covariance():
    data = _rand_data(15, 25, seed=10)
    out = shrink_cov(data, k=1)
    eigvals = np.linalg.eigvalsh(out)
    assert np.isfinite(out).all()
    assert eigvals.shape == (25,)
    assert np.all(eigvals >= -1e-8)


def test_shrink_cov_rejects_singular_covariance_when_columns_are_identical():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(20, 1))
    data = np.repeat(x, 5, axis=1)
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data, k=0)


def test_shrink_cov_accepts_nonzero_k_for_precentered_data_and_returns_symmetric_matrix():
    data = _rand_data(28, 9, seed=12)
    centered = data - np.mean(data, axis=0)
    out = shrink_cov(centered, k=3)
    assert out.shape == (9, 9)
    assert _is_symmetric(out)
    assert np.isfinite(out).all()
