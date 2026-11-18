import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _random_data(n, p, seed=0, mean_shift=0.0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p) + mean_shift


def test_shrink_cov_returns_square_matrix_same_shape():
    data = _random_data(20, 5, seed=1)
    out = shrink_cov(data, k=0)
    assert out.shape == (5, 5)


def test_shrink_cov_output_is_symmetric():
    data = _random_data(18, 6, seed=2)
    out = shrink_cov(data, k=0)
    assert np.allclose(out, out.T, atol=1e-12)


def test_shrink_cov_output_is_finite():
    data = _random_data(24, 8, seed=3)
    out = shrink_cov(data, k=0)
    assert np.isfinite(out).all()


def test_shrink_cov_k_none_matches_explicit_centering_with_k1():
    data = _random_data(22, 7, seed=4, mean_shift=5.0)
    centered = data - np.mean(data, axis=0)

    out_none = shrink_cov(data, k=None)
    out_centered = shrink_cov(centered, k=1)

    assert np.allclose(out_none, out_centered, atol=1e-10, rtol=1e-10)


def test_shrink_cov_k0_differs_from_demeaned_version_on_shifted_data():
    data = _random_data(22, 7, seed=5, mean_shift=3.0)

    out_k0 = shrink_cov(data, k=0)
    out_none = shrink_cov(data, k=None)

    assert not np.allclose(out_k0, out_none)


def test_shrink_cov_k_positive_adjusts_effective_sample_size():
    data = _random_data(25, 6, seed=6)
    centered = data - np.mean(data, axis=0)

    out_k1 = shrink_cov(centered, k=1)
    out_k2 = shrink_cov(centered, k=2)

    assert out_k1.shape == out_k2.shape == (6, 6)
    assert not np.allclose(out_k1, out_k2)


def test_shrink_cov_handles_p_greater_than_n():
    data = _random_data(15, 20, seed=7)
    out = shrink_cov(data, k=0)
    assert out.shape == (20, 20)
    assert np.allclose(out, out.T, atol=1e-12)


def test_shrink_cov_handles_p_equals_n():
    data = _random_data(18, 18, seed=8)
    out = shrink_cov(data, k=0)
    assert out.shape == (18, 18)
    assert np.allclose(out, out.T, atol=1e-12)


def test_shrink_cov_minimum_allowed_effective_sample_size_n12():
    data = _random_data(13, 4, seed=9)
    out = shrink_cov(data, k=1)  # effective sample size = 12
    assert out.shape == (4, 4)


def test_shrink_cov_rejects_non_2d_input_vector():
    data = np.arange(20.0)
    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(data, k=0)


def test_shrink_cov_rejects_non_2d_input_3d_array():
    data = np.zeros((2, 3, 4))
    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(data, k=0)


def test_shrink_cov_rejects_effective_sample_size_below_12():
    data = _random_data(12, 5, seed=10)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data, k=1)  # effective sample size = 11


def test_shrink_cov_rejects_too_large_k():
    data = _random_data(12, 5, seed=11)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data, k=13)


def test_shrink_cov_raises_value_error_for_singular_covariance():
    data = np.ones((12, 4))
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data, k=0)


def test_shrink_cov_with_zero_mean_data_and_k_none_equals_k1():
    rng = np.random.RandomState(12)
    data = rng.randn(30, 5)
    data = data - data.mean(axis=0, keepdims=True)

    out_none = shrink_cov(data, k=None)
    out_k1 = shrink_cov(data, k=1)

    assert np.allclose(out_none, out_k1, atol=1e-10, rtol=1e-10)
