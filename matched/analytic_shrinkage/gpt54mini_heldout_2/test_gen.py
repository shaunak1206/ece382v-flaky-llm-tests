import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _centered_data(n, p, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    return x - x.mean(axis=0)


def test_shrink_cov_requires_2d_input():
    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(np.arange(12))


def test_shrink_cov_requires_effective_sample_size_at_least_12_with_default_k():
    data = np.ones((12, 3))
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data)


def test_shrink_cov_requires_effective_sample_size_at_least_12_with_explicit_k():
    data = np.ones((12, 3))
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data, k=0)


def test_shrink_cov_output_shape_matches_number_of_variables():
    data = _centered_data(20, 5, seed=1)
    out = shrink_cov(data)
    assert out.shape == (5, 5)


def test_shrink_cov_is_symmetric_for_p_le_n():
    data = _centered_data(30, 8, seed=2)
    out = shrink_cov(data)
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_is_symmetric_for_p_greater_than_n():
    data = _centered_data(15, 25, seed=3)
    out = shrink_cov(data)
    assert out.shape == (25, 25)
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_with_k_none_matches_precentered_data_with_k_one():
    data = _centered_data(40, 7, seed=4)

    out_default = shrink_cov(data)
    out_precentered = shrink_cov(data, k=1)

    assert np.allclose(out_default, out_precentered, atol=1e-10, rtol=1e-10)


def test_shrink_cov_k_zero_differs_from_default_on_noncentered_data():
    rng = np.random.default_rng(5)
    data = rng.standard_normal((25, 6)) + 5.0

    out_default = shrink_cov(data)
    out_nocenter = shrink_cov(data, k=0)

    assert not np.allclose(out_default, out_nocenter)


def test_shrink_cov_is_homogeneous_of_degree_two():
    data = _centered_data(35, 9, seed=6)
    c = 3.5

    out = shrink_cov(data)
    out_scaled = shrink_cov(c * data)

    assert np.allclose(out_scaled, (c ** 2) * out, atol=1e-8, rtol=1e-7)


def test_shrink_cov_accepts_integer_input_and_returns_float_matrix():
    data = np.arange(60).reshape(20, 3)
    out = shrink_cov(data)

    assert out.shape == (3, 3)
    assert np.issubdtype(out.dtype, np.floating)


def test_shrink_cov_changes_the_sample_covariance_on_generic_data():
    data = _centered_data(30, 6, seed=7)
    n = data.shape[0] - 1
    sample_cov = data.T @ data / n
    out = shrink_cov(data)

    assert not np.allclose(out, sample_cov)


def test_shrink_cov_handles_p_greater_than_n_with_large_output_dimension():
    data = _centered_data(14, 40, seed=8)
    out = shrink_cov(data)

    assert out.shape == (40, 40)
    assert np.isfinite(out).all()


def test_shrink_cov_raises_for_singular_sample_covariance():
    # Rank-1 data produces zero sample eigenvalues, which the implementation
    # treats as singular.
    data = np.zeros((20, 5))
    data[:, 0] = np.arange(20)

    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data, k=0)
