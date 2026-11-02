"""Unit tests for nonlinshrink.shrink_cov (Ledoit and Wolf 2018 analytic nonlinear shrinkage)."""

import numpy as np
import pytest

from nonlinshrink import shrink_cov


def _gaussian(n, p, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p))


def test_output_shape_is_p_by_p():
    sigma = shrink_cov(_gaussian(30, 5))
    assert sigma.shape == (5, 5)


def test_output_is_float_ndarray():
    sigma = shrink_cov(_gaussian(30, 4))
    assert isinstance(sigma, np.ndarray)
    assert np.issubdtype(sigma.dtype, np.floating)


def test_output_is_symmetric():
    sigma = shrink_cov(_gaussian(40, 6, seed=1))
    assert np.allclose(sigma, sigma.T, atol=1e-12)


def test_output_is_positive_definite_when_p_le_n():
    sigma = shrink_cov(_gaussian(40, 6, seed=2))
    assert np.linalg.eigvalsh(sigma).min() > 0


def test_rejects_one_dimensional_input():
    with pytest.raises(AssertionError):
        shrink_cov(np.ones(20))


def test_rejects_three_dimensional_input():
    with pytest.raises(AssertionError):
        shrink_cov(np.ones((20, 3, 2)))


def test_default_k_needs_at_least_13_rows():
    # default k demeans and subtracts one, so n_eff = n - 1 must be >= 12
    with pytest.raises(AssertionError):
        shrink_cov(_gaussian(12, 3))
    assert shrink_cov(_gaussian(13, 3)).shape == (3, 3)


def test_k_zero_needs_at_least_12_rows():
    assert shrink_cov(_gaussian(12, 3), k=0).shape == (3, 3)
    with pytest.raises(AssertionError):
        shrink_cov(_gaussian(11, 3), k=0)


def test_k_reduces_effective_sample_size_by_k():
    # k=2 leaves n_eff = n - 2, so 13 rows is too few and 14 is enough
    with pytest.raises(AssertionError):
        shrink_cov(_gaussian(13, 3), k=2)
    assert shrink_cov(_gaussian(14, 3), k=2).shape == (3, 3)


def test_default_k_is_invariant_to_constant_offset():
    data = _gaussian(30, 4, seed=3)
    assert np.allclose(shrink_cov(data + 7.5), shrink_cov(data), rtol=1e-8, atol=1e-10)


def test_k_zero_does_not_demean():
    data = _gaussian(30, 4, seed=4) + 5.0
    assert not np.allclose(shrink_cov(data, k=0), shrink_cov(data))


def test_k_one_on_centered_data_matches_default():
    data = _gaussian(30, 4, seed=5) + 3.0
    centered = data - data.mean(axis=0)
    assert np.allclose(shrink_cov(centered, k=1), shrink_cov(data), rtol=1e-8, atol=1e-10)


def test_scales_quadratically_with_data():
    data = _gaussian(30, 4, seed=6)
    assert np.allclose(shrink_cov(3.0 * data), 9.0 * shrink_cov(data), rtol=1e-7, atol=1e-10)


def test_invariant_to_row_order():
    data = _gaussian(30, 4, seed=7)
    perm = np.random.default_rng(70).permutation(30)
    assert np.allclose(shrink_cov(data[perm]), shrink_cov(data), rtol=1e-8, atol=1e-10)


def test_column_permutation_permutes_result():
    data = _gaussian(30, 5, seed=8)
    perm = np.array([3, 0, 4, 1, 2])
    sigma = shrink_cov(data)
    sigma_perm = shrink_cov(data[:, perm])
    assert np.allclose(sigma_perm, sigma[np.ix_(perm, perm)], rtol=1e-8, atol=1e-10)


def test_shares_eigenvectors_with_sample_covariance():
    data = _gaussian(30, 5, seed=9)
    centered = data - data.mean(axis=0)
    sample_cov = centered.T @ centered / (30 - 1)
    sigma = shrink_cov(data)
    assert np.allclose(sigma @ sample_cov, sample_cov @ sigma, atol=1e-10)


def test_recovers_identity_for_large_sample():
    sigma = shrink_cov(_gaussian(4000, 4, seed=10))
    assert np.allclose(sigma, np.eye(4), atol=0.1)


def test_p_greater_than_n_gives_full_rank_estimate():
    sigma = shrink_cov(_gaussian(20, 40, seed=11))
    assert sigma.shape == (40, 40)
    assert np.all(np.isfinite(sigma))
    assert np.linalg.eigvalsh(sigma).min() > 1e-8


def test_constant_column_raises_singular_error():
    data = _gaussian(20, 3, seed=12)
    data[:, 2] = 3.0
    with pytest.raises(ValueError, match="singular"):
        shrink_cov(data)


def test_collinear_columns_raise_singular_error():
    data = _gaussian(20, 3, seed=13)
    data[:, 2] = data[:, 0] + data[:, 1]
    with pytest.raises(ValueError, match="singular"):
        shrink_cov(data)


def test_does_not_modify_input():
    data = _gaussian(30, 4, seed=14) + 2.0
    original = data.copy()
    shrink_cov(data)
    shrink_cov(data, k=0)
    assert np.array_equal(data, original)


def test_integer_input_matches_float_input():
    ints = np.random.default_rng(15).integers(0, 10, size=(30, 4))
    assert np.allclose(shrink_cov(ints), shrink_cov(ints.astype(float)), rtol=1e-10, atol=1e-12)
