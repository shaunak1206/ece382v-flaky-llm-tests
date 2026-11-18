import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _centered_pair_data(n_pairs=7, p=4):
    # Exactly zero-mean columns by construction: stacked +/- rows.
    base = np.arange(1, n_pairs * p + 1, dtype=float).reshape(n_pairs, p)
    data = np.vstack([base, -base])
    assert np.allclose(data.mean(axis=0), 0.0)
    return data


def test_shrink_cov_returns_square_symmetric_matrix_for_k_none():
    data = _centered_pair_data(n_pairs=7, p=4)  # n=14, effective n=13
    out = shrink_cov(data)

    assert out.shape == (4, 4)
    assert np.allclose(out, out.T, atol=1e-12, rtol=1e-12)


def test_shrink_cov_does_not_mutate_input_when_k_none():
    data = _centered_pair_data(n_pairs=7, p=4)
    original = data.copy()

    _ = shrink_cov(data)

    assert np.array_equal(data, original)


def test_shrink_cov_does_not_mutate_input_when_k_zero():
    data = _centered_pair_data(n_pairs=7, p=4)
    original = data.copy()

    _ = shrink_cov(data, k=0)

    assert np.array_equal(data, original)


def test_shrink_cov_k_none_equals_k_one_for_exactly_centered_data():
    data = _centered_pair_data(n_pairs=7, p=4)

    out_none = shrink_cov(data, k=None)
    out_one = shrink_cov(data, k=1)

    assert np.allclose(out_none, out_one, atol=1e-12, rtol=1e-12)


def test_shrink_cov_k_zero_and_k_one_are_different_for_noncentered_data():
    # k=0 uses n, while k=1 uses n-1; for non-centered data they should differ.
    data = np.arange(1, 57, dtype=float).reshape(14, 4)

    out_zero = shrink_cov(data, k=0)
    out_one = shrink_cov(data, k=1)

    assert out_zero.shape == out_one.shape == (4, 4)
    assert not np.allclose(out_zero, out_one, atol=1e-12, rtol=1e-12)


def test_shrink_cov_p_greater_than_n_branch_returns_correct_shape():
    np.random.seed(0)
    data = np.random.RandomState(0).randn(12, 15)

    out = shrink_cov(data, k=0)

    assert out.shape == (15, 15)
    assert np.allclose(out, out.T, atol=1e-12, rtol=1e-12)


def test_shrink_cov_p_greater_than_n_branch_is_deterministic():
    np.random.seed(0)
    data = np.random.RandomState(0).randn(12, 15)

    out1 = shrink_cov(data, k=0)
    out2 = shrink_cov(data, k=0)

    assert np.allclose(out1, out2, atol=1e-12, rtol=1e-12)


def test_shrink_cov_output_is_finite_for_well_conditioned_input():
    data = _centered_pair_data(n_pairs=7, p=4)

    out = shrink_cov(data)

    assert np.isfinite(out).all()


def test_shrink_cov_handles_unequal_column_scales():
    # Deterministic, nontrivial covariance structure.
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [3.0, 4.0, 1.0],
            [4.0, 3.0, 2.0],
            [5.0, 6.0, 1.0],
            [6.0, 5.0, 0.0],
            [7.0, 8.0, 3.0],
            [8.0, 7.0, 2.0],
            [9.0, 10.0, 1.0],
            [10.0, 9.0, 0.0],
            [11.0, 12.0, 3.0],
            [12.0, 11.0, 2.0],
            [13.0, 14.0, 1.0],
        ],
        dtype=float,
    )

    out = shrink_cov(data, k=None)

    assert out.shape == (3, 3)
    assert np.allclose(out, out.T, atol=1e-12, rtol=1e-12)


def test_shrink_cov_raises_for_non_2d_input():
    data = np.arange(14, dtype=float)

    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(data)


def test_shrink_cov_raises_for_effective_sample_size_too_small_k_none():
    data = np.zeros((12, 3), dtype=float)

    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data, k=None)


def test_shrink_cov_raises_for_effective_sample_size_too_small_k_one():
    data = np.zeros((12, 3), dtype=float)

    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data, k=1)


def test_shrink_cov_raises_value_error_for_singular_matrix():
    # Third column is identically zero, so the covariance is singular.
    data = np.array(
        [
            [1.0, 2.0, 0.0],
            [2.0, 3.0, 0.0],
            [3.0, 4.0, 0.0],
            [4.0, 5.0, 0.0],
            [5.0, 6.0, 0.0],
            [6.0, 7.0, 0.0],
            [7.0, 8.0, 0.0],
            [8.0, 9.0, 0.0],
            [9.0, 10.0, 0.0],
            [10.0, 11.0, 0.0],
            [11.0, 12.0, 0.0],
            [12.0, 13.0, 0.0],
            [13.0, 14.0, 0.0],
            [14.0, 15.0, 0.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data, k=0)
