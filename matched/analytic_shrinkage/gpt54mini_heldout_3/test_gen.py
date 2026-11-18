import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _rng(seed=0):
    return np.random.default_rng(seed)


def _random_data(n, p, seed=0, shift=0.0, scale=1.0):
    rng = _rng(seed)
    return shift + scale * rng.standard_normal((n, p))


def test_shrink_cov_requires_2d_input():
    x = np.arange(20.0)
    with pytest.raises(AssertionError, match="2d array"):
        shrink_cov(x)


def test_shrink_cov_requires_sample_size_at_least_12_with_default_k():
    data = _random_data(12, 3, seed=1)
    # Default k=None subtracts 1 from the effective sample size.
    with pytest.raises(AssertionError, match="n must be >= 12"):
        shrink_cov(data)


def test_shrink_cov_requires_sample_size_at_least_12_with_explicit_k():
    data = _random_data(11, 3, seed=2)
    with pytest.raises(AssertionError, match="n must be >= 12"):
        shrink_cov(data, k=0)


def test_shrink_cov_matches_centered_data_when_k_is_one():
    data = _random_data(20, 5, seed=3, shift=7.5)
    centered = data - np.mean(data, axis=0)

    out_default = shrink_cov(data)
    out_centered = shrink_cov(centered, k=1)

    assert np.allclose(out_default, out_centered, atol=1e-10, rtol=1e-10)


def test_shrink_cov_output_is_symmetric_for_p_le_n():
    data = _random_data(24, 8, seed=4)
    out = shrink_cov(data, k=0)

    assert out.shape == (8, 8)
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_output_is_symmetric_for_p_greater_than_n():
    data = _random_data(15, 25, seed=5)
    out = shrink_cov(data, k=0)

    assert out.shape == (25, 25)
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_returns_finite_values_for_full_rank_input():
    data = _random_data(30, 10, seed=6)
    out = shrink_cov(data, k=0)

    assert np.isfinite(out).all()


def test_shrink_cov_is_deterministic_for_same_input():
    data = _random_data(18, 6, seed=7)
    out1 = shrink_cov(data, k=0)
    out2 = shrink_cov(data, k=0)

    assert np.allclose(out1, out2, atol=0.0, rtol=0.0)


def test_shrink_cov_no_demeaning_differs_from_default_on_shifted_data():
    data = _random_data(22, 7, seed=8, shift=100.0)

    out_no_centering = shrink_cov(data, k=0)
    out_default = shrink_cov(data)  # centers internally

    assert not np.allclose(out_no_centering, out_default)


def test_shrink_cov_raises_value_error_on_singular_covariance_with_duplicate_columns():
    base = _random_data(20, 4, seed=9)
    data = np.column_stack([base, base[:, [0]]])  # duplicated column -> singular

    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data, k=0)


def test_shrink_cov_raises_value_error_on_singular_covariance_with_zero_column():
    data = _random_data(18, 5, seed=10)
    data[:, 2] = 0.0

    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data, k=0)


def test_shrink_cov_works_for_high_dimensional_case_p_greater_than_n():
    data = _random_data(14, 30, seed=11)
    out = shrink_cov(data, k=0)

    assert out.shape == (30, 30)
    assert np.isfinite(out).all()
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_preserves_positive_diagonal_entries_on_valid_input():
    data = _random_data(25, 9, seed=12)
    out = shrink_cov(data, k=0)

    assert np.all(np.diag(out) > 0)
