import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _random_data(n, p, seed=0):
    rs = np.random.RandomState(seed)
    return rs.randn(n, p)


def test_output_shape_is_p_by_p():
    data = _random_data(50, 5)
    out = shrink_cov(data)
    assert out.shape == (5, 5)


def test_returns_numpy_array():
    data = _random_data(40, 4)
    out = shrink_cov(data)
    assert isinstance(out, np.ndarray)


def test_output_is_symmetric():
    data = _random_data(60, 6, seed=1)
    out = shrink_cov(data)
    np.testing.assert_allclose(out, out.T, rtol=1e-10, atol=1e-12)


def test_output_is_positive_definite_low_dim():
    data = _random_data(80, 5, seed=2)
    out = shrink_cov(data)
    assert np.linalg.eigvalsh(out).min() > 0


def test_output_is_finite_high_dim():
    data = _random_data(20, 30, seed=3)
    out = shrink_cov(data)
    assert out.shape == (30, 30)
    assert np.all(np.isfinite(out))


def test_output_is_positive_definite_high_dim():
    data = _random_data(20, 30, seed=4)
    out = shrink_cov(data)
    assert np.linalg.eigvalsh(out).min() > 0


def test_default_demeaning_invariant_to_shift():
    data = _random_data(50, 4, seed=5)
    out_raw = shrink_cov(data)
    out_shifted = shrink_cov(data + 100.0)
    np.testing.assert_allclose(out_shifted, out_raw, rtol=1e-6, atol=1e-8)


def test_k_one_on_demeaned_data_matches_default():
    data = _random_data(50, 4, seed=6)
    demeaned = data - data.mean(axis=0)
    out_default = shrink_cov(data)
    out_k1 = shrink_cov(demeaned, k=1)
    np.testing.assert_allclose(out_k1, out_default, rtol=1e-10, atol=1e-12)


def test_k_zero_differs_from_default_when_mean_nonzero():
    data = _random_data(50, 4, seed=7) + 3.0
    out_default = shrink_cov(data)
    out_k0 = shrink_cov(data, k=0)
    assert not np.allclose(out_k0, out_default)


def test_k_two_runs_and_returns_valid_matrix():
    data = _random_data(50, 4, seed=8)
    demeaned = data - data.mean(axis=0)
    out = shrink_cov(demeaned, k=2)
    assert out.shape == (4, 4)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, out.T, rtol=1e-10, atol=1e-12)


def test_eigenvectors_shared_with_sample_covariance():
    # The shrunk matrix is diagonal in the sample eigenbasis, so it commutes
    # with the sample covariance matrix.
    data = _random_data(60, 5, seed=9)
    n = data.shape[0]
    centered = data - data.mean(axis=0)
    sample = centered.T @ centered / (n - 1)
    out = shrink_cov(data)
    np.testing.assert_allclose(out @ sample, sample @ out, rtol=1e-6, atol=1e-8)


def test_invariant_to_row_permutation():
    data = _random_data(50, 4, seed=10)
    perm = np.random.RandomState(11).permutation(data.shape[0])
    np.testing.assert_allclose(
        shrink_cov(data[perm]), shrink_cov(data), rtol=1e-8, atol=1e-10
    )


def test_scale_equivariance():
    data = _random_data(50, 4, seed=12)
    c = 3.0
    np.testing.assert_allclose(
        shrink_cov(c * data), (c ** 2) * shrink_cov(data), rtol=1e-6, atol=1e-8
    )


def test_input_not_modified():
    data = _random_data(50, 4, seed=13)
    original = data.copy()
    shrink_cov(data)
    np.testing.assert_array_equal(data, original)


def test_deterministic_repeated_calls():
    data = _random_data(50, 4, seed=14)
    np.testing.assert_array_equal(shrink_cov(data), shrink_cov(data))


def test_large_sample_close_to_sample_covariance():
    # With many observations and few variables the shrinkage is negligible.
    data = _random_data(20000, 3, seed=15)
    n = data.shape[0]
    centered = data - data.mean(axis=0)
    sample = centered.T @ centered / (n - 1)
    out = shrink_cov(data)
    np.testing.assert_allclose(out, sample, rtol=0.05, atol=0.05)


def test_1d_input_raises_assertion():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(20, dtype=float))


def test_3d_input_raises_assertion():
    with pytest.raises(AssertionError):
        shrink_cov(np.zeros((20, 3, 2)))


def test_too_few_observations_default_k_raises():
    # n=12 with default demeaning gives effective sample size 11 < 12.
    data = _random_data(12, 3, seed=16)
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_minimum_effective_sample_size_accepted():
    # n=13 with default demeaning gives effective sample size 12.
    data = _random_data(13, 3, seed=17)
    out = shrink_cov(data)
    assert out.shape == (3, 3)


def test_k_zero_allows_exactly_twelve_observations():
    data = _random_data(12, 3, seed=18)
    out = shrink_cov(data, k=0)
    assert out.shape == (3, 3)


def test_constant_column_raises_value_error():
    data = _random_data(40, 4, seed=19)
    data[:, 1] = 7.0
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_duplicate_columns_raise_value_error():
    data = _random_data(40, 4, seed=20)
    data[:, 3] = data[:, 0]
    with pytest.raises(ValueError):
        shrink_cov(data)
