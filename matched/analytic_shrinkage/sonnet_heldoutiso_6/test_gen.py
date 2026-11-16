import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _random_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p)


def test_output_shape_p_less_than_n():
    data = _random_data(50, 5)
    result = shrink_cov(data)
    assert result.shape == (5, 5)


def test_output_is_symmetric():
    data = _random_data(60, 8, seed=1)
    result = shrink_cov(data)
    np.testing.assert_allclose(result, result.T, rtol=1e-8, atol=1e-10)


def test_output_is_finite_and_positive_definite():
    data = _random_data(80, 10, seed=2)
    result = shrink_cov(data)
    assert np.all(np.isfinite(result))
    eigvals = np.linalg.eigvalsh((result + result.T) / 2)
    assert np.all(eigvals > 0)


def test_one_dimensional_input_raises():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(50, dtype=float))


def test_three_dimensional_input_raises():
    with pytest.raises(AssertionError):
        shrink_cov(np.zeros((20, 3, 2)))


def test_too_few_samples_default_k_raises():
    # 12 rows with k=None gives effective sample size 11 < 12
    data = _random_data(12, 2, seed=3)
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_minimum_sample_size_accepted_with_k_zero():
    data = _random_data(12, 2, seed=4)
    result = shrink_cov(data, k=0)
    assert result.shape == (2, 2)
    assert np.all(np.isfinite(result))


def test_too_few_samples_with_k_raises():
    data = _random_data(13, 2, seed=5)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=2)


def test_default_k_equals_k1_on_demeaned_data():
    data = _random_data(40, 4, seed=6)
    demeaned = data - data.mean(axis=0)
    expected = shrink_cov(demeaned, k=1)
    actual = shrink_cov(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_k_zero_differs_from_default_on_nonzero_mean_data():
    data = _random_data(40, 4, seed=7) + 5.0
    default = shrink_cov(data)
    no_demean = shrink_cov(data, k=0)
    assert not np.allclose(default, no_demean)


def test_k_changes_effective_sample_size():
    data = _random_data(50, 4, seed=8)
    r1 = shrink_cov(data, k=1)
    r2 = shrink_cov(data, k=2)
    assert r1.shape == r2.shape
    assert not np.allclose(r1, r2)


def test_singular_matrix_raises_value_error():
    data = _random_data(40, 3, seed=9)
    data = np.hstack([data, data[:, :1]])  # duplicated column -> singular
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_zero_column_raises_value_error():
    data = _random_data(40, 3, seed=10)
    data[:, 1] = 0.0
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_high_dimensional_p_greater_than_n():
    data = _random_data(20, 30, seed=11)
    result = shrink_cov(data, k=0)
    assert result.shape == (30, 30)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, result.T, rtol=1e-8, atol=1e-10)


def test_does_not_modify_input():
    data = _random_data(40, 5, seed=12) + 2.0
    original = data.copy()
    shrink_cov(data)
    np.testing.assert_array_equal(data, original)


def test_deterministic():
    data = _random_data(40, 5, seed=13)
    np.testing.assert_array_equal(shrink_cov(data), shrink_cov(data))


def test_scale_equivariance():
    data = _random_data(60, 6, seed=14)
    c = 3.0
    base = shrink_cov(data)
    scaled = shrink_cov(c * data)
    np.testing.assert_allclose(scaled, c ** 2 * base, rtol=1e-5, atol=1e-8)


def test_row_permutation_invariance():
    data = _random_data(50, 5, seed=15)
    perm = np.random.RandomState(99).permutation(50)
    np.testing.assert_allclose(shrink_cov(data), shrink_cov(data[perm]),
                               rtol=1e-6, atol=1e-8)


def test_shrinkage_reduces_condition_number():
    data = _random_data(100, 50, seed=16)
    centered = data - data.mean(axis=0)
    sample = centered.T.dot(centered) / 99
    shrunk = shrink_cov(data)
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample)


def test_close_to_identity_for_identity_covariance_large_sample():
    data = _random_data(500, 5, seed=17)
    result = shrink_cov(data)
    assert np.max(np.abs(result - np.eye(5))) < 0.3
