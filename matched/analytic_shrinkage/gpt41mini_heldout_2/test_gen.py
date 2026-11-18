import numpy as np
import pytest
from nonlinshrink.nonlinshrink import shrink_cov, EPS


def generate_data(n, p, seed=0):
    np.random.seed(seed)
    return np.random.randn(n, p)


def test_output_shape_small_dim():
    n, p = 50, 10
    data = generate_data(n, p)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)


def test_output_symmetric():
    n, p = 100, 20
    data = generate_data(n, p)
    shrunk = shrink_cov(data)
    assert np.allclose(shrunk, shrunk.T, atol=1e-10)


def test_no_demean_k_zero():
    n, p = 60, 12
    data = generate_data(n, p)
    # With k=0, data remains unchanged
    cov_with_demean = shrink_cov(data)
    cov_no_demean = shrink_cov(data, k=0)
    # They should differ, since demeaning changes effective sample size and data
    assert not np.allclose(cov_with_demean, cov_no_demean)


def test_k_greater_than_one_adjusts_effective_sample_size():
    n, p = 70, 15
    data = generate_data(n, p)
    # With k=5, effective sample size = n - 5
    cov_k5 = shrink_cov(data, k=5)
    cov_k1 = shrink_cov(data, k=1)
    # They should differ
    assert not np.allclose(cov_k5, cov_k1)


def test_error_on_1d_array():
    data = np.array([1, 2, 3, 4])
    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(data)


def test_error_if_effective_sample_size_less_than_12():
    # n=12, k=1 => n-k=11 < 12 triggers error
    n, p = 12, 5
    data = generate_data(n, p)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data)  # default k=1 => 12-1=11 < 12


def test_no_error_if_effective_sample_size_exactly_12():
    # n=13, k=1 => n-k=12 passes
    n, p = 13, 5
    data = generate_data(n, p)
    # Should not raise
    shrink_cov(data)


def test_singular_matrix_raises_value_error():
    # Create data that yields singular covariance: set two cols equal
    n, p = 50, 10
    data = generate_data(n, p)
    data[:, 1] = data[:, 0]  # second column equals first col → singular covariance
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data)


def test_shrink_cov_output_positive_semidefinite():
    n, p = 100, 20
    data = generate_data(n, p)
    shrunk = shrink_cov(data)
    eigvals = np.linalg.eigvalsh(shrunk)
    # Shrunk covariance should be PSD; eigenvalues ≥ 0 (allow small neg tolerance for numerical error)
    assert np.all(eigvals > -1e-10)


def test_nonlinshrink_with_large_dimensions():
    n, p = 150, 140
    data = generate_data(n, p)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)


def test_nonlinshrink_high_dimensional_p_greater_than_n():
    n, p = 90, 120
    data = generate_data(n, p)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)


def test_result_is_finite():
    n, p = 100, 20
    data = generate_data(n, p)
    shrunk = shrink_cov(data)
    assert np.all(np.isfinite(shrunk))


def test_shrink_cov_with_constant_column():
    # Add a constant column to data, which should reduce rank and cause singularity error
    n, p = 50, 10
    data = generate_data(n, p)
    data[:, 0] = 3.14  # constant column
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_shrink_cov_with_highly_collinear_columns():
    n, p = 60, 10
    data = generate_data(n, p)
    data[:, 1] = data[:, 0] * 2 + np.random.normal(0, 1e-12, size=n)  # almost collinear
    # Might or might not raise error depending on precision - test that code runs or raises appropriately
    try:
        shrink_cov(data)
    except ValueError as ve:
        assert "singular" in str(ve).lower()


def test_shrink_cov_with_k_none_demeans_data():
    n, p = 40, 5
    data = generate_data(n, p)
    out1 = shrink_cov(data)
    # Manually demean data and pass k=0 should match closely (not exact due to effective n)
    demeaned = data - np.mean(data, axis=0)
    out2 = shrink_cov(demeaned, k=0)
    # They differ because effective sample sizes differ (k=1 vs k=0)
    assert not np.allclose(out1, out2)


def test_shrink_cov_constant_all_columns():
    # Data matrix all same value, covariance zero matrix except singular
    n, p = 50, 10
    data = np.ones((n, p)) * 7
    # With default k=1 demeaning, covariance is zero matrix singular → raises error
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data)


def test_behavior_when_all_columns_are_zero():
    n, p = 50, 5
    data = np.zeros((n, p))
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data)


def test_repeatability_given_fixed_seed():
    n, p = 80, 8
    data = generate_data(n, p)
    shrunk1 = shrink_cov(data)
    shrunk2 = shrink_cov(data)
    assert np.allclose(shrunk1, shrunk2)


def test_nonlinshrink_with_k_equal_n_minus_12():
    n, p = 30, 10
    data = generate_data(n, p)
    # k = n-12 makes effective sample size = 12, the minimum allowed
    k = n - 12
    shrunk = shrink_cov(data, k=k)
    assert shrunk.shape == (p, p)


def test_nonlinshrink_raises_if_k_too_large():
    n, p = 25, 5
    data = generate_data(n, p)
    k = n - 11  # effective n = 11 < 12 triggers error
    with pytest.raises(AssertionError):
        shrink_cov(data, k=k)
