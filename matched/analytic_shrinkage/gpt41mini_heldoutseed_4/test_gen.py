import numpy as np
import pytest
from nonlinshrink.nonlinshrink import shrink_cov, EPS


def generate_data(n, p, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p))


def test_shrink_cov_basic_shape():
    data = generate_data(50, 10)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (10, 10)
    assert np.allclose(shrunk, shrunk.T)  # Symmetric matrix


def test_shrink_cov_demeans_default():
    data = generate_data(50, 10)
    # Calculate sample covariance manually with demeaned data
    data_demeaned = data - np.mean(data, axis=0)
    sample_cov = np.dot(data_demeaned.T, data_demeaned) / (data.shape[0] - 1)
    shrunk = shrink_cov(data)
    # Shrunk covariance is not equal to sample covariance but should have similar trace range
    assert np.isclose(np.trace(shrunk), pytest.approx(np.trace(sample_cov), rel=0.5))


def test_shrink_cov_no_demean_k_zero():
    data = generate_data(50, 10)
    # Sample covariance without demeaning, divisor = n
    sample_cov = np.dot(data.T, data) / data.shape[0]
    shrunk = shrink_cov(data, k=0)
    assert shrunk.shape == sample_cov.shape
    assert np.allclose(shrunk, shrunk.T)
    # Trace should be roughly similar to sample_cov (no demeaning)
    assert np.isclose(np.trace(shrunk), pytest.approx(np.trace(sample_cov), rel=0.5))


def test_shrink_cov_k_greater_than_one_dem_meaned_once():
    data = generate_data(50, 8)
    # Subtract mean then pass k=5 so effective sample size = n-k
    data_demeaned = data - np.mean(data, axis=0)
    shrunk = shrink_cov(data_demeaned, k=5)
    assert shrunk.shape == (8, 8)
    assert np.allclose(shrunk, shrunk.T)


def test_shrink_cov_effective_sample_size_check():
    # n - k must >= 12; so n=15, k=3 is ok, k=4 too big and should assert
    data = generate_data(15, 5)
    shrink_cov(data, k=3)  # should be ok
    with pytest.raises(AssertionError):
        shrink_cov(data, k=4)


def test_shrink_cov_input_not_2d():
    data = np.arange(10)  # 1d array
    with pytest.raises(AssertionError) as e:
        shrink_cov(data)
    assert 'input must be a 2d array' in str(e.value)


def test_shrink_cov_singular_matrix_raises():
    # Construct matrix with zero variance variable (singular)
    data = generate_data(20, 5)
    data[:, 0] = 1.0  # zero variance column
    with pytest.raises(ValueError) as e:
        shrink_cov(data)
    assert "Matrix is singular" in str(e.value)


def test_shrink_cov_high_dimensional_p_greater_than_n():
    # p > n case, e.g. n=20, p=25
    data = generate_data(20, 25)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (25, 25)
    assert np.allclose(shrunk, shrunk.T)


def test_shrink_cov_returns_positive_semidefinite():
    # Covariance matrix should be positive semidefinite
    data = generate_data(40, 10)
    shrunk = shrink_cov(data)
    eigs = np.linalg.eigvalsh(shrunk)
    assert np.all(eigs >= -1e-10)  # allow small negative eps for numerical


def test_shrink_cov_consistency_for_same_data():
    data = generate_data(60, 8, seed=42)
    shrunk1 = shrink_cov(data)
    shrunk2 = shrink_cov(data)
    assert np.allclose(shrunk1, shrunk2)


def test_shrink_cov_effect_of_k_none_vs_k_1():
    data = generate_data(50, 7)
    # Calling with k=None should demean and use k=1 internally
    shrunk_none = shrink_cov(data)
    shrunk_one = shrink_cov(data, k=1)
    # Outcome matrices should be close but not necessarily exact because k=1 assumes data demeaned already
    assert shrunk_none.shape == shrunk_one.shape
    assert np.allclose(shrunk_none, shrunk_one, atol=1e-5)


def test_shrink_cov_small_sample_size_raises():
    data = generate_data(13, 4)
    with pytest.raises(AssertionError):
        # with default k=1, effective sample size 12 is ok, 13-1=12
        shrink_cov(data, k=2)  # 13-2=11 < 12, should raise


def test_shrink_cov_large_data():
    n, p = 200, 50
    data = generate_data(n, p)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)
    # Test symmetry
    assert np.allclose(shrunk, shrunk.T)


def test_shrink_cov_diagonal_structure_for_independent_data():
    # If variables are independent, sample cov is near diagonal
    rng = np.random.default_rng(123)
    data = rng.normal(size=(100, 5))
    shrunk = shrink_cov(data)
    off_diag_sum = np.sum(np.abs(shrunk - np.diag(np.diag(shrunk))))
    # Nonlinear shrinkage should preserve approximate diagonal dominance
    assert off_diag_sum < 0.5


def test_shrink_cov_input_all_zeros():
    data = np.zeros((20, 6))
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data)


def test_shrink_cov_input_constant_columns():
    data = np.ones((30, 3))
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data)


def test_shrink_cov_monotonic_eigenvalues():
    data = generate_data(50, 8)
    shrunk = shrink_cov(data)
    eigvals = np.linalg.eigvalsh(shrunk)
    # Eigenvalues of covariance matrix should be non-negative and in ascending order
    assert np.all(eigvals[:-1] <= eigvals[1:])
    assert np.all(eigvals >= -EPS)


def test_shrink_cov_returns_real_values():
    data = generate_data(60, 10)
    shrunk = shrink_cov(data)
    # Covariance matrix elements should be all real numbers (no complex)
    assert not np.iscomplexobj(shrunk)


def test_shrink_cov_handles_float32_input():
    data = generate_data(50, 7).astype(np.float32)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (7, 7)
    # Output should be float64 (due to numpy defaults)
    assert shrunk.dtype == np.float64


def test_shrink_cov_correct_output_for_scaled_data():
    data = generate_data(40, 5)
    scale = 5.0
    data_scaled = data * scale
    shrunk_orig = shrink_cov(data)
    shrunk_scaled = shrink_cov(data_scaled)
    # Shrunk covariance should scale by scale^2
    ratio = shrunk_scaled / (shrunk_orig * scale ** 2)
    # Ratios close to one
    assert np.allclose(ratio, 1, rtol=1e-5, atol=1e-6)


def test_shrink_cov_returns_numpy_array():
    data = generate_data(30, 4)
    shrunk = shrink_cov(data)
    assert isinstance(shrunk, np.ndarray)


def test_shrink_cov_input_with_nan_raises():
    data = generate_data(20, 5)
    data[0, 0] = np.nan
    with pytest.raises(np.linalg.LinAlgError):
        shrink_cov(data)
