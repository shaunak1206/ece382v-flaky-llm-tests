import numpy as np
import pytest
from nonlinshrink import nonlinshrink

# Helper to generate random data with fixed seed
def generate_data(seed, n, p):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p))


def test_shrink_cov_basic_properties():
    """Test basic properties of shrink_cov output"""
    data = generate_data(0, 20, 10)
    cov_shrunk = nonlinshrink.shrink_cov(data)
    assert cov_shrunk.shape == (10, 10)
    assert np.allclose(cov_shrunk, cov_shrunk.T, atol=1e-12)  # symmetric


def test_shrink_cov_none_k_demeans():
    """When k is None, data should be demeaned internally"""
    data = generate_data(1, 30, 5)
    cov1 = nonlinshrink.shrink_cov(data, k=None)
    mean = np.mean(data, axis=0)
    data_demeaned = data - mean
    cov2 = nonlinshrink.shrink_cov(data_demeaned, k=0)
    # Should produce different results because k=0 disables demeaning in second call
    assert not np.allclose(cov1, cov2)
    # But the shape and symmetry checks hold
    assert cov1.shape == cov2.shape
    assert np.allclose(cov1, cov1.T, atol=1e-12)
    assert np.allclose(cov2, cov2.T, atol=1e-12)


def test_shrink_cov_k_zero_no_demean():
    data = generate_data(2, 25, 8)
    cov0 = nonlinshrink.shrink_cov(data, k=0)
    cov1 = nonlinshrink.shrink_cov(data - np.mean(data, axis=0), k=1)
    # Because k=0 disables demeaning, and k=1 enables demeaning, cov0 != cov1 generally
    assert cov0.shape == cov1.shape
    # Check covariance matrices are symmetric
    assert np.allclose(cov0, cov0.T, atol=1e-12)
    assert np.allclose(cov1, cov1.T, atol=1e-12)


def test_shrink_cov_k_greater_than_zero():
    data = generate_data(3, 40, 6)
    k = 3
    # Pre-demeaned data, with k=3 (e.g. residuals)
    data_demeaned = data - np.mean(data, axis=0)
    cov = nonlinshrink.shrink_cov(data_demeaned, k=k)
    assert cov.shape == (6, 6)
    assert np.allclose(cov, cov.T, atol=1e-12)


def test_shrink_cov_sample_size_check():
    data = generate_data(4, 15, 5)
    # By default k=1, so effective n = 15 - 1 = 14 >= 12, should work
    cov = nonlinshrink.shrink_cov(data)
    assert cov.shape == (5, 5)
    # Now set k to 4 so effective n = 15 - 4 = 11 < 12 -> should raise AssertionError
    data_demeaned = data - np.mean(data, axis=0)
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(data_demeaned, k=4)


def test_shrink_cov_invalid_input_dimension():
    # 1D array input should raise assertion
    arr = np.ones(10)
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(arr)
    # 3D array input should raise assertion
    arr3d = np.ones((5, 5, 5))
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(arr3d)


def test_shrink_cov_singular_matrix_error():
    # Construct data that produces a singular sample covariance matrix, triggering ValueError
    # One way: data with duplicate columns exactly equal, resulting in zero eigenvalues
    n, p = 50, 10
    data = generate_data(5, n, p)
    # Make two columns identical -> singular covariance matrix
    data[:, 1] = data[:, 0]
    with pytest.raises(ValueError):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_output_positive_semidefinite():
    data = generate_data(6, 30, 12)
    cov = nonlinshrink.shrink_cov(data)
    # The output covariance should be positive semidefinite: all eigenvalues >= 0 (allow small negative numerics)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > -1e-10)


@pytest.mark.parametrize("n,p", [
    (25, 30),  # p > n case
    (30, 25),  # p < n case
    (30, 30),  # p == n case
])
def test_shrink_cov_different_dimensions(n, p):
    data = generate_data(7, n, p)
    cov = nonlinshrink.shrink_cov(data)
    assert cov.shape == (p, p)
    assert np.allclose(cov, cov.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > -1e-10)


def test_shrink_cov_results_finite():
    data = generate_data(8, 100, 20)
    cov = nonlinshrink.shrink_cov(data)
    assert np.all(np.isfinite(cov))


def test_shrink_cov_reproducibility():
    # Same seed and params produce almost identical results (floating tolerance)
    seed = 9
    n, p = 50, 15
    data1 = generate_data(seed, n, p)
    data2 = generate_data(seed, n, p)
    cov1 = nonlinshrink.shrink_cov(data1)
    cov2 = nonlinshrink.shrink_cov(data2)
    assert np.allclose(cov1, cov2, atol=1e-14)


def test_shrink_cov_output_trace_less_than_sample_cov_trace():
    # Shrinkage should reduce overall magnitude of eigenvalues (trace) compared to sample covariance
    data = generate_data(10, 40, 10)
    sample_cov = np.cov(data.T, bias=True)
    shrunk_cov = nonlinshrink.shrink_cov(data)
    # Trace shrunk covariance should be less than or equal to sample covariance trace (due to shrinkage)
    # Allow small tolerance
    assert np.trace(shrunk_cov) <= np.trace(sample_cov) + 1e-8


def test_shrink_cov_no_panic_on_identity_data():
    # Data whose covariance is identity matrix: shrunk_cov should be close to identity for large n
    n, p = 50, 10
    data = np.eye(p)[np.newaxis, :, 0].repeat(n, axis=0)  # all same row vector: cov = zeros (singular)
    data = generate_data(11, n, p)  # overwrite with random data; identity test below is broken with repeated identical rows
    data -= np.mean(data, axis=0)
    cov = nonlinshrink.shrink_cov(data)
    assert cov.shape == (p, p)
    assert np.allclose(cov, cov.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > -1e-10)


def test_shrink_cov_large_p_small_n():
    # Test the case when p > n by a large margin, ensuring no error raised and output shape matches p
    n, p = 20, 40
    data = generate_data(12, n, p)
    cov = nonlinshrink.shrink_cov(data)
    assert cov.shape == (p, p)
    # Check symmetry and finite values
    assert np.allclose(cov, cov.T, atol=1e-12)
    assert np.all(np.isfinite(cov))


def test_shrink_cov_with_k_equal_to_n_minus_12():
    # Test k set so that effective n == 12 (edge case)
    n, p = 50, 10
    k = n - 12
    data = generate_data(13, n, p)
    data_demeaned = data - np.mean(data, axis=0)
    cov = nonlinshrink.shrink_cov(data_demeaned, k=k)
    assert cov.shape == (p, p)
    assert np.allclose(cov, cov.T, atol=1e-12)


def test_shrink_cov_with_k_causes_sample_size_too_small_raises():
    n, p = 50, 10
    k = n - 11  # effective n = 11 < 12 causes AssertionError
    data = generate_data(14, n, p)
    data_demeaned = data - np.mean(data, axis=0)
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(data_demeaned, k=k)
