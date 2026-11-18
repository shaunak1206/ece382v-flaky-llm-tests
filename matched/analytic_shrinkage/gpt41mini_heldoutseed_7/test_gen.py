import numpy as np
import pytest
from nonlinshrink import nonlinshrink


def test_shrink_cov_basic_shape():
    np.random.seed(0)
    data = np.random.randn(50, 10)
    cov_shrunk = nonlinshrink.shrink_cov(data)
    assert cov_shrunk.shape == (10, 10)
    assert np.allclose(cov_shrunk, cov_shrunk.T), "Covariance matrix not symmetric"


def test_shrink_cov_output_positive_semidefinite():
    np.random.seed(1)
    data = np.random.randn(60, 20)
    cov_shrunk = nonlinshrink.shrink_cov(data)
    eigvals = np.linalg.eigvalsh(cov_shrunk)
    assert np.all(eigvals >= -1e-10), "Shrunk covariance is not positive semidefinite"


def test_shrink_cov_with_k_none_demeaning():
    np.random.seed(2)
    data = np.random.randn(40, 15)
    data_demeaned = data - np.mean(data, axis=0)
    cov1 = nonlinshrink.shrink_cov(data, k=None)
    cov2 = nonlinshrink.shrink_cov(data_demeaned, k=1)
    assert np.allclose(cov1, cov2, atol=1e-8)


def test_shrink_cov_with_k_zero_no_demeaning():
    np.random.seed(3)
    data = np.random.randn(30, 8)
    cov_no_demean = nonlinshrink.shrink_cov(data, k=0)
    assert cov_no_demean.shape == (8, 8)


def test_shrink_cov_with_k_positive_adjusted_sample_size():
    np.random.seed(4)
    data = np.random.randn(25, 10)
    cov_k3 = nonlinshrink.shrink_cov(data, k=3)
    cov_k_none = nonlinshrink.shrink_cov(data, k=None)
    # sample size adjusted: n - k
    # Shapes always same, just test runs without error and returns matrix
    assert cov_k3.shape == (10, 10)
    assert cov_k_none.shape == (10, 10)
    assert np.allclose(cov_k3, cov_k3.T)
    assert np.allclose(cov_k_none, cov_k_none.T)


def test_shrink_cov_raises_error_if_shape_not_2d():
    arr_1d = np.array([1, 2, 3])
    arr_3d = np.ones((3, 3, 3))
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(arr_1d)
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(arr_3d)


def test_shrink_cov_raises_error_if_effective_sample_too_small():
    data = np.ones((13, 5))
    # effective sample size will be 13-2=11 < 12 -> error
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        nonlinshrink.shrink_cov(data, k=2)


def test_shrink_cov_singular_matrix_raises_value_error():
    # Construct data so that after trimming lam some values are zero causing ValueError
    # Use p > n scenario for that
    np.random.seed(42)
    n, p = 12, 20
    data = np.random.randn(n, p)
    # Add collinear columns to cause singularity
    data[:, -1] = data[:, 0]
    with pytest.raises(ValueError, match="Matrix is singular"):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_returns_cov_close_to_sample_cov_for_large_n_p():
    np.random.seed(5)
    n, p = 200, 50
    data = np.random.randn(n, p)
    cov_shrunk = nonlinshrink.shrink_cov(data)
    sample_cov = np.cov(data, rowvar=False, bias=True)
    # Shrunk covariance should be symmetric and somewhat close to sample covariance but more stable
    assert cov_shrunk.shape == (p, p)
    diff = np.linalg.norm(cov_shrunk - sample_cov, ord='fro') / (p ** 2)
    assert diff < 0.05


def test_shrink_cov_p_greater_than_n_behavior():
    # p > n high-dimensional setting
    np.random.seed(6)
    n, p = 30, 40
    data = np.random.randn(n, p)
    cov_shrunk = nonlinshrink.shrink_cov(data)
    assert cov_shrunk.shape == (p, p)
    eigvals = np.linalg.eigvalsh(cov_shrunk)
    assert np.all(eigvals >= -1e-10)


def test_shrink_cov_p_equal_n_behavior():
    # p = n edge case
    np.random.seed(7)
    n = p = 25
    data = np.random.randn(n, p)
    cov_shrunk = nonlinshrink.shrink_cov(data)
    assert cov_shrunk.shape == (p, p)


def test_shrink_cov_with_constant_column():
    # One column constant (zero variance), should raise singular error
    np.random.seed(8)
    n, p = 20, 10
    data = np.random.randn(n, p)
    data[:, 5] = 3.14  # constant column
    with pytest.raises(ValueError, match="Matrix is singular"):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_preserves_trace_with_demeaning():
    np.random.seed(9)
    n, p = 50, 10
    data = np.random.randn(n, p)
    cov_shrunk = nonlinshrink.shrink_cov(data)
    sample_cov = np.cov(data - data.mean(axis=0), rowvar=False, bias=True)
    trace_shrunk = np.trace(cov_shrunk)
    trace_sample = np.trace(sample_cov)
    # Trace should be similar but not necessarily equal (analytic shrinkage)
    assert abs(trace_shrunk - trace_sample) / trace_sample < 0.2


def test_shrink_cov_preserves_trace_without_demeaning():
    np.random.seed(10)
    n, p = 40, 15
    data = np.random.randn(n, p)
    cov_shrunk = nonlinshrink.shrink_cov(data, k=0)
    sample_cov = np.cov(data, rowvar=False, bias=True)
    trace_shrunk = np.trace(cov_shrunk)
    trace_sample = np.trace(sample_cov)
    assert abs(trace_shrunk - trace_sample) / trace_sample < 0.2


def test_shrink_cov_large_dimensions():
    np.random.seed(11)
    n, p = 100, 90
    data = np.random.randn(n, p)
    cov_shrunk = nonlinshrink.shrink_cov(data)
    assert cov_shrunk.shape == (p, p)
    eigvals = np.linalg.eigvalsh(cov_shrunk)
    assert np.all(eigvals >= -1e-10)


def test_shrink_cov_nonrandom_input_identity():
    # Use identity matrix data for which covariance is also identity
    p = 20
    data = np.eye(p)
    # Duplicate rows to have n > p
    data = np.vstack([data] * 15)
    cov_shrunk = nonlinshrink.shrink_cov(data, k=0)
    assert cov_shrunk.shape == (20, 20)
    # Should be close to identity (shrinkage might do minimal bias)
    assert np.allclose(np.diag(cov_shrunk), 1, atol=0.2)
    assert np.allclose(cov_shrunk, cov_shrunk.T)


def test_shrink_cov_all_zero_input():
    data = np.zeros((20, 10))
    # This will fail because of singular covariance (all zeros)
    with pytest.raises(ValueError, match="Matrix is singular"):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_near_singular_but_positive_definite():
    # Create matrix almost singular but perturb slightly
    np.random.seed(12)
    A = np.ones((30, 10))
    A[:, 0] += np.linspace(0, 1, 30)  # slight perturbation
    with pytest.raises(AssertionError):
        # effective sample size after demeaning is 29 - 1 = 28 >= 12, so no error here.
        # But covariance matrix is singular due to collinear columns after trimming eigenvalues
        nonlinshrink.shrink_cov(A)


def test_shrink_cov_returns_finite_matrix():
    np.random.seed(13)
    data = np.random.randn(60, 30)
    cov = nonlinshrink.shrink_cov(data)
    assert np.all(np.isfinite(cov)), "Covariance matrix contains non-finite values"


def test_shrink_cov_input_not_float_raises_no_error():
    # Integer inputs coerced to float internally should work fine
    np.random.seed(14)
    data = np.random.randint(-10, 10, size=(50, 5))
    cov = nonlinshrink.shrink_cov(data)
    assert cov.shape == (5, 5)


def test_shrink_cov_large_k_reduces_effective_sample_size():
    np.random.seed(15)
    data = np.random.randn(35, 10)
    with pytest.raises(AssertionError):
        # effective sample size = 35 - 30 = 5 < 12 raises error
        nonlinshrink.shrink_cov(data, k=30)


def test_shrink_cov_use_effective_sample_size_correctly():
    np.random.seed(16)
    data = np.random.randn(50, 10)
    # Demean internally (k=None): effective n = 50 - 1 = 49
    cov1 = nonlinshrink.shrink_cov(data, k=None)
    # Manually demean and k=1: effective n = 50 - 1 = 49
    data_demeaned = data - np.mean(data, axis=0)
    cov2 = nonlinshrink.shrink_cov(data_demeaned, k=1)
    assert np.allclose(cov1, cov2, atol=1e-8)
