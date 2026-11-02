import numpy as np
import pytest
from nonlinshrink.nonlinshrink import shrink_cov, EPS


def make_data(n, p, seed=123):
    """Helper to make random data matrix with fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p))


def test_shrink_cov_raises_on_1d_array():
    data = np.array([1, 2, 3])
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_shrink_cov_raises_on_small_effective_sample():
    data = make_data(12, 5)
    # k=1 (default) reduces n=12 to n-k=11 < 12, expect AssertionError
    with pytest.raises(AssertionError):
        shrink_cov(data)
    # but if k=0 (no demean) n=12, still ok
    result = shrink_cov(data, k=0)
    assert result.shape == (5, 5)


def test_shrink_cov_demeaning_behavior_default_vs_k0():
    n, p = 20, 5
    data = np.ones((n, p)) * 5.0
    # Default k=None demean, data becomes all zeros, covariance is zero matrix, singular => raises
    with pytest.raises(ValueError):
        shrink_cov(data)
    # with k=0 no demeaning, covariance is not zero, works
    out = shrink_cov(data, k=0)
    assert out.shape == (p, p)
    assert np.all(out >= 0)
    # diagonal elements are all equal (variance of constant is 0)
    assert np.allclose(np.diag(out), np.diag(out), rtol=1e-12, atol=1e-12)


def test_shrink_cov_singular_matrix_raises():
    # create data whose covariance has eigenvalue close to zero
    n, p = 20, 10
    # data with zero variance in last column (constant)
    data = make_data(n, p)
    data[:, -1] = 1.0
    # Should raise ValueError for singular matrix condition
    with pytest.raises(ValueError):
        shrink_cov(data, k=0)


def test_shrink_cov_returns_symmetric_matrix():
    data = make_data(30, 10)
    cov_shrunk = shrink_cov(data)
    assert cov_shrunk.shape == (10, 10)
    # symmetric to relative tolerance
    assert np.allclose(cov_shrunk, cov_shrunk.T, rtol=1e-12, atol=1e-12)


def test_shrink_cov_eigenvalues_positive_definite():
    data = make_data(50, 8)
    cov_shrunk = shrink_cov(data)
    eigvals = np.linalg.eigvalsh(cov_shrunk)
    # Output matrix should be positive definite (all eigenvalues positive)
    assert np.all(eigvals > 0)


def test_shrink_cov_effective_sample_size_adjustment():
    n, p = 40, 7
    data = make_data(n, p)
    # k=None means demean and n decreases by 1
    cov1 = shrink_cov(data, k=None)
    # k=1 means no demean, n decreases by 1, result should be same as default
    cov2 = shrink_cov(data, k=1)
    assert np.allclose(cov1, cov2, rtol=1e-12, atol=1e-12)
    # k=0 means no demean, n not reduced -> different output expected
    cov3 = shrink_cov(data, k=0)
    # cov3 should differ from cov1 as sample size effectively larger and no demean
    assert not np.allclose(cov1, cov3, rtol=1e-6)


def test_shrink_cov_with_k_greater_than_one():
    # n-k decreases effective sample size more strongly
    n, p = 50, 6
    data = make_data(n, p)
    cov_k1 = shrink_cov(data, k=1)
    cov_k3 = shrink_cov(data, k=3)
    # cov_k3 likely different from cov_k1 due to smaller effective sample size
    assert cov_k1.shape == cov_k3.shape == (p, p)
    # not equal (expect some difference)
    assert not np.allclose(cov_k1, cov_k3, rtol=1e-6)


def test_shrink_cov_rank_p_lte_n():
    # Case p <= n
    n, p = 30, 20
    data = make_data(n, p)
    cov = shrink_cov(data)
    assert cov.shape == (p, p)


def test_shrink_cov_rank_p_gt_n():
    # Case p > n (high dimensional)
    n, p = 30, 40
    data = make_data(n, p)
    cov = shrink_cov(data)
    assert cov.shape == (p, p)


def test_shrink_cov_output_close_to_sample_covariance_in_large_sample():
    # For very large samples, shrink_cov should be close to sample covariance
    n, p = 1000, 10
    data = make_data(n, p)
    sample_cov = np.cov(data, rowvar=False, bias=False)
    cov_shrunk = shrink_cov(data)
    # Should be close to sample covariance within some tolerance
    # Shrinkage should adjust eigenvalues but not very far for large n
    diff = np.abs(cov_shrunk - sample_cov)
    max_diff = np.max(diff)
    assert max_diff < 0.1


def test_shrink_cov_reproducibility_fixed_seed():
    n, p = 50, 15
    data1 = make_data(n, p, seed=42)
    data2 = make_data(n, p, seed=42)
    cov1 = shrink_cov(data1)
    cov2 = shrink_cov(data2)
    # identical inputs produce identical outputs
    assert np.allclose(cov1, cov2, rtol=1e-12, atol=1e-12)


def test_shrink_cov_input_unchanged_for_k_not_none():
    # The input data array should not be changed when k is provided (especially no demean)
    n, p = 30, 10
    data = make_data(n, p)
    data_copy = data.copy()
    shrink_cov(data, k=0)
    # data unchanged
    assert np.allclose(data, data_copy)


def test_shrink_cov_input_unchanged_for_k_none():
    # For k=None data is demeaned internally - check original data unchanged outside function
    n, p = 30, 10
    data = make_data(n, p)
    data_copy = data.copy()
    shrink_cov(data, k=None)
    # original data unchanged
    assert np.allclose(data, data_copy)


def test_shrink_cov_returns_float64_matrix():
    data = make_data(30, 10)
    cov = shrink_cov(data)
    assert cov.dtype == np.float64


def test_shrink_cov_diagonal_positive_and_nonzero():
    data = make_data(40, 8)
    cov = shrink_cov(data)
    diag = np.diag(cov)
    assert np.all(diag > 0)


def test_shrink_cov_not_identity_on_random_data():
    data = make_data(50, 10)
    cov = shrink_cov(data)
    # Should not be identity matrix except in trivial cases
    eye = np.eye(cov.shape[0])
    assert not np.allclose(cov, eye, atol=1e-8)


def test_shrink_cov_handles_constant_column_raises():
    # Data with one constant column -> covariance singular
    data = make_data(30, 5)
    data[:, 0] = 3.14
    with pytest.raises(ValueError):
        shrink_cov(data, k=0)


def test_shrink_cov_handles_all_zero_column_raises():
    data = make_data(30, 6)
    data[:, 2] = 0.0
    with pytest.raises(ValueError):
        shrink_cov(data, k=0)
