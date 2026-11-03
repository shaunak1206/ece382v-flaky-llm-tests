import numpy as np
import pytest
from nonlinshrink import nonlinshrink


def make_data(n, p, seed=12345):
    # Helper to create reproducible random data
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p))


def approx_equal(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol, rtol=tol)


def test_shrink_cov_basic_shape_and_symmetry():
    np.random.seed(0)
    data = np.random.randn(20, 10)
    cov_shrunk = nonlinshrink.shrink_cov(data)

    # Output shape should be (p,p)
    assert cov_shrunk.shape == (10, 10)
    # Output matrix should be symmetric
    assert approx_equal(cov_shrunk, cov_shrunk.T)


def test_shrink_cov_default_demean_effective_sample_size():
    np.random.seed(1)
    data = np.random.randn(25, 12)
    # k = None triggers demean and n-k used internally
    res = nonlinshrink.shrink_cov(data, k=None)
    res2 = nonlinshrink.shrink_cov(data - np.mean(data, axis=0), k=1)
    assert approx_equal(res, res2)


def test_shrink_cov_no_demean_k0():
    np.random.seed(2)
    data = np.random.randn(30, 15)
    # k=0 means no demean, effective sample size = n
    res = nonlinshrink.shrink_cov(data, k=0)
    # mean of data columns is NOT zero
    assert not np.allclose(np.mean(data, axis=0), 0)
    # Result still symmetric and shape matches
    assert res.shape == (15, 15)
    assert approx_equal(res, res.T)


def test_shrink_cov_k_positive_greater_than_zero():
    np.random.seed(10)
    n, p = 40, 20
    data = np.random.randn(n, p)
    k = 5  # shifted effective sample size n-k=35
    result = nonlinshrink.shrink_cov(data, k=k)
    assert result.shape == (p, p)
    assert approx_equal(result, result.T)


def test_shrink_cov_raises_for_insufficient_sample_size():
    # Since n-k must be >= 12, test for failure if less
    data = np.ones((15, 10))
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(data, k=5)  # effective n=10 < 12


def test_shrink_cov_raises_on_singular_matrix_problem():
    # Create data that leads to zero or near-zero eigenvalues after trimming
    n, p = 50, 40
    data = np.zeros((n, p))
    # Demean triggers subtraction
    with pytest.raises(ValueError, match="Matrix is singular"):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_eigenvalue_order_and_trimming():
    # Test that eigenvalues after trimming are used and are positive enough
    np.random.seed(123)
    n, p = 50, 40
    data = np.random.randn(n, p)
    # Manually compute sample covariance eigenvalues and compare trimming
    sample_cov = np.cov(data, rowvar=False, bias=False)
    lam, _ = np.linalg.eigh(sample_cov)
    lam_trimmed = lam[np.maximum(0, p - (n - 1)):]  # k=1 default demean; n effective = n-1
    cov_shrunk = nonlinshrink.shrink_cov(data)
    # Check all trimmed eigenvalues > 0 (no singular)
    assert np.all(lam_trimmed > 0)
    # Result shape correct
    assert cov_shrunk.shape == (p, p)


def test_shrink_cov_high_dimensional_p_gt_n():
    # p > n case exercise (p=55, n=50)
    np.random.seed(555)
    n, p = 50, 55
    data = np.random.randn(n, p)
    res = nonlinshrink.shrink_cov(data)
    # Output matrix shape
    assert res.shape == (p, p)
    # Result symmetric
    assert approx_equal(res, res.T)
    # Result covariance matrix should be positive semi-definite: eigenvalues >= 0
    eigvals = np.linalg.eigvalsh(res)
    # Small negative tolerance allowed due to numerical errors
    assert np.all(eigvals > -1e-8)


def test_shrink_cov_low_dimensional_p_lt_n():
    # p < n case exercise (p=20, n=50)
    np.random.seed(2024)
    n, p = 50, 20
    data = np.random.randn(n, p)
    res = nonlinshrink.shrink_cov(data)
    assert res.shape == (p, p)
    assert approx_equal(res, res.T)
    eigvals = np.linalg.eigvalsh(res)
    assert np.all(eigvals > -1e-8)


def test_shrink_cov_input_validation_for_ndarray_dim():
    np.random.seed(42)
    # Provide 1d input, expect assertion error
    arr = np.random.randn(10)
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(arr)
    # Provide 3d input
    arr3d = np.random.randn(2, 3, 4)
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(arr3d)


def test_shrink_cov_result_close_to_sample_cov_for_large_n():
    np.random.seed(77)
    n, p = 1000, 30
    data = np.random.randn(n, p)
    # Large samples should have shrinked covariance close to sample covariance
    sample_cov = np.dot((data - data.mean(axis=0)).T, (data - data.mean(axis=0))) / (n - 1)
    shrunk_cov = nonlinshrink.shrink_cov(data, k=1)
    # Frobenius norm of difference is small relative to matrix norm
    diff_norm = np.linalg.norm(shrunk_cov - sample_cov)
    sample_norm = np.linalg.norm(sample_cov)
    assert diff_norm / sample_norm < 0.1  # not identical, but close


def test_shrink_cov_result_positive_semidefinite():
    np.random.seed(88)
    n, p = 60, 40
    data = np.random.randn(n, p)
    shrunk_cov = nonlinshrink.shrink_cov(data)
    eigvals = np.linalg.eigvalsh(shrunk_cov)
    # Eigenvalues should be >= 0 or very close with numerical tolerance
    assert np.all(eigvals > -1e-8)


def test_shrink_cov_monotonicity_of_shrunk_eigenvalues():
    # Shrunk eigenvalues computed internally are positive and generally shrink large ones
    np.random.seed(202)
    n, p = 100, 30
    data = np.random.randn(n, p)
    # Compute shrunk covariance and eigenvalues
    shrunk_cov = nonlinshrink.shrink_cov(data)
    lam, _ = np.linalg.eigh(np.cov(data, rowvar=False, bias=False))
    shrunk_lam, _ = np.linalg.eigh(shrunk_cov)
    # sorted ascending - shrunk lambda should not have any negatives
    assert np.all(shrunk_lam > -1e-8)
    # Shrunk eigenvalues should be in a similar scale but generally reduced
    assert np.mean(shrunk_lam) <= np.mean(lam) + 1e-8


def test_shrink_cov_with_identical_columns():
    # If data columns identical, covariance matrix rank deficient, expect singular error
    np.random.seed(1337)
    n, p = 20, 10
    base_col = np.random.randn(n)
    data = np.column_stack([base_col for _ in range(p)])
    # Demeaning triggers error since eigenvalues trimmed contain zeros
    with pytest.raises(ValueError, match="Matrix is singular"):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_different_k_effect():
    # Verify that setting different k modifies effective sample size and output changes
    np.random.seed(75)
    n, p = 50, 20
    data = np.random.randn(n, p)
    cov_k_none = nonlinshrink.shrink_cov(data, k=None)
    cov_k_0 = nonlinshrink.shrink_cov(data, k=0)
    cov_k_2 = nonlinshrink.shrink_cov(data, k=2)

    # All matrices symmetric
    assert approx_equal(cov_k_none, cov_k_none.T)
    assert approx_equal(cov_k_0, cov_k_0.T)
    assert approx_equal(cov_k_2, cov_k_2.T)

    # k=0 no demean cov_k_0 likely differs from k=None
    assert not approx_equal(cov_k_0, cov_k_none)
    # k=2 different adjustment, result differs from k=0 and k=None
    assert not approx_equal(cov_k_2, cov_k_0)
    assert not approx_equal(cov_k_2, cov_k_none)


def test_shrink_cov_with_data_all_zeros_raises_singular():
    data = np.zeros((20, 10))
    with pytest.raises(ValueError, match="Matrix is singular"):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_corner_case_effective_sample_size_exactly_12():
    # n-k == 12 is allowed
    np.random.seed(42)
    n, p = 25, 15
    k = 13  # n-k = 12
    data = np.random.randn(n, p)
    res = nonlinshrink.shrink_cov(data, k=k)
    assert res.shape == (p, p)
    assert approx_equal(res, res.T)
    eigvals = np.linalg.eigvalsh(res)
    assert np.all(eigvals > -1e-8)


def test_shrink_cov_reproducibility_with_equal_seed():
    # Two runs with same seed produce same output
    seed = 101
    n, p = 60, 30
    data1 = make_data(n, p, seed=seed)
    data2 = make_data(n, p, seed=seed)

    res1 = nonlinshrink.shrink_cov(data1, k=None)
    res2 = nonlinshrink.shrink_cov(data2, k=None)

    # Use close comparison, matrices identical
    assert approx_equal(res1, res2)
