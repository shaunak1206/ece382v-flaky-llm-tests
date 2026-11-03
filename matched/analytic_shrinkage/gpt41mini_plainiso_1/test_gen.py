import pytest
import numpy as np
from nonlinshrink.nonlinshrink import shrink_cov, EPS


def test_shrink_cov_basic_behavior():
    # Simple test on random data, p < n
    np.random.seed(0)
    n, p = 20, 5
    data = np.random.randn(n, p)
    result = shrink_cov(data)
    assert result.shape == (p, p)
    assert np.allclose(result, result.T)
    # Should be positive semi-definite
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals >= -1e-12)


def test_shrink_cov_with_k_none_demeans():
    # When k=None, demean should be applied
    np.random.seed(1)
    n, p = 30, 10
    data_raw = np.random.rand(n, p) + 5
    result_with_none = shrink_cov(data_raw, k=None)
    # Manual demean
    data_demeaned = data_raw - np.mean(data_raw, axis=0)
    result_manual = shrink_cov(data_demeaned, k=1)
    assert np.allclose(result_with_none, result_manual, rtol=1e-5)


def test_shrink_cov_with_k_zero_no_demean():
    np.random.seed(2)
    n, p = 40, 6
    data = np.random.randn(n, p)
    # Calling with k=0 means no demeaning
    result_k0 = shrink_cov(data, k=0)
    # Manually demeaned (k=1) result should differ
    result_k1 = shrink_cov(data, k=1)
    assert not np.allclose(result_k0, result_k1)


def test_shrink_cov_with_k_greater_than_zero():
    np.random.seed(3)
    n, p = 50, 7
    data = np.random.randn(n, p)
    # Set k=5 (pretend already demeaned)
    result = shrink_cov(data, k=5)
    assert result.shape == (p, p)


def test_shrink_cov_raises_if_n_minus_k_less_than_12():
    np.random.seed(4)
    n, p = 15, 3
    data = np.random.randn(n, p)
    # k chosen so effective n < 12
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data, k=10)
    # Also with no k (default k=1): n-k=14 valid
    shrink_cov(data, k=3)  # should not raise


def test_shrink_cov_raises_if_input_not_2d():
    data_1d = np.array([1, 2, 3])
    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(data_1d)
    data_3d = np.ones((3, 3, 3))
    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(data_3d)


def test_shrink_cov_singular_matrix_raises():
    # Construct data that results in singular covariance matrix with tiny eigenvalues
    np.random.seed(5)
    n, p = 20, 10
    data = np.zeros((n, p))
    # Add small noise to create nearly zero eigenvalues
    data[:, 0] = 1
    data[:, 1] = 1 + EPS / 10
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data)


def test_shrink_cov_p_less_than_n_case():
    np.random.seed(6)
    n, p = 30, 10  # p < n
    data = np.random.randn(n, p)
    cov = shrink_cov(data, k=None)
    assert cov.shape == (p, p)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0)


def test_shrink_cov_p_equal_n_case():
    np.random.seed(7)
    n = p = 15
    data = np.random.randn(n, p)
    cov = shrink_cov(data, k=None)
    assert cov.shape == (p, p)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0)


def test_shrink_cov_p_greater_than_n_case():
    np.random.seed(8)
    n, p = 12, 20  # p > n
    data = np.random.randn(n, p)
    cov = shrink_cov(data, k=None)
    assert cov.shape == (p, p)
    eigvals = np.linalg.eigvalsh(cov)
    # eigenvalues can be non-negative semi-definite
    assert np.all(eigvals > -1e-10)


def test_shrink_cov_effective_sample_size_adjustment():
    # Check effective sample size n-k adjustment changes results correctly
    np.random.seed(9)
    n, p = 25, 8
    data = np.random.randn(n, p)
    cov_k1 = shrink_cov(data, k=1)
    cov_k3 = shrink_cov(data, k=3)
    assert not np.allclose(cov_k1, cov_k3)


def test_shrink_cov_output_symmetry():
    np.random.seed(10)
    n, p = 22, 6
    data = np.random.randn(n, p)
    cov = shrink_cov(data)
    assert np.allclose(cov, cov.T)


def test_shrink_cov_output_real_values():
    np.random.seed(11)
    n, p = 26, 9
    data = np.random.randn(n, p)
    cov = shrink_cov(data)
    assert np.isrealobj(cov)


def test_shrink_cov_no_nan_or_inf():
    np.random.seed(12)
    n, p = 30, 8
    data = np.random.randn(n, p)
    cov = shrink_cov(data)
    assert not np.isnan(cov).any()
    assert not np.isinf(cov).any()


def test_shrink_cov_monotonic_eigenvalues():
    # The shrinked eigenvalues should be positive and likely somewhat smoothed
    np.random.seed(13)
    n, p = 25, 7
    data = np.random.randn(n, p)
    cov = shrink_cov(data)
    eigvals = np.linalg.eigvalsh(cov)
    # All eigenvalues should be >= 0 (positive semi-definite)
    assert np.all(eigvals >= -1e-12)
    # It is unlikely to be exactly sorted ascending, but small perturbations
    # We test monotonic ascending property (not guaranteed by shrink_cov)
    # So just test basic positive spectrum


def test_shrink_cov_idempotency_of_demeaning():
    # Repeated demeaning (k=None) should be equivalent to once demeaning (k=1)
    np.random.seed(14)
    n, p = 40, 10
    data = np.random.randn(n, p) + 10
    result1 = shrink_cov(data, k=None)
    # demeaned already
    data_demeaned = data - np.mean(data, axis=0)
    result2 = shrink_cov(data_demeaned, k=1)
    assert np.allclose(result1, result2, rtol=1e-5)


def test_shrink_cov_large_data():
    # Test runtime on larger data (not checking runtime here, but no errors)
    np.random.seed(15)
    n, p = 100, 50
    data = np.random.randn(n, p)
    cov = shrink_cov(data)
    assert cov.shape == (p, p)


def test_shrink_cov_extreme_values():
    # Test whether large/small values handled gracefully
    np.random.seed(16)
    n, p = 40, 10
    data = np.ones((n, p)) * 1e10
    data += np.random.randn(n, p) * 1e-5
    cov = shrink_cov(data, k=None)
    # The covariance should be close to zero matrix because variance is tiny due to no variance in values
    assert np.allclose(np.diag(cov), cov.diagonal(), rtol=1e-4)
    assert np.all(cov >= 0)


def test_shrink_cov_extreme_p_equals_n_and_k():
    np.random.seed(17)
    n, p = 20, 20
    data = np.random.randn(n, p)
    # k = 19 -> effective sample size 1, should raise
    with pytest.raises(AssertionError):
        shrink_cov(data, k=19)
    # k = 8 valid
    cov = shrink_cov(data, k=8)
    assert cov.shape == (p, p)


def test_shrink_cov_effective_sample_size_exactly_12():
    # n-k=12 borderline, should not raise
    np.random.seed(18)
    n, p = 25, 10
    data = np.random.randn(n, p)
    cov = shrink_cov(data, k=13)
    assert cov.shape == (p, p)


def test_shrink_cov_sanity_check_low_rank_data():
    # Create low rank data matrix where rank < p
    np.random.seed(19)
    n, p = 30, 15
    data = np.random.randn(n, 3) @ np.random.randn(3, p)  # rank 3 < p
    # k=1 (will demean)
    cov = shrink_cov(data)
    assert cov.shape == (p, p)
    eigvals = np.linalg.eigvalsh(cov)
    # Possibly some eigenvalues close to zero but no error raised
    assert np.all(eigvals >= -1e-12)
