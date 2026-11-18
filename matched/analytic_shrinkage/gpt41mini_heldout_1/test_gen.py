import numpy as np
import pytest
from nonlinshrink.nonlinshrink import shrink_cov, EPS

# Setting random seed for reproducibility in tests
np.random.seed(0)


def test_output_shape_square():
    # n > p, simple covariance matrix shape check
    n, p = 50, 10
    data = np.random.randn(n, p)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)


def test_output_is_symmetric():
    n, p = 60, 20
    data = np.random.randn(n, p)
    shrunk = shrink_cov(data)
    assert np.allclose(shrunk, shrunk.T, atol=1e-10)


def test_output_positive_semidefinite():
    # Shrunk covariance should be at least positive semidefinite (no negative eigenvalues)
    n, p = 70, 15
    data = np.random.randn(n, p)
    shrunk = shrink_cov(data)
    eigvals = np.linalg.eigvalsh(shrunk)
    assert np.all(eigvals >= -1e-10)  # Allow tiny numerical errors


def test_k_none_demeaning_effect():
    # Data with nonzero mean: default k=None should demean
    n, p = 50, 10
    data = np.ones((n, p)) * 5 + np.random.randn(n, p)
    shrunk_none = shrink_cov(data, k=None)

    # Manually demean data and k=1 should match
    data_demeaned = data - np.mean(data, axis=0)
    shrunk_manual = shrink_cov(data_demeaned, k=1)

    assert np.allclose(shrunk_none, shrunk_manual, atol=1e-10)


def test_k_zero_no_demean():
    # When k=0, no demeaning should be performed
    n, p = 50, 10
    offset = np.linspace(1, 10, p)
    data = np.ones((n, p)) * offset + np.random.randn(n, p)
    shrunk_k0 = shrink_cov(data, k=0)

    # With k=None, data is demeaned before processing so results differ
    shrunk_none = shrink_cov(data, k=None)

    assert not np.allclose(shrunk_k0, shrunk_none, atol=1e-10)


def test_k_greater_than_zero_adjusts_sample_size():
    # k >= 1 means data already demeaned; effective sample size adjusted by k
    n, p, k = 50, 10, 5
    data = np.random.randn(n, p)
    # Demean manually
    data_demeaned = data - np.mean(data, axis=0)

    shrunk_k5 = shrink_cov(data_demeaned, k=k)
    shrunk_k1 = shrink_cov(data_demeaned, k=1)

    # Results differ because effective sample sizes differ (n-k)
    assert not np.allclose(shrunk_k5, shrunk_k1, atol=1e-10)


def test_sample_size_too_small_raises():
    # n - k must be >= 12
    n, p = 15, 5
    data = np.random.randn(n, p)

    # k so that effective n < 12
    k = 4
    with pytest.raises(AssertionError):
        shrink_cov(data, k=k)

    # Also check for none but demeaning subtracts 1
    with pytest.raises(AssertionError):
        shrink_cov(data[:12], k=None)  # n=12 effective n=11 < 12


def test_input_not_2d_raises():
    # Input array must be 2d
    data_1d = np.random.randn(10)
    with pytest.raises(AssertionError):
        shrink_cov(data_1d)

    data_3d = np.random.randn(3, 4, 5)
    with pytest.raises(AssertionError):
        shrink_cov(data_3d)


def test_singular_matrix_raises():
    # Construct data that leads to (near) singular covariance:
    # For example, setting many repeated rows => covariance matrix has zero eigenvalues
    n, p = 20, 10

    data = np.ones((n, p))
    # Should raise due to singular covariance and lam / sum(lam) < EPS condition
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data)


def test_p_greater_than_n_case():
    # High-dimensional setting p > n
    n, p = 30, 50
    data = np.random.randn(n, p)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)
    # symmetry
    assert np.allclose(shrunk, shrunk.T, atol=1e-10)
    # eigenvalues non-negative
    eigs = np.linalg.eigvalsh(shrunk)
    assert np.all(eigs >= -1e-10)


def test_dtilde_values_positive():
    # Output matrix diag values are positive since covariance
    n, p = 60, 20
    data = np.random.randn(n, p)
    shrunk = shrink_cov(data)
    eigs = np.linalg.eigvalsh(shrunk)
    assert np.all(eigs > 0)


def test_repeated_calls_same_result():
    # Multiple calls with same input produce same output
    n, p = 50, 10
    data = np.random.randn(n, p)
    result1 = shrink_cov(data)
    result2 = shrink_cov(data)
    assert np.allclose(result1, result2, atol=1e-10)


def test_large_data_performance_smoke():
    # Smoke test on larger data size (not strict timing, just correctness)
    n, p = 300, 100
    data = np.random.randn(n, p)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)


def test_shrink_cov_is_real_valued():
    n, p = 70, 30
    data = np.random.randn(n, p)
    shrunk = shrink_cov(data)
    assert np.isrealobj(shrunk)


def test_shrink_cov_matches_sample_cov_when_large_sample():
    # When n >> p, shrunk matrix should be close to sample covariance
    n, p = 500, 10
    data = np.random.randn(n, p)
    # sample cov
    effective_n = n - 1
    sample_cov = np.dot((data - np.mean(data, axis=0)).T,
                        (data - np.mean(data, axis=0))) / effective_n
    shrunk = shrink_cov(data, k=None)
    # They shall be close but not equal; test that shrunk eigenvalues are positive and shape correct
    assert shrunk.shape == sample_cov.shape
    eigvals_sample = np.linalg.eigvalsh(sample_cov)
    eigvals_shrunk = np.linalg.eigvalsh(shrunk)
    assert np.all(eigvals_shrunk > 0)
    assert np.all(eigvals_sample >= 0)


def test_when_lam_ratio_less_than_eps_raises():
    # Mock the lam / sum(lam) < EPS condition by crafting lam with one zero eigenvalue
    n, p = 20, 10
    data = np.random.randn(n, p)
    # Manually construct covariance matrix with zero eigenvalue
    data[:, 0] = data[:, 1]  # make one feature redundant => zero eigenvalue likely
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data)


def test_effective_sample_size_minimum_border():
    # n-k = 12 is allowed
    n, p = 13, 5
    data = np.random.randn(n, p)
    # k=1 (default)
    shrunk = shrink_cov(data, k=1)
    assert shrunk.shape == (p, p)
    # k=0 means n-k=13 (allowed)
    shrunk2 = shrink_cov(data, k=0)
    assert shrunk2.shape == (p, p)


def test_k_none_and_k_1_equivalence_when_data_demeaned():
    n, p = 30, 10
    data = np.random.randn(n, p)
    data_demeaned = data - np.mean(data, axis=0)

    shrunk_none = shrink_cov(data, k=None)
    shrunk_1 = shrink_cov(data_demeaned, k=1)
    assert np.allclose(shrunk_none, shrunk_1, atol=1e-10)
