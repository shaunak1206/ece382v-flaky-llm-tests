import numpy as np
import pytest
from nonlinshrink import nonlinshrink


def base_data_seed(seed=12345):
    """Generate reproducible test data with fixed PRNG seed."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(50, 10))


def test_input_2d_enforcement():
    # Non-2D input should raise AssertionError
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(np.array([1, 2, 3]))
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(np.array([[[1]]]))


def test_effective_sample_size_none_and_k():
    data = base_data_seed()
    # k is None: data demeaned; effective n = n -1
    cov1 = nonlinshrink.shrink_cov(data, k=None)
    # k=0: no demean; effective n = n
    cov2 = nonlinshrink.shrink_cov(data, k=0)
    # k=1: no demean; n = n - 1
    cov3 = nonlinshrink.shrink_cov(data, k=1)
    # They differ since demean changes covariance
    assert not np.allclose(cov1, cov2)
    assert np.allclose(cov1, cov3, atol=1e-6)


def test_output_shape_is_square(data=None):
    if data is None:
        data = base_data_seed()
    cov = nonlinshrink.shrink_cov(data)
    p = data.shape[1]
    assert cov.shape == (p, p)


def test_output_cov_symmetric():
    data = base_data_seed()
    cov = nonlinshrink.shrink_cov(data)
    assert np.allclose(cov, cov.T, atol=1e-12)


def test_raise_if_effective_sample_size_too_small():
    data = base_data_seed()
    # Use k that makes effective n < 12
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(data, k=40)


def test_singular_matrix_detection():
    # Construct data with colinear columns => singular covariance
    data = np.ones((20, 5))
    # Add small noise to all but one column to avoid exact singularity
    data[:, 1:] = np.eye(20, 4)
    # Since eigenvalues will contain zeros, expect ValueError ("Matrix is singular")
    with pytest.raises(ValueError):
        nonlinshrink.shrink_cov(data)


def test_output_positive_semidefinite():
    data = base_data_seed()
    cov = nonlinshrink.shrink_cov(data)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-10)  # allow small negative numerical errors


def test_output_similarity_to_sample_cov_when_large_n():
    # When n >> p, shrinkage effect diminishes, output close to sample cov
    rng = np.random.default_rng(2023)
    n = 500
    p = 20
    data = rng.standard_normal(size=(n, p))
    # Default k=None means demean
    shrunk_cov = nonlinshrink.shrink_cov(data)
    sample_cov = (data - data.mean(0)).T @ (data - data.mean(0)) / (n - 1)
    # Shrunk covariance close to sample covariance, but slightly different
    assert np.allclose(shrunk_cov, sample_cov, rtol=1e-2, atol=1e-2)
    assert not np.allclose(shrunk_cov, sample_cov, atol=1e-6)  # not identical


def test_behavior_when_p_greater_than_n():
    # p > n triggers the else branch
    rng = np.random.default_rng(999)
    n = 30
    p = 50
    data = rng.standard_normal(size=(n, p))
    shrunk_cov = nonlinshrink.shrink_cov(data)
    # Output shape match p x p
    assert shrunk_cov.shape == (p, p)
    # Symmetric PSD
    eigvals = np.linalg.eigvalsh(shrunk_cov)
    assert np.all(eigvals >= -1e-10)


def test_effect_of_k_parameter_none_vs_zero():
    data = base_data_seed()
    # Explicitly test k=None vs k=0 effect on demean and output
    out_none = nonlinshrink.shrink_cov(data, k=None)
    out_zero = nonlinshrink.shrink_cov(data, k=0)
    # Generally, with demeaned data vs non-demeaned data covariances differ
    assert not np.allclose(out_none, out_zero)


def test_output_positive_definite_for_random_covariance():
    # Generate covariance from multivariate normal with known covariance
    rng = np.random.default_rng(42)
    p = 15
    cov_true = np.eye(p) * 3.0 + 0.3  # diagonally dominant
    mean = np.zeros(p)
    data = rng.multivariate_normal(mean, cov_true, size=100)
    shrunk = nonlinshrink.shrink_cov(data)
    eigvals = np.linalg.eigvalsh(shrunk)
    assert np.all(eigvals > -1e-14)


def test_no_modification_of_input_data():
    data = base_data_seed()
    data_copy = data.copy()
    _ = nonlinshrink.shrink_cov(data, k=None)
    # data should stay unchanged outside function since demean inside does copy
    assert np.allclose(data, data_copy)


def test_function_produces_consistent_result_on_same_input():
    # Multiple calls produce identical output
    data = base_data_seed(555)
    res1 = nonlinshrink.shrink_cov(data, k=None)
    res2 = nonlinshrink.shrink_cov(data, k=None)
    assert np.allclose(res1, res2)


def test_small_covariance_matrix_k0():
    # Small covariance matrix with no demean
    data = base_data_seed()
    cov_k0 = nonlinshrink.shrink_cov(data, k=0)
    n, p = data.shape
    # effective sample size = n
    expected_n = n
    # Check symmetry and positive semi-definite
    assert cov_k0.shape == (p, p)
    assert np.allclose(cov_k0, cov_k0.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(cov_k0)
    assert np.all(eigvals >= -1e-10)


def test_k_greater_than_one_behavior():
    # k > 1 means data already demeaned; effective sample size = n - k
    data = base_data_seed()
    # Use k=5, effective n = 50-5=45 >=12 satisfies the assertion
    cov_k5 = nonlinshrink.shrink_cov(data, k=5)
    assert isinstance(cov_k5, np.ndarray)
    assert cov_k5.shape[0] == cov_k5.shape[1] == data.shape[1]


def test_extreme_values_do_not_break():
    # Data with large magnitude values
    rng = np.random.default_rng(777)
    data = rng.normal(0, 1e6, size=(50, 10))
    cov = nonlinshrink.shrink_cov(data)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-10)


def test_identity_data():
    # Data drawn from identity covariance (i.i.d standard normal)
    rng = np.random.default_rng(2024)
    data = rng.standard_normal(size=(60, 15))
    cov = nonlinshrink.shrink_cov(data)
    p = data.shape[1]
    assert cov.shape == (p, p)
    # Should be symmetric and PSD
    assert np.allclose(cov, cov.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(cov)
    assert np.all(eigs >= -1e-10)


def test_nonlinshrink_eps_threshold_behavior():
    # Construct very small eigenvalues to test singular matrix detection threshold EPS
    # We'll craft a covariance matrix with some eigenvalues extremely small
    
    n, p = 30, 10
    # Create data with mostly unit variance and some columns almost constant
    data = np.random.RandomState(10).normal(0, 1, size=(n, p))
    data[:, 0] = 1e-14  # almost constant column, causing near singularity
    
    # k=None applies demean and reduces n by 1 => effective n = 29 still >= 12
    with pytest.raises(ValueError):
        _ = nonlinshrink.shrink_cov(data, k=None)


def test_k_equal_n_minus_12_min_sample_size():
    # Pick k = n-12 so effective n = 12 minimum allowed
    data = base_data_seed()
    n, p = data.shape
    k = n - 12
    cov = nonlinshrink.shrink_cov(data, k=k)
    assert cov.shape == (p, p)


def test_reject_k_too_large():
    # k that leads to effective sample size < 12 rejected
    data = base_data_seed()
    n, p = data.shape
    k = n - 11  # effective n = 11 <12, should error
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(data, k=k)
