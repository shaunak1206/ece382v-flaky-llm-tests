import numpy as np
import pytest

from nonlinshrink import nonlinshrink


def test_shrink_cov_basic_shape():
    np.random.seed(0)
    data = np.random.randn(50, 10)
    cov = nonlinshrink.shrink_cov(data)
    assert cov.shape == (10, 10)
    assert np.allclose(cov, cov.T, atol=1e-12), "Covariance not symmetric"


def test_shrink_cov_demeaning_equivalence():
    np.random.seed(1)
    data = np.random.randn(100, 5)

    cov_default = nonlinshrink.shrink_cov(data)
    cov_k1 = nonlinshrink.shrink_cov(data, k=1)  # no demeaning, effective n = n-1
    # k=1 means data is assumed demeaned, so we manually demean and pass k=1
    data_demeaned = data - np.mean(data, axis=0)
    cov_manual = nonlinshrink.shrink_cov(data_demeaned, k=1)

    # Default (k=None) demeans and then uses effective n = n-1, so should match manual demean + k=1
    assert np.allclose(cov_default, cov_manual, atol=1e-12)
    assert not np.allclose(cov_default, cov_k1), "k=1 without demeaning should differ"


def test_shrink_cov_k0_no_demean():
    np.random.seed(2)
    data = np.random.randn(60, 8)
    cov_k0 = nonlinshrink.shrink_cov(data, k=0)  # no demeaning, effective n = n
    data_demeaned = data - np.mean(data, axis=0)
    cov_default = nonlinshrink.shrink_cov(data)  # demean + effective n = n-1
    # They should differ
    assert not np.allclose(cov_k0, cov_default)


def test_shrink_cov_k_greater_than_1_adjusts_effective_n():
    np.random.seed(3)
    data = np.random.randn(30, 7)
    # k = 5 means no demeaning, effective n = n-5
    cov_k5 = nonlinshrink.shrink_cov(data, k=5)
    # k = 0 means no demeaning, effective n = n
    cov_k0 = nonlinshrink.shrink_cov(data, k=0)
    assert cov_k5.shape == (7, 7)
    # Should differ due to effective sample size
    assert not np.allclose(cov_k5, cov_k0)


def test_shrink_cov_raises_for_small_sample_size():
    # small n - k resulting in < 12
    data = np.random.randn(13, 5)
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(data, k=2)  # 13 - 2 = 11 < 12


def test_shrink_cov_raises_for_non_2d_input():
    data = np.random.randn(10)
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_singular_matrix_raises():
    # Create data that leads to singular covariance by having repeated columns
    n, p = 20, 10
    data = np.random.randn(n, p)
    # Set last column equal to first column (exact linear dependence)
    data[:, -1] = data[:, 0]
    # Expect ValueError about matrix singular
    with pytest.raises(ValueError):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_returns_positive_semidefinite():
    np.random.seed(4)
    data = np.random.randn(50, 10)
    cov = nonlinshrink.shrink_cov(data)
    eigvals = np.linalg.eigvalsh(cov)
    # eigenvalues should be >= 0 or close to it, allowing numerical errors
    assert np.all(eigvals > -1e-12), "Covariance matrix not positive semi-definite"


def test_shrink_cov_high_dimensional_p_gt_n():
    np.random.seed(5)
    n = 30
    p = 50
    data = np.random.randn(n, p)
    cov = nonlinshrink.shrink_cov(data)
    assert cov.shape == (p, p)
    # Check symmetry
    assert np.allclose(cov, cov.T, atol=1e-12)


def test_shrink_cov_output_close_to_sample_cov_for_large_n():
    # For very large sample size, nonlinear shrinkage should be close to sample covariance
    np.random.seed(6)
    n = 1000
    p = 10
    data = np.random.randn(n, p)
    sample_cov = (data - np.mean(data, axis=0)).T @ (data - np.mean(data, axis=0)) / (n - 1)
    cov_shrunk = nonlinshrink.shrink_cov(data)
    # They need not be extremely close, but fairly close as shrinkage impact decreases with large n
    diff = np.linalg.norm(cov_shrunk - sample_cov, ord='fro') / np.linalg.norm(sample_cov, ord='fro')
    assert diff < 0.1


def test_shrink_cov_with_identical_rows():
    # Data with identical rows results in singular covariance and should raise
    data = np.ones((15, 5))
    with pytest.raises(ValueError):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_with_constant_columns():
    # Data with constant column should cause a singular eigenvalue near zero
    np.random.seed(7)
    data = np.random.randn(20, 6)
    data[:, 0] = 1.0  # constant column
    with pytest.raises(ValueError):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_non_numeric_input():
    data = np.array([["a", "b"], ["c", "d"]])
    with pytest.raises(TypeError):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_large_values_stability():
    np.random.seed(8)
    n, p = 100, 10
    data = np.random.randn(n, p) * 1e6
    cov = nonlinshrink.shrink_cov(data)
    assert cov.shape == (p, p)


def test_shrink_cov_effective_sample_size_change_affects_result():
    np.random.seed(9)
    data = np.random.randn(30, 7)
    cov_k1 = nonlinshrink.shrink_cov(data, k=1)
    cov_k3 = nonlinshrink.shrink_cov(data, k=3)
    assert not np.allclose(cov_k1, cov_k3)


def test_shrink_cov_k_none_demeans_data():
    np.random.seed(10)
    data = np.random.randn(40, 5) + 100
    cov_none = nonlinshrink.shrink_cov(data)
    # If demeaned manually and k=1, should produce same result as k=None
    data_demeaned = data - np.mean(data, axis=0)
    cov_manual = nonlinshrink.shrink_cov(data_demeaned, k=1)
    assert np.allclose(cov_none, cov_manual)


def test_shrink_cov_identity_data():
    # Construct data with identity covariance
    np.random.seed(11)
    p = 6
    n = 50
    data = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=n)
    cov_shrunk = nonlinshrink.shrink_cov(data)
    # The shrunk covariance should be close to identity
    eigvals = np.linalg.eigvalsh(cov_shrunk)
    assert np.all(eigvals > 0)
    assert np.allclose(np.diag(cov_shrunk), np.ones(p), atol=0.25)


def test_shrink_cov_effective_sample_size_error_k_too_large():
    # k too large so n - k < 12
    data = np.random.randn(50, 5)
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(data, k=50)


def test_shrink_cov_accepts_float_data():
    # data as float32 works without error
    np.random.seed(12)
    data = np.random.randn(60, 10).astype(np.float32)
    cov = nonlinshrink.shrink_cov(data)
    assert cov.shape == (10, 10)


def test_shrink_cov_with_nan_raises():
    data = np.random.randn(20, 5)
    data[0, 0] = np.nan
    with pytest.raises(Exception):
        nonlinshrink.shrink_cov(data)


def test_shrink_cov_with_inf_raises():
    data = np.random.randn(20, 5)
    data[1, 1] = np.inf
    with pytest.raises(Exception):
        nonlinshrink.shrink_cov(data)
