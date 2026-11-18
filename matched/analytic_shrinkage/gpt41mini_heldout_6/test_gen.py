import numpy as np
import pytest
from nonlinshrink import nonlinshrink


def test_output_shape_square_covariance():
    np.random.seed(0)
    data = np.random.randn(50, 10)
    shrunk = nonlinshrink.shrink_cov(data)
    assert shrunk.shape == (10, 10)


def test_output_symmetric():
    np.random.seed(1)
    data = np.random.randn(60, 15)
    shrunk = nonlinshrink.shrink_cov(data)
    assert np.allclose(shrunk, shrunk.T, atol=1e-10)


def test_shrink_cov_no_demean_k0():
    # k=0 disables demeaning
    np.random.seed(2)
    data = np.random.randn(40, 5)
    result_0 = nonlinshrink.shrink_cov(data, k=0)
    # If we demean manually and use k=1, result should differ
    data_demeaned = data - np.mean(data, axis=0)
    result_1 = nonlinshrink.shrink_cov(data_demeaned, k=1)
    # They should not be identical
    assert not np.allclose(result_0, result_1)


def test_shrink_cov_pre_demeaned_data():
    # Supply k >= 1 to indicate data already demeaned
    np.random.seed(3)
    data = np.random.randn(30, 7)
    data_demeaned = data - np.mean(data, axis=0)
    result = nonlinshrink.shrink_cov(data_demeaned, k=1)
    # Result is symmetric square matrix shape (p,p)
    assert result.shape == (7, 7)
    assert np.allclose(result, result.T)


def test_assert_2d_input():
    arr_1d = np.array([1, 2, 3])
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(arr_1d)


def test_assert_sample_size_too_small():
    data = np.ones((12, 3))
    # k=None => effective n = 12 - 1 = 11 < 12 triggers assert
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(data)
    # k=0 => effective n = 12, should pass
    # but covariance is zero matrix and eigenvalues will be zero -> singular matrix error
    with pytest.raises(ValueError):
        nonlinshrink.shrink_cov(data, k=0)


def test_singular_matrix_detection():
    # Construct data with zero variance in one dimension => zero eigenvalue
    data = np.ones((20, 5))
    # k=0 means no demeaning, effective n =20
    with pytest.raises(ValueError):
        nonlinshrink.shrink_cov(data, k=0)


def test_large_p_greater_than_n():
    # Here p > n testing code branch with p > n
    np.random.seed(10)
    data = np.random.randn(10, 15)  # n=10, p=15
    # Provide demeaned data with correct k
    data = data - np.mean(data, axis=0)

    cov = nonlinshrink.shrink_cov(data, k=1)
    assert cov.shape == (15, 15)
    assert np.allclose(cov, cov.T, atol=1e-10)


def test_large_n_greater_than_p():
    np.random.seed(11)
    data = np.random.randn(30, 8)  # n=30, p=8
    cov = nonlinshrink.shrink_cov(data)
    assert cov.shape == (8, 8)
    assert np.allclose(cov, cov.T, atol=1e-10)


def test_effective_sample_size_computation():
    np.random.seed(11)
    data = np.random.randn(15, 4)

    # k=None means demean, n_eff = n - 1
    cov1 = nonlinshrink.shrink_cov(data, k=None)

    # k=1 means pre-demeaned data, subtract k from n
    data_demeaned = data - np.mean(data, axis=0)
    cov2 = nonlinshrink.shrink_cov(data_demeaned, k=1)

    # They should be different matrices due to data and k handling
    assert not np.allclose(cov1, cov2)


def test_output_positive_semidefinite():
    # Check that shrunk covariance matrix is positive semi-definite
    np.random.seed(20)
    data = np.random.randn(25, 10)
    shrunk = nonlinshrink.shrink_cov(data)
    eigvals = np.linalg.eigvalsh(shrunk)
    # Should be all >= 0 to numerical precision (allow tiny negative due to float)
    assert np.all(eigvals > -1e-12)


def test_result_close_to_sample_cov_for_large_n():
    # For very large n compared to p, shrunk covariance approaches sample covariance
    np.random.seed(21)
    n, p = 200, 10
    data = np.random.randn(n, p)
    sample_cov = np.dot((data - np.mean(data, axis=0)).T, (data - np.mean(data, axis=0))) / (n - 1)
    shrunk_cov = nonlinshrink.shrink_cov(data)
    diff_norm = np.linalg.norm(shrunk_cov - sample_cov, ord='fro')
    # Difference should be relatively small for large n
    assert diff_norm < 1.0


def test_behavior_with_k_greater_than_1():
    # k specifies how many regressors or degrees of freedom were used in pre-processing
    np.random.seed(22)
    data = np.random.randn(50, 12)
    data_demeaned = data - np.mean(data, axis=0)
    # k=5 simulates pre-demeaned or residual data with 5 regressors
    shrunk1 = nonlinshrink.shrink_cov(data_demeaned, k=5)
    shrunk2 = nonlinshrink.shrink_cov(data_demeaned, k=1)
    # Should differ due to different effective sample size
    assert not np.allclose(shrunk1, shrunk2)


def test_float32_input():
    # Input data with dtype float32, function should work and output float64 by default
    np.random.seed(1)
    data = np.random.randn(50, 5).astype(np.float32)
    shrunk = nonlinshrink.shrink_cov(data)
    assert shrunk.dtype == np.float64
    assert shrunk.shape == (5, 5)


def test_zero_variance_column_after_demeaning():
    # One column is constant => zero variance after demeaning => singular matrix
    data = np.random.randn(20, 4)
    data[:, 0] = 1.0  # constant column
    with pytest.raises(ValueError):
        nonlinshrink.shrink_cov(data)


def test_constant_column_no_demeaning_k0():
    # If k=0 no demeaning; constant column will have nonzero variance so might pass check
    data = np.random.randn(20, 4)
    data[:, 0] = 1.0  # constant column
    shrunk = nonlinshrink.shrink_cov(data, k=0)
    assert shrunk.shape == (4, 4)
    assert np.allclose(shrunk, shrunk.T)


def test_sparse_data_matrix():
    # data with many zeros - test stability and no error
    np.random.seed(30)
    data = np.random.randn(40, 6)
    data[np.random.rand(*data.shape) < 0.7] = 0
    shrunk = nonlinshrink.shrink_cov(data)
    assert shrunk.shape == (6, 6)
    assert np.allclose(shrunk, shrunk.T)


def test_input_int_array_casting():
    # Integer input array should be converted properly (usually float needed for mean...)
    data = np.array([[1, 2, 3], [4, 5, 6]] * 10, dtype=int)
    # With k=0 no demean, but effective n=20 (20-0=20 >=12)
    # but sample size too small for test? 20-0=20 fine -> test passes.
    shrunk = nonlinshrink.shrink_cov(data, k=0)
    assert shrunk.shape == (3, 3)
    assert np.allclose(shrunk, shrunk.T)


def test_row_equals_one_raises_assertion():
    data = np.ones((1, 3))
    # n=1 - k=1 effective sample size 0 < 12 triggers assertion
    with pytest.raises(AssertionError):
        nonlinshrink.shrink_cov(data)


def test_column_equals_one_runs():
    # Test data with p=1 (single variable)
    np.random.seed(40)
    data = np.random.randn(25, 1)
    shrunk = nonlinshrink.shrink_cov(data)
    assert shrunk.shape == (1, 1)
    assert shrunk[0, 0] >= 0


def test_nan_values_in_data_raises():
    data = np.random.randn(20, 4)
    data[0, 0] = np.nan
    with pytest.raises(Exception):
        # may raise AssertionError or a numpy error when computing mean or dot product
        nonlinshrink.shrink_cov(data)


def test_inf_values_in_data_raises():
    data = np.random.randn(20, 4)
    data[0, 0] = np.inf
    with pytest.raises(Exception):
        nonlinshrink.shrink_cov(data)
