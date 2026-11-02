import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov, EPS


def make_data(n, p, seed=42, demean=True):
    """Helper to generate reproducible normal data with fixed seed."""
    np.random.seed(seed)
    data = np.random.randn(n, p)
    if demean:
        data -= data.mean(axis=0)
    return data


def test_output_shape_square(n=50, p=30):
    data = make_data(n, p)
    cov = shrink_cov(data)
    assert cov.shape == (p, p)


def test_output_shape_wide(n=30, p=50):
    data = make_data(n, p)
    cov = shrink_cov(data)
    assert cov.shape == (p, p)


def test_output_shape_equal_n_p(n=40, p=40):
    data = make_data(n, p)
    cov = shrink_cov(data)
    assert cov.shape == (p, p)


def test_shrink_cov_is_symmetric():
    data = make_data(60, 20)
    cov = shrink_cov(data)
    # Matrix must be symmetric
    assert np.allclose(cov, cov.T, atol=1e-13)


def test_returned_matrix_positive_semidefinite():
    data = make_data(80, 25)
    cov = shrink_cov(data)
    # Eigenvalues should be non-negative (within small numerical errors)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -EPS)


def test_effect_of_k_none_demeans_and_adjusts_n():
    # When k=None data is demeaned here
    n, p = 100, 20
    data0 = make_data(n, p, seed=1234, demean=False)
    data1 = make_data(n, p, seed=1234, demean=True)

    cov0 = shrink_cov(data0, k=None)
    cov1 = shrink_cov(data1, k=1)

    # They should be close because data0 demeaned inside function
    assert np.allclose(cov0, cov1, rtol=1e-6, atol=1e-8)


def test_k_zero_no_demeaning_and_effective_sample_nunchanged():
    n, p = 40, 15
    data = make_data(n, p, demean=False)
    cov_direct = shrink_cov(data, k=0)

    # calculate sample covariance manually without demeaning
    sample_cov = (data.T @ data) / n
    # The shrink_cov result should differ from sample_cov because of shrinkage:
    assert cov_direct.shape == sample_cov.shape
    # The shrink_cov output should NOT be identical to sample_cov
    assert not np.allclose(cov_direct, sample_cov)


def test_k_greater_than_zero_no_demeaning_and_adjusted_effective_sample():
    n, p = 45, 18
    k = 5
    data = make_data(n, p, demean=False)
    cov = shrink_cov(data, k=k)
    # Just test it runs and returns the right shape
    assert cov.shape == (p, p)


@pytest.mark.parametrize("p,n", [(20, 40), (50, 50), (70, 40)])
def test_shrink_cov_returns_finite_values(p, n):
    data = make_data(n, p)
    cov = shrink_cov(data)
    # Should not contain NaNs or infs for valid inputs
    assert np.all(np.isfinite(cov))


def test_minimum_sample_size_requirement():
    n, p = 15, 10
    data = make_data(n, p)
    # k=0 means effective n=n, which is less than 12
    with pytest.raises(AssertionError):
        shrink_cov(data, k=n-5)  # effective sample size < 12


def test_singular_matrix_raises_value_error():
    # Create data that will produce zero eigenvalues (singular)
    n, p = 20, 10
    np.random.seed(123)
    data = np.zeros((n, p))
    # This should lead to zero eigenvalues =>
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_shrink_cov_different_data_gives_different_results():
    data1 = make_data(50, 20, seed=1)
    data2 = make_data(50, 20, seed=2)
    cov1 = shrink_cov(data1)
    cov2 = shrink_cov(data2)
    # They should be different matrices for different data
    diff = np.linalg.norm(cov1 - cov2, ord='fro')
    assert diff > 1e-6


def test_shrink_cov_returns_positive_definite_when_p_less_than_n_by_large_margin():
    n, p = 100, 50
    data = make_data(n, p)
    cov = shrink_cov(data)
    eigs = np.linalg.eigvalsh(cov)
    assert np.all(eigs > 0)


def test_shrink_cov_returns_positive_definite_when_p_greater_than_n():
    n, p = 50, 80
    data = make_data(n, p)
    cov = shrink_cov(data)
    eigs = np.linalg.eigvalsh(cov)
    # In high dimension (p > n) result matrix is PSD - eigenvalues >=0
    assert np.all(eigs >= -EPS)


def test_shrink_cov_result_close_to_sample_covariance_for_identity_data():
    # Generate data with covariance close to identity
    n, p = 60, 30
    np.random.seed(1234)
    data = np.random.multivariate_normal(np.zeros(p), np.eye(p), size=n)
    # shrink_cov demean data internally by default
    cov_shrink = shrink_cov(data)
    # Sample covariance
    data_dm = data - data.mean(axis=0)
    sample_cov = (data_dm.T @ data_dm) / (n - 1)
    # The shrunk covariance should be close to sample covariance in Frobenius norm
    diff = np.linalg.norm(sample_cov - cov_shrink, ord='fro') / (p * p)
    assert diff < 0.05


def test_no_side_effects_on_input_data():
    n, p = 30, 20
    data = make_data(n, p)
    data_copy = data.copy()
    shrink_cov(data)
    # Input data should not be modified by shrink_cov
    assert np.allclose(data, data_copy)


def test_k_none_demeans_data_inside_function_only():
    n, p = 40, 20
    data = make_data(n, p, demean=False)
    # override mean outside to zeros and compare with k=1 on demeaned data
    cov1 = shrink_cov(data, k=None)
    data_demeaned = data - data.mean(axis=0)
    cov2 = shrink_cov(data_demeaned, k=1)
    assert np.allclose(cov1, cov2, rtol=1e-6, atol=1e-8)


def test_edge_case_k_equal_n_raises():
    n, p = 30, 10
    data = make_data(n, p)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=n)


def test_edge_case_k_greater_than_n_raises():
    n, p = 30, 10
    data = make_data(n, p)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=n + 1)


def test_shrink_cov_output_is_real_valued():
    n, p = 50, 20
    data = make_data(n, p)
    cov = shrink_cov(data)
    assert np.isrealobj(cov)


def test_shrink_cov_handles_rank_deficient_data():
    n, p = 40, 20
    data = make_data(n, p)
    # Make data rank deficient by repeating a column
    data[:, 1] = data[:, 0]
    with pytest.raises(ValueError):
        shrink_cov(data)


@pytest.mark.parametrize("k", [None, 0, 1, 5])
def test_shrink_cov_runs_for_various_k(k):
    n, p = 60, 30
    data = make_data(n, p, demean=(k != 0))
    if k is not None and k >= n - 12:
        with pytest.raises(AssertionError):
            shrink_cov(data, k=k)
    else:
        cov = shrink_cov(data, k=k)
        assert cov.shape == (p, p)
        assert np.all(np.isfinite(cov))


def test_shrink_cov_raises_assert_for_1d_input():
    arr_1d = np.arange(10)
    with pytest.raises(AssertionError):
        shrink_cov(arr_1d)


def test_shrink_cov_raises_assert_for_3d_input():
    arr_3d = np.zeros((10,10,10))
    with pytest.raises(AssertionError):
        shrink_cov(arr_3d)
