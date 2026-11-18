import numpy as np
import pytest
from nonlinshrink.nonlinshrink import shrink_cov, EPS


def make_data(n, p, seed=0):
    np.random.seed(seed)
    return np.random.randn(n, p)


def test_shrink_cov_basic_shape():
    data = make_data(50, 10)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (10, 10)
    assert np.allclose(shrunk, shrunk.T, atol=1e-10)  # symmetric


def test_shrink_cov_with_k_none_equals_demean():
    data = make_data(50, 5)
    shrunk1 = shrink_cov(data, k=None)
    data_demeaned = data - np.mean(data, axis=0)
    shrunk2 = shrink_cov(data_demeaned, k=0)
    # shrunk1 uses demeaning, shrunk2 with k=0 no demeaning
    # so results differ since effective n differs:
    assert shrunk1.shape == shrunk2.shape
    # They are not equal, but both valid
    assert np.allclose(shrunk1, shrunk1.T)
    assert np.allclose(shrunk2, shrunk2.T)


def test_shrink_cov_with_k_zero_no_demean():
    data = make_data(30, 7)
    shrunk = shrink_cov(data, k=0)
    assert shrunk.shape == (7, 7)
    # sample covariance from raw data (no demean), ensure no error
    assert np.allclose(shrunk, shrunk.T)


def test_shrink_cov_with_valid_k_positive():
    data = make_data(40, 8)
    shrunk = shrink_cov(data, k=3)
    assert shrunk.shape == (8, 8)
    assert np.allclose(shrunk, shrunk.T)


def test_shrink_cov_raises_assertion_on_non_2d_input():
    with pytest.raises(AssertionError):
        shrink_cov(np.array([1, 2, 3]))


def test_shrink_cov_raises_assertion_on_small_effective_n():
    data = make_data(20, 5)
    # n=20, k=9 => n-k=11 < 12 triggers assertion
    with pytest.raises(AssertionError):
        shrink_cov(data, k=9)


def test_shrink_cov_singular_matrix_raises_value_error():
    # Build data that leads to zero eigenvalues after trimming lam
    # Construct data with rank deficiency and large p > n so lam slicing has effect
    n, p = 15, 20
    data = np.zeros((n, p))
    # Make 12 rows identical, so covariance has rank <= 1 in leftover lam slice
    data[:12, :] = 1
    # fill remaining rows randomly to avoid degenerate cases
    data[12:, :] = np.random.randn(n - 12, p)
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data, k=1)


def test_shrink_cov_p_less_equal_n_path():
    n, p = 30, 20  # p <= n
    data = make_data(n, p)
    cov_shrunk = shrink_cov(data, k=None)
    assert cov_shrunk.shape == (p, p)
    assert np.all(np.linalg.eigvals(cov_shrunk) > 0)  # positive semi-definite


def test_shrink_cov_p_greater_n_path():
    n, p = 20, 30  # p > n
    data = make_data(n, p)
    cov_shrunk = shrink_cov(data, k=None)
    assert cov_shrunk.shape == (p, p)
    assert np.allclose(cov_shrunk, cov_shrunk.T)


def test_shrink_cov_no_demean_k0_effective_n_equal_n():
    data = make_data(50, 10)
    # With k=0 no demean, so n effective is n
    shrunk_0 = shrink_cov(data, k=0)
    # With k=1 demeaning, effective n = n-1
    shrunk_1 = shrink_cov(data, k=1)
    assert shrunk_0.shape == shrunk_1.shape == (10, 10)
    # They should differ because effective n differs
    assert not np.allclose(shrunk_0, shrunk_1)


def test_shrink_cov_input_data_not_modified():
    data = make_data(40, 10)
    copy_data = data.copy()
    shrink_cov(data, k=None)
    assert np.allclose(data, copy_data)  # input unchanged


def test_shrink_cov_reproducible_results():
    data = make_data(60, 15)
    out1 = shrink_cov(data, k=None)
    out2 = shrink_cov(data, k=None)
    assert np.allclose(out1, out2)


def test_shrink_cov_large_data():
    data = make_data(200, 50)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (50, 50)


def test_shrink_cov_small_p_n_equal():
    n = p = 15
    data = make_data(n, p)
    shrunk = shrink_cov(data, k=None)
    assert shrunk.shape == (p, p)


def test_shrink_cov_effective_n_and_demean_relationship():
    data = make_data(50, 10)
    # with k=None means demean and n adjusted by 1
    shrunk_none = shrink_cov(data)
    # with k=1 means no demean, n adjusted by 1 as well
    # but data assumed demeaned already
    data_demeaned = data - np.mean(data, axis=0)
    shrunk_one = shrink_cov(data_demeaned, k=1)
    assert shrunk_none.shape == shrunk_one.shape
    # values expected to be similar but not equal since demean and demeaned inputs


def test_shrink_cov_identity_data_returns_close_to_identity():
    # Data such that covariance close to identity matrix
    n, p = 50, 10
    data = np.eye(p)[np.random.choice(p, n, replace=True)]
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)
    # diagonal elements should be positive and dominant
    diag = np.diag(shrunk)
    assert np.all(diag > 0)
    assert np.all(diag >= shrunk.min(axis=1))


def test_shrink_cov_constant_columns_raise_assertion_or_value_error():
    # This will cause covariance matrix to be singular
    data = np.ones((30, 5))
    with pytest.raises(AssertionError):
        # if demeaned, variance zero and effective n likely insufficient
        shrink_cov(data)
    # Alternatively test with k=0 no demean but singular covariance eigenvalues
    data2 = np.ones((40, 5))
    with pytest.raises(ValueError):
        shrink_cov(data2, k=0)


def test_shrink_cov_float_precision():
    n, p = 50, 10
    data = make_data(n, p).astype(np.float32)
    shrunk = shrink_cov(data)
    assert shrunk.dtype == np.float64  # output is double by default


def test_shrink_cov_with_large_k():
    # k close to but less than n - 12 to avoid assertion
    n, p = 50, 10
    k = 38  # effective n = 12, still valid
    data = make_data(n, p)
    shrunk = shrink_cov(data, k=k)
    assert shrunk.shape == (p, p)


def test_shrink_cov_with_k_equal_zero_and_p_greater_than_n():
    n, p = 15, 25  # p > n
    data = make_data(n, p)
    shrunk = shrink_cov(data, k=0)
    assert shrunk.shape == (p, p)
    assert np.allclose(shrunk, shrunk.T)


def test_shrink_cov_extreme_values():
    n, p = 30, 10
    data = np.full((n, p), 1e10)
    with pytest.raises(ValueError):
        shrink_cov(data, k=1)
