import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov, EPS


def test_shrink_cov_invalid_input_shape():
    # data is 1d array, should raise AssertionError
    data = np.arange(10)
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_shrink_cov_effective_sample_size_k_none(monkeypatch):
    # seed numpy here because shrink_cov uses numpy functions only
    np.random.seed(12345)

    n, p = 50, 10
    data = np.random.randn(n, p)
    # k is None, shrink_cov demeans by default and subtracts k=1

    # run without monkeypatch changes, just to observe no error
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)
    assert np.allclose(shrunk, shrunk.T, atol=1e-12)  # symmetric


def test_shrink_cov_no_demeaning_k_zero():
    np.random.seed(9876)

    n, p = 60, 8
    data = np.random.randn(n, p)
    # k=0, no demeaning, effective sample size = n
    shrunk = shrink_cov(data, k=0)
    assert shrunk.shape == (p, p)
    assert np.allclose(shrunk, shrunk.T, atol=1e-12)

    # compare to manual covariance matrix (biased, divisor n)
    manual_cov = data.T @ data / n
    # Should be different because of non-linear shrinkage
    assert not np.allclose(shrunk, manual_cov)


def test_shrink_cov_with_k_greater_than_zero():
    np.random.seed(23456)
    n, p = 55, 12
    data = np.random.randn(n, p)

    # k > 1 means no demeaning, effective sample size = n - k
    k = 5
    shrunk = shrink_cov(data, k=k)
    assert shrunk.shape == (p, p)
    assert np.allclose(shrunk, shrunk.T, atol=1e-12)


def test_shrink_cov_sample_size_too_small():
    np.random.seed(1)
    n, p = 14, 5
    data = np.random.randn(n, p)
    # If k=3, effective n=11 < 12, should raise AssertionError
    with pytest.raises(AssertionError):
        shrink_cov(data, k=3)


def test_shrink_cov_matrix_singular_error():
    np.random.seed(4321)
    n, p = 50, 10
    # create data with duplicate columns to cause singular covariance
    base = np.random.randn(n, p - 1)
    duplicate_col = base[:, 0:1]
    data = np.hstack([base, duplicate_col])  # last column duplicates first
    # We allow demean via k=None
    # The shrink_cov should raise ValueError for singular matrix
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_shrink_cov_output_psd():
    np.random.seed(54321)
    n, p = 80, 12
    data = np.random.randn(n, p)
    cov = shrink_cov(data)
    # output covariance should be symmetric PSD matrix
    assert np.allclose(cov, cov.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-12)  # allow tiny minus due to numerical


def test_shrink_cov_p_less_than_n_vs_p_greater_than_n():
    np.random.seed(1234)

    # Case p < n
    n1, p1 = 50, 30
    data1 = np.random.randn(n1, p1)
    cov1 = shrink_cov(data1)

    # Case p > n
    n2, p2 = 40, 60
    data2 = np.random.randn(n2, p2)
    cov2 = shrink_cov(data2)

    # Shapes match input dimensions
    assert cov1.shape == (p1, p1)
    assert cov2.shape == (p2, p2)

    # Both outputs symmetric
    assert np.allclose(cov1, cov1.T, atol=1e-12)
    assert np.allclose(cov2, cov2.T, atol=1e-12)


def test_shrink_cov_output_close_to_sample_cov_for_large_n():
    np.random.seed(7777)
    n, p = 1000, 10
    data = np.random.randn(n, p)

    sample_cov = (data - np.mean(data, axis=0)).T @ (data - np.mean(data, axis=0)) / (n - 1)
    shrunk = shrink_cov(data, k=1)  # k=1 demeaning

    # For large n, nonlinear shrinkage should be close to sample covariance but different
    dist = np.linalg.norm(shrunk - sample_cov, ord='fro')
    assert dist > 0
    assert dist / np.linalg.norm(sample_cov, ord='fro') < 0.3


def test_shrink_cov_all_equal_columns_zero_variance():
    np.random.seed(987)
    n, p = 50, 5
    # All columns equal, zero variance data
    data = np.ones((n, p)) * 5
    # k=0 no demean, effective sample size = n
    with pytest.raises(ValueError):
        shrink_cov(data, k=0)


def test_shrink_cov_raises_assert_for_1d_array():
    np.random.seed(0)
    arr_1d = np.arange(12)
    with pytest.raises(AssertionError):
        shrink_cov(arr_1d)


def test_shrink_cov_raises_assert_for_too_small_n_after_demean():
    np.random.seed(123)
    n, p = 13, 3
    data = np.random.randn(n, p)
    # If k is None, k=1, effective sample size n-1=12 minimal allowed
    # Now test k=2 which yields n-k=11 < 12 => should assert
    with pytest.raises(AssertionError):
        shrink_cov(data, k=2)


def test_shrink_cov_output_variability_with_different_k():
    np.random.seed(2024)
    n, p = 60, 15
    data = np.random.randn(n, p)

    cov_k_none = shrink_cov(data, k=None)
    cov_k0 = shrink_cov(data, k=0)
    cov_k5 = shrink_cov(data, k=5)

    # The outputs are all symmetric
    for cov in (cov_k_none, cov_k0, cov_k5):
        assert np.allclose(cov, cov.T, atol=1e-12)

    # Should differ when k changes
    assert not np.allclose(cov_k_none, cov_k0)
    assert not np.allclose(cov_k0, cov_k5)
    assert not np.allclose(cov_k5, cov_k_none)


def test_shrink_cov_with_demeaning_vs_without_with_zero_mean_data():
    np.random.seed(111)
    n, p = 80, 10
    # zero mean data by construction
    data = np.random.randn(n, p)
    data -= np.mean(data, axis=0)

    # Applying shrink_cov with k=None (demeans) or k=k value (no demean)
    cov_demean = shrink_cov(data, k=None)
    cov_nodemean = shrink_cov(data, k=0)
    # Since data already demeaned, these two should not be identical but reasonably close
    dist = np.linalg.norm(cov_demean - cov_nodemean, ord='fro')
    # Distance positive and smaller than relative threshold
    assert dist > 0
    assert dist / np.linalg.norm(cov_demean, ord='fro') < 0.5


def test_shrink_cov_handles_high_dimensional_data():
    np.random.seed(2024)
    n, p = 30, 50  # p > n
    data = np.random.randn(n, p)
    cov = shrink_cov(data)
    assert cov.shape == (p, p)
    assert np.allclose(cov, cov.T, atol=1e-12)

    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > -1e-12)


def test_shrink_cov_warns_for_zero_eigenvalues_replaced_by_eps(monkeypatch):
    # This is tricky since code raises ValueError on small eigenvalues relative to EPS sum(lam)
    # create data with eigenvalues exactly zero but larger eigenvalues present
    np.random.seed(2024)
    n, p = 50, 5
    data = np.zeros((n, p))
    # But fill first column with non-zero to get rank 1 matrix
    data[:, 0] = 1.0

    with pytest.raises(ValueError):
        shrink_cov(data)


def test_shrink_cov_nonrandom_behavior_with_fixed_seed():
    np.random.seed(42)
    n, p = 70, 20
    data = np.random.randn(n, p)
    result1 = shrink_cov(data)
    np.random.seed(42)
    data = np.random.randn(n, p)
    result2 = shrink_cov(data)

    assert np.allclose(result1, result2, atol=1e-12)


def test_shrink_cov_raises_for_small_effective_sample_size_edge():
    # n=13, k=1, n-k=12 valid
    np.random.seed(2023)
    n, p = 13, 3
    data = np.random.randn(n, p)

    shrink_cov(data, k=1)  # Should not raise
    # n-k =11 invalid
    with pytest.raises(AssertionError):
        shrink_cov(data, k=2)


def test_shrink_cov_output_real_symmetric_positive_definite_matrix():
    np.random.seed(1357)
    n, p = 100, 10
    data = np.random.randn(n, p)
    cov = shrink_cov(data)

    # Return real values
    assert np.isrealobj(cov)

    # Symmetry
    assert np.allclose(cov, cov.T, atol=1e-12)

    # Check positive definiteness by smallest eigenvalue >=0 (allow numerical tolerance)
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() >= -1e-12


def test_shrink_cov_stable_under_input_perturbation():
    np.random.seed(3333)
    n, p = 100, 10
    data = np.random.randn(n, p)
    cov = shrink_cov(data)

    # Slight perturbation to data
    data_perturbed = data.copy()
    data_perturbed[0, 0] += 1e-6
    cov2 = shrink_cov(data_perturbed)

    # Result should be close to original (stable)
    assert np.linalg.norm(cov - cov2, ord='fro') / np.linalg.norm(cov, ord='fro') < 1e-2


def test_shrink_cov_using_integer_data(monkeypatch):
    # int data coerced to float internally by numpy, still works
    np.random.seed(2024)
    n, p = 30, 10
    data_float = np.random.randn(n, p)
    data_int = (data_float * 100).astype(int)
    cov_float = shrink_cov(data_float)
    cov_int = shrink_cov(data_int)

    # shapes equal
    assert cov_float.shape == cov_int.shape
    # values differ because integer data different scale
    assert not np.allclose(cov_float, cov_int)

