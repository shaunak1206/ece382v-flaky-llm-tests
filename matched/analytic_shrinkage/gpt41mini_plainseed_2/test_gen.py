import numpy as np
import pytest
from nonlinshrink.nonlinshrink import shrink_cov, EPS


def make_data(n, p, seed=123):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p))


def test_shrink_cov_basic_shape():
    data = make_data(50, 20)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (20, 20)
    assert np.allclose(shrunk, shrunk.T)
    assert np.all(np.linalg.eigvals(shrunk) >= -1e-10)  # semi-positive definite


def test_shrink_cov_no_demean_k0():
    data = make_data(60, 15)
    shrunk = shrink_cov(data, k=0)
    # Should not demean, so mean along axis 0 should not be approx zero
    means = np.mean(data, axis=0)
    assert not np.allclose(means, 0)
    # Output shape correct
    assert shrunk.shape == (15, 15)


def test_shrink_cov_with_k_gt_1():
    data = make_data(40, 10)
    shrunk = shrink_cov(data, k=3)  # data assumed pre-demeaned w/ 3 regressors used
    assert shrunk.shape == (10, 10)


def test_shrink_cov_raises_if_sample_size_eff_too_small():
    data = make_data(13, 5)
    # k=2 makes effective n=11 < 12, should raise assertion
    with pytest.raises(AssertionError):
        shrink_cov(data, k=2)


def test_shrink_cov_input_not_2d_raises():
    data = np.array([1, 2, 3])
    with pytest.raises(AssertionError):
        shrink_cov(data)
    data = np.ones((3, 3, 3))
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_shrink_cov_singular_matrix_raises_valueerror():
    # Construct a matrix with zero eigenvalues after trimming lam
    # Use p=20, n=15 and k=2 => effective n=13 so p-n = 5
    # Use repeated columns so covariance is singular
    p = 20
    n = 17
    k = 3  # effective n = 14
    # Construct data rank deficient to cause zero eigenvalues after trimming lam
    data = np.ones((n, p))
    with pytest.raises(ValueError):
        shrink_cov(data, k=k)


def test_shrink_cov_p_less_than_n_behavior():
    n, p = 50, 30
    data = make_data(n, p)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)
    # The output matrix should be symmetric and positive semi-definite
    eigvals = np.linalg.eigvalsh(shrunk)
    assert np.all(eigvals >= -1e-10)


def test_shrink_cov_p_equal_n_behavior():
    p = 40
    n = 40 + 2  # k=1 default, effective n=41
    data = make_data(n, p)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)
    eigvals = np.linalg.eigvalsh(shrunk)
    assert np.all(eigvals >= -1e-10)


def test_shrink_cov_p_greater_than_n_behavior():
    n, p = 30, 50
    data = make_data(n, p)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)
    eigvals = np.linalg.eigvalsh(shrunk)
    assert np.all(eigvals >= -1e-10)


def test_shrink_cov_output_positive_semidefinite():
    data = make_data(80, 25)
    shrunk = shrink_cov(data)
    eigvals = np.linalg.eigvalsh(shrunk)
    assert np.all(eigvals >= -1e-10)


def test_shrink_cov_no_change_k_equal_n():
    # If k=n, then effective n=0, should raise assertion error
    data = make_data(15, 5)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=15)


def test_shrink_cov_with_constant_columns():
    # some columns are constant, causing zero variance -> singularity
    data = make_data(40, 10)
    data[:, 0] = 5
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_shrink_cov_non_integer_k_raises():
    data = make_data(20, 10)
    with pytest.raises(TypeError):
        # k must be an int or None, passing a float should error in n-k step
        shrink_cov(data, k=1.5)


def test_shrink_cov_return_type_and_dtype():
    data = make_data(100, 30)
    shrunk = shrink_cov(data)
    assert isinstance(shrunk, np.ndarray)
    assert shrunk.dtype == np.float64


def test_shrink_cov_independence_from_mean_shift():
    data = make_data(70, 15)
    shifted_data = data + 10  # Add constant shift, should not affect shrunk covariance if demeaned
    shrunk1 = shrink_cov(data)
    shrunk2 = shrink_cov(shifted_data)
    assert np.allclose(shrunk1, shrunk2, atol=1e-10)


def test_shrink_cov_effect_of_k_param():
    data = make_data(50, 20)
    shrunk_k_none = shrink_cov(data, k=None)
    shrunk_k_0 = shrink_cov(data, k=0)
    shrunk_k_2 = shrink_cov(data, k=2)
    assert shrunk_k_none.shape == (20, 20)
    assert shrunk_k_0.shape == (20, 20)
    assert shrunk_k_2.shape == (20, 20)
    # k=0 result differs from default k=None since demean differs
    assert not np.allclose(shrunk_k_none, shrunk_k_0)
    # k=2 differs as well
    assert not np.allclose(shrunk_k_none, shrunk_k_2)


def test_shrink_cov_input_all_zeros():
    data = np.zeros((30, 15))
    # mean subtraction doesn't change zeros, effective n = 29
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_shrink_cov_with_small_eigenvalues_but_above_eps():
    # construct data with small eigenvalues > EPS threshold after trimming
    n, p = 50, 40
    base = np.eye(p) * 1e-5
    noise = make_data(n, p) * 1e-3
    data = noise + base
    shrunk = shrink_cov(data)
    assert shrunk.shape == (p, p)


def test_shrink_cov_large_matrix_runtime():
    # Smoke test for larger matrix (not timing, just checking it runs)
    data = make_data(200, 150)
    shrunk = shrink_cov(data)
    assert shrunk.shape == (150, 150)


def test_shrink_cov_variance_preservation():
    data = make_data(60, 20)
    shrunk = shrink_cov(data)
    # Diagonal should be positive and generally similar scale to sample variance
    sample_var = np.var(data, axis=0, ddof=1)
    shrunk_diag = np.diag(shrunk)
    assert np.all(shrunk_diag > 0)
    # Shrunk variances are roughly on the same order of magnitude as sample variance
    assert np.allclose(np.mean(shrunk_diag), np.mean(sample_var), rtol=0.5)


def test_shrink_cov_symmetry_and_real():
    data = make_data(40, 10)
    shrunk = shrink_cov(data)
    # Matrix should be real symmetric
    assert np.isrealobj(shrunk)
    assert np.allclose(shrunk, shrunk.T)
