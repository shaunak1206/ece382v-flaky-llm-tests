import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def make_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p)


def make_correlated_data(n, p, seed=1):
    rng = np.random.RandomState(seed)
    a = rng.randn(p, p)
    return rng.randn(n, p).dot(a)


def test_output_shape_low_dim():
    data = make_data(50, 5)
    out = shrink_cov(data)
    assert out.shape == (5, 5)


def test_output_is_symmetric():
    data = make_correlated_data(80, 6)
    out = shrink_cov(data)
    assert np.allclose(out, out.T, atol=1e-10)


def test_output_is_finite_and_real():
    data = make_correlated_data(60, 8)
    out = shrink_cov(data)
    assert np.all(np.isfinite(out))
    assert not np.iscomplexobj(out)


def test_output_positive_definite():
    data = make_correlated_data(100, 10)
    out = shrink_cov(data)
    eig = np.linalg.eigvalsh((out + out.T) / 2)
    assert np.all(eig > 0)


def test_one_dimensional_input_raises():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(50, dtype=float))


def test_three_dimensional_input_raises():
    with pytest.raises(AssertionError):
        shrink_cov(np.zeros((20, 3, 2)))


def test_too_few_samples_default_k_raises():
    # n=12 with k=None -> effective n = 11 < 12
    data = make_data(12, 3)
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_minimum_sample_size_boundary_ok():
    # n=13 with k=None -> effective n = 12, allowed
    data = make_data(13, 3)
    out = shrink_cov(data)
    assert out.shape == (3, 3)
    # n=12 with k=0 -> effective n = 12, allowed
    out0 = shrink_cov(make_data(12, 3), k=0)
    assert out0.shape == (3, 3)


def test_large_k_reduces_sample_size_and_raises():
    data = make_data(20, 3)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=10)


def test_singular_matrix_raises_value_error():
    data = make_data(50, 4)
    data[:, 0] = 0.0
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_high_dimensional_case_p_greater_than_n():
    data = make_data(20, 30)
    out = shrink_cov(data)
    assert out.shape == (30, 30)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out.T, atol=1e-8)


def test_default_k_equals_k1_on_demeaned_data():
    data = make_correlated_data(70, 5) + 3.0
    demeaned = data - data.mean(axis=0)
    out_default = shrink_cov(data)
    out_k1 = shrink_cov(demeaned, k=1)
    assert np.allclose(out_default, out_k1, rtol=1e-8, atol=1e-10)


def test_k0_does_not_demean():
    data = make_correlated_data(70, 5) + 5.0
    out_default = shrink_cov(data)
    out_k0 = shrink_cov(data, k=0)
    assert not np.allclose(out_default, out_k0)


def test_input_not_modified():
    data = make_correlated_data(40, 4) + 2.0
    copy = data.copy()
    shrink_cov(data)
    assert np.array_equal(data, copy)


def test_deterministic():
    data = make_correlated_data(60, 6)
    assert np.array_equal(shrink_cov(data), shrink_cov(data))


def test_commutes_with_sample_covariance():
    # The estimator keeps the eigenvectors of the sample covariance.
    data = make_correlated_data(100, 5)
    centered = data - data.mean(axis=0)
    sample = centered.T.dot(centered) / (data.shape[0] - 1)
    out = shrink_cov(data)
    lhs = sample.dot(out)
    rhs = out.dot(sample)
    scale = np.abs(lhs).max()
    assert np.allclose(lhs, rhs, atol=1e-8 * scale)


def test_scale_equivariance():
    data = make_correlated_data(80, 5)
    c = 3.0
    out = shrink_cov(data)
    out_scaled = shrink_cov(c * data)
    assert np.allclose(out_scaled, c ** 2 * out, rtol=1e-6, atol=1e-8)


def test_column_permutation_equivariance():
    data = make_correlated_data(90, 6)
    perm = np.array([3, 1, 5, 0, 2, 4])
    out = shrink_cov(data)
    out_perm = shrink_cov(data[:, perm])
    assert np.allclose(out_perm, out[np.ix_(perm, perm)], rtol=1e-5, atol=1e-8)


def test_close_to_identity_for_large_sample_identity_covariance():
    data = make_data(5000, 4, seed=3)
    out = shrink_cov(data)
    assert np.allclose(out, np.eye(4), atol=0.2)


def test_condition_number_improved_over_sample_covariance():
    n, p = 100, 50
    data = make_data(n, p, seed=5)
    centered = data - data.mean(axis=0)
    sample = centered.T.dot(centered) / (n - 1)
    out = shrink_cov(data)
    cond_sample = np.linalg.cond(sample)
    cond_shrunk = np.linalg.cond((out + out.T) / 2)
    assert cond_shrunk < cond_sample
