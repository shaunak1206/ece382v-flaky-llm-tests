import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _data(n=100, p=5, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p)


def test_output_shape_n_greater_than_p():
    out = shrink_cov(_data(100, 5))
    assert out.shape == (5, 5)


def test_output_is_symmetric():
    out = shrink_cov(_data(80, 6, seed=1))
    assert np.allclose(out, out.T, atol=1e-10)


def test_output_is_positive_definite():
    out = shrink_cov(_data(80, 6, seed=2))
    assert np.all(np.linalg.eigvalsh(out) > 0)


def test_output_is_finite():
    out = shrink_cov(_data(60, 8, seed=3))
    assert np.all(np.isfinite(out))


def test_non_2d_input_raises():
    with pytest.raises(AssertionError):
        shrink_cov(np.random.RandomState(0).randn(50))
    with pytest.raises(AssertionError):
        shrink_cov(np.random.RandomState(0).randn(5, 5, 5))


def test_too_few_samples_raises():
    with pytest.raises(AssertionError):
        shrink_cov(_data(11, 3))
    # with k=None one dof is lost, so n=12 is not enough
    with pytest.raises(AssertionError):
        shrink_cov(_data(12, 3))


def test_minimum_sample_size_accepted():
    # n=13 with default demeaning -> effective n = 12
    out = shrink_cov(_data(13, 3))
    assert out.shape == (3, 3)
    # n=12 with k=0 -> effective n = 12
    out0 = shrink_cov(_data(12, 3), k=0)
    assert out0.shape == (3, 3)


def test_k_reduces_effective_sample_size():
    data = _data(14, 3)
    assert shrink_cov(data, k=2).shape == (3, 3)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=3)


def test_default_equals_manual_demean_with_k1():
    data = _data(60, 4, seed=4) + 5.0
    demeaned = data - data.mean(axis=0)
    a = shrink_cov(data)
    b = shrink_cov(demeaned, k=1)
    assert np.allclose(a, b, atol=1e-8)


def test_singular_matrix_raises_value_error():
    data = _data(50, 3, seed=5)
    data[:, 2] = data[:, 0]
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_input_not_modified():
    data = _data(40, 4, seed=6) + 2.0
    copy = data.copy()
    shrink_cov(data)
    assert np.array_equal(data, copy)
    shrink_cov(data, k=0)
    assert np.array_equal(data, copy)


def test_commutes_with_sample_covariance():
    # Shrunk estimator keeps the sample eigenvectors.
    data = _data(100, 5, seed=7)
    centered = data - data.mean(axis=0)
    sample = centered.T.dot(centered) / (data.shape[0] - 1)
    out = shrink_cov(data)
    assert np.allclose(sample.dot(out), out.dot(sample), atol=1e-8)


def test_high_dimensional_case_p_greater_than_n():
    out = shrink_cov(_data(20, 40, seed=8))
    assert out.shape == (40, 40)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(out) > 0)


def test_close_to_identity_for_large_sample_identity_population():
    data = _data(3000, 5, seed=9)
    out = shrink_cov(data)
    assert np.allclose(out, np.eye(5), atol=0.2)


def test_shrinkage_improves_conditioning():
    data = _data(40, 10, seed=10)
    centered = data - data.mean(axis=0)
    sample = centered.T.dot(centered) / (data.shape[0] - 1)
    out = shrink_cov(data)
    assert np.linalg.cond(out) < np.linalg.cond(sample)


def test_scale_equivariance():
    data = _data(60, 5, seed=11)
    c = 3.0
    a = shrink_cov(c * data)
    b = (c ** 2) * shrink_cov(data)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-8)


def test_row_permutation_invariance():
    data = _data(60, 5, seed=12)
    perm = np.random.RandomState(99).permutation(data.shape[0])
    a = shrink_cov(data)
    b = shrink_cov(data[perm])
    assert np.allclose(a, b, atol=1e-8)


def test_deterministic():
    data = _data(50, 6, seed=13)
    assert np.array_equal(shrink_cov(data), shrink_cov(data))


def test_diagonal_entries_positive_with_unequal_variances():
    rng = np.random.RandomState(14)
    data = rng.randn(200, 4) * np.array([1.0, 2.0, 3.0, 0.5])
    out = shrink_cov(data)
    assert np.all(np.diag(out) > 0)
    # variance ordering should be broadly preserved
    d = np.diag(out)
    assert d[2] > d[3]
