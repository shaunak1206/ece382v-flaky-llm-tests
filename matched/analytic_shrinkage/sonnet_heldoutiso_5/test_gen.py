import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _make_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p)


def test_output_shape_p_less_than_n():
    data = _make_data(100, 10)
    out = shrink_cov(data)
    assert isinstance(out, np.ndarray)
    assert out.shape == (10, 10)


def test_output_shape_p_greater_than_n():
    data = _make_data(20, 30, seed=1)
    out = shrink_cov(data)
    assert out.shape == (30, 30)


def test_output_shape_p_greater_than_n_k0():
    data = _make_data(20, 30, seed=2)
    out = shrink_cov(data, k=0)
    assert out.shape == (30, 30)
    assert np.all(np.isfinite(out))


def test_output_is_symmetric():
    data = _make_data(150, 12, seed=3)
    out = shrink_cov(data)
    assert np.allclose(out, out.T, atol=1e-10)


def test_output_is_positive_definite():
    data = _make_data(120, 15, seed=4)
    out = shrink_cov(data)
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(eigvals > 0)


def test_output_positive_definite_high_dimensional():
    data = _make_data(25, 40, seed=5)
    out = shrink_cov(data)
    assert np.all(np.isfinite(out))
    assert np.all(np.linalg.eigvalsh(out) > 0)


def test_non_2d_input_raises():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(50, dtype=float))
    with pytest.raises(AssertionError):
        shrink_cov(np.zeros((20, 3, 2)))


def test_small_sample_raises_default_k():
    # n=12 with k=None gives effective n=11 < 12
    data = _make_data(12, 3, seed=6)
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_small_sample_raises_k0():
    data = _make_data(11, 3, seed=7)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=0)


def test_minimum_sample_sizes_accepted():
    out1 = shrink_cov(_make_data(13, 3, seed=8))
    assert out1.shape == (3, 3)
    out2 = shrink_cov(_make_data(12, 3, seed=9), k=0)
    assert out2.shape == (3, 3)


def test_larger_k_reduces_effective_sample_size():
    data = _make_data(14, 3, seed=10)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=3)


def test_singular_matrix_raises_value_error():
    data = _make_data(50, 4, seed=11)
    data = np.hstack([data, data[:, :1]])  # duplicated column -> singular
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_constant_column_raises_value_error():
    data = _make_data(50, 4, seed=12)
    data[:, 2] = 3.0  # zero variance after demeaning
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_default_equals_manual_demeaning_with_k1():
    data = _make_data(60, 6, seed=13) + 5.0
    centered = data - data.mean(axis=0)
    a = shrink_cov(data)
    b = shrink_cov(centered, k=1)
    assert np.allclose(a, b, rtol=1e-8, atol=1e-10)


def test_k_changes_result():
    centered = _make_data(40, 5, seed=14)
    centered = centered - centered.mean(axis=0)
    a = shrink_cov(centered, k=0)
    b = shrink_cov(centered, k=1)
    assert not np.allclose(a, b)


def test_input_not_mutated():
    data = _make_data(50, 5, seed=15) + 2.0
    copy = data.copy()
    shrink_cov(data)
    assert np.array_equal(data, copy)
    shrink_cov(data, k=0)
    assert np.array_equal(data, copy)


def test_deterministic():
    data = _make_data(80, 8, seed=16)
    assert np.array_equal(shrink_cov(data), shrink_cov(data))


def test_scale_equivariance():
    data = _make_data(100, 8, seed=17)
    c = 3.0
    a = shrink_cov(c * data)
    b = c ** 2 * shrink_cov(data)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-8)


def test_commutes_with_sample_covariance():
    data = _make_data(100, 6, seed=18)
    centered = data - data.mean(axis=0)
    sample = centered.T.dot(centered) / (100 - 1)
    out = shrink_cov(data)
    assert np.allclose(out.dot(sample), sample.dot(out), atol=1e-8)


def test_reduces_eigenvalue_dispersion():
    data = _make_data(100, 50, seed=19)
    centered = data - data.mean(axis=0)
    sample = centered.T.dot(centered) / (100 - 1)
    out = shrink_cov(data)
    ev_s = np.linalg.eigvalsh(sample)
    ev_o = np.linalg.eigvalsh(out)
    assert ev_o.max() / ev_o.min() < ev_s.max() / ev_s.min()


def test_close_to_identity_for_large_sample():
    data = _make_data(3000, 5, seed=20)
    out = shrink_cov(data)
    assert np.max(np.abs(out - np.eye(5))) < 0.2
