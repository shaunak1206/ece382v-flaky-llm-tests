import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _make_data(n, p):
    return np.random.randn(n, p)


def test_output_shape_low_dim():
    np.random.seed(0)
    data = _make_data(60, 5)
    out = shrink_cov(data)
    assert out.shape == (5, 5)
    assert np.all(np.isfinite(out))


def test_output_symmetric():
    np.random.seed(1)
    data = _make_data(80, 6)
    out = shrink_cov(data)
    assert np.allclose(out, out.T, atol=1e-10)


def test_output_positive_definite_low_dim():
    np.random.seed(2)
    data = _make_data(100, 8)
    out = shrink_cov(data)
    eigs = np.linalg.eigvalsh((out + out.T) / 2)
    assert np.all(eigs > 0)


def test_high_dimensional_case_p_greater_than_n():
    np.random.seed(3)
    data = _make_data(20, 30)
    out = shrink_cov(data)
    assert out.shape == (30, 30)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out.T, atol=1e-8)
    eigs = np.linalg.eigvalsh((out + out.T) / 2)
    assert np.all(eigs > 0)


def test_non_2d_input_raises_assertion():
    np.random.seed(4)
    with pytest.raises(AssertionError):
        shrink_cov(np.random.randn(50))
    with pytest.raises(AssertionError):
        shrink_cov(np.random.randn(5, 5, 5))


def test_small_sample_size_raises_assertion_default_k():
    np.random.seed(5)
    # n=12 rows, k defaults to 1 -> effective n = 11 < 12
    with pytest.raises(AssertionError):
        shrink_cov(_make_data(12, 3))


def test_minimum_sample_size_accepted():
    np.random.seed(6)
    # 13 rows, k=1 -> effective n = 12, allowed
    out = shrink_cov(_make_data(13, 3))
    assert out.shape == (3, 3)
    # 12 rows, k=0 -> effective n = 12, allowed
    out0 = shrink_cov(_make_data(12, 3), k=0)
    assert out0.shape == (3, 3)


def test_k_reduces_effective_sample_size():
    np.random.seed(7)
    data = _make_data(20, 3)
    out = shrink_cov(data, k=8)  # effective n = 12
    assert out.shape == (3, 3)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=9)  # effective n = 11


def test_default_equals_manual_demean_with_k1():
    np.random.seed(8)
    data = _make_data(50, 4) + 3.0
    demeaned = data - data.mean(axis=0)
    a = shrink_cov(data)
    b = shrink_cov(demeaned, k=1)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-8)


def test_k_zero_does_not_demean():
    np.random.seed(9)
    data = _make_data(50, 4) + 5.0
    a = shrink_cov(data)
    b = shrink_cov(data, k=0)
    assert not np.allclose(a, b, rtol=1e-3, atol=1e-3)


def test_singular_matrix_raises_value_error():
    np.random.seed(10)
    data = _make_data(50, 4)
    data[:, 2] = 0.0
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_input_not_modified():
    np.random.seed(11)
    data = _make_data(40, 5) + 1.0
    copy = data.copy()
    shrink_cov(data)
    shrink_cov(data, k=0)
    assert np.array_equal(data, copy)


def test_deterministic_for_same_input():
    np.random.seed(12)
    data = _make_data(40, 5)
    a = shrink_cov(data)
    b = shrink_cov(data)
    assert np.array_equal(a, b)


def test_scale_equivariance():
    np.random.seed(13)
    data = _make_data(60, 5)
    c = 3.0
    a = shrink_cov(data)
    b = shrink_cov(c * data)
    assert np.allclose(b, c ** 2 * a, rtol=1e-6, atol=1e-8)


def test_row_order_invariance():
    np.random.seed(14)
    data = _make_data(60, 5)
    a = shrink_cov(data)
    b = shrink_cov(data[::-1].copy())
    assert np.allclose(a, b, rtol=1e-6, atol=1e-8)


def test_orthogonal_rotation_equivariance():
    np.random.seed(15)
    data = _make_data(80, 5)
    q, _ = np.linalg.qr(np.random.randn(5, 5))
    a = shrink_cov(data)
    b = shrink_cov(data.dot(q))
    assert np.allclose(b, q.T.dot(a).dot(q), rtol=1e-5, atol=1e-7)


def test_commutes_with_sample_covariance():
    np.random.seed(16)
    data = _make_data(100, 6)
    demeaned = data - data.mean(axis=0)
    n = data.shape[0] - 1
    sample = demeaned.T.dot(demeaned) / n
    out = shrink_cov(data)
    comm = sample.dot(out) - out.dot(sample)
    assert np.allclose(comm, 0.0, atol=1e-8)


def test_single_variable():
    np.random.seed(17)
    data = _make_data(50, 1)
    out = shrink_cov(data)
    assert out.shape == (1, 1)
    assert np.isfinite(out[0, 0])
    assert out[0, 0] > 0


def test_shrinks_eigenvalue_dispersion_for_identity_truth():
    np.random.seed(18)
    n, p = 200, 50
    data = _make_data(n, p)
    demeaned = data - data.mean(axis=0)
    sample = demeaned.T.dot(demeaned) / (n - 1)
    sample_eigs = np.linalg.eigvalsh(sample)
    out_eigs = np.linalg.eigvalsh(shrink_cov(data))
    assert (out_eigs.max() - out_eigs.min()) < (sample_eigs.max() - sample_eigs.min())
