import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def make_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p)


def test_output_shape_p_less_than_n():
    data = make_data(100, 5)
    out = shrink_cov(data)
    assert out.shape == (5, 5)


def test_output_shape_p_greater_than_n():
    data = make_data(20, 40, seed=1)
    out = shrink_cov(data)
    assert out.shape == (40, 40)


def test_output_is_symmetric():
    data = make_data(150, 8, seed=2)
    out = shrink_cov(data)
    assert np.allclose(out, out.T, atol=1e-10)


def test_output_is_finite_and_real():
    data = make_data(80, 10, seed=3)
    out = shrink_cov(data)
    assert np.all(np.isfinite(out))
    assert np.isrealobj(out)


def test_output_positive_definite():
    data = make_data(200, 6, seed=4)
    out = shrink_cov(data)
    eigvals = np.linalg.eigvalsh((out + out.T) / 2)
    assert np.all(eigvals > 0)


def test_high_dimensional_symmetric_and_finite():
    data = make_data(30, 60, seed=5)
    out = shrink_cov(data)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out.T, atol=1e-8)


def test_1d_input_raises_assertion():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(50, dtype=float))


def test_3d_input_raises_assertion():
    with pytest.raises(AssertionError):
        shrink_cov(np.zeros((20, 3, 2)))


def test_small_sample_size_raises_default_k():
    # n=12 with k=None gives effective n=11 < 12
    data = make_data(12, 3, seed=6)
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_small_sample_size_boundary_ok_with_k0():
    # n=12 with k=0 gives effective n=12, which is allowed
    data = make_data(12, 3, seed=7)
    out = shrink_cov(data, k=0)
    assert out.shape == (3, 3)


def test_k_reduces_effective_sample_size_below_threshold():
    data = make_data(20, 3, seed=8)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=10)


def test_constant_column_raises_value_error():
    data = make_data(100, 4, seed=9)
    data[:, 2] = 1.0
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_default_equals_k1_on_demeaned_data():
    data = make_data(100, 5, seed=10)
    demeaned = data - data.mean(axis=0)
    out_default = shrink_cov(data)
    out_k1 = shrink_cov(demeaned, k=1)
    assert np.allclose(out_default, out_k1, atol=1e-10)


def test_input_not_modified():
    data = make_data(60, 4, seed=11)
    copy = data.copy()
    shrink_cov(data)
    shrink_cov(data, k=0)
    assert np.array_equal(data, copy)


def test_deterministic():
    data = make_data(70, 6, seed=12)
    assert np.array_equal(shrink_cov(data), shrink_cov(data.copy()))


def test_scale_equivariance():
    data = make_data(120, 5, seed=13)
    c = 3.0
    out = shrink_cov(data)
    out_scaled = shrink_cov(c * data)
    assert np.allclose(out_scaled, c ** 2 * out, rtol=1e-6, atol=1e-8)


def test_column_permutation_equivariance():
    data = make_data(100, 6, seed=14)
    perm = np.array([3, 1, 5, 0, 2, 4])
    out = shrink_cov(data)
    out_perm = shrink_cov(data[:, perm])
    assert np.allclose(out_perm, out[np.ix_(perm, perm)], atol=1e-8)


def test_near_identity_for_large_sample_identity_covariance():
    data = make_data(3000, 4, seed=15)
    out = shrink_cov(data)
    assert np.allclose(out, np.eye(4), atol=0.2)


def test_shrinkage_improves_condition_number():
    n, p = 100, 50
    data = make_data(n, p, seed=16)
    demeaned = data - data.mean(axis=0)
    sample = demeaned.T.dot(demeaned) / (n - 1)
    shrunk = shrink_cov(data)
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample)


def test_trace_roughly_preserved():
    data = make_data(500, 10, seed=17)
    demeaned = data - data.mean(axis=0)
    sample = demeaned.T.dot(demeaned) / (500 - 1)
    shrunk = shrink_cov(data)
    assert np.trace(shrunk) == pytest.approx(np.trace(sample), rel=0.25)
