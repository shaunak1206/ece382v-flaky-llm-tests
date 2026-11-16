import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def make_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, p)


def test_output_shape_p_less_than_n():
    data = make_data(100, 10)
    out = shrink_cov(data)
    assert out.shape == (10, 10)


def test_output_shape_p_greater_than_n():
    data = make_data(20, 30, seed=1)
    out = shrink_cov(data)
    assert out.shape == (30, 30)


def test_output_is_symmetric():
    data = make_data(80, 15, seed=2)
    out = shrink_cov(data)
    assert np.allclose(out, out.T, atol=1e-10)


def test_output_is_finite_and_positive_definite():
    data = make_data(120, 20, seed=3)
    out = shrink_cov(data)
    assert np.all(np.isfinite(out))
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(eigvals > 0)


def test_high_dimensional_output_finite_and_positive_definite():
    data = make_data(20, 30, seed=4)
    out = shrink_cov(data)
    assert np.all(np.isfinite(out))
    eigvals = np.linalg.eigvalsh((out + out.T) / 2)
    assert np.all(eigvals > 0)


def test_one_dimensional_input_raises_assertion():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(50, dtype=float))


def test_three_dimensional_input_raises_assertion():
    with pytest.raises(AssertionError):
        shrink_cov(np.zeros((20, 3, 2)))


def test_sample_size_too_small_default_k():
    # n=12 with default k=None gives effective sample size 11 < 12
    data = make_data(12, 3, seed=5)
    with pytest.raises(AssertionError):
        shrink_cov(data)


def test_sample_size_too_small_k_zero():
    data = make_data(11, 3, seed=6)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=0)


def test_minimum_sample_size_accepted():
    data = make_data(12, 3, seed=7)
    out = shrink_cov(data, k=0)
    assert out.shape == (3, 3)
    assert np.all(np.isfinite(out))

    data = make_data(13, 3, seed=8)
    out = shrink_cov(data)
    assert out.shape == (3, 3)


def test_singular_duplicate_column_raises_value_error():
    data = make_data(50, 4, seed=9)
    data[:, 3] = data[:, 0]
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_zero_column_raises_value_error():
    data = make_data(50, 4, seed=10)
    data[:, 2] = 0.0
    with pytest.raises(ValueError):
        shrink_cov(data, k=0)


def test_default_k_equals_manual_demeaning_with_k1():
    data = make_data(60, 8, seed=11) + 5.0
    out_default = shrink_cov(data)
    out_manual = shrink_cov(data - data.mean(axis=0), k=1)
    assert np.allclose(out_default, out_manual)


def test_default_is_invariant_to_mean_shift():
    data = make_data(60, 8, seed=12)
    shifted = data + np.arange(8) * 10.0
    assert np.allclose(shrink_cov(data), shrink_cov(shifted), atol=1e-8)


def test_k_zero_differs_from_default_k():
    data = make_data(40, 5, seed=13)
    centered = data - data.mean(axis=0)
    out_k0 = shrink_cov(centered, k=0)
    out_k1 = shrink_cov(centered, k=1)
    assert not np.allclose(out_k0, out_k1)


def test_does_not_modify_input():
    data = make_data(50, 6, seed=14)
    original = data.copy()
    shrink_cov(data)
    shrink_cov(data, k=0)
    assert np.array_equal(data, original)


def test_deterministic():
    data = make_data(50, 6, seed=15)
    assert np.array_equal(shrink_cov(data), shrink_cov(data))


def test_scale_equivariance():
    data = make_data(70, 7, seed=16)
    c = 3.0
    out = shrink_cov(data)
    out_scaled = shrink_cov(c * data)
    assert np.allclose(out_scaled, c ** 2 * out, rtol=1e-6, atol=1e-8)


def test_close_to_identity_for_large_sample_identity_covariance():
    data = make_data(2000, 10, seed=17)
    out = shrink_cov(data)
    assert np.allclose(out, np.eye(10), atol=0.15)


def test_shrinkage_reduces_condition_number():
    data = make_data(100, 50, seed=18)
    centered = data - data.mean(axis=0)
    sample_cov = centered.T @ centered / 99
    out = shrink_cov(data)
    cond_sample = np.linalg.cond(sample_cov)
    cond_shrunk = np.linalg.cond(out)
    assert cond_shrunk < cond_sample
