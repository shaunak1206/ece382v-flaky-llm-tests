import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def test_shrink_cov_rejects_non_2d_input():
    x = np.arange(12.0)
    with pytest.raises(AssertionError, match="2d array"):
        shrink_cov(x)


def test_shrink_cov_rejects_small_effective_sample_size_default_k():
    data = np.arange(11 * 3.0).reshape(11, 3)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data)


def test_shrink_cov_rejects_small_effective_sample_size_with_k():
    data = np.arange(14 * 3.0).reshape(14, 3)
    # Effective sample size is 11, so it should fail.
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data, k=3)


def test_shrink_cov_returns_square_matrix_same_dimension():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(20, 5))
    out = shrink_cov(data)
    assert out.shape == (5, 5)


def test_shrink_cov_output_is_symmetric_for_p_le_n():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(24, 6))
    out = shrink_cov(data)
    assert np.allclose(out, out.T, atol=1e-10)


def test_shrink_cov_output_is_symmetric_for_p_gt_n():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(12, 15))
    out = shrink_cov(data, k=0)
    assert out.shape == (15, 15)
    assert np.allclose(out, out.T, atol=1e-10)


def test_shrink_cov_output_is_finite_for_p_gt_n():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(12, 15))
    out = shrink_cov(data, k=0)
    assert np.isfinite(out).all()


def test_shrink_cov_default_demeaning_matches_explicit_k1_on_centered_data():
    rng = np.random.default_rng(4)
    data = rng.normal(size=(20, 4))
    data = data - data.mean(axis=0, keepdims=True)

    out_default = shrink_cov(data)
    out_k1 = shrink_cov(data, k=1)

    assert np.allclose(out_default, out_k1, atol=1e-10, rtol=1e-10)


def test_shrink_cov_k0_differs_from_default_when_data_is_not_centered():
    rng = np.random.default_rng(5)
    data = rng.normal(loc=3.0, scale=2.0, size=(25, 4))

    out_default = shrink_cov(data)
    out_k0 = shrink_cov(data, k=0)

    assert not np.allclose(out_default, out_k0)


def test_shrink_cov_differs_from_sample_covariance_on_generic_data():
    rng = np.random.default_rng(6)
    data = rng.normal(size=(30, 5))
    sample_cov = np.cov(data, rowvar=False, bias=False)
    out = shrink_cov(data)

    assert out.shape == sample_cov.shape
    assert not np.allclose(out, sample_cov)


def test_shrink_cov_is_idempotent_on_exactly_centered_input_with_k1():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(22, 3))
    data = data - data.mean(axis=0, keepdims=True)

    out1 = shrink_cov(data, k=1)
    out2 = shrink_cov(data, k=1)

    assert np.allclose(out1, out2, atol=1e-12)


def test_shrink_cov_handles_more_variables_than_observations():
    rng = np.random.default_rng(8)
    data = rng.normal(size=(13, 20))
    out = shrink_cov(data, k=0)

    assert out.shape == (20, 20)
    assert np.isfinite(out).all()
    assert np.allclose(out, out.T, atol=1e-10)


def test_shrink_cov_raises_on_singular_matrix():
    # Perfectly collinear columns lead to a singular covariance structure.
    rng = np.random.default_rng(9)
    base = rng.normal(size=(20, 1))
    data = np.hstack([base, base, base])

    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data, k=0)
