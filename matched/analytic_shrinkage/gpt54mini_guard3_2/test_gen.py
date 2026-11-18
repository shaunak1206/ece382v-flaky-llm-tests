import numpy as np
import pytest

from nonlinshrink.nonlinshrink import EPS, fsopt, loss_fr, loss_mv, prial, shrink_cov


def test_shrink_cov_rejects_non_2d_input():
    data = np.arange(12.0)
    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(data)


def test_shrink_cov_rejects_small_effective_sample_size_default_k():
    data = np.arange(22.0).reshape(11, 2)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data)


def test_shrink_cov_rejects_small_effective_sample_size_with_k():
    data = np.arange(26.0).reshape(13, 2)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data, k=2)


def test_shrink_cov_returns_symmetric_matrix():
    data = np.arange(48.0).reshape(12, 4)
    out = shrink_cov(data, k=0)
    assert out.shape == (4, 4)
    assert np.allclose(out, out.T, atol=1e-12, rtol=0)


def test_shrink_cov_output_is_finite_for_well_behaved_data():
    data = np.arange(60.0).reshape(15, 4)
    out = shrink_cov(data, k=0)
    assert np.isfinite(out).all()


def test_shrink_cov_default_k_demeans_data_and_differs_from_k_zero():
    data = np.arange(60.0).reshape(15, 4)
    out_default = shrink_cov(data)
    out_k0 = shrink_cov(data, k=0)
    assert out_default.shape == out_k0.shape == (4, 4)
    assert not np.allclose(out_default, out_k0)


def test_shrink_cov_k_none_equivalent_to_explicit_k_one():
    data = np.arange(60.0).reshape(15, 4)
    out_none = shrink_cov(data, k=None)
    out_k1 = shrink_cov(data, k=1)
    assert np.allclose(out_none, out_k1, atol=1e-12, rtol=1e-12)


def test_shrink_cov_k_zero_matches_no_demeaning_behavior():
    data = np.arange(60.0).reshape(15, 4)
    out_k0 = shrink_cov(data, k=0)

    n = data.shape[0]
    sample_cov = np.dot(data.T, data) / n
    lam, u = np.linalg.eigh(sample_cov)
    lam = lam[np.maximum(0, data.shape[1] - n):]
    assert not np.any(lam / np.sum(lam) < EPS)
    assert out_k0.shape == (4, 4)
    assert np.allclose(out_k0, out_k0.T, atol=1e-12, rtol=0)


def test_shrink_cov_high_dimensional_output_shape_and_symmetry():
    data = np.arange(150.0).reshape(15, 10)
    out = shrink_cov(data, k=0)
    assert out.shape == (10, 10)
    assert np.allclose(out, out.T, atol=1e-12, rtol=0)


def test_shrink_cov_high_dimensional_case_p_greater_than_n():
    data = np.arange(120.0).reshape(12, 15)
    out = shrink_cov(data, k=0)
    assert out.shape == (15, 15)
    assert np.isfinite(out).all()
    assert np.allclose(out, out.T, atol=1e-12, rtol=0)


def test_shrink_cov_identity_like_data_is_stable():
    data = np.eye(12, 4)
    out = shrink_cov(data, k=0)
    assert out.shape == (4, 4)
    assert np.isfinite(out).all()
    assert np.allclose(out, out.T, atol=1e-12, rtol=0)
    assert np.all(np.diag(out) > 0)


def test_fsopt_returns_symmetric_matrix():
    sample = np.array(
        [[2.0, 0.5, 0.1],
         [0.5, 1.5, 0.2],
         [0.1, 0.2, 1.0]]
    )
    sigma = np.array(
        [[1.0, 0.2, 0.0],
         [0.2, 2.0, 0.1],
         [0.0, 0.1, 1.5]]
    )
    out = fsopt(sample, sigma)
    assert out.shape == sample.shape
    assert np.allclose(out, out.T, atol=1e-12, rtol=0)


def test_loss_mv_is_nonnegative_for_equal_matrices():
    sigma = np.array(
        [[2.0, 0.3],
         [0.3, 1.0]]
    )
    val = loss_mv(sigma, sigma)
    assert np.isfinite(val)
    assert val >= -1e-12


def test_loss_fr_zero_for_equal_matrices():
    sigma = np.array(
        [[2.0, 0.3],
         [0.3, 1.0]]
    )
    assert loss_fr(sigma, sigma) == pytest.approx(0.0, abs=1e-15)


def test_prial_is_one_when_estimator_equals_fsopt():
    sample = np.array(
        [[2.0, 0.5, 0.1],
         [0.5, 1.5, 0.2],
         [0.1, 0.2, 1.0]]
    )
    sigma = np.array(
        [[1.0, 0.2, 0.0],
         [0.2, 2.0, 0.1],
         [0.0, 0.1, 1.5]]
    )
    sigma_hat = fsopt(sample, sigma)
    value = prial(sample, sigma_hat, sigma)
    assert value == pytest.approx(1.0, abs=1e-12, rel=1e-12)


def test_prial_is_zero_when_estimator_equals_sample():
    sample = np.array(
        [[2.0, 0.5, 0.1],
         [0.5, 1.5, 0.2],
         [0.1, 0.2, 1.0]]
    )
    sigma = np.array(
        [[1.0, 0.2, 0.0],
         [0.2, 2.0, 0.1],
         [0.0, 0.1, 1.5]]
    )
    value = prial(sample, sample, sigma)
    assert value == pytest.approx(0.0, abs=1e-12, rel=1e-12)
