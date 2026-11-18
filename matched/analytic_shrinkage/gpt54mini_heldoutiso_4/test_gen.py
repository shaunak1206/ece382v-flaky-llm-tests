import numpy as np
import pytest

from nonlinshrink.nonlinshrink import EPS, fsopt, loss_fr, loss_mv, prial, shrink_cov


def _spd_matrix(p, seed=0, cond=5.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(p, p))
    q, _ = np.linalg.qr(a)
    eigs = np.linspace(1.0, cond, p)
    return q @ np.diag(eigs) @ q.T


def _random_data(n, p, seed=0, mean_shift=0.0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, p))
    if mean_shift != 0.0:
        data = data + mean_shift
    return data


def test_shrink_cov_rejects_non_2d_input():
    x = np.arange(15.0)
    with pytest.raises(AssertionError, match="2d array"):
        shrink_cov(x)


def test_shrink_cov_rejects_small_effective_sample_size_default_k():
    data = _random_data(11, 4, seed=1)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data)


def test_shrink_cov_rejects_small_effective_sample_size_with_k():
    data = _random_data(13, 4, seed=2)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data, k=2)  # effective n = 11


def test_shrink_cov_returns_symmetric_matrix_and_correct_shape():
    data = _random_data(18, 6, seed=3)
    out = shrink_cov(data)
    assert out.shape == (6, 6)
    assert np.allclose(out, out.T, atol=1e-10)


def test_shrink_cov_default_k_centers_data_equivalent_to_manual_centering_shape_branch():
    raw = _random_data(20, 5, seed=4, mean_shift=4.0)
    centered = raw - raw.mean(axis=0)
    out1 = shrink_cov(raw)
    out2 = shrink_cov(centered, k=1)
    assert out1.shape == (5, 5)
    assert out2.shape == (5, 5)
    assert np.allclose(out1, out2, atol=1e-10)


def test_shrink_cov_k_zero_uses_raw_data_without_demeaning():
    data = _random_data(20, 5, seed=5, mean_shift=3.0)
    out_k0 = shrink_cov(data, k=0)
    out_default = shrink_cov(data)
    assert out_k0.shape == (5, 5)
    assert not np.allclose(out_k0, out_default)


def test_shrink_cov_k_one_differs_from_default_due_to_no_demeaning_and_effective_sample_size():
    data = _random_data(20, 5, seed=6, mean_shift=2.5)
    out_k1 = shrink_cov(data, k=1)
    out_default = shrink_cov(data)
    assert out_k1.shape == (5, 5)
    assert not np.allclose(out_k1, out_default)


def test_shrink_cov_handles_p_less_equal_n_branch():
    data = _random_data(25, 10, seed=7)
    out = shrink_cov(data)
    assert out.shape == (10, 10)
    assert np.isfinite(out).all()


def test_shrink_cov_handles_p_greater_than_n_branch():
    data = _random_data(13, 20, seed=8)
    out = shrink_cov(data, k=1)  # effective n = 12, so p > n
    assert out.shape == (20, 20)
    assert np.isfinite(out).all()
    assert np.allclose(out, out.T, atol=1e-10)


def test_shrink_cov_raises_on_singular_covariance_matrix():
    rng = np.random.default_rng(9)
    base = rng.normal(size=(20, 4))
    data = np.column_stack([base, base[:, 0]])  # duplicate column => singular sample cov
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data, k=0)


def test_shrink_cov_identity_like_data_returns_finite_matrix():
    data = np.eye(12)
    out = shrink_cov(data, k=0)
    assert out.shape == (12, 12)
    assert np.isfinite(out).all()
    assert np.allclose(out, out.T, atol=1e-10)


def test_fsopt_returns_symmetric_matrix_and_preserves_shape():
    sample = _spd_matrix(4, seed=10)
    sigma = _spd_matrix(4, seed=11)
    out = fsopt(sample, sigma)
    assert out.shape == (4, 4)
    assert np.allclose(out, out.T, atol=1e-10)


def test_loss_mv_and_loss_fr_are_nonnegative_for_identical_matrices():
    sigma = _spd_matrix(5, seed=12)
    assert loss_fr(sigma, sigma) == pytest.approx(0.0)
    assert loss_mv(sigma, sigma) == pytest.approx(0.0, abs=1e-10)


def test_prial_is_one_for_fully_optimal_estimator():
    sample = _spd_matrix(4, seed=13)
    sigma = _spd_matrix(4, seed=14)
    sigma_hat = fsopt(sample, sigma)
    assert prial(sample, sigma_hat, sigma) == pytest.approx(1.0, rel=1e-10, abs=1e-10)


def test_prial_is_zero_when_using_sample_covariance():
    sample = _spd_matrix(4, seed=15)
    sigma = _spd_matrix(4, seed=16)
    assert prial(sample, sample, sigma) == pytest.approx(0.0, rel=1e-10, abs=1e-10)


def test_shrink_cov_output_is_finite_for_random_high_dimensional_input():
    data = _random_data(14, 30, seed=17)
    out = shrink_cov(data, k=1)
    assert out.shape == (30, 30)
    assert np.isfinite(out).all()
    assert np.all(np.diag(out) >= 0)


def test_eps_constant_small_positive_value():
    assert EPS > 0
    assert EPS < 1e-6
