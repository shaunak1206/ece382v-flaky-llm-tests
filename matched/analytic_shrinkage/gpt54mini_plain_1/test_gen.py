import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _random_data(n, p, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p))


def test_shrink_cov_returns_square_matrix_same_dimension():
    x = _random_data(20, 5, seed=1)
    s = shrink_cov(x)
    assert s.shape == (5, 5)


def test_shrink_cov_output_is_symmetric():
    x = _random_data(18, 6, seed=2)
    s = shrink_cov(x)
    assert np.allclose(s, s.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_output_has_finite_entries():
    x = _random_data(25, 8, seed=3)
    s = shrink_cov(x)
    assert np.isfinite(s).all()


def test_shrink_cov_accepts_p_greater_than_n():
    x = _random_data(15, 20, seed=4)
    s = shrink_cov(x, k=0)
    assert s.shape == (20, 20)
    assert np.allclose(s, s.T, atol=1e-10, rtol=1e-10)
    assert np.isfinite(s).all()


def test_shrink_cov_accepts_square_input():
    x = _random_data(16, 16, seed=5)
    s = shrink_cov(x, k=0)
    assert s.shape == (16, 16)
    assert np.allclose(s, s.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_raises_on_non_2d_input():
    x = np.arange(20.0)
    with pytest.raises(AssertionError, match="2d array"):
        shrink_cov(x)


def test_shrink_cov_raises_when_effective_sample_size_too_small_with_default_k():
    x = _random_data(12, 3, seed=6)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(x)


def test_shrink_cov_raises_when_effective_sample_size_too_small_with_k():
    x = _random_data(13, 3, seed=7)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(x, k=2)


def test_shrink_cov_raises_on_singular_matrix():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(20, 4))
    x[:, 3] = x[:, 0]  # duplicate column => singular sample covariance
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(x, k=0)


def test_shrink_cov_is_invariant_to_constant_shift_when_k_none():
    x = _random_data(22, 7, seed=9)
    shifted = x + 10.0
    s1 = shrink_cov(x)
    s2 = shrink_cov(shifted)
    assert np.allclose(s1, s2, atol=1e-10, rtol=1e-10)


def test_shrink_cov_k_none_matches_k_one_for_centered_data():
    x = _random_data(24, 6, seed=10)
    x = x - x.mean(axis=0)
    s_none = shrink_cov(x)
    s_one = shrink_cov(x, k=1)
    assert np.allclose(s_none, s_one, atol=1e-10, rtol=1e-10)


def test_shrink_cov_k_zero_differs_from_default_on_centered_data():
    x = _random_data(24, 6, seed=11)
    x = x - x.mean(axis=0)
    s_default = shrink_cov(x)
    s_zero = shrink_cov(x, k=0)
    assert s_default.shape == s_zero.shape
    assert not np.allclose(s_default, s_zero, atol=1e-12, rtol=1e-12)


def test_shrink_cov_scales_quadratically_with_input_scaling():
    x = _random_data(20, 5, seed=12)
    c = 3.5
    s1 = shrink_cov(x, k=0)
    s2 = shrink_cov(c * x, k=0)
    assert np.allclose(s2, (c ** 2) * s1, atol=1e-8, rtol=1e-8)


def test_shrink_cov_returns_positive_diagonal_entries():
    x = _random_data(30, 9, seed=13)
    s = shrink_cov(x, k=0)
    assert np.all(np.diag(s) > 0)
