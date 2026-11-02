import numpy as np
import pytest

from nonlinshrink import shrink_cov


def _rng(seed):
    return np.random.RandomState(seed)


def _scalar_expected(sum_sq, n_eff):
    """Closed form of shrink_cov for p = 1, derived from the analytic formula.

    With p = 1 the kernel terms reduce to H_f = 0 and f = 3 / (4 sqrt(5) h lam),
    so the shrunk variance is lam / ((3 pi / (4 sqrt 5) n^(-2/3))^2 + (1 - 1/n)^2).
    """
    lam = sum_sq / n_eff
    a = 3.0 * np.pi / (4.0 * np.sqrt(5.0)) * n_eff ** (-2.0 / 3.0)
    b = 1.0 - 1.0 / n_eff
    return lam / (a ** 2 + b ** 2)


def test_p1_default_matches_closed_form():
    x = _rng(1).standard_normal((40, 1)) * 2.0 + 3.0
    xc = x - x.mean(axis=0)
    expected = _scalar_expected(np.sum(xc ** 2), 39)
    out = shrink_cov(x)
    assert out.shape == (1, 1)
    np.testing.assert_allclose(out[0, 0], expected, rtol=1e-9)


def test_p1_k_zero_matches_closed_form():
    x = _rng(2).standard_normal((40, 1)) * 2.0 + 3.0
    expected = _scalar_expected(np.sum(x ** 2), 40)
    np.testing.assert_allclose(shrink_cov(x, k=0)[0, 0], expected, rtol=1e-9)


def test_p1_k_positive_matches_closed_form():
    x = _rng(3).standard_normal((40, 1)) * 2.0 + 3.0
    expected = _scalar_expected(np.sum(x ** 2), 37)
    np.testing.assert_allclose(shrink_cov(x, k=3)[0, 0], expected, rtol=1e-9)


def test_returns_p_by_p_ndarray():
    for p in (2, 3, 7):
        x = _rng(p).standard_normal((30, p))
        out = shrink_cov(x)
        assert isinstance(out, np.ndarray)
        assert out.shape == (p, p)


def test_output_is_symmetric():
    x = _rng(4).standard_normal((50, 6))
    out = shrink_cov(x)
    np.testing.assert_allclose(out, out.T, rtol=1e-10, atol=1e-12)


def test_output_positive_definite_when_p_below_n():
    x = _rng(5).standard_normal((60, 10))
    out = shrink_cov(x)
    assert np.linalg.eigvalsh(out).min() > 0


def test_shares_eigenvectors_with_sample_covariance():
    # The estimator is u diag(d) u^T with u the sample eigenvectors, so it
    # must commute with the (demeaned) sample covariance.
    x = _rng(6).standard_normal((50, 5))
    xc = x - x.mean(axis=0)
    sample = xc.T @ xc / 49
    out = shrink_cov(x)
    np.testing.assert_allclose(out @ sample, sample @ out, rtol=1e-6, atol=1e-8)


def test_default_invariant_to_location_shift():
    x = _rng(7).standard_normal((45, 4))
    np.testing.assert_allclose(shrink_cov(x + 10.0), shrink_cov(x),
                               rtol=1e-8, atol=1e-10)


def test_default_equals_explicit_k1_on_demeaned_data():
    x = _rng(8).standard_normal((45, 4)) + 2.0
    xc = x - x.mean(axis=0)
    np.testing.assert_allclose(shrink_cov(x), shrink_cov(xc, k=1),
                               rtol=1e-9, atol=1e-12)


def test_scale_equivariance_quadratic():
    x = _rng(9).standard_normal((40, 5))
    c = 3.0
    np.testing.assert_allclose(shrink_cov(c * x), c ** 2 * shrink_cov(x),
                               rtol=1e-8, atol=1e-10)


def test_invariant_to_row_order():
    x = _rng(10).standard_normal((40, 5))
    perm = _rng(11).permutation(40)
    np.testing.assert_allclose(shrink_cov(x[perm]), shrink_cov(x),
                               rtol=1e-8, atol=1e-10)


def test_equivariant_to_column_permutation():
    x = _rng(12).standard_normal((40, 5))
    perm = np.array([3, 0, 4, 1, 2])
    out = shrink_cov(x)
    out_perm = shrink_cov(x[:, perm])
    np.testing.assert_allclose(out_perm, out[np.ix_(perm, perm)],
                               rtol=1e-8, atol=1e-10)


def test_does_not_modify_input():
    x = _rng(13).standard_normal((30, 4))
    original = x.copy()
    shrink_cov(x)
    shrink_cov(x, k=0)
    shrink_cov(x, k=2)
    np.testing.assert_array_equal(x, original)


def test_rejects_non_2d_input():
    with pytest.raises(AssertionError, match="2d"):
        shrink_cov(np.ones(20))
    with pytest.raises(AssertionError):
        shrink_cov(np.ones((4, 5, 3)))


def test_default_requires_12_effective_observations():
    x = _rng(14).standard_normal((12, 2))  # 12 - 1 = 11 effective
    with pytest.raises(AssertionError, match="12"):
        shrink_cov(x)


def test_default_accepts_13_observations():
    x = _rng(15).standard_normal((13, 2))  # 13 - 1 = 12 effective
    out = shrink_cov(x)
    assert out.shape == (2, 2)
    assert np.all(np.isfinite(out))


def test_k_zero_allows_12_observations():
    x = _rng(16).standard_normal((12, 2))  # k = 0 keeps all 12
    out = shrink_cov(x, k=0)
    assert out.shape == (2, 2)
    assert np.all(np.isfinite(out))


def test_k_reduces_effective_sample_size_below_threshold():
    x = _rng(17).standard_normal((13, 2))  # 13 - 2 = 11 < 12
    with pytest.raises(AssertionError):
        shrink_cov(x, k=2)


def test_duplicate_columns_raise_singular():
    x = _rng(18).standard_normal((30, 4))
    x[:, 3] = x[:, 0]
    with pytest.raises(ValueError, match="singular"):
        shrink_cov(x)


def test_constant_column_raises_singular():
    x = _rng(19).standard_normal((30, 4))
    x[:, 1] = 5.0
    with pytest.raises(ValueError, match="singular"):
        shrink_cov(x)


def test_high_dimensional_output_finite_and_symmetric():
    x = _rng(20).standard_normal((15, 40))  # p > n
    out = shrink_cov(x)
    assert out.shape == (40, 40)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, out.T, rtol=1e-10, atol=1e-12)


def test_high_dimensional_output_positive_semidefinite():
    x = _rng(21).standard_normal((20, 50))  # p > n
    eig = np.linalg.eigvalsh(shrink_cov(x))
    assert eig.min() > -1e-10 * eig.max()


def test_repeated_calls_are_identical():
    x = _rng(22).standard_normal((30, 4))
    np.testing.assert_allclose(shrink_cov(x), shrink_cov(x), rtol=0, atol=0)
