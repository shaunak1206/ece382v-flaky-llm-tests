import numpy as np
import pytest

import nonlinshrink.nonlinshrink as shrink_module
from nonlinshrink.nonlinshrink import shrink_cov


def _data(n, p, seed):
    """Deterministic Gaussian-like data matrix built from an explicit seed."""
    return np.random.RandomState(seed).randn(n, p)


def _sample_cov_centered(X):
    """Sample covariance with the same convention as shrink_cov(X) for k=None."""
    Xc = X - X.mean(axis=0)
    n = X.shape[0] - 1
    return Xc.T.dot(Xc) / n


def test_output_shape_is_p_by_p():
    np.random.seed(0)
    X = _data(n=50, p=5, seed=0)
    out = shrink_cov(X)
    assert out.shape == (5, 5)


def test_returns_float_ndarray():
    np.random.seed(1)
    X = _data(n=40, p=4, seed=1)
    out = shrink_cov(X)
    assert isinstance(out, np.ndarray)
    assert np.issubdtype(out.dtype, np.floating)


def test_output_is_symmetric():
    np.random.seed(2)
    X = _data(n=60, p=6, seed=2)
    out = shrink_cov(X)
    assert np.allclose(out, out.T, rtol=0, atol=1e-10)


def test_output_is_positive_definite_when_n_greater_than_p():
    np.random.seed(3)
    X = _data(n=80, p=6, seed=3)
    out = shrink_cov(X)
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(np.isfinite(eigvals))
    assert np.all(eigvals > 0)


def test_output_commutes_with_sample_covariance():
    # The estimator shares eigenvectors with the sample covariance,
    # so the two matrices must commute.
    np.random.seed(4)
    X = _data(n=60, p=5, seed=4)
    S = _sample_cov_centered(X)
    out = shrink_cov(X)
    assert np.allclose(out.dot(S), S.dot(out), rtol=1e-6, atol=1e-8)


def test_default_demeans_same_as_explicit_k1_on_centered_data():
    np.random.seed(5)
    X = _data(n=50, p=4, seed=5)
    Xc = X - X.mean(axis=0)
    out_default = shrink_cov(X)
    out_explicit = shrink_cov(Xc, k=1)
    assert np.allclose(out_default, out_explicit, rtol=1e-10, atol=1e-12)


def test_default_is_invariant_to_location_shift():
    np.random.seed(6)
    X = _data(n=50, p=4, seed=6)
    shift = np.array([10.0, -3.0, 0.5, 100.0])
    out_original = shrink_cov(X)
    out_shifted = shrink_cov(X + shift)
    assert np.allclose(out_original, out_shifted, rtol=1e-8, atol=1e-10)


def test_k_zero_does_not_demean():
    # With k=0 the data is used as given (no centering), so for data with a
    # nonzero mean the result must differ from the default demeaning path.
    np.random.seed(7)
    X = _data(n=50, p=4, seed=7) + 5.0
    out_default = shrink_cov(X)
    out_k0 = shrink_cov(X, k=0)
    assert not np.allclose(out_default, out_k0, rtol=1e-3, atol=1e-6)


def test_k_one_on_uncentered_data_differs_from_centered_data():
    np.random.seed(8)
    X = _data(n=50, p=4, seed=8) + 2.0
    Xc = X - X.mean(axis=0)
    out_uncentered = shrink_cov(X, k=1)
    out_centered = shrink_cov(Xc, k=1)
    assert not np.allclose(out_uncentered, out_centered, rtol=1e-3, atol=1e-6)


def test_scale_equivariance_quadratic_in_scale():
    # Covariance estimator must scale by c**2 when data is scaled by c.
    np.random.seed(9)
    X = _data(n=60, p=5, seed=9)
    c = 3.0
    out = shrink_cov(X)
    out_scaled = shrink_cov(c * X)
    assert np.allclose(out_scaled, (c ** 2) * out, rtol=1e-7, atol=1e-10)


def test_column_permutation_equivariance():
    np.random.seed(10)
    X = _data(n=60, p=5, seed=10)
    perm = [2, 0, 4, 1, 3]
    out = shrink_cov(X)
    out_perm = shrink_cov(X[:, perm])
    expected = out[np.ix_(perm, perm)]
    assert np.allclose(out_perm, expected, rtol=1e-7, atol=1e-10)


def test_row_order_invariance():
    np.random.seed(11)
    X = _data(n=60, p=5, seed=11)
    out = shrink_cov(X)
    out_reversed = shrink_cov(X[::-1].copy())
    assert np.allclose(out, out_reversed, rtol=1e-8, atol=1e-10)


def test_input_array_is_not_modified():
    np.random.seed(12)
    X = _data(n=50, p=4, seed=12)
    X_before = X.copy()
    shrink_cov(X)
    shrink_cov(X, k=1)
    assert np.array_equal(X, X_before)


def test_repeated_calls_give_identical_results():
    np.random.seed(13)
    X = _data(n=50, p=4, seed=13)
    first = shrink_cov(X)
    second = shrink_cov(X)
    assert np.array_equal(first, second)


def test_rejects_non_2d_input():
    np.random.seed(14)
    X = np.random.RandomState(14).randn(20, 3, 2)
    with pytest.raises(AssertionError):
        shrink_cov(X)


def test_rejects_too_few_effective_observations_default_path():
    # 11 rows with default demeaning gives effective n = 10 < 12.
    np.random.seed(15)
    X = _data(n=11, p=3, seed=15)
    with pytest.raises(AssertionError):
        shrink_cov(X)


def test_minimum_effective_sample_size_boundary_default_path():
    # 13 rows with default demeaning gives effective n = 12, which is allowed.
    np.random.seed(16)
    X = _data(n=13, p=3, seed=16)
    out = shrink_cov(X)
    assert out.shape == (3, 3)
    assert np.all(np.isfinite(out))


def test_k_zero_sample_size_boundary():
    # With k=0 the effective sample size equals n.
    np.random.seed(17)
    X_ok = _data(n=12, p=3, seed=17)
    assert shrink_cov(X_ok, k=0).shape == (3, 3)
    X_bad = _data(n=11, p=3, seed=17)
    with pytest.raises(AssertionError):
        shrink_cov(X_bad, k=0)


def test_singular_sample_covariance_raises_value_error():
    # A column that is an exact multiple of another makes the sample
    # covariance singular.
    np.random.seed(18)
    X = _data(n=30, p=4, seed=18)
    X[:, 2] = 2.0 * X[:, 0]
    with pytest.raises(ValueError):
        shrink_cov(X)


def test_high_dimensional_case_p_greater_than_n():
    np.random.seed(19)
    X = _data(n=40, p=60, seed=19)
    out = shrink_cov(X)
    assert out.shape == (60, 60)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out.T, rtol=0, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(out) > 0)


def test_high_dimensional_null_space_directions_share_one_value():
    # For p > n the p - n directions with zero sample variance all receive
    # the same shrunk value, so the diagonal in the sample eigenbasis is constant there.
    np.random.seed(20)
    n, p = 40, 60
    X = _data(n=n, p=p, seed=20)
    S = _sample_cov_centered(X)
    _, u = np.linalg.eigh(S)
    out = shrink_cov(X)
    d = np.diag(u.T.dot(out).dot(u))
    null_block = d[: p - (n - 1)]
    assert np.allclose(null_block, null_block[0], rtol=1e-6, atol=1e-8)


def test_equal_sample_eigenvalues_give_scalar_multiple_of_identity():
    # Data whose sample covariance is exactly the identity has equal
    # eigenvalues, so the estimate must be a scalar multiple of I.
    np.random.seed(21)
    n, p = 30, 5
    Q, _ = np.linalg.qr(np.random.RandomState(21).randn(n, p))
    X = Q * np.sqrt(n - 1)
    out = shrink_cov(X, k=1)
    off_diag = out - np.diag(np.diag(out))
    assert np.allclose(off_diag, 0.0, atol=1e-8)
    diag = np.diag(out)
    assert np.allclose(diag, diag[0], rtol=1e-6, atol=1e-10)


def test_monkeypatched_eps_forces_singular_error(monkeypatch):
    # With a large tolerance, well-conditioned data is treated as singular.
    # The smallest eigenvalue ratio is at most 1/p = 0.2 < 0.5 for p = 5.
    np.random.seed(22)
    X = _data(n=50, p=5, seed=22)
    monkeypatch.setattr(shrink_module, "EPS", 0.5)
    with pytest.raises(ValueError):
        shrink_cov(X)


def test_monkeypatched_eps_zero_matches_default_on_well_conditioned_data(monkeypatch):
    np.random.seed(23)
    X = _data(n=50, p=5, seed=23)
    out_default = shrink_cov(X)
    monkeypatch.setattr(shrink_module, "EPS", 0.0)
    out_patched = shrink_cov(X)
    assert np.allclose(out_default, out_patched, rtol=1e-10, atol=1e-12)
