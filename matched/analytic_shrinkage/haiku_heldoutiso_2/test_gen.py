import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def _make_data(n, p, seed=0):
    rng = np.random.RandomState(seed)
    return rng.standard_normal((n, p))


def test_output_shape_matches_number_of_variables():
    X = _make_data(n=50, p=6)
    result = shrink_cov(X)
    assert result.shape == (6, 6)


def test_output_is_numpy_array():
    X = _make_data(n=50, p=4)
    result = shrink_cov(X)
    assert isinstance(result, np.ndarray)


def test_output_is_symmetric():
    X = _make_data(n=60, p=5, seed=1)
    result = shrink_cov(X)
    np.testing.assert_allclose(result, result.T, rtol=1e-10, atol=1e-12)


def test_output_is_positive_definite_when_n_greater_than_p():
    X = _make_data(n=80, p=8, seed=2)
    result = shrink_cov(X)
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals > 0)


def test_output_is_finite():
    X = _make_data(n=40, p=10, seed=3)
    result = shrink_cov(X)
    assert np.all(np.isfinite(result))


def test_one_dimensional_input_raises_assertion():
    x = np.arange(50, dtype=float)
    with pytest.raises(AssertionError):
        shrink_cov(x)


def test_three_dimensional_input_raises_assertion():
    X = np.zeros((20, 3, 2))
    with pytest.raises(AssertionError):
        shrink_cov(X)


def test_default_sample_size_boundary_passes_at_13_observations():
    # k=None subtracts 1, so n=13 gives effective n=12, which is allowed.
    X = _make_data(n=13, p=3, seed=4)
    result = shrink_cov(X)
    assert result.shape == (3, 3)


def test_default_sample_size_below_minimum_raises():
    # n=12 gives effective n=11 < 12.
    X = _make_data(n=12, p=3, seed=5)
    with pytest.raises(AssertionError):
        shrink_cov(X)


def test_k_zero_sample_size_boundary():
    # k=0 keeps effective n equal to n, so n=12 is allowed and n=11 is not.
    X_ok = _make_data(n=12, p=3, seed=6)
    assert shrink_cov(X_ok, k=0).shape == (3, 3)

    X_bad = _make_data(n=11, p=3, seed=6)
    with pytest.raises(AssertionError):
        shrink_cov(X_bad, k=0)


def test_k_positive_reduces_effective_sample_size():
    # With k=2, n=13 gives effective n=11 which is too small.
    X = _make_data(n=13, p=3, seed=7)
    with pytest.raises(AssertionError):
        shrink_cov(X, k=2)


def test_default_demeaning_matches_predemeaned_data_with_k_one():
    X = _make_data(n=70, p=5, seed=8) + 3.5
    demeaned = X - X.mean(axis=0)
    np.testing.assert_allclose(
        shrink_cov(X),
        shrink_cov(demeaned, k=1),
        rtol=1e-10,
        atol=1e-12,
    )


def test_k_zero_differs_from_default_on_demeaned_data():
    # Same centred data, but k=0 uses n instead of n-1 in the sample covariance.
    X = _make_data(n=60, p=4, seed=9)
    demeaned = X - X.mean(axis=0)
    default_result = shrink_cov(demeaned, k=1)
    zero_result = shrink_cov(demeaned, k=0)
    assert not np.allclose(default_result, zero_result)


def test_input_is_not_modified():
    X = _make_data(n=50, p=4, seed=10)
    original = X.copy()
    shrink_cov(X)
    np.testing.assert_array_equal(X, original)


def test_repeated_calls_are_deterministic():
    X = _make_data(n=50, p=4, seed=11)
    first = shrink_cov(X)
    second = shrink_cov(X)
    np.testing.assert_array_equal(first, second)


def test_row_permutation_invariance():
    X = _make_data(n=60, p=5, seed=12)
    rng = np.random.RandomState(99)
    perm = rng.permutation(X.shape[0])
    np.testing.assert_allclose(
        shrink_cov(X),
        shrink_cov(X[perm]),
        rtol=1e-8,
        atol=1e-10,
    )


def test_orthogonal_variable_rotation_equivariance():
    X = _make_data(n=80, p=5, seed=13)
    rng = np.random.RandomState(21)
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    rotated = shrink_cov(X @ Q)
    expected = Q.T @ shrink_cov(X) @ Q
    np.testing.assert_allclose(rotated, expected, rtol=1e-6, atol=1e-8)


def test_scale_equivariance():
    X = _make_data(n=70, p=4, seed=14)
    c = 3.0
    np.testing.assert_allclose(
        shrink_cov(c * X),
        (c ** 2) * shrink_cov(X),
        rtol=1e-6,
        atol=1e-8,
    )


def test_singular_sample_covariance_raises_value_error():
    X = _make_data(n=50, p=5, seed=15)
    X[:, 0] = 0.0  # zero-variance column makes the covariance singular
    with pytest.raises(ValueError):
        shrink_cov(X)


def test_high_dimensional_case_p_greater_than_n():
    X = _make_data(n=20, p=40, seed=16)
    result = shrink_cov(X)
    assert result.shape == (40, 40)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, result.T, rtol=1e-8, atol=1e-10)


def test_shrinkage_reduces_eigenvalue_spread():
    # Nonlinear shrinkage pulls extreme sample eigenvalues toward the center.
    X = _make_data(n=30, p=20, seed=17)
    centered = X - X.mean(axis=0)
    sample_cov = centered.T @ centered / (X.shape[0] - 1)
    sample_eigs = np.linalg.eigvalsh(sample_cov)
    shrunk_eigs = np.linalg.eigvalsh(shrink_cov(X))
    assert np.ptp(shrunk_eigs) < np.ptp(sample_eigs)


def test_large_sample_recovers_identity_roughly():
    # With many observations from an identity-covariance Gaussian,
    # the shrunk estimate should be close to the identity.
    X = _make_data(n=4000, p=4, seed=18)
    result = shrink_cov(X)
    np.testing.assert_allclose(result, np.eye(4), atol=0.15)
