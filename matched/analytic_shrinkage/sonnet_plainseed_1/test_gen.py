import numpy as np
import pytest

import nonlinshrink
from nonlinshrink.nonlinshrink import shrink_cov, prial, fsopt, loss_mv, loss_fr


def make_data(n, p, seed=0, cov_scale=None):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, p)
    if cov_scale is not None:
        x = x * np.asarray(cov_scale)
    return x


def make_spd(p, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(p, p)
    return np.dot(a, a.T) + p * np.eye(p)


def test_package_exports_functions():
    assert nonlinshrink.shrink_cov is shrink_cov
    assert nonlinshrink.prial is prial
    assert nonlinshrink.fsopt is fsopt
    assert nonlinshrink.loss_mv is loss_mv
    assert nonlinshrink.loss_fr is loss_fr


def test_shrink_cov_output_shape_low_dim():
    data = make_data(50, 6, seed=1)
    out = shrink_cov(data)
    assert out.shape == (6, 6)
    assert np.all(np.isfinite(out))


def test_shrink_cov_output_is_symmetric():
    data = make_data(60, 8, seed=2)
    out = shrink_cov(data)
    np.testing.assert_allclose(out, out.T, rtol=1e-8, atol=1e-10)


def test_shrink_cov_output_is_positive_definite():
    data = make_data(60, 8, seed=3, cov_scale=np.arange(1, 9))
    out = shrink_cov(data)
    eigvals = np.linalg.eigvalsh((out + out.T) / 2.0)
    assert np.all(eigvals > 0)


def test_shrink_cov_default_equals_prior_demeaned_with_k1():
    data = make_data(40, 5, seed=4) + 3.0
    demeaned = data - np.mean(data, axis=0)
    a = shrink_cov(data)
    b = shrink_cov(demeaned, k=1)
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


def test_shrink_cov_does_not_mutate_input():
    data = make_data(30, 4, seed=5) + 1.5
    original = data.copy()
    shrink_cov(data)
    shrink_cov(data, k=0)
    np.testing.assert_array_equal(data, original)


def test_shrink_cov_is_deterministic():
    data = make_data(45, 7, seed=6)
    a = shrink_cov(data)
    b = shrink_cov(data.copy())
    np.testing.assert_array_equal(a, b)


def test_shrink_cov_scale_equivariance():
    data = make_data(50, 5, seed=7)
    c = 2.0
    a = shrink_cov(data)
    b = shrink_cov(c * data)
    np.testing.assert_allclose(b, (c ** 2) * a, rtol=1e-6, atol=1e-8)


def test_shrink_cov_close_to_sample_cov_when_n_much_larger_than_p():
    data = make_data(2000, 3, seed=8, cov_scale=np.array([1.0, 2.0, 3.0]))
    sample = np.cov(data.T)
    out = shrink_cov(data)
    rel = np.linalg.norm(out - sample) / np.linalg.norm(sample)
    assert rel < 0.1


def test_shrink_cov_reduces_condition_number_for_identity_truth():
    data = make_data(40, 10, seed=9)
    sample = np.cov(data.T)
    out = shrink_cov(data)
    assert np.linalg.cond(out) < np.linalg.cond(sample)


def test_shrink_cov_high_dimensional_p_greater_than_n():
    n, p = 15, 20
    data = make_data(n, p, seed=10)
    out = shrink_cov(data, k=0)
    assert out.shape == (p, p)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, out.T, rtol=1e-8, atol=1e-10)
    eigvals = np.linalg.eigvalsh((out + out.T) / 2.0)
    assert np.all(eigvals > 0)


def test_shrink_cov_minimum_sample_size_boundaries_accepted():
    # k=0: effective n = 12 is allowed
    out0 = shrink_cov(make_data(12, 3, seed=11), k=0)
    assert out0.shape == (3, 3)
    # k=None: effective n = n - 1 = 12 is allowed for n = 13
    out1 = shrink_cov(make_data(13, 3, seed=12))
    assert out1.shape == (3, 3)


def test_shrink_cov_sample_size_too_small_raises():
    with pytest.raises(AssertionError):
        shrink_cov(make_data(12, 3, seed=13))  # effective n = 11
    with pytest.raises(AssertionError):
        shrink_cov(make_data(11, 3, seed=14), k=0)


def test_shrink_cov_large_k_reduces_effective_sample_size():
    data = make_data(15, 3, seed=15)
    out = shrink_cov(data, k=3)  # effective n = 12
    assert out.shape == (3, 3)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=4)  # effective n = 11


def test_shrink_cov_requires_2d_input():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(30, dtype=float))
    with pytest.raises(AssertionError):
        shrink_cov(np.zeros((20, 3, 2)))


def test_shrink_cov_singular_matrix_raises_value_error():
    data = make_data(30, 4, seed=16)
    data[:, 1] = data[:, 0]
    with pytest.raises(ValueError, match="singular"):
        shrink_cov(data)


def test_loss_fr_known_values():
    sigma = np.eye(3)
    assert loss_fr(sigma, sigma) == pytest.approx(0.0)
    # delta = I, trace(delta ** 2) / p = 3 / 3
    assert loss_fr(2 * sigma, sigma) == pytest.approx(1.0)


def test_loss_mv_known_values():
    sigma = np.eye(3)
    assert loss_mv(sigma, sigma) == pytest.approx(0.0, abs=1e-12)
    # scale invariant
    assert loss_mv(2 * sigma, sigma) == pytest.approx(0.0, abs=1e-12)
    assert loss_mv(np.diag([1.0, 2.0, 3.0]), sigma) == pytest.approx(26.0 / 121.0)


def test_fsopt_diagonal_returns_true_diagonal():
    sample = np.diag([1.0, 2.0, 3.0])
    sigma = np.diag([4.0, 5.0, 6.0])
    out = fsopt(sample, sigma)
    np.testing.assert_allclose(out, sigma, atol=1e-10)


def test_prial_oracle_is_one_and_sample_is_zero():
    sigma = make_spd(5, seed=17)
    rng = np.random.RandomState(18)
    p = rng.randn(5, 5) * 0.3
    sample = sigma + (p + p.T) / 2.0
    oracle = fsopt(sample, sigma)
    assert prial(sample, oracle, sigma) == pytest.approx(1.0)
    assert prial(sample, sample, sigma) == pytest.approx(0.0, abs=1e-9)
