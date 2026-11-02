import numpy as np
import pytest

from nonlinshrink import shrink_cov


def make_data(n, p, seed=0, scale=1.0):
    rng = np.random.RandomState(seed)
    return scale * rng.randn(n, p)


def test_output_shape_p_less_than_n():
    data = make_data(100, 10)
    out = shrink_cov(data)
    assert out.shape == (10, 10)


def test_output_shape_p_greater_than_n():
    data = make_data(15, 30)
    out = shrink_cov(data)
    assert out.shape == (30, 30)


def test_output_is_finite_low_dim():
    out = shrink_cov(make_data(80, 8, seed=1))
    assert np.all(np.isfinite(out))


def test_output_is_finite_high_dim():
    out = shrink_cov(make_data(20, 40, seed=2))
    assert np.all(np.isfinite(out))


def test_output_is_symmetric():
    out = shrink_cov(make_data(60, 12, seed=3))
    np.testing.assert_allclose(out, out.T, atol=1e-10)


def test_output_is_symmetric_high_dim():
    out = shrink_cov(make_data(20, 35, seed=4))
    np.testing.assert_allclose(out, out.T, atol=1e-10)


def test_output_positive_definite_low_dim():
    out = shrink_cov(make_data(100, 15, seed=5))
    eigvals = np.linalg.eigvalsh((out + out.T) / 2)
    assert np.all(eigvals > 0)


def test_output_positive_definite_high_dim():
    out = shrink_cov(make_data(20, 40, seed=6))
    eigvals = np.linalg.eigvalsh((out + out.T) / 2)
    assert np.all(eigvals > 0)


def test_default_demeans_data():
    data = make_data(80, 6, seed=7)
    shifted = data + np.array([5.0, -3.0, 10.0, 0.5, -7.0, 2.0])
    np.testing.assert_allclose(shrink_cov(data), shrink_cov(shifted), rtol=1e-6, atol=1e-8)


def test_k_one_on_demeaned_data_matches_default():
    data = make_data(80, 6, seed=8)
    demeaned = data - data.mean(axis=0)
    np.testing.assert_allclose(
        shrink_cov(demeaned, k=1), shrink_cov(data), rtol=1e-6, atol=1e-8
    )


def test_k_zero_does_not_demean():
    data = make_data(80, 6, seed=9) + 4.0
    out_k0 = shrink_cov(data, k=0)
    out_default = shrink_cov(data)
    assert not np.allclose(out_k0, out_default)


def test_k_changes_effective_sample_size():
    data = make_data(80, 6, seed=10)
    demeaned = data - data.mean(axis=0)
    out_k1 = shrink_cov(demeaned, k=1)
    out_k3 = shrink_cov(demeaned, k=3)
    assert out_k1.shape == out_k3.shape
    assert not np.allclose(out_k1, out_k3)


def test_scale_equivariance():
    data = make_data(70, 9, seed=11)
    c = 3.0
    base = shrink_cov(data)
    scaled = shrink_cov(c * data)
    np.testing.assert_allclose(scaled, (c ** 2) * base, rtol=1e-5)


def test_input_not_modified():
    data = make_data(50, 7, seed=12)
    original = data.copy()
    shrink_cov(data)
    np.testing.assert_array_equal(data, original)


def test_deterministic():
    data = make_data(50, 7, seed=13)
    np.testing.assert_array_equal(shrink_cov(data), shrink_cov(data))


def test_one_dimensional_input_raises():
    with pytest.raises(AssertionError):
        shrink_cov(np.arange(50, dtype=float))


def test_three_dimensional_input_raises():
    with pytest.raises(AssertionError):
        shrink_cov(np.zeros((20, 3, 2)))


def test_too_few_samples_raises():
    with pytest.raises(AssertionError):
        shrink_cov(make_data(11, 3, seed=14), k=0)


def test_effective_sample_size_after_k_too_small_raises():
    # n=13 with default demeaning gives effective n=12 (ok); k=2 gives 11 (fail)
    data = make_data(13, 3, seed=15)
    out = shrink_cov(data)
    assert out.shape == (3, 3)
    with pytest.raises(AssertionError):
        shrink_cov(data, k=2)


def test_minimum_sample_size_accepted():
    out = shrink_cov(make_data(12, 2, seed=16), k=0)
    assert out.shape == (2, 2)
    assert np.all(np.isfinite(out))


def test_singular_matrix_raises_value_error():
    data = make_data(50, 4, seed=17)
    data[:, 3] = data[:, 0]  # duplicated column -> singular covariance
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_shrinkage_improves_conditioning():
    data = make_data(60, 40, seed=18)
    demeaned = data - data.mean(axis=0)
    sample_cov = demeaned.T.dot(demeaned) / 59
    shrunk = shrink_cov(data)
    cond_sample = np.linalg.cond(sample_cov)
    cond_shrunk = np.linalg.cond(shrunk)
    assert cond_shrunk < cond_sample


def test_close_to_identity_with_many_samples():
    data = make_data(5000, 4, seed=19)
    out = shrink_cov(data)
    np.testing.assert_allclose(out, np.eye(4), atol=0.15)


def test_shares_eigenvectors_with_sample_covariance():
    data = make_data(100, 6, seed=20)
    demeaned = data - data.mean(axis=0)
    sample_cov = demeaned.T.dot(demeaned) / 99
    _, u = np.linalg.eigh(sample_cov)
    rotated = u.T.dot(shrink_cov(data)).dot(u)
    off_diag = rotated - np.diag(np.diag(rotated))
    assert np.max(np.abs(off_diag)) < 1e-8
