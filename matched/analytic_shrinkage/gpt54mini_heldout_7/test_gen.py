import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def make_spd_data(n, p, seed=0, mean_shift=True):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    if mean_shift:
        x = x + rng.normal(loc=2.0, scale=0.5, size=(1, p))
    return x


def test_shrink_cov_returns_square_matrix_same_shape():
    data = make_spd_data(20, 5, seed=1)
    out = shrink_cov(data)
    assert out.shape == (5, 5)


def test_shrink_cov_output_is_symmetric():
    data = make_spd_data(24, 6, seed=2)
    out = shrink_cov(data)
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_deterministic_for_same_input():
    data = make_spd_data(18, 4, seed=3)
    out1 = shrink_cov(data)
    out2 = shrink_cov(data.copy())
    assert np.allclose(out1, out2, atol=1e-12, rtol=1e-12)


def test_shrink_cov_preserves_identity_for_identity_like_data():
    data = np.eye(12)
    out = shrink_cov(data)
    assert out.shape == (12, 12)
    assert np.allclose(out, out.T, atol=1e-10)
    assert np.all(np.isfinite(out))


def test_shrink_cov_default_demeaning_matches_explicit_k1_on_raw_data():
    data = make_spd_data(20, 5, seed=4, mean_shift=True)
    out_default = shrink_cov(data)
    out_k1 = shrink_cov(data, k=1)
    assert np.allclose(out_default, out_k1, atol=1e-12, rtol=1e-12)


def test_shrink_cov_k0_differs_from_default_when_data_has_nonzero_mean():
    data = make_spd_data(22, 5, seed=5, mean_shift=True)
    out_default = shrink_cov(data)
    out_k0 = shrink_cov(data, k=0)
    assert out_default.shape == out_k0.shape == (5, 5)
    assert not np.allclose(out_default, out_k0)


def test_shrink_cov_k0_matches_manual_centering_removed_effective_sample_size():
    data = make_spd_data(24, 6, seed=6, mean_shift=False)
    centered = data - np.mean(data, axis=0)
    out_default = shrink_cov(data)
    out_manual = shrink_cov(centered, k=1)
    assert np.allclose(out_default, out_manual, atol=1e-12, rtol=1e-12)


def test_shrink_cov_handles_high_dimensional_case_p_greater_than_n():
    data = make_spd_data(14, 20, seed=7)
    out = shrink_cov(data)
    assert out.shape == (20, 20)
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)
    assert np.all(np.isfinite(out))


def test_shrink_cov_high_dimensional_case_with_explicit_k():
    data = make_spd_data(16, 18, seed=8)
    out = shrink_cov(data, k=2)  # effective n = 14
    assert out.shape == (18, 18)
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)


def test_shrink_cov_raises_assertion_for_non_2d_input():
    data = np.arange(20)
    with pytest.raises(AssertionError, match="input must be a 2d array"):
        shrink_cov(data)


def test_shrink_cov_raises_assertion_when_effective_sample_size_too_small():
    data = make_spd_data(11, 3, seed=9)
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        shrink_cov(data)


def test_shrink_cov_raises_value_error_for_singular_matrix_due_to_duplicate_column():
    rng = np.random.default_rng(10)
    base = rng.standard_normal((20, 2))
    data = np.column_stack([base, base[:, [0]]])  # duplicate one column -> singular covariance
    with pytest.raises(ValueError, match="Matrix is singular"):
        shrink_cov(data)


def test_shrink_cov_handles_more_columns_than_rows_and_returns_finite_values():
    data = make_spd_data(15, 25, seed=11)
    out = shrink_cov(data, k=1)
    assert out.shape == (25, 25)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out.T, atol=1e-10, rtol=1e-10)
