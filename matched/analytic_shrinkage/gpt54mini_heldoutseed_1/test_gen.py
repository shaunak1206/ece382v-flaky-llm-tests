import numpy as np
import pytest

import nonlinshrink.nonlinshrink as ns


def _orthonormal_matrix(p):
    return np.eye(p)


def test_shrink_cov_rejects_non_2d_input():
    x = np.arange(20.0)
    with pytest.raises(AssertionError, match="2d array"):
        ns.shrink_cov(x)


def test_shrink_cov_requires_minimum_effective_sample_size_with_k_none():
    data = np.ones((12, 3))
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        ns.shrink_cov(data)


def test_shrink_cov_requires_minimum_effective_sample_size_with_k_adjustment():
    data = np.ones((12, 3))
    with pytest.raises(AssertionError, match="sample size n must be >= 12"):
        ns.shrink_cov(data, k=1)


def test_shrink_cov_returns_symmetric_matrix_same_shape_p_le_n():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(20, 5))

    out = ns.shrink_cov(data, k=0)

    assert out.shape == (5, 5)
    assert np.allclose(out, out.T, atol=1e-12)
    assert np.isfinite(out).all()


def test_shrink_cov_returns_symmetric_matrix_same_shape_p_gt_n():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(15, 20))

    out = ns.shrink_cov(data, k=1)

    assert out.shape == (20, 20)
    assert np.allclose(out, out.T, atol=1e-12)
    assert np.isfinite(out).all()


def test_shrink_cov_k_none_and_k_zero_differ_on_noncentered_data():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(18, 4)) + 10.0

    out_default = ns.shrink_cov(data)
    out_k0 = ns.shrink_cov(data, k=0)

    assert not np.allclose(out_default, out_k0)


def test_shrink_cov_is_deterministic_for_same_inputs():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(22, 6))

    out1 = ns.shrink_cov(data, k=0)
    out2 = ns.shrink_cov(data, k=0)

    assert np.allclose(out1, out2)


def test_shrink_cov_handles_precentered_data_with_k_one():
    rng = np.random.default_rng(4)
    data = rng.normal(size=(16, 5))
    data -= data.mean(axis=0)

    out = ns.shrink_cov(data, k=1)

    assert out.shape == (5, 5)
    assert np.allclose(out, out.T, atol=1e-12)
    assert np.isfinite(out).all()


def test_shrink_cov_raises_for_singular_spectrum(monkeypatch):
    data = np.arange(60.0).reshape(15, 4)

    def fake_eigh(_):
        lam = np.array([1.0, 1.0, 1.0, 1e-14])
        u = _orthonormal_matrix(4)
        return lam, u

    monkeypatch.setattr(ns.np.linalg, "eigh", fake_eigh)

    with pytest.raises(ValueError, match="Matrix is singular"):
        ns.shrink_cov(data, k=0)


def test_shrink_cov_branch_p_le_n_uses_mocked_eigh(monkeypatch):
    data = np.arange(100.0).reshape(20, 5)

    def fake_eigh(_):
        lam = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u = _orthonormal_matrix(5)
        return lam, u

    monkeypatch.setattr(ns.np.linalg, "eigh", fake_eigh)

    out = ns.shrink_cov(data, k=0)

    assert out.shape == (5, 5)
    assert np.allclose(out, out.T, atol=1e-12)
    assert np.isfinite(out).all()


def test_shrink_cov_branch_p_gt_n_uses_mocked_eigh(monkeypatch):
    data = np.arange(300.0).reshape(15, 20)

    def fake_eigh(_):
        lam = np.linspace(1.0, 20.0, 20)
        u = _orthonormal_matrix(20)
        return lam, u

    monkeypatch.setattr(ns.np.linalg, "eigh", fake_eigh)

    out = ns.shrink_cov(data, k=1)

    assert out.shape == (20, 20)
    assert np.allclose(out, out.T, atol=1e-12)
    assert np.isfinite(out).all()
    assert np.all(np.diag(out) > 0)


def test_shrink_cov_does_not_modify_input_array():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(20, 5))
    original = data.copy()

    _ = ns.shrink_cov(data, k=0)

    assert np.array_equal(data, original)


def test_shrink_cov_with_centered_data_and_k_none_matches_manual_centering_shape():
    rng = np.random.default_rng(6)
    data = rng.normal(size=(20, 5))
    centered = data - data.mean(axis=0)

    out = ns.shrink_cov(data)
    out_centered = ns.shrink_cov(centered, k=1)

    assert out.shape == out_centered.shape == (5, 5)
    assert np.isfinite(out).all()
    assert np.isfinite(out_centered).all()
