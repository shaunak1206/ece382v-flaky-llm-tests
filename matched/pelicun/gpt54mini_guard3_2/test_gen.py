import math
from types import SimpleNamespace

import numpy as np
import pytest

import pelicun.uq as uq


def _fake_minimize_return_x(x):
    return SimpleNamespace(x=np.asarray(x, dtype=float), fun=0.0, message="ok")


def test_tmvn_rvs_unbounded_scalar_returns_expected_shape(monkeypatch):
    np.random.seed(123)
    monkeypatch.setattr(
        uq.multivariate_normal,
        "rvs",
        lambda mean=None, cov=None, size=None: np.full((size, 1), 7.0),
    )

    samples = uq.tmvn_rvs(mu=0.0, COV=1.0, size=3)

    assert samples.shape == (3, 1)
    assert np.all(samples == 7.0)


def test_tmvn_rvs_unbounded_multivariate_delegates_to_mvn_sampler(monkeypatch):
    np.random.seed(123)
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])

    def fake_rvs(mean=None, cov=None, size=None):
        assert size == 2
        return expected

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=np.array([0.0, 1.0]), COV=np.eye(2), size=2)

    assert np.array_equal(out, expected)


def test_tmvn_rvs_scalar_inputs_are_promoted_to_one_dim(monkeypatch):
    np.random.seed(123)
    captured = {}

    def fake_rvs(mean=None, cov=None, size=None):
        captured["mean"] = np.asarray(mean)
        captured["cov"] = np.asarray(cov)
        return np.array([[5.0], [6.0]])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=1.5, COV=2.5, size=2)

    assert out.shape == (2, 1)
    assert captured["mean"].shape == (1,)
    assert captured["cov"].shape == (1,)
    assert np.allclose(captured["mean"], [1.5])
    assert np.allclose(captured["cov"], [2.5])


def test_tmvn_rvs_rejection_sampling_collects_until_enough_samples(monkeypatch):
    np.random.seed(123)
    call_sizes = []

    def fake_density(mu, COV, lower=None, upper=None):
        return 0.5, 0.0

    def fake_rvs(mean=None, cov=None, size=None):
        call_sizes.append(size)
        return np.array(
            [
                [-1.0],
                [0.0],
                [0.5],
                [2.0],
                [3.0],
                [4.0],
            ]
        )[:size]

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-0.1, upper=1.0, size=2)

    assert out.shape == (2, 1)
    assert np.all((out > -0.1) & (out < 1.0))
    assert call_sizes[0] >= 2


def test_tmvn_rvs_rejection_sampling_multidimensional_filters_rows(monkeypatch):
    np.random.seed(123)

    def fake_density(mu, COV, lower=None, upper=None):
        return 0.5, 0.0

    def fake_rvs(mean=None, cov=None, size=None):
        return np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [2.0, 2.0],
                [0.5, 0.6],
            ]
        )[:size]

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2), lower=[0.0, 0.0], upper=[1.0, 1.0], size=2)

    assert out.shape == (2, 2)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_tmvn_rvs_warns_for_high_rejection_rate(monkeypatch):
    np.random.seed(123)

    def fake_density(mu, COV, lower=None, upper=None):
        return 5e-4, 0.0

    def fake_rvs(mean=None, cov=None, size=None):
        return np.array([[0.1], [0.2], [0.3], [0.4]])[:size]

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    with pytest.warns(UserWarning, match="rejection rate"):
        out = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=1)

    assert out.shape == (1, 1)


def test_tmvn_rvs_raises_when_alpha_is_too_small(monkeypatch):
    np.random.seed(123)

    def fake_density(mu, COV, lower=None, upper=None):
        return 1e-6, 2e-8  # alpha <= 100 * eps_alpha

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    with pytest.raises(ValueError, match="density of the joint probability"):
        uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=1)


def test_tmvn_MLE_univariate_no_truncation_uses_minimize_result(monkeypatch):
    np.random.seed(123)
    samples = np.array([1.0, 2.0, 3.0, 4.0])

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        # Choose standardized parameters that map back to the sample mean/std.
        x = np.array([0.0, 1.0 / math.sqrt(4.0 / 3.0)])
        return _fake_minimize_return_x(x)

    monkeypatch.setattr(uq, "minimize", fake_minimize)

    mu, cov = uq.tmvn_MLE(samples)

    assert mu == pytest.approx(np.mean(samples), rel=1e-12, abs=1e-12)
    assert cov == pytest.approx(np.var(samples, ddof=1), rel=1e-12, abs=1e-12)


def test_tmvn_MLE_multivariate_no_truncation_shapes(monkeypatch):
    np.random.seed(123)
    samples = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        # Standardized values that map back to the sample mean and sample std.
        x = np.array([0.0, 0.0, 1.0 / math.sqrt(3.0 / 2.0), 1.0 / math.sqrt(3.0 / 2.0)])
        return _fake_minimize_return_x(x)

    monkeypatch.setattr(uq, "minimize", fake_minimize)

    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.allclose(mu, np.mean(samples, axis=1))
    assert np.allclose(np.diag(cov), np.var(samples, axis=1, ddof=1))


def test_tmvn_MLE_returns_zero_covariance_off_diagonal_when_samples_are_uncorrelated(monkeypatch):
    np.random.seed(123)
    samples = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 9.0, 8.0, 7.0]])

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        # Keep the standardized parameters at the origin and scale=1.
        x = np.array([0.0, 0.0, 1.0 / math.sqrt(4.0 / 3.0), 1.0 / math.sqrt(4.0 / 3.0)])
        return _fake_minimize_return_x(x)

    monkeypatch.setattr(uq, "minimize", fake_minimize)

    mu, cov = uq.tmvn_MLE(samples)

    assert cov.shape == (2, 2)
    assert cov[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert cov[1, 0] == pytest.approx(0.0, abs=1e-12)


def test_tmvn_MLE_with_truncation_passes_bounds_to_orthotope_density(monkeypatch):
    np.random.seed(123)
    samples = np.array([0.2, 0.3, 0.4, 0.5])
    seen = {"calls": []}

    def fake_density(mu, COV, lower=None, upper=None):
        seen["calls"].append((np.asarray(mu).copy(), np.asarray(COV).copy(), np.asarray(lower).copy(), np.asarray(upper).copy()))
        return 1.0, 0.0

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        return _fake_minimize_return_x(np.array([0.0, 1.0]))

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq, "minimize", fake_minimize)

    mu, cov = uq.tmvn_MLE(samples, tr_lower=np.array([0.0]), tr_upper=np.array([1.0]))

    assert len(seen["calls"]) >= 1
    assert mu == pytest.approx(np.mean(samples), rel=1e-12, abs=1e-12)
    assert cov > 0.0


def test_tmvn_MLE_with_alpha_limit_can_force_regularization(monkeypatch):
    np.random.seed(123)
    samples = np.array([1.0, 1.1, 1.2, 1.3])

    def fake_density(mu, COV, lower=None, upper=None):
        return 0.01, 0.0

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        return _fake_minimize_return_x(np.array([0.0, 1.0]))

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq, "minimize", fake_minimize)

    mu, cov = uq.tmvn_MLE(samples, tr_lower=np.array([0.0]), tr_upper=np.array([10.0]), alpha_lim=0.1)

    assert mu == pytest.approx(np.mean(samples), rel=1e-12, abs=1e-12)
    assert cov > 0.0


def test_tmvn_MLE_with_censoring_uses_detection_limits(monkeypatch):
    np.random.seed(123)
    samples = np.array([1.0, 1.2, 1.4, 1.6])

    calls = []

    def fake_density(mu, COV, lower=None, upper=None):
        calls.append((np.asarray(lower).copy(), np.asarray(upper).copy()))
        return 1.0, 0.0

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        return _fake_minimize_return_x(np.array([0.0, 1.0]))

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq, "minimize", fake_minimize)

    mu, cov = uq.tmvn_MLE(
        samples,
        censored_count=1,
        det_lower=np.array([1.0]),
        det_upper=np.array([1.5]),
    )

    assert len(calls) >= 1
    assert mu == pytest.approx(np.mean(samples), rel=1e-12, abs=1e-12)
    assert cov > 0.0
