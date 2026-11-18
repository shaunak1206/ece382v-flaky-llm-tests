import warnings

import numpy as np
import pytest

import pelicun.uq as uq


def test_tmvn_rvs_no_bounds_scalar_returns_requested_size(monkeypatch):
    def fake_rvs(mean, cov, size=None):
        return np.full((size,), 3.14)

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    samples = uq.tmvn_rvs(1.0, 4.0, size=5)

    assert samples.shape == (5,)
    assert np.all(samples == 3.14)


def test_tmvn_rvs_no_bounds_multivariate_shape(monkeypatch):
    def fake_rvs(mean, cov, size=None):
        return np.arange(size * len(mean), dtype=float).reshape(size, len(mean))

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    mu = np.array([0.0, 1.0])
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])
    samples = uq.tmvn_rvs(mu, cov, size=4)

    assert samples.shape == (4, 2)
    assert np.array_equal(samples[0], np.array([0.0, 1.0]))


def test_tmvn_rvs_rejects_when_alpha_too_small(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (0.001, 0.00002))

    with pytest.raises(ValueError, match="density of the joint probability distribution"):
        uq.tmvn_rvs(np.array([0.0]), np.array([1.0]), lower=[-1.0], upper=[1.0], size=1)


def test_tmvn_rvs_warns_for_low_alpha(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (5e-4, 1e-7))

    calls = []

    def fake_rvs(mean, cov, size=None):
        calls.append(size)
        return np.array([[0.0], [0.1], [0.2], [0.3]])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        samples = uq.tmvn_rvs(np.array([0.0]), np.array([1.0]), lower=[-1.0], upper=[1.0], size=2)

    assert samples.shape == (2, 1)
    assert calls
    assert any("rejection rate" in str(w.message).lower() for w in rec)


def test_tmvn_rvs_rejection_sampling_filters_out_of_bounds(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (0.5, 1e-6))

    raw_batches = [
        np.array([[-2.0], [0.0], [2.0], [0.5]]),
        np.array([[0.25], [1.5], [0.75], [-0.5]]),
    ]
    call_count = {"n": 0}

    def fake_rvs(mean, cov, size=None):
        arr = raw_batches[min(call_count["n"], len(raw_batches) - 1)]
        call_count["n"] += 1
        return arr[:size]

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    samples = uq.tmvn_rvs(np.array([0.0]), np.array([1.0]), lower=[0.0], upper=[1.0], size=3)

    assert samples.shape == (3, 1)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_tmvn_rvs_passes_through_two_dimensional_bounds(monkeypatch):
    def fake_density(mu, cov, lower, upper):
        assert np.allclose(lower, np.array([-1.0, -2.0]))
        assert np.allclose(upper, np.array([1.0, 2.0]))
        return 1.0, 1e-12

    def fake_rvs(mean, cov, size=None):
        return np.array([[0.0, 0.0]])

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    samples = uq.tmvn_rvs(np.array([0.0, 0.0]), np.eye(2), lower=[-1.0, -2.0], upper=[1.0, 2.0], size=1)

    assert samples.shape == (1, 2)


def test_mvn_orthotope_density_1d_calls_mvndst_with_expected_shapes(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["lower"] = np.array(lower)
        captured["upper"] = np.array(upper)
        captured["infin"] = np.array(infin)
        captured["correl"] = correl
        return 0.01, 0.5, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    alpha, eps = uq.mvn_orthotope_density(0.0, 4.0, lower=-1.0, upper=2.0)

    assert alpha == 0.5
    assert eps == 0.01
    assert captured["lower"].shape == (1,)
    assert captured["upper"].shape == (1,)
    assert captured["infin"].shape == (1,)
    assert captured["correl"] == 0


def test_mvn_orthotope_density_multivariate_uses_lower_triangle_correlations(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["lower"] = np.array(lower)
        captured["upper"] = np.array(upper)
        captured["infin"] = np.array(infin)
        captured["correl"] = np.array(correl)
        return 0.02, 0.4, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    mu = np.array([0.0, 1.0])
    cov = np.array([[1.0, 0.3], [0.3, 4.0]])
    alpha, eps = uq.mvn_orthotope_density(mu, cov, lower=[-1.0, 0.0], upper=[1.0, 2.0])

    assert alpha == 0.4
    assert eps == 0.02
    assert captured["lower"].shape == (2,)
    assert captured["upper"].shape == (2,)
    assert captured["infin"].shape == (2,)
    assert captured["correl"].shape == (1,)
    assert np.isclose(captured["correl"][0], 0.3 / (1.0 * 2.0))


def test_tmvn_MLE_univariate_returns_sample_mean_and_variance(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 1e-12))
    monkeypatch.setattr(
        uq,
        "minimize",
        lambda *args, **kwargs: type("O", (), {"x": np.array([0.0, 1.0]), "fun": 0.0, "message": "ok"})(),
    )

    samples = np.array([1.0, 2.0, 3.0, 4.0])
    mu, cov = uq.tmvn_MLE(samples)

    assert np.isclose(mu, np.mean(samples))
    assert np.isclose(cov, np.var(samples, ddof=1))


def test_tmvn_MLE_multivariate_returns_reasonable_shapes(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 1e-12))

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        return type("O", (), {"x": np.array(x0), "fun": 0.0, "message": "ok"})()

    monkeypatch.setattr(uq, "minimize", fake_minimize)

    samples = np.array([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]])
    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))


def test_tmvn_MLE_handles_zero_variance_dimension(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 1e-12))

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        return type("O", (), {"x": np.array(x0), "fun": 0.0, "message": "ok"})()

    monkeypatch.setattr(uq, "minimize", fake_minimize)

    samples = np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]])
    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert cov[0, 0] > 0.0
    assert cov[1, 1] > 0.0


def test_tmvn_MLE_enforces_truncation_alpha_limit(monkeypatch):
    calls = {"n": 0}

    def fake_density(*args, **kwargs):
        calls["n"] += 1
        return 1e-8, 1e-10

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(
        uq,
        "minimize",
        lambda *args, **kwargs: type("O", (), {"x": np.array([0.0, 1.0]), "fun": 0.0, "message": "ok"})(),
    )

    samples = np.array([1.0, 2.0, 3.0])

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mu, cov = uq.tmvn_MLE(samples, tr_lower=np.array([0.0]), tr_upper=np.array([1.0]), alpha_lim=1e-4)

    assert calls["n"] >= 1
    assert mu.shape == ()
    assert cov.shape == ()
    assert any("prescribed minimum limit" in str(w.message).lower() for w in rec)


def test_tmvn_MLE_with_censoring_passes_detection_limits(monkeypatch):
    captured = {}

    def fake_density(mu, cov, lower, upper):
        key = "det" if lower is not None and upper is not None and np.all(np.asarray(lower) < np.asarray(upper)) else "tr"
        captured.setdefault(key, []).append((np.array(lower), np.array(upper)))
        return 1.0, 1e-12

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        return type("O", (), {"x": np.array(x0), "fun": 0.0, "message": "ok"})()

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq, "minimize", fake_minimize)

    samples = np.array([1.0, 1.2, 0.8, 1.1])
    mu, cov = uq.tmvn_MLE(
        samples,
        censored_count=1,
        det_lower=np.array([0.5]),
        det_upper=np.array([1.5]),
        tr_lower=np.array([0.0]),
        tr_upper=np.array([2.0]),
    )

    assert mu.shape == ()
    assert cov.shape == ()
    assert captured
    assert any(np.allclose(l, np.array([0.5])) and np.allclose(u, np.array([1.5])) for l, u in captured.get("det", []))


def test_tmvn_MLE_returns_finite_values_for_simple_data(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 1e-12))

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        return type("O", (), {"x": np.array(x0), "fun": 0.0, "message": "ok"})()

    monkeypatch.setattr(uq, "minimize", fake_minimize)

    samples = np.array([[0.5, 1.5, 2.5], [2.0, 2.5, 3.0]])
    mu, cov = uq.tmvn_MLE(samples, alpha_lim=1e-6)

    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
    assert cov.shape == (2, 2)
