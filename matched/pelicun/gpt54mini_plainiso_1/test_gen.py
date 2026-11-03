import warnings

import numpy as np
import pytest

import pelicun.uq as uq


def test_tmvn_rvs_univariate_no_bounds_calls_mvn_and_returns_array(monkeypatch):
    called = {}

    def fake_rvs(mean, cov, size):
        called["mean"] = np.asarray(mean)
        called["cov"] = np.asarray(cov)
        called["size"] = size
        return np.full(size, 7.0)

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=1.5, COV=2.5, size=4)

    assert np.allclose(out, np.full(4, 7.0))
    assert called["mean"].shape == (1,)
    assert called["cov"].shape == (1,)
    assert called["size"] == 4


def test_tmvn_rvs_multivariate_no_bounds_returns_expected_shape(monkeypatch):
    def fake_rvs(mean, cov, size):
        return np.arange(size * len(mean), dtype=float).reshape(size, len(mean))

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=[0.0, 1.0], COV=np.eye(2), size=3)

    assert out.shape == (3, 2)
    assert np.array_equal(out[0], np.array([0.0, 1.0]))


def test_tmvn_rvs_truncated_accepts_samples_within_bounds(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    def fake_rvs(mean, cov, size):
        # First two are inside the box, last one is outside.
        if size == 2:
            return np.array([[0.1, 0.2], [0.3, 0.4]])
        return np.array([[0.1, 0.2], [0.3, 0.4], [10.0, 10.0]])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2), lower=[0.0, 0.0], upper=[1.0, 1.0], size=2)

    assert out.shape == (2, 2)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_tmvn_rvs_truncated_respects_scalar_input_and_returns_1d_samples(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(
        uq.multivariate_normal,
        "rvs",
        lambda mean, cov, size: np.array([-0.25, 0.1, 0.2, 0.3]),
    )

    out = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-0.5, upper=0.5, size=2)

    assert out.shape == (2,)
    assert np.all(out >= -0.5)
    assert np.all(out <= 0.5)


def test_tmvn_rvs_raises_value_error_when_alpha_too_small(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-5, 2e-4))

    with pytest.raises(ValueError, match="too small"):
        uq.tmvn_rvs(mu=[0.0], COV=np.eye(1), lower=[-1.0], upper=[1.0], size=1)


def test_tmvn_rvs_warns_for_high_rejection_rate(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (5e-4, 1e-8))

    def fake_rvs(mean, cov, size):
        # Always return samples inside bounds.
        return np.zeros((size, 2))

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    with pytest.warns(UserWarning, match="rejection rate"):
        out = uq.tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2), lower=[-1.0, -1.0], upper=[1.0, 1.0], size=3)

    assert out.shape == (3, 2)


def test_mvn_orthotope_density_passes_through_mvndst_for_univariate(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["lower"] = np.asarray(lower)
        captured["upper"] = np.asarray(upper)
        captured["infin"] = np.asarray(infin)
        captured["correl"] = correl
        return 0.12, 0.34, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    alpha, eps = uq.mvn_orthotope_density(mu=0.0, COV=4.0, lower=-1.0, upper=1.0)

    assert alpha == 0.34
    assert eps == 0.12
    assert captured["lower"].shape == (1,)
    assert captured["upper"].shape == (1,)
    assert captured["infin"].shape == (1,)
    assert captured["correl"] == 0


def test_mvn_orthotope_density_multivariate_uses_lower_triangle_correlations(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["lower"] = np.asarray(lower)
        captured["upper"] = np.asarray(upper)
        captured["infin"] = np.asarray(infin)
        captured["correl"] = np.asarray(correl)
        return 0.01, 0.99, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    COV = np.array([[4.0, 1.2], [1.2, 9.0]])
    alpha, eps = uq.mvn_orthotope_density(mu=[0.0, 0.0], COV=COV, lower=[-1.0, -2.0], upper=[1.0, 2.0])

    assert alpha == 0.99
    assert eps == 0.01
    assert captured["correl"].shape == (1,)
    assert np.allclose(captured["correl"], np.array([1.2 / (2.0 * 3.0)]))


def test_tmvn_mle_returns_sample_moments_when_no_truncation(monkeypatch):
    samples = np.array([1.0, 2.0, 3.0, 4.0])
    # Keep the optimizer from changing the initial guess.
    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: pytest.fail("minimize should not be called"))
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    mu, cov = uq.tmvn_MLE(samples)

    assert np.isclose(mu, np.mean(samples))
    assert np.isclose(cov, np.var(samples, ddof=1))


def test_tmvn_mle_multivariate_no_truncation_returns_expected_shapes(monkeypatch):
    samples = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: pytest.fail("minimize should not be called"))
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.allclose(mu, np.array([2.0, 20.0]))
    assert np.allclose(cov, np.array([[1.0, 10.0], [10.0, 100.0]]))


def test_tmvn_mle_handles_zero_variance_in_univariate_case(monkeypatch):
    samples = np.array([2.0, 2.0, 2.0])

    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: pytest.fail("minimize should not be called"))
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    mu, cov = uq.tmvn_MLE(samples)

    assert np.isclose(mu, 2.0)
    assert cov > 0.0


def test_tmvn_mle_truncated_branch_passes_adjusted_limits_to_density(monkeypatch):
    samples = np.array([0.2, 0.4, 0.6, 0.8])
    calls = []

    def fake_density(mu, cov, lower=None, upper=None):
        calls.append((np.asarray(mu), np.asarray(cov), lower, upper))
        return 1.0, 0.0

    class DummyOptimizeResult:
        def __init__(self, x):
            self.x = x
            self.message = "ok"
            self.fun = 0.0

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        # Ensure the objective is callable and returns finite values.
        assert np.isfinite(fun(x0, *args))
        return DummyOptimizeResult(x0)

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq, "minimize", fake_minimize)

    mu, cov = uq.tmvn_MLE(samples, tr_lower=np.array([0.0]), tr_upper=np.array([1.0]))

    assert np.isfinite(mu)
    assert np.isfinite(cov)
    assert len(calls) >= 1
    # The standardized truncation limits are used inside the likelihood.
    assert calls[0][2] is not None
    assert calls[0][3] is not None


def test_tmvn_mle_with_censored_data_uses_detection_limits(monkeypatch):
    samples = np.array([0.2, 0.4, 0.6, 0.8])
    density_calls = []

    def fake_density(mu, cov, lower=None, upper=None):
        density_calls.append((lower, upper))
        return 1.0, 0.0

    class DummyOptimizeResult:
        def __init__(self, x):
            self.x = x
            self.message = "ok"
            self.fun = 0.0

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(
        uq,
        "minimize",
        lambda fun, x0, args=(), method=None, options=None: DummyOptimizeResult(x0),
    )

    mu, cov = uq.tmvn_MLE(
        samples,
        censored_count=2,
        det_lower=np.array([0.0]),
        det_upper=np.array([1.0]),
    )

    assert np.isfinite(mu)
    assert np.isfinite(cov)
    # density is evaluated at least for truncation/censoring calculations
    assert len(density_calls) >= 1


def test_tmvn_mle_raises_when_truncation_density_is_too_small(monkeypatch):
    samples = np.array([0.1, 0.2, 0.3])

    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-6, 1e-4))
    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: pytest.fail("minimize should not be called"))

    with pytest.raises(ValueError, match="too small"):
        uq.tmvn_MLE(samples, tr_lower=np.array([10.0]), tr_upper=np.array([11.0]))
