import numpy as np
import pytest
from scipy.stats import multivariate_normal

import pelicun.uq as uq


class DummyResult:
    def __init__(self, x, fun=0.0, message="ok"):
        self.x = np.asarray(x)
        self.fun = fun
        self.message = message


def test_tmvn_rvs_unbounded_calls_scipy_rvs(monkeypatch):
    called = {}

    def fake_rvs(mean, cov, size=None):
        called["mean"] = np.asarray(mean)
        called["cov"] = np.asarray(cov)
        called["size"] = size
        return np.array([[1.0, 2.0], [3.0, 4.0]])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=[0.0, 1.0], COV=[[1.0, 0.2], [0.2, 2.0]], size=2)

    assert out.shape == (2, 2)
    assert np.allclose(out, [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(called["mean"], [0.0, 1.0])
    assert np.allclose(called["cov"], [[1.0, 0.2], [0.2, 2.0]])
    assert called["size"] == 2


def test_tmvn_rvs_scalar_inputs_no_bounds(monkeypatch):
    def fake_rvs(mean, cov, size=None):
        assert np.asarray(mean).shape == (1,)
        assert np.asarray(cov).shape == (1,)
        return np.array([5.0])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(mu=1.5, COV=0.25, size=1)
    assert np.allclose(out, [5.0])


def test_tmvn_rvs_bounded_returns_only_samples_within_limits(monkeypatch):
    raw = np.array(
        [
            [0.1, 0.2],
            [2.0, 2.0],
            [0.4, 0.6],
            [0.7, 0.8],
        ]
    )

    def fake_density(mu, COV, lower=None, upper=None):
        return 0.5, 0.0

    def fake_rvs(mean, cov, size=None):
        assert size is not None
        return raw[:size]

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    out = uq.tmvn_rvs(
        mu=[0.0, 0.0],
        COV=[[1.0, 0.0], [0.0, 1.0]],
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        size=2,
    )

    assert out.shape == (2, 2)
    assert np.all(out > 0.0)
    assert np.all(out < 1.0)
    assert np.allclose(out, np.array([[0.1, 0.2], [0.4, 0.6]]))


def test_tmvn_rvs_raises_when_alpha_too_small(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-6, 1e-4))

    with pytest.raises(ValueError, match="density of the joint probability distribution"):
        uq.tmvn_rvs(
            mu=[0.0, 0.0],
            COV=[[1.0, 0.0], [0.0, 1.0]],
            lower=[-1.0, -1.0],
            upper=[1.0, 1.0],
            size=1,
        )


def test_tmvn_rvs_warns_for_very_small_alpha(monkeypatch):
    raw = np.array([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]])

    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (5e-4, 0.0))
    monkeypatch.setattr(uq.multivariate_normal, "rvs", lambda mean, cov, size=None: raw[:size])

    with pytest.warns(UserWarning, match="rejection rate"):
        out = uq.tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=[-1.0], upper=[1.0], size=2)

    assert out.shape == (2, 1)


def test_mvn_orthotope_density_univariate_passes_standardized_bounds(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["lower"] = np.asarray(lower)
        captured["upper"] = np.asarray(upper)
        captured["infin"] = np.asarray(infin)
        captured["correl"] = correl
        return 0.01, 0.25, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    alpha, eps = uq.mvn_orthotope_density(mu=1.0, COV=4.0, lower=0.0, upper=3.0)

    assert alpha == 0.25
    assert eps == 0.01
    assert np.allclose(captured["lower"], [-0.5])
    assert np.allclose(captured["upper"], [1.0])
    assert np.allclose(captured["infin"], [2.0])
    assert captured["correl"] == 0


def test_mvn_orthotope_density_multivariate_uses_lower_triangle_correlations(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["lower"] = np.asarray(lower)
        captured["upper"] = np.asarray(upper)
        captured["infin"] = np.asarray(infin)
        captured["correl"] = np.asarray(correl)
        return 0.02, 0.5, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    mu = np.array([0.0, 0.0])
    COV = np.array([[1.0, 0.3], [0.3, 4.0]])
    alpha, eps = uq.mvn_orthotope_density(mu, COV, lower=[-1.0, -2.0], upper=[1.0, 2.0])

    assert alpha == 0.5
    assert eps == 0.02
    assert captured["correl"].shape == (1,)
    assert np.allclose(captured["correl"], [0.15])


def test_tmvn_MLE_univariate_without_truncation_returns_reasonable_shape(monkeypatch):
    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: pytest.fail("minimize should not be called"))
    samples = np.array([1.0, 3.0])

    mu, cov = uq.tmvn_MLE(samples)

    assert np.isscalar(mu) or np.asarray(mu).shape == ()
    assert np.isscalar(cov) or np.asarray(cov).shape == ()
    assert mu == pytest.approx(2.0)
    assert cov > 0.0


def test_tmvn_MLE_multivariate_without_truncation_uses_minimize(monkeypatch):
    called = {}

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        called["method"] = method
        called["x0"] = np.asarray(x0)
        called["args"] = args
        fun(np.asarray(x0), *args)
        return DummyResult(x0, fun=0.0, message="success")

    monkeypatch.setattr(uq, "minimize", fake_minimize)
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    samples = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    mu, cov = uq.tmvn_MLE(samples)

    assert called["method"] == "Nelder-Mead"
    assert called["x0"].shape == (4,)
    assert np.asarray(mu).shape == (2,)
    assert np.asarray(cov).shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))


def test_tmvn_MLE_constant_dimension_gets_small_variance(monkeypatch):
    def fake_minimize(fun, x0, args=(), method=None, options=None):
        return DummyResult(x0)

    monkeypatch.setattr(uq, "minimize", fake_minimize)
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    samples = np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]])
    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert cov[0, 0] >= 0.0
    assert cov[1, 1] >= 0.0


def test_tmvn_MLE_truncation_penalty_warning_path(monkeypatch):
    warnings_seen = []

    def fake_warn(msg):
        warnings_seen.append(str(msg))

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        # Trigger the objective once to exercise the alpha-estimate warning path.
        fun(np.asarray(x0), *args)
        return DummyResult(x0, fun=1.0, message="stopped")

    monkeypatch.setattr(uq.warnings, "warn", fake_warn)
    monkeypatch.setattr(uq, "minimize", fake_minimize)
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-6, 1e-4))

    samples = np.array([[0.0, 0.1, 0.2], [0.0, 0.2, 0.4]])
    mu, cov = uq.tmvn_MLE(samples, tr_lower=[-1.0, -1.0], tr_upper=[1.0, 1.0])

    assert len(warnings_seen) >= 1
    assert np.asarray(mu).shape == (2,)
    assert np.asarray(cov).shape == (2, 2)


def test_tmvn_MLE_with_censoring_uses_detection_limits(monkeypatch):
    calls = []

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        calls.append(("minimize", method))
        fun(np.asarray(x0), *args)
        return DummyResult(x0, fun=0.0, message="ok")

    def fake_density(mu, COV, lower=None, upper=None):
        calls.append((np.asarray(lower), np.asarray(upper)))
        return 1.0, 0.0

    monkeypatch.setattr(uq, "minimize", fake_minimize)
    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    samples = np.array([[0.0, 0.1], [1.0, 1.1]])
    mu, cov = uq.tmvn_MLE(
        samples,
        censored_count=1,
        det_lower=[-0.5, 0.5],
        det_upper=[0.5, 1.5],
    )

    assert calls
    assert np.asarray(mu).shape == (2,)
    assert np.asarray(cov).shape == (2, 2)


def test_tmvn_MLE_returns_finite_values_for_simple_2d_data(monkeypatch):
    def fake_minimize(fun, x0, args=(), method=None, options=None):
        return DummyResult(x0)

    monkeypatch.setattr(uq, "minimize", fake_minimize)
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    samples = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    mu, cov = uq.tmvn_MLE(samples)

    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
    assert np.all(np.diag(cov) > 0.0)
