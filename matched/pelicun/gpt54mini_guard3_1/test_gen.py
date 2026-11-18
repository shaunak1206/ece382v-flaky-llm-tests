import numpy as np
import pytest

from pelicun import uq


def test_tmvn_rvs_untruncated_scalar_shape_and_seed():
    np.random.seed(1234)

    samples = uq.tmvn_rvs(mu=0.0, COV=1.0, size=5)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (5,)
    expected = np.array([0.47143516, -1.19097569, 1.43270697, -0.3126519, -0.72058873])
    np.testing.assert_allclose(samples, expected, rtol=0, atol=1e-8)


def test_tmvn_rvs_untruncated_multivariate_shape():
    np.random.seed(4321)

    mu = np.array([0.0, 1.0])
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])

    samples = uq.tmvn_rvs(mu=mu, COV=cov, size=4)

    assert samples.shape == (4, 2)


def test_tmvn_rvs_truncated_returns_values_within_bounds(monkeypatch):
    np.random.seed(7)

    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    mu = np.array([0.0, 0.0])
    cov = np.eye(2)
    lower = np.array([-0.5, -0.25])
    upper = np.array([0.5, 0.25])

    samples = uq.tmvn_rvs(mu=mu, COV=cov, lower=lower, upper=upper, size=10)

    assert samples.shape == (10, 2)
    assert np.all(samples >= lower - 1e-12)
    assert np.all(samples <= upper + 1e-12)


def test_tmvn_rvs_truncated_1d_scalar_bounds(monkeypatch):
    np.random.seed(11)

    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    samples = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=6)

    assert samples.shape == (6, 1)
    assert np.all(samples >= -1.0 - 1e-12)
    assert np.all(samples <= 1.0 + 1e-12)


def test_tmvn_rvs_raises_when_alpha_too_small(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-6, 2e-8))

    with pytest.raises(ValueError, match="density of the joint probability distribution"):
        uq.tmvn_rvs(mu=np.array([0.0]), COV=np.array([[1.0]]), lower=np.array([10.0]), upper=np.array([11.0]), size=1)


def test_tmvn_rvs_warns_for_extreme_rejection_rate(monkeypatch):
    np.random.seed(9)

    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (5e-4, 0.0))

    with pytest.warns(UserWarning, match="rejection rate"):
        samples = uq.tmvn_rvs(mu=np.array([0.0]), COV=np.array([[1.0]]), lower=np.array([0.0]), upper=np.array([0.1]), size=1)

    assert samples.shape == (1, 1)


def test_tmvn_MLE_univariate_degenerate_sample_returns_near_zero_variance():
    samples = np.array([2.0])

    mu, cov = uq.tmvn_MLE(samples)

    assert pytest.approx(mu, abs=1e-12) == 2.0
    assert np.isscalar(cov) or np.asarray(cov).shape == ()
    assert pytest.approx(float(cov), rel=0, abs=1e-15) == 9.98e-13


def test_tmvn_MLE_univariate_multiple_samples_no_truncation():
    samples = np.array([1.0, 2.0, 3.0])

    mu, cov = uq.tmvn_MLE(samples)

    assert pytest.approx(mu, abs=1e-12) == 2.0
    assert pytest.approx(float(cov), abs=1e-12) == 2.0 / 3.0


def test_tmvn_MLE_multivariate_without_minimization(monkeypatch):
    samples = np.array([[0.0, 1.0], [2.0, 3.0]])

    def fail_minimize(*args, **kwargs):
        raise AssertionError("minimize should not be called when nsamples <= ndims")

    monkeypatch.setattr(uq, "minimize", fail_minimize)

    mu, cov = uq.tmvn_MLE(samples)

    np.testing.assert_allclose(mu, np.array([1.0, 2.0]), atol=1e-12)
    assert cov.shape == (2, 2)
    np.testing.assert_allclose(cov, np.diag([1.0, 1.0]), atol=1e-12)


def test_tmvn_MLE_multivariate_calls_minimize_and_uses_returned_params(monkeypatch):
    samples = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])

    class Result:
        def __init__(self, x):
            self.x = x
            self.message = "ok"
            self.fun = 0.0

    def fake_minimize(fun, x0, args=(), **kwargs):
        x = np.array(x0, dtype=float)
        x[0] = 0.1
        x[1] = -0.2
        x[2] = 1.1
        x[3] = 0.9
        return Result(x)

    monkeypatch.setattr(uq, "minimize", fake_minimize)

    mu, cov = uq.tmvn_MLE(samples)

    expected_mu = np.array([2.0, 3.0]) + np.array([0.1, -0.2]) * np.array([1.63299316, 1.63299316])
    expected_sig = np.array([1.1, 0.9]) * np.array([1.63299316, 1.63299316])

    np.testing.assert_allclose(mu, expected_mu, atol=1e-8)
    np.testing.assert_allclose(np.diag(cov), expected_sig**2, atol=1e-8)
    assert cov.shape == (2, 2)


def test_tmvn_MLE_multivariate_constant_dimension_keeps_positive_variance(monkeypatch):
    samples = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])

    class Result:
        def __init__(self, x):
            self.x = x
            self.message = "ok"
            self.fun = 0.0

    def fake_minimize(fun, x0, args=(), **kwargs):
        return Result(np.array(x0, dtype=float))

    monkeypatch.setattr(uq, "minimize", fake_minimize)

    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert cov[0, 0] > 0.0
    assert np.isfinite(cov).all()


def test_tmvn_MLE_warns_when_correlation_is_nan(monkeypatch):
    samples = np.array([[1.0, 1.0], [2.0, 2.0]])

    warnings_seen = []

    def fake_show_warning(msg):
        warnings_seen.append(msg)

    class Result:
        def __init__(self, x):
            self.x = x
            self.message = "ok"
            self.fun = 0.0

    def fake_minimize(fun, x0, args=(), **kwargs):
        return Result(np.array(x0, dtype=float))

    monkeypatch.setattr(uq, "show_warning", fake_show_warning)
    monkeypatch.setattr(uq, "minimize", fake_minimize)

    uq.tmvn_MLE(samples)

    assert any("correlation matrix" in msg for msg in warnings_seen)


def test_tmvn_MLE_truncation_alpha_limit_forces_warning_and_penalty(monkeypatch):
    samples = np.array([[0.0, 0.1], [0.0, 0.1]])

    def fake_mvndst(lower, upper, infin, correl):
        return (1e-12, 1e-14, 0)

    calls = {"n": 0}

    def fake_minimize(fun, x0, args=(), **kwargs):
        calls["n"] += 1
        val = fun(np.array(x0, dtype=float), *args)
        assert val == 1e10
        return type("Result", (), {"x": np.array(x0, dtype=float), "message": "ok", "fun": val})()

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)
    monkeypatch.setattr(uq, "minimize", fake_minimize)

    with pytest.warns(UserWarning, match="density of the joint probability distribution"):
        mu, cov = uq.tmvn_MLE(
            samples,
            tr_lower=np.array([-1.0, -1.0]),
            tr_upper=np.array([1.0, 1.0]),
            alpha_lim=1e-3,
        )

    assert calls["n"] == 1
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
