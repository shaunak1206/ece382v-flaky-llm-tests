import warnings

import numpy as np
import pytest

import pelicun.uq as uq


def test_tmvn_rvs_univariate_no_bounds_uses_multivariate_normal(monkeypatch):
    calls = []

    def fake_rvs(mean=None, cov=None, size=None):
        calls.append((np.asarray(mean), np.asarray(cov), size))
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    samples = uq.tmvn_rvs(mu=0.5, COV=2.0, size=3)

    assert np.allclose(samples, np.array([1.0, 2.0, 3.0]))
    assert len(calls) == 1
    assert calls[0][0].shape == (1,)
    assert calls[0][1].shape == (1,)


def test_tmvn_rvs_multivariate_no_bounds(monkeypatch):
    def fake_rvs(mean=None, cov=None, size=None):
        assert np.allclose(mean, np.array([1.0, -1.0]))
        assert np.allclose(cov, np.array([[2.0, 0.3], [0.3, 1.5]]))
        return np.array([[10.0, 11.0], [12.0, 13.0]])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    samples = uq.tmvn_rvs(mu=[1.0, -1.0], COV=[[2.0, 0.3], [0.3, 1.5]], size=2)

    assert samples.shape == (2, 2)
    assert np.allclose(samples, np.array([[10.0, 11.0], [12.0, 13.0]]))


def test_tmvn_rvs_scalar_inputs_are_promoted(monkeypatch):
    def fake_density(mu, COV, lower=None, upper=None):
        return 1.0, 0.0

    def fake_rvs(mean=None, cov=None, size=None):
        assert np.asarray(mean).shape == (1,)
        assert np.asarray(cov).shape == (1,)
        return np.array([0.1, 0.2, 0.3, 0.4])

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    samples = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=3)
    assert samples.shape == (3,)


def test_tmvn_rvs_rejects_when_alpha_too_small(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (0.001, 0.00002))

    with pytest.raises(ValueError, match="density of the joint probability distribution"):
        uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=1)


def test_tmvn_rvs_warns_when_alpha_is_small(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (5e-4, 1e-7))

    def fake_rvs(mean=None, cov=None, size=None):
        return np.array([[0.0], [0.5], [2.0], [-0.2], [0.3], [0.7], [0.9], [1.1]])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        samples = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=2)

    assert samples.shape == (2, 1)
    assert any("higher than 0.999" in str(w.message) for w in rec)


def test_tmvn_rvs_filters_out_of_bounds_samples(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    calls = []

    def fake_rvs(mean=None, cov=None, size=None):
        calls.append(size)
        if len(calls) == 1:
            return np.array([[2.0], [0.2], [1.5], [-2.0], [0.3]])
        return np.array([[0.4], [0.6], [0.8], [0.1], [0.9]])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    samples = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=0.0, upper=1.0, size=3)

    assert samples.shape == (3, 1)
    assert np.all((samples > 0.0) & (samples < 1.0))
    assert len(calls) >= 1


def test_tmvn_orthotope_density_univariate_passes_expected_args(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["lower"] = np.asarray(lower)
        captured["upper"] = np.asarray(upper)
        captured["infin"] = np.asarray(infin)
        captured["correl"] = correl
        return 0.01, 0.23, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    alpha, eps = uq.mvn_orthotope_density(mu=1.0, COV=4.0, lower=0.0, upper=3.0)

    assert alpha == pytest.approx(0.23)
    assert eps == pytest.approx(0.01)
    assert np.allclose(captured["lower"], np.array([-0.5]))
    assert np.allclose(captured["upper"], np.array([1.0]))
    assert np.allclose(captured["infin"], np.array([2.0]))
    assert captured["correl"] == 0


def test_tmvn_orthotope_density_multivariate_uses_lower_triangular_correlations(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["correl"] = np.asarray(correl)
        return 0.02, 0.33, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    COV = np.array([[4.0, 1.0], [1.0, 9.0]])
    alpha, eps = uq.mvn_orthotope_density(mu=[0.0, 0.0], COV=COV, lower=[-1.0, -1.0], upper=[1.0, 1.0])

    assert alpha == pytest.approx(0.33)
    assert eps == pytest.approx(0.02)
    assert captured["correl"].shape == (1,)
    assert captured["correl"][0] == pytest.approx(1.0 / 6.0)


def test_tmvn_mle_univariate_calls_minimize_and_returns_unbiased_result(monkeypatch):
    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: pytest.fail("minimize should not be called"))
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    samples = np.array([1.0, 2.0, 3.0, 4.0])
    mu, cov = uq.tmvn_MLE(samples)

    assert mu == pytest.approx(np.mean(samples))
    assert cov == pytest.approx(np.var(samples, ddof=1))


def test_tmvn_mle_univariate_uses_minimize_for_numerous_samples(monkeypatch):
    calls = {"count": 0, "x0": None}

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        calls["count"] += 1
        calls["x0"] = np.array(x0)
        return type("R", (), {"x": np.array([0.0, 1.0]), "fun": 0.0, "message": "ok"})()

    monkeypatch.setattr(uq, "minimize", fake_minimize)
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    samples = np.array([0.9, 1.0, 1.1, 1.2, 0.95])
    mu, cov = uq.tmvn_MLE(samples)

    assert calls["count"] == 1
    assert calls["x0"].shape == (2,)
    assert mu == pytest.approx(np.mean(samples), rel=1e-6)
    assert cov > 0.0


def test_tmvn_mle_multivariate_returns_covariance_shape(monkeypatch):
    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: pytest.fail("minimize should not be called"))
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    samples = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
        ]
    )
    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))


def test_tmvn_mle_multivariate_handles_zero_variance_dimension(monkeypatch):
    monkeypatch.setattr(uq, "minimize", lambda *args, **kwargs: pytest.fail("minimize should not be called"))
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    samples = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0],
        ]
    )
    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert cov[0, 0] > 0.0
    assert cov[1, 1] > 0.0


def test_tmvn_mle_truncation_and_censoring_passed_to_internal_density(monkeypatch):
    density_calls = []

    def fake_density(mu, COV, lower=None, upper=None):
        density_calls.append((np.asarray(mu), np.asarray(COV), lower, upper))
        return 1.0, 0.0

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        fun(np.array(x0), *args, enforce_bounds=True)
        return type("R", (), {"x": np.array(x0), "fun": 0.0, "message": "ok"})()

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)
    monkeypatch.setattr(uq, "minimize", fake_minimize)

    samples = np.array([1.0, 1.2, 0.8, 1.1])
    mu, cov = uq.tmvn_MLE(
        samples,
        tr_lower=0.0,
        tr_upper=2.0,
        censored_count=1,
        det_lower=0.1,
        det_upper=1.9,
    )

    assert len(density_calls) >= 2
    assert mu == pytest.approx(np.mean(samples), rel=1e-6)
    assert cov > 0.0


def test_tmvn_mle_single_sample_returns_finite_values(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    samples = np.array([2.5])
    mu, cov = uq.tmvn_MLE(samples)

    assert np.isfinite(mu)
    assert np.isfinite(cov)
    assert cov > 0.0
