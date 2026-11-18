import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import pelicun.uq as uq


def test_tmvn_rvs_untruncated_delegates_to_multivariate_normal(monkeypatch):
    calls = {}

    def fake_rvs(mean, cov, size=None, **kwargs):
        calls["mean"] = np.asarray(mean)
        calls["cov"] = np.asarray(cov)
        calls["size"] = size
        return np.array([[1.0, 2.0], [3.0, 4.0]])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    samples = uq.tmvn_rvs(mu=[0.0, 1.0], COV=[[1.0, 0.2], [0.2, 2.0]], size=2)

    assert samples.shape == (2, 2)
    assert np.allclose(samples, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(calls["mean"], np.array([0.0, 1.0]))
    assert np.allclose(calls["cov"], np.array([[1.0, 0.2], [0.2, 2.0]]))
    assert calls["size"] == 2


def test_tmvn_rvs_scalar_inputs_are_promoted(monkeypatch):
    def fake_mvn_orthotope_density(mu, cov, lower=None, upper=None):
        assert mu.shape == (1,)
        assert np.asarray(cov).shape == (1,)
        return 1.0, 0.0

    def fake_rvs(mean, cov, size=None, **kwargs):
        assert np.asarray(mean).shape == (1,)
        assert np.asarray(cov).shape == (1,)
        return np.array([0.25, 0.75])

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_mvn_orthotope_density)
    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    samples = uq.tmvn_rvs(mu=0.0, COV=1.0, lower=-1.0, upper=1.0, size=2)

    assert samples.shape == (2,)
    assert np.allclose(samples, np.array([0.25, 0.75]))


def test_tmvn_rvs_rejects_when_alpha_too_small(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-6, 2e-8))

    with pytest.raises(ValueError, match="density of the joint probability distribution"):
        uq.tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=[10.0], upper=[11.0], size=1)


def test_tmvn_rvs_warns_for_high_rejection_rate(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1e-4, 0.0))

    def fake_rvs(mean, cov, size=None, **kwargs):
        return np.array([[0.5], [0.6], [0.7], [0.8], [0.9]])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        samples = uq.tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=[0.0], upper=[1.0], size=2)

    assert samples.shape == (2, 1)
    assert any("rejection rate" in str(w.message).lower() for w in rec)


def test_tmvn_rvs_accepts_all_samples_from_first_draw(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))
    calls = {"n": 0}

    def fake_rvs(mean, cov, size=None, **kwargs):
        calls["n"] += 1
        return np.array([[0.1], [0.2], [0.3]])

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    samples = uq.tmvn_rvs(mu=[0.0], COV=[[1.0]], lower=[0.0], upper=[1.0], size=2)

    assert calls["n"] == 1
    assert samples.shape == (2, 1)
    assert np.allclose(samples[:, 0], np.array([0.1, 0.2]))


def test_tmvn_rvs_multidimensional_filters_by_all_bounds(monkeypatch):
    monkeypatch.setattr(uq, "mvn_orthotope_density", lambda *args, **kwargs: (1.0, 0.0))

    def fake_rvs(mean, cov, size=None, **kwargs):
        return np.array(
            [
                [0.1, 0.1],
                [0.9, 0.9],
                [0.5, -1.0],
                [0.2, 0.8],
            ]
        )

    monkeypatch.setattr(uq.multivariate_normal, "rvs", fake_rvs)

    samples = uq.tmvn_rvs(mu=[0.0, 0.0], COV=np.eye(2), lower=[0.0, 0.0], upper=[1.0, 1.0], size=3)

    assert samples.shape == (3, 2)
    assert np.all((samples >= 0.0) & (samples <= 1.0))


def test_mvn_orthotope_density_univariate_uses_mvndst(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["lower"] = np.asarray(lower)
        captured["upper"] = np.asarray(upper)
        captured["infin"] = np.asarray(infin)
        captured["correl"] = correl
        return 0.0123, 0.456, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    alpha, eps = uq.mvn_orthotope_density(mu=0.0, COV=4.0, lower=-2.0, upper=2.0)

    assert alpha == pytest.approx(0.456)
    assert eps == pytest.approx(0.0123)
    assert captured["lower"].shape == (1,)
    assert captured["upper"].shape == (1,)
    assert captured["infin"].tolist() == [2.0]
    assert captured["correl"] == 0


def test_mvn_orthotope_density_multivariate_passes_lower_triangle_correlations(monkeypatch):
    captured = {}

    def fake_mvndst(lower, upper, infin, correl):
        captured["correl"] = np.asarray(correl)
        return 0.1, 0.2, 0

    monkeypatch.setattr(uq, "mvndst", fake_mvndst)

    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    uq.mvn_orthotope_density(mu=[0.0, 1.0], COV=cov, lower=[-1.0, 0.0], upper=[1.0, 2.0])

    assert captured["correl"].shape == (1,)
    assert captured["correl"][0] == pytest.approx(0.5 / np.sqrt(2.0))


def test_tmvn_MLE_univariate_one_sample_returns_sample_mean(monkeypatch):
    sample = np.array([3.5])

    mu, cov = uq.tmvn_MLE(sample)

    assert np.asarray(mu).shape == ()
    assert mu == pytest.approx(3.5)
    assert np.asarray(cov).shape == ()
    assert cov > 0


def test_tmvn_MLE_univariate_multiple_samples_returns_reasonable_shape(monkeypatch):
    sample = np.array([1.0, 2.0, 3.0])

    mu, cov = uq.tmvn_MLE(sample)

    assert np.asarray(mu).shape == ()
    assert np.asarray(cov).shape == ()
    assert np.isfinite(mu)
    assert np.isfinite(cov)
    assert cov > 0


def test_tmvn_MLE_multivariate_no_truncation_returns_shapes(monkeypatch):
    samples = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])

    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
    assert np.all(np.diag(cov) > 0)


def test_tmvn_MLE_multivariate_zero_variance_dimension_is_regularized(monkeypatch):
    samples = np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]])

    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.diag(cov) > 0)
    assert np.all(np.isfinite(cov))


def test_tmvn_MLE_accepts_truncation_and_uses_density(monkeypatch):
    calls = {"n": 0}

    def fake_density(mu, cov, lower=None, upper=None):
        calls["n"] += 1
        return 1.0, 0.0

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    samples = np.array([0.1, 0.2, 0.3])
    mu, cov = uq.tmvn_MLE(samples, tr_lower=np.array([-1.0]), tr_upper=np.array([1.0]))

    assert calls["n"] >= 1
    assert mu.shape == ()
    assert cov.shape == ()
    assert np.isfinite(mu)
    assert np.isfinite(cov)


def test_tmvn_MLE_with_censoring_uses_detection_limits(monkeypatch):
    calls = []

    def fake_density(mu, cov, lower=None, upper=None):
        calls.append((np.asarray(mu).copy(), np.asarray(cov).copy(), lower, upper))
        return 1.0, 0.0

    monkeypatch.setattr(uq, "mvn_orthotope_density", fake_density)

    samples = np.array([0.0, 0.0, 0.0, 0.0])
    mu, cov = uq.tmvn_MLE(
        samples,
        censored_count=2,
        det_lower=np.array([-1.0]),
        det_upper=np.array([1.0]),
    )

    assert len(calls) >= 2
    assert mu.shape == ()
    assert cov.shape == ()
    assert np.isfinite(mu)
    assert np.isfinite(cov)


def test_tmvn_MLE_raises_on_nan_inputs(monkeypatch):
    samples = np.array([1.0, np.nan, 2.0])

    mu, cov = uq.tmvn_MLE(samples)

    assert np.isnan(mu) or np.isfinite(mu)
    assert np.isfinite(cov) or np.isnan(cov)


def test_tmvn_MLE_handles_two_dimensional_single_sample(monkeypatch):
    samples = np.array([[1.0], [2.0]])

    mu, cov = uq.tmvn_MLE(samples)

    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(cov))
