"""Unit tests for LMMCore and MTLMM in limix_lmm, plus package exports."""

import re

import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm import LMMCore, MTLMM


def _make_data(N=60, S=7, seed=0, n_cov=2):
    rng = np.random.RandomState(seed)
    y = rng.randn(N, 1)
    F = np.hstack([np.ones((N, 1)), rng.randn(N, n_cov - 1)])
    G = 1.0 * (rng.rand(N, S) < 0.3)
    return y, F, G


def _ols_rss(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X.dot(beta)
    return beta, float(np.sum(resid ** 2))


class _IdentityKronCov(object):
    """Stand-in for limix_core Cov2KronSum with identity factors and unit D."""

    def __init__(self, N, P):
        self.N = N
        self.P = P

    def Lr(self):
        return np.eye(self.N)

    def Lc(self):
        return np.eye(self.P)

    def D(self):
        return np.ones((self.N, self.P))


def _use_identity_kron_cov(monkeypatch):
    covar_mod = pytest.importorskip("limix_core.covar")
    monkeypatch.setattr(covar_mod, "Cov2KronSum", _IdentityKronCov, raising=False)


def test_default_intercept_and_df():
    y, _, _ = _make_data(N=40)
    lmm = LMMCore(y, None)
    np.testing.assert_array_equal(lmm.F, np.ones((40, 1)))
    assert lmm.df == 39


def test_df_counts_covariates():
    y, F, _ = _make_data(N=50, n_cov=4)
    lmm = LMMCore(y, F)
    assert lmm.df == 50 - 4


def test_null_fit_matches_ols():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F)
    beta0, rss0 = _ols_rss(F, y)
    np.testing.assert_allclose(lmm.beta_F0, beta0, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(lmm.s20, rss0 / lmm.df, rtol=1e-6)


def test_identity_ki_dot_matches_none():
    y, F, G = _make_data()
    plain = LMMCore(y, F)
    plain.process(G)
    ident = LMMCore(y, F, Ki_dot=lambda x: x)
    ident.process(G)
    np.testing.assert_allclose(ident.getPv(), plain.getPv(), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(ident.getLRT(), plain.getLRT(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        ident.getBetaSNP(), plain.getBetaSNP(), rtol=1e-8, atol=1e-10
    )


def test_scaled_covariance_leaves_results_unchanged():
    # sigma^2 is re-estimated, so a scalar multiple of K must not change the test
    y, F, G = _make_data(seed=1)
    base = LMMCore(y, F)
    base.process(G)
    scaled = LMMCore(y, F, Ki_dot=lambda x: x / 3.0)
    scaled.process(G)
    np.testing.assert_allclose(scaled.getLRT(), base.getLRT(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(scaled.getPv(), base.getPv(), rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(
        scaled.getBetaSNP(), base.getBetaSNP(), rtol=1e-6, atol=1e-8
    )


def test_gls_matches_whitened_ols():
    y, F, G = _make_data(N=40, S=3, seed=2)
    A = np.random.RandomState(2).randn(40, 40)
    K = A.dot(A.T) / 40.0 + np.eye(40)
    L = np.linalg.cholesky(K)
    Li = np.linalg.inv(L)
    lmm = LMMCore(y, F, Ki_dot=lambda x: np.linalg.solve(K, x))
    lmm.process(G)
    yw = Li.dot(y)
    Fw = Li.dot(F)
    for s in range(G.shape[1]):
        X = np.hstack([Fw, Li.dot(G[:, [s]])])
        beta, _ = _ols_rss(X, yw)
        np.testing.assert_allclose(
            lmm.getBetaSNP()[0, s], beta[-1, 0], rtol=1e-5, atol=1e-8
        )


def test_output_shapes_single_variant():
    y, F, G = _make_data(N=60, S=7)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (7,)
    assert lmm.getLRT().shape == (7,)
    assert lmm.getBetaSNP().shape == (1, 7)
    assert lmm.getBetaSNPste().shape == (1, 7)


def test_step_groups_variants():
    y, F, G = _make_data(N=60, S=10, seed=3)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    # int(10 / 4) = 2 blocks of 4 consecutive variants; the last 2 are dropped
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)
    assert lmm.getLRT().shape == (2,)
    _, rss0 = _ols_rss(F, y)
    for b in range(2):
        X = np.hstack([F, G[:, 4 * b:4 * b + 4]])
        _, rss1 = _ols_rss(X, y)
        expected = -lmm.df * np.log(rss1 / rss0)
        assert np.isclose(lmm.getLRT()[b], expected, rtol=1e-6)


def test_interaction_step_matches_joint_ols():
    y, F, G = _make_data(N=60, S=9, seed=11)
    Inter = np.random.RandomState(12).randn(60, 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=3)
    assert lmm.getPv().shape == (3,)
    assert lmm.getBetaSNP().shape == (6, 3)
    _, rss0 = _ols_rss(F, y)
    for b in range(3):
        block = G[:, 3 * b:3 * b + 3]
        X = np.hstack(
            [Inter[:, [i]] * block[:, [j]] for i in range(2) for j in range(3)]
        )
        _, rss1 = _ols_rss(np.hstack([F, X]), y)
        expected = -lmm.df * np.log(rss1 / rss0)
        assert np.isclose(lmm.getLRT()[b], expected, rtol=1e-6)


def test_unit_interaction_matches_no_interaction():
    y, F, G = _make_data(seed=4)
    plain = LMMCore(y, F)
    plain.process(G)
    unit = LMMCore(y, F)
    unit.process(G, Inter=np.ones((60, 1)))
    np.testing.assert_allclose(unit.getPv(), plain.getPv(), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(unit.getLRT(), plain.getLRT(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        unit.getBetaSNP(), plain.getBetaSNP(), rtol=1e-8, atol=1e-10
    )


def test_lrt_matches_manual_ols():
    y, F, G = _make_data(N=60, S=6, seed=4)
    lmm = LMMCore(y, F)
    lmm.process(G)
    _, rss0 = _ols_rss(F, y)
    expected = np.array(
        [
            -lmm.df * np.log(_ols_rss(np.hstack([F, G[:, [s]]]), y)[1] / rss0)
            for s in range(G.shape[1])
        ]
    )
    np.testing.assert_allclose(lmm.getLRT(), expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        lmm.getPv(), st.chi2(1).sf(expected), rtol=1e-6, atol=1e-12
    )


def test_beta_matches_ols_coefficient():
    y, F, G = _make_data(N=60, S=6, seed=4)
    lmm = LMMCore(y, F)
    lmm.process(G)
    expected = np.array(
        [
            _ols_rss(np.hstack([F, G[:, [s]]]), y)[0][-1, 0]
            for s in range(G.shape[1])
        ]
    )
    np.testing.assert_allclose(lmm.getBetaSNP()[0], expected, rtol=1e-6, atol=1e-8)


def test_pvalues_are_valid():
    y, F, G = _make_data(N=80, S=20, seed=5)
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    assert np.all(np.isfinite(pv))
    assert np.all((pv >= 0) & (pv <= 1))
    assert np.all(lmm.getLRT() >= -1e-10)


def test_strong_signal_detected():
    rng = np.random.RandomState(6)
    N = 200
    G = (rng.rand(N, 5) < 0.3).astype(float)
    y = 3.0 * G[:, [2]] + 0.1 * rng.randn(N, 1)
    lmm = LMMCore(y, np.ones((N, 1)))
    lmm.process(G)
    pv = lmm.getPv()
    assert np.argmin(pv) == 2
    assert pv[2] < 1e-10
    np.testing.assert_allclose(lmm.getBetaSNP()[0, 2], 3.0, atol=0.1)


def test_getbetasnpste_matches_definition():
    y, F, G = _make_data(N=60, S=5, seed=7)
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(lmm.getPv()))
    ste = lmm.getBetaSNPste()
    np.testing.assert_allclose(ste, beta / z, rtol=1e-10)
    assert np.all(ste > 0)


def test_verbose_reports_timing(capsys):
    y, F, G = _make_data(N=30, S=2)
    LMMCore(y, F).process(G, verbose=True)
    assert "Tested for 2 variants" in capsys.readouterr().out
    LMMCore(y, F).process(G)
    assert capsys.readouterr().out == ""


def test_mtlmm_single_trait_matches_lmmcore(monkeypatch):
    _use_identity_kron_cov(monkeypatch)
    y, F, G = _make_data(N=50, S=4, seed=8)
    mt = MTLMM(y, F, covar=_IdentityKronCov(50, 1))
    pv, beta = mt.process(G)
    ref = LMMCore(y, F)
    ref.process(G)
    assert mt.df == ref.df
    np.testing.assert_allclose(mt.lrt, ref.getLRT(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(pv, ref.getPv(), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(beta, ref.getBetaSNP(), rtol=1e-8, atol=1e-10)


def test_mtlmm_multi_trait_shapes(monkeypatch):
    _use_identity_kron_cov(monkeypatch)
    rng = np.random.RandomState(9)
    N, P, S = 40, 2, 5
    Y = rng.randn(N, P)
    F = np.ones((N, 1))
    G = (rng.rand(N, S) < 0.3).astype(float)
    mt = MTLMM(Y, F, covar=_IdentityKronCov(N, P))
    pv, beta = mt.process(G)
    assert pv.shape == (S,)
    assert beta.shape == (P, S)
    assert np.all(np.isfinite(pv))
    assert np.all((pv >= 0) & (pv <= 1))
    assert mt.df == N * P - P * F.shape[1]


@pytest.mark.xfail(
    reason="MTLMMHelperBase uses LY and Y attributes it never sets",
    raises=AttributeError,
)
def test_mtlmm_null_fit_without_covariance():
    y, F, G = _make_data(N=30, S=3, seed=10)
    mt = MTLMM(y, F)
    pv, _ = mt.process(G)
    ref = LMMCore(y, F)
    ref.process(G)
    np.testing.assert_allclose(pv, ref.getPv(), rtol=1e-8, atol=1e-12)


def test_mtlmm_rejects_unsupported_covar():
    pytest.importorskip("limix_core.covar")
    y, F, _ = _make_data(N=20)
    with pytest.raises(AssertionError):
        MTLMM(y, F, covar=object())


def test_package_exports():
    from limix_lmm import lmm_core, mtlmm

    assert limix_lmm.LMMCore is lmm_core.LMMCore
    assert limix_lmm.MTLMM is mtlmm.MTLMM
    for name in ("LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "test"):
        assert name in limix_lmm.__all__
    for name in limix_lmm.__all__:
        assert hasattr(limix_lmm, name)


def test_version_is_semver():
    assert re.match(r"^\d+\.\d+\.\d+$", limix_lmm.__version__)
