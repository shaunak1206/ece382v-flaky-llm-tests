import numpy as np
import pytest
from numpy.random import RandomState
import scipy.stats as st

import limix_lmm
from limix_lmm.lmm_core import LMMCore


def _data(N=60, S=8, seed=0):
    rs = RandomState(seed)
    y = rs.randn(N, 1)
    F = np.concatenate([np.ones((N, 1)), rs.randn(N, 1)], 1)
    G = 1.0 * (rs.rand(N, S) < 0.3) + 0.5 * rs.randn(N, S)
    Inter = rs.randn(N, 2)
    return y, F, G, Inter


def _ols(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X.dot(b)
    return b, float(r.T.dot(r))


def test_package_metadata_and_exports():
    assert limix_lmm.__version__ == "0.1.2"
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip",
                 "__version__", "test"]:
        assert name in limix_lmm.__all__
        assert hasattr(limix_lmm, name)


def test_lmmcore_and_mtlmm_exported_classes():
    from limix_lmm.mtlmm import MTLMM
    assert limix_lmm.LMMCore is LMMCore
    assert limix_lmm.MTLMM is MTLMM
    assert isinstance(MTLMM, type)


def test_init_default_covariates_is_intercept():
    y, _, _, _ = _data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (y.shape[0], 1)
    assert np.all(lmm.F == 1)
    assert lmm.df == y.shape[0] - 1


def test_init_degrees_of_freedom_with_covariates():
    y, F, _, _ = _data()
    lmm = LMMCore(y, F)
    assert lmm.df == y.shape[0] - F.shape[1]


def test_null_model_matches_ols():
    y, F, _, _ = _data()
    lmm = LMMCore(y, F)
    b, rss = _ols(y, F)
    np.testing.assert_allclose(np.asarray(lmm.beta_F0).ravel(), b.ravel(), rtol=1e-6)
    np.testing.assert_allclose(lmm.s20, rss / lmm.df, rtol=1e-6)


def test_process_output_shapes_step1():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (G.shape[1],)
    assert lmm.getLRT().shape == (G.shape[1],)
    assert lmm.getBetaSNP().shape == (1, G.shape[1])


def test_betas_lrt_and_pv_match_ols_step1():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    _, rss0 = _ols(y, F)
    for s in range(G.shape[1]):
        X = np.concatenate([F, G[:, [s]]], 1)
        b, rss1 = _ols(y, X)
        np.testing.assert_allclose(lmm.getBetaSNP()[0, s], b[-1, 0], rtol=1e-5, atol=1e-8)
        lrt = -lmm.df * np.log(rss1 / rss0)
        np.testing.assert_allclose(lmm.getLRT()[s], lrt, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(lmm.getPv(), st.chi2(1).sf(lmm.getLRT()), rtol=1e-8)


def test_pvalues_are_valid_probabilities():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    assert np.all(pv >= 0) and np.all(pv <= 1)
    assert np.all(np.isfinite(pv))


def test_step_shapes_and_joint_test_matches_ols():
    y, F, G, _ = _data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    assert lmm.getPv().shape == (4,)
    assert lmm.getBetaSNP().shape == (2, 4)
    _, rss0 = _ols(y, F)
    X = np.concatenate([F, G[:, 2:4]], 1)
    b, rss1 = _ols(y, X)
    np.testing.assert_allclose(lmm.getBetaSNP()[:, 1], b[-2:, 0], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(lmm.getLRT()[1], -lmm.df * np.log(rss1 / rss0), rtol=1e-5)
    np.testing.assert_allclose(lmm.getPv()[1], st.chi2(2).sf(lmm.getLRT()[1]), rtol=1e-6)


def test_step_drops_incomplete_trailing_block():
    y, F, G, _ = _data(S=7)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    assert lmm.getPv().shape == (3,)
    assert lmm.getBetaSNP().shape == (2, 3)


def test_interaction_step1_matches_ols_on_product():
    y, F, G, Inter = _data()
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getBetaSNP().shape == (Inter.shape[1], G.shape[1])
    s = 3
    Xg = Inter * G[:, s:s + 1]
    b, _ = _ols(y, np.concatenate([F, Xg], 1))
    np.testing.assert_allclose(lmm.getBetaSNP()[:, s], b[-2:, 0], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(lmm.getPv(), st.chi2(2).sf(lmm.getLRT()), rtol=1e-8)


def test_interaction_with_step_shapes():
    y, F, G, Inter = _data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)
    assert lmm.getBetaSNP().shape == (4, 4)
    assert lmm.getPv().shape == (4,)
    assert lmm.getLRT().shape == (4,)


def test_identity_Ki_dot_equals_no_Ki_dot():
    y, F, G, _ = _data()
    a = LMMCore(y, F)
    b = LMMCore(y, F, lambda x: x)
    a.process(G)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(a.getPv(), b.getPv(), rtol=1e-6, atol=1e-12)


def test_scaled_Ki_dot_keeps_betas_and_lrt():
    y, F, G, _ = _data()
    a = LMMCore(y, F)
    b = LMMCore(y, F, lambda x: 2.0 * x)
    a.process(G)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(a.getLRT(), b.getLRT(), rtol=1e-6, atol=1e-10)


def test_planted_effect_is_detected():
    rs = RandomState(3)
    N = 200
    G = rs.randn(N, 5)
    F = np.ones((N, 1))
    y = 2.0 * G[:, [2]] + 0.5 * rs.randn(N, 1)
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    assert np.argmin(pv) == 2
    assert pv[2] < 1e-10
    assert abs(lmm.getBetaSNP()[0, 2] - 2.0) < 0.3


def test_beta_ste_consistent_with_lrt():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    beta = lmm.getBetaSNP()
    assert ste.shape == beta.shape
    assert np.all(ste > 0)
    np.testing.assert_allclose(np.abs(beta / ste) ** 2, lmm.getLRT()[None, :], rtol=1e-4)


def test_verbose_prints_summary(capsys):
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for %d variants" % G.shape[1] in out


def test_quiet_by_default(capsys):
    y, F, G, _ = _data()
    LMMCore(y, F).process(G)
    assert capsys.readouterr().out == ""
