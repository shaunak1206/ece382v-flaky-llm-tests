import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm import LMMCore, MTLMM
from limix_lmm.lmm_core import LMMCore as LMMCoreDirect


def _data(N=60, S=8, k=2, seed=0):
    rs = np.random.RandomState(seed)
    y = rs.randn(N, 1)
    F = np.concatenate([np.ones((N, 1)), rs.randn(N, k - 1)], 1)
    G = 1.0 * (rs.rand(N, S) < 0.3) + 0.1 * rs.randn(N, S)
    return y, F, G, rs


def _rss(y, X):
    beta = np.linalg.lstsq(X, y[:, 0], rcond=None)[0]
    r = y[:, 0] - X.dot(beta)
    return beta, float(r.dot(r))


def test_package_exports():
    assert limix_lmm.LMMCore is LMMCoreDirect
    assert limix_lmm.MTLMM is MTLMM
    assert isinstance(limix_lmm.__version__, str)
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip",
                 "__version__", "test"]:
        assert name in limix_lmm.__all__
        assert hasattr(limix_lmm, name)


def test_mtlmm_is_class():
    assert isinstance(MTLMM, type)
    assert MTLMM.__name__ == "MTLMM"


def test_init_default_F_is_ones():
    y, _, _, _ = _data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (y.shape[0], 1)
    assert np.all(lmm.F == 1)
    assert lmm.df == y.shape[0] - 1


def test_init_df_with_covariates():
    y, F, _, _ = _data(N=50, k=3)
    lmm = LMMCore(y, F)
    assert lmm.df == 50 - 3


def test_null_beta_matches_ols():
    y, F, _, _ = _data()
    lmm = LMMCore(y, F)
    beta, _ = _rss(y, F)
    np.testing.assert_allclose(np.ravel(lmm.beta_F0), beta, rtol=1e-6, atol=1e-8)


def test_null_s2_matches_ols():
    y, F, _, _ = _data()
    lmm = LMMCore(y, F)
    _, rss = _rss(y, F)
    np.testing.assert_allclose(lmm.s20, rss / lmm.df, rtol=1e-6)


def test_process_shapes_step1():
    y, F, G, _ = _data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (8,)
    assert lmm.getLRT().shape == (8,)
    assert lmm.getBetaSNP().shape == (1, 8)
    assert lmm.getBetaSNPste().shape == (1, 8)


def test_beta_matches_ols_per_variant():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    for s in range(G.shape[1]):
        X = np.concatenate([F, G[:, [s]]], 1)
        beta, _ = _rss(y, X)
        np.testing.assert_allclose(lmm.getBetaSNP()[0, s], beta[-1],
                                   rtol=1e-5, atol=1e-8)


def test_lrt_and_pv_match_ols():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    _, rss0 = _rss(y, F)
    df = y.shape[0] - F.shape[1]
    for s in range(G.shape[1]):
        _, rss1 = _rss(y, np.concatenate([F, G[:, [s]]], 1))
        lrt = -df * np.log(rss1 / rss0)
        np.testing.assert_allclose(lmm.getLRT()[s], lrt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(lmm.getPv()[s], st.chi2(1).sf(lrt),
                                   rtol=1e-5, atol=1e-12)


def test_pvalues_in_unit_interval_and_lrt_nonnegative():
    y, F, G, _ = _data(S=20)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert np.all(lmm.getPv() >= 0) and np.all(lmm.getPv() <= 1)
    assert np.all(lmm.getLRT() >= -1e-10)


def test_step_shapes_and_floor():
    y, F, G, _ = _data(S=5)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (2, 2)
    assert lmm.getLRT().shape == (2,)


def test_step_joint_test_matches_ols():
    y, F, G, _ = _data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G, step=3)
    _, rss0 = _rss(y, F)
    df = y.shape[0] - F.shape[1]
    for s in range(2):
        X = np.concatenate([F, G[:, 3 * s:3 * (s + 1)]], 1)
        beta, rss1 = _rss(y, X)
        np.testing.assert_allclose(lmm.getBetaSNP()[:, s], beta[F.shape[1]:],
                                   rtol=1e-5, atol=1e-8)
        lrt = -df * np.log(rss1 / rss0)
        np.testing.assert_allclose(lmm.getPv()[s], st.chi2(3).sf(lrt),
                                   rtol=1e-5, atol=1e-12)


def test_inter_shapes():
    y, F, G, rs = _data(S=6)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getBetaSNP().shape == (2, 6)
    assert lmm.getPv().shape == (6,)
    lmm.process(G, Inter=Inter, step=2)
    assert lmm.getBetaSNP().shape == (4, 3)
    assert lmm.getPv().shape == (3,)


def test_inter_of_ones_equals_no_inter():
    y, F, G, _ = _data()
    Inter = np.ones((y.shape[0], 1))
    a = LMMCore(y, F)
    a.process(G)
    b = LMMCore(y, F)
    b.process(G, Inter=Inter)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(a.getPv(), b.getPv(), rtol=1e-6, atol=1e-10)


def test_identity_Ki_dot_equals_none():
    y, F, G, _ = _data()
    a = LMMCore(y, F)
    a.process(G)
    b = LMMCore(y, F, lambda x: x)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(a.getLRT(), b.getLRT(), rtol=1e-8, atol=1e-10)


def test_scaled_Ki_dot_keeps_betas_and_lrt():
    y, F, G, _ = _data()
    a = LMMCore(y, F)
    a.process(G)
    b = LMMCore(y, F, lambda x: 2.0 * x)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(a.getLRT(), b.getLRT(), rtol=1e-6, atol=1e-9)


def test_beta_ste_positive_and_consistent_with_pv():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    assert np.all(np.isfinite(ste))
    assert np.all(ste > 0)
    z = np.sign(lmm.getBetaSNP()) * np.sqrt(st.chi2(1).isf(lmm.getPv()))
    np.testing.assert_allclose(ste, lmm.getBetaSNP() / z, rtol=1e-8)


def test_verbose_prints(capsys):
    y, F, G, _ = _data(S=4)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 4 variants" in out
    lmm.process(G, verbose=False)
    assert capsys.readouterr().out == ""
