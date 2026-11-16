import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm import LMMCore, MTLMM


def _data(N=60, S=8, k=2, seed=0):
    rs = np.random.RandomState(seed)
    F = np.concatenate([np.ones((N, 1)), rs.randn(N, k - 1)], 1)
    G = 1.0 * (rs.rand(N, S) < 0.3) + 0.0
    G = G + 0.1 * rs.randn(N, S)
    y = rs.randn(N, 1) + 0.5 * G[:, [0]]
    return y, F, G, rs


def _rss(y, D):
    b = np.linalg.lstsq(D, y[:, 0], rcond=None)[0]
    r = y[:, 0] - D.dot(b)
    return float(r.dot(r)), b


def test_package_exports():
    assert limix_lmm.LMMCore is LMMCore
    assert limix_lmm.MTLMM is MTLMM
    assert isinstance(limix_lmm.__version__, str)
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip",
                 "__version__", "test"]:
        assert name in limix_lmm.__all__
        assert hasattr(limix_lmm, name)


def test_init_default_covariates_is_ones():
    y, _, _, _ = _data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (y.shape[0], 1)
    assert np.all(lmm.F == 1)
    assert lmm.df == y.shape[0] - 1


def test_init_df_with_covariates():
    y, F, _, _ = _data(k=3)
    lmm = LMMCore(y, F)
    assert lmm.df == y.shape[0] - 3
    assert lmm.Ki_dot is None


def test_null_model_matches_ols():
    y, F, _, _ = _data()
    lmm = LMMCore(y, F)
    rss, b = _rss(y, F)
    np.testing.assert_allclose(np.ravel(lmm.beta_F0), b, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(lmm.s20, rss / lmm.df, rtol=1e-6)


def test_process_shapes_step1():
    y, F, G, _ = _data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (8,)
    assert lmm.getLRT().shape == (8,)
    assert lmm.getBetaSNP().shape == (1, 8)
    assert lmm.getBetaSNPste().shape == (1, 8)


def test_process_shapes_with_step_and_remainder():
    y, F, G, _ = _data(S=10)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (2,)
    assert lmm.getLRT().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_process_shapes_with_inter():
    y, F, G, rs = _data(S=6)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getPv().shape == (6,)
    assert lmm.getBetaSNP().shape == (2, 6)


def test_process_shapes_with_inter_and_step():
    y, F, G, rs = _data(S=8)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=4)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (8, 2)
    assert np.all(np.isfinite(lmm.getLRT()))


def test_beta_matches_ols_step1():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    for s in range(G.shape[1]):
        D = np.concatenate([F, G[:, [s]]], 1)
        _, b = _rss(y, D)
        np.testing.assert_allclose(beta[0, s], b[-1], rtol=1e-5, atol=1e-8)


def test_lrt_and_pv_match_ols():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    rss0, _ = _rss(y, F)
    expected = []
    for s in range(G.shape[1]):
        rss1, _ = _rss(y, np.concatenate([F, G[:, [s]]], 1))
        expected.append(-lmm.df * np.log(rss1 / rss0))
    expected = np.array(expected)
    np.testing.assert_allclose(lmm.getLRT(), expected, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(lmm.getPv(), st.chi2(1).sf(expected),
                               rtol=1e-5, atol=1e-12)


def test_interaction_step1_matches_ols():
    y, F, G, rs = _data(S=5)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    rss0, _ = _rss(y, F)
    for s in range(5):
        X = Inter * G[:, [s]]
        rss1, b = _rss(y, np.concatenate([F, X], 1))
        np.testing.assert_allclose(lmm.getBetaSNP()[:, s], b[F.shape[1]:],
                                   rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(lmm.getLRT()[s],
                                   -lmm.df * np.log(rss1 / rss0), rtol=1e-5)


def test_interaction_step_gt1_rss_invariant_to_column_order():
    y, F, G, rs = _data(S=6)
    Inter = rs.randn(y.shape[0], 2)
    step = 3
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=step)
    rss0, _ = _rss(y, F)
    for s in range(2):
        cols = []
        for j in range(step):
            for i in range(2):
                cols.append(Inter[:, i] * G[:, step * s + j])
        X = np.stack(cols, 1)
        rss1, _ = _rss(y, np.concatenate([F, X], 1))
        np.testing.assert_allclose(lmm.getLRT()[s],
                                   -lmm.df * np.log(rss1 / rss0), rtol=1e-5)
    np.testing.assert_allclose(lmm.getPv(),
                               st.chi2(6).sf(lmm.getLRT()), rtol=1e-5)


def test_identity_Ki_dot_equals_no_Ki_dot():
    y, F, G, _ = _data()
    a = LMMCore(y, F)
    b = LMMCore(y, F, lambda x: x)
    a.process(G)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-8)
    np.testing.assert_allclose(a.getLRT(), b.getLRT(), rtol=1e-8)
    np.testing.assert_allclose(a.getPv(), b.getPv(), rtol=1e-8)


def test_scaled_Ki_dot_leaves_beta_and_lrt_unchanged():
    y, F, G, _ = _data()
    a = LMMCore(y, F)
    b = LMMCore(y, F, lambda x: 2.0 * x)
    a.process(G)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-6)
    np.testing.assert_allclose(a.getLRT(), b.getLRT(), rtol=1e-6)


def test_pvalues_in_unit_interval_and_lrt_nonnegative():
    y, F, G, _ = _data(S=20, seed=3)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert np.all(lmm.getPv() >= 0) and np.all(lmm.getPv() <= 1)
    assert np.all(lmm.getLRT() >= -1e-8)


def test_beta_ste_consistent_with_beta_and_pv():
    y, F, G, _ = _data(S=6, seed=5)
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    ste = lmm.getBetaSNPste()
    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    np.testing.assert_allclose(ste, beta / z, rtol=1e-6)
    assert np.all(np.isfinite(ste))
    assert np.all(ste > 0)


def test_verbose_prints_message(capsys):
    y, F, G, _ = _data(S=4)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 4 variants" in out


def test_process_silent_by_default(capsys):
    y, F, G, _ = _data(S=4)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert capsys.readouterr().out == ""


def test_mtlmm_is_class():
    assert isinstance(MTLMM, type)
    assert MTLMM.__name__ == "MTLMM"
