import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm import LMMCore, MTLMM
from limix_lmm.lmm_core import LMMCore as LMMCoreDirect


def _data(N=60, S=8, seed=0):
    rs = np.random.RandomState(seed)
    y = rs.randn(N, 1)
    F = np.concatenate([np.ones((N, 1)), rs.randn(N, 1)], axis=1)
    G = 1.0 * (rs.rand(N, S) < 0.3) + 0.1 * rs.randn(N, S)
    return y, F, G, rs


def _ols(y, D):
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    r = y - D.dot(beta)
    return beta, float(r.T.dot(r))


def test_package_exports():
    assert limix_lmm.LMMCore is LMMCoreDirect
    assert limix_lmm.MTLMM is MTLMM
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip",
                 "__version__", "test"]:
        assert name in limix_lmm.__all__
        assert hasattr(limix_lmm, name)


def test_version_string():
    assert limix_lmm.__version__ == "0.1.2"


def test_init_df_and_null_attributes():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    assert lmm.df == y.shape[0] - F.shape[1]
    assert lmm.Kiy.shape == y.shape
    assert lmm.KiF.shape == F.shape
    assert lmm.FKiF.shape == (2, 2)


def test_init_F_none_uses_intercept():
    y, F, G, _ = _data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (y.shape[0], 1)
    assert np.all(lmm.F == 1)
    assert lmm.df == y.shape[0] - 1


def test_null_model_matches_ols():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    beta, rss = _ols(y, F)
    np.testing.assert_allclose(np.asarray(lmm.beta_F0).ravel(), beta.ravel(), rtol=1e-6)
    np.testing.assert_allclose(float(np.squeeze(lmm.s20)), rss / lmm.df, rtol=1e-6)


def test_process_output_shapes_step1():
    y, F, G, _ = _data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (8,)
    assert lmm.getLRT().shape == (8,)
    assert lmm.getBetaSNP().shape == (1, 8)
    assert lmm.getBetaSNPste().shape == (1, 8)


def test_process_matches_ols_single_variants():
    y, F, G, _ = _data(S=5)
    lmm = LMMCore(y, F)
    lmm.process(G)
    _, rss0 = _ols(y, F)
    for s in range(G.shape[1]):
        D = np.concatenate([F, G[:, [s]]], axis=1)
        beta, rss1 = _ols(y, D)
        np.testing.assert_allclose(lmm.getBetaSNP()[0, s], beta[-1, 0], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(lmm.getLRT()[s], -lmm.df * np.log(rss1 / rss0),
                                   rtol=1e-5, atol=1e-8)


def test_pvalues_consistent_with_lrt_chi2():
    y, F, G, _ = _data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    assert np.all(pv >= 0) and np.all(pv <= 1)
    np.testing.assert_allclose(pv, st.chi2(1).sf(lmm.getLRT()), rtol=1e-8)


def test_step_greater_than_one_shapes_and_truncation():
    y, F, G, _ = _data(S=9)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    # int(9 / 4) == 2 complete blocks
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)
    assert lmm.getLRT().shape == (2,)


def test_step_joint_test_matches_ols():
    y, F, G, _ = _data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G, step=3)
    _, rss0 = _ols(y, F)
    for s in range(2):
        D = np.concatenate([F, G[:, 3 * s:3 * (s + 1)]], axis=1)
        beta, rss1 = _ols(y, D)
        np.testing.assert_allclose(lmm.getBetaSNP()[:, s], beta[2:, 0], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(lmm.getLRT()[s], -lmm.df * np.log(rss1 / rss0),
                                   rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(lmm.getPv()[s], st.chi2(3).sf(lmm.getLRT()[s]),
                                   rtol=1e-6)


def test_interaction_step1_matches_ols():
    y, F, G, rs = _data(S=4)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getBetaSNP().shape == (2, 4)
    assert lmm.getPv().shape == (4,)
    _, rss0 = _ols(y, F)
    for s in range(4):
        X = Inter * G[:, [s]]
        D = np.concatenate([F, X], axis=1)
        beta, rss1 = _ols(y, D)
        np.testing.assert_allclose(lmm.getBetaSNP()[:, s], beta[2:, 0], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(lmm.getLRT()[s], -lmm.df * np.log(rss1 / rss0),
                                   rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(lmm.getPv()[s], st.chi2(2).sf(lmm.getLRT()[s]),
                                   rtol=1e-6)


def test_interaction_with_step_shapes():
    y, F, G, rs = _data(S=8)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)
    assert lmm.getBetaSNP().shape == (4, 4)
    assert lmm.getPv().shape == (4,)
    assert lmm.getLRT().shape == (4,)


def test_identity_Ki_dot_equals_no_Ki_dot():
    y, F, G, _ = _data(S=5)
    a = LMMCore(y, F)
    b = LMMCore(y, F, Ki_dot=lambda x: x.copy())
    a.process(G)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-8)
    np.testing.assert_allclose(a.getPv(), b.getPv(), rtol=1e-8)
    np.testing.assert_allclose(a.getLRT(), b.getLRT(), rtol=1e-8, atol=1e-10)


def test_scaled_Ki_dot_keeps_betas_and_tests():
    y, F, G, _ = _data(S=5)
    a = LMMCore(y, F)
    b = LMMCore(y, F, Ki_dot=lambda x: 0.5 * x)
    a.process(G)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-6)
    np.testing.assert_allclose(a.getLRT(), b.getLRT(), rtol=1e-6, atol=1e-9)


def test_beta_ste_is_positive_and_finite():
    y, F, G, _ = _data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    assert np.all(np.isfinite(ste))
    assert np.all(ste > 0)


def test_verbose_prints_summary(capsys):
    y, F, G, _ = _data(S=3)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 3 variants" in out


def test_getters_before_process_raise():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    with pytest.raises(AttributeError):
        lmm.getPv()
    with pytest.raises(AttributeError):
        lmm.getBetaSNP()
    with pytest.raises(AttributeError):
        lmm.getLRT()
