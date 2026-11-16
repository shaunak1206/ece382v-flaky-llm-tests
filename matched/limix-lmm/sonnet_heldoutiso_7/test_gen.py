import numpy as np
import pytest

import limix_lmm
from limix_lmm import LMMCore, MTLMM, LMM
from limix_lmm.lmm_core import LMMCore as LMMCoreDirect


def _data(N=60, S=8, k=2, seed=0):
    rs = np.random.RandomState(seed)
    y = rs.randn(N, 1)
    F = np.concatenate([np.ones((N, 1)), rs.randn(N, k - 1)], 1)
    G = 1.0 * (rs.rand(N, S) < 0.3) + 0.0 * rs.randn(N, S)
    G = G + 0.1 * rs.randn(N, S)
    return y, F, G, rs


def _ols(y, D):
    beta = np.linalg.solve(D.T.dot(D), D.T.dot(y))
    res = y - D.dot(beta)
    s2 = float(res.T.dot(res)) / (y.shape[0] - D.shape[1])
    return beta, s2


def test_package_exports():
    assert limix_lmm.LMMCore is LMMCoreDirect
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip",
                 "__version__", "test"]:
        assert name in limix_lmm.__all__
        assert hasattr(limix_lmm, name)


def test_version_string():
    assert limix_lmm.__version__ == "0.1.2"
    assert LMM is not None and MTLMM is not None


def test_init_default_covariates():
    y, _, _, _ = _data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (y.shape[0], 1)
    assert np.all(lmm.F == 1)
    assert lmm.df == y.shape[0] - 1


def test_init_df_with_covariates():
    y, F, _, _ = _data(N=50, k=3)
    lmm = LMMCore(y, F)
    assert lmm.df == 50 - 3
    assert lmm.Ki_dot is None


def test_null_model_matches_ols():
    y, F, _, _ = _data()
    lmm = LMMCore(y, F)
    beta, s2 = _ols(y, F)
    assert np.allclose(np.asarray(lmm.beta_F0).ravel(), beta.ravel())
    assert np.allclose(float(np.asarray(lmm.s20).squeeze()), s2)


def test_process_shapes_step1():
    y, F, G, _ = _data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (8,)
    assert lmm.getLRT().shape == (8,)
    assert lmm.getBetaSNP().shape == (1, 8)


def test_process_shapes_step_and_inter():
    y, F, G, rs = _data(S=10)
    lmm = LMMCore(y, F)
    lmm.process(G, step=3)
    assert lmm.getPv().shape == (3,)
    assert lmm.getBetaSNP().shape == (3, 3)

    Inter = rs.randn(y.shape[0], 2)
    lmm.process(G, Inter=Inter)
    assert lmm.getPv().shape == (10,)
    assert lmm.getBetaSNP().shape == (2, 10)

    lmm.process(G, Inter=Inter, step=2)
    assert lmm.getPv().shape == (5,)
    assert lmm.getBetaSNP().shape == (4, 5)


def test_beta_and_lrt_match_ols_no_inter():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    _, s20 = _ols(y, F)
    df = y.shape[0] - F.shape[1]
    for s in range(G.shape[1]):
        D = np.concatenate([F, G[:, s:s + 1]], 1)
        beta, s2 = _ols(y, D)
        assert np.allclose(lmm.getBetaSNP()[0, s], beta[-1, 0])
        assert np.allclose(lmm.getLRT()[s], -df * np.log(s2 / s20))


def test_pvalues_consistent_with_chi2():
    import scipy.stats as st
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    expected = st.chi2(1).sf(lmm.getLRT())
    assert np.allclose(lmm.getPv(), expected)
    assert np.all(lmm.getPv() >= 0) and np.all(lmm.getPv() <= 1)


def test_inter_step1_matches_ols():
    y, F, G, rs = _data()
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    for s in range(G.shape[1]):
        X = Inter * G[:, s:s + 1]
        beta, _ = _ols(y, np.concatenate([F, X], 1))
        assert np.allclose(lmm.getBetaSNP()[:, s], beta[F.shape[1]:, 0])


def test_step_joint_test_matches_ols():
    y, F, G, _ = _data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    for s in range(4):
        X = G[:, 2 * s:2 * s + 2]
        beta, _ = _ols(y, np.concatenate([F, X], 1))
        assert np.allclose(lmm.getBetaSNP()[:, s], beta[F.shape[1]:, 0])


def test_identity_Ki_dot_equals_none():
    y, F, G, _ = _data()
    a = LMMCore(y, F)
    b = LMMCore(y, F, lambda x: x.copy())
    a.process(G)
    b.process(G)
    assert np.allclose(a.getPv(), b.getPv())
    assert np.allclose(a.getBetaSNP(), b.getBetaSNP())
    assert np.allclose(a.getLRT(), b.getLRT())


def test_scaled_Ki_dot_invariance():
    # Scaling the covariance only rescales sigma^2, not betas or LRT
    y, F, G, _ = _data()
    a = LMMCore(y, F)
    b = LMMCore(y, F, lambda x: x / 2.0)
    a.process(G)
    b.process(G)
    assert np.allclose(a.getBetaSNP(), b.getBetaSNP())
    assert np.allclose(a.getLRT(), b.getLRT())
    assert np.allclose(a.getPv(), b.getPv())


def test_strong_signal_gives_small_pvalue():
    rs = np.random.RandomState(3)
    N = 200
    F = np.ones((N, 1))
    G = rs.randn(N, 5)
    y = 2.0 * G[:, [2]] + 0.1 * rs.randn(N, 1)
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    assert pv[2] < 1e-10
    assert np.argmin(pv) == 2
    assert abs(lmm.getBetaSNP()[0, 2] - 2.0) < 0.1


def test_beta_ste_shape_and_positive():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    assert ste.shape == lmm.getBetaSNP().shape
    assert np.all(np.isfinite(ste))
    assert np.all(ste > 0)


def test_beta_ste_consistent_with_z():
    import scipy.stats as st
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    ste = lmm.getBetaSNPste()
    z = beta / ste
    assert np.allclose(z ** 2, st.chi2(1).isf(lmm.getPv()))


def test_verbose_prints(capsys):
    y, F, G, _ = _data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 6 variants" in out


def test_quiet_by_default(capsys):
    y, F, G, _ = _data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert capsys.readouterr().out == ""


def test_getters_before_process_fail():
    y, F, _, _ = _data()
    lmm = LMMCore(y, F)
    with pytest.raises(AttributeError):
        lmm.getPv()
    with pytest.raises(AttributeError):
        lmm.getBetaSNP()
