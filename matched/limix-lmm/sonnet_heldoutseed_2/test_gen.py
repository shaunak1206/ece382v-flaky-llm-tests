import numpy as np
import pytest

import limix_lmm
from limix_lmm import LMMCore, MTLMM
from limix_lmm.lmm_core import LMMCore as LMMCoreDirect


def _data(N=60, S=8, seed=0):
    rs = np.random.RandomState(seed)
    y = rs.randn(N, 1)
    F = np.concatenate([np.ones((N, 1)), rs.randn(N, 1)], 1)
    G = 1.0 * (rs.rand(N, S) < 0.3) + rs.randn(N, S) * 0.1
    return y, F, G, rs


def _ols(y, F, X):
    A = np.concatenate([F, X], 1)
    coef = np.linalg.lstsq(A, y[:, 0], rcond=None)[0]
    res = y[:, 0] - A.dot(coef)
    return coef, float(res.dot(res))


def test_package_exports():
    assert LMMCore is LMMCoreDirect
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip",
                 "__version__", "test"]:
        assert name in limix_lmm.__all__
        assert hasattr(limix_lmm, name)
    assert limix_lmm.__version__ == "0.1.2"


def test_mtlmm_is_exposed_class():
    assert isinstance(MTLMM, type)
    assert limix_lmm.MTLMM is MTLMM


def test_init_df_and_attributes():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    assert lmm.df == y.shape[0] - F.shape[1]
    assert lmm.Ki_dot is None
    assert lmm.F is F
    assert lmm.y is y


def test_init_F_none_uses_intercept():
    y, _, _, _ = _data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (y.shape[0], 1)
    assert np.allclose(lmm.F, 1.0)
    assert lmm.df == y.shape[0] - 1


def test_null_model_matches_ols():
    y, F, _, _ = _data()
    lmm = LMMCore(y, F)
    coef, rss = _ols(y, F, np.zeros((y.shape[0], 0)))
    assert np.allclose(np.ravel(lmm.beta_F0), coef)
    assert np.allclose(lmm.s20, rss / lmm.df)


def test_process_output_shapes_step1():
    y, F, G, _ = _data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (8,)
    assert lmm.getLRT().shape == (8,)
    assert lmm.getBetaSNP().shape == (1, 8)
    assert lmm.getBetaSNPste().shape == (1, 8)


def test_pvalues_valid_range():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    assert np.all(pv >= 0) and np.all(pv <= 1)
    assert np.all(np.isfinite(pv))


def test_beta_and_lrt_match_ols_step1():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    _, rss0 = _ols(y, F, np.zeros((y.shape[0], 0)))
    for s in range(G.shape[1]):
        coef, rss1 = _ols(y, F, G[:, [s]])
        assert np.allclose(lmm.getBetaSNP()[0, s], coef[-1])
        expected_lrt = -lmm.df * np.log(rss1 / rss0)
        assert np.allclose(lmm.getLRT()[s], expected_lrt)


def test_step_two_joint_test_matches_ols():
    y, F, G, _ = _data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    assert lmm.getPv().shape == (4,)
    assert lmm.getBetaSNP().shape == (2, 4)
    _, rss0 = _ols(y, F, np.zeros((y.shape[0], 0)))
    coef, rss1 = _ols(y, F, G[:, 2:4])
    assert np.allclose(lmm.getBetaSNP()[:, 1], coef[-2:])
    assert np.allclose(lmm.getLRT()[1], -lmm.df * np.log(rss1 / rss0))


def test_step_truncates_remainder_variants():
    y, F, G, _ = _data(S=9)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_interaction_step1_matches_ols():
    y, F, G, rs = _data(S=6)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getBetaSNP().shape == (2, 6)
    assert lmm.getPv().shape == (6,)
    for s in [0, 3]:
        X = Inter * G[:, [s]]
        coef, _ = _ols(y, F, X)
        assert np.allclose(lmm.getBetaSNP()[:, s], coef[-2:])


def test_interaction_with_step_shapes():
    y, F, G, rs = _data(S=8)
    Inter = rs.randn(y.shape[0], 2)
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
    assert np.allclose(a.getPv(), b.getPv())
    assert np.allclose(a.getBetaSNP(), b.getBetaSNP())
    assert np.allclose(a.getLRT(), b.getLRT())


def test_scaled_Ki_dot_keeps_betas_and_lrt():
    y, F, G, _ = _data()
    a = LMMCore(y, F)
    b = LMMCore(y, F, lambda x: x / 2.0)
    a.process(G)
    b.process(G)
    assert np.allclose(a.getBetaSNP(), b.getBetaSNP())
    assert np.allclose(a.getLRT(), b.getLRT())
    assert np.allclose(a.getPv(), b.getPv())


def test_beta_ste_consistent_with_lrt():
    y, F, G, _ = _data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    beta = lmm.getBetaSNP()
    # for a single-df test z^2 == lrt (chi2 with 1 dof)
    z = beta / ste
    assert np.allclose(z ** 2, lmm.getLRT(), rtol=1e-4, atol=1e-8)
    assert np.all(ste > 0)


def test_strong_signal_gives_small_pvalue():
    rs = np.random.RandomState(3)
    N = 200
    g = rs.randn(N, 1)
    y = 2.0 * g + 0.1 * rs.randn(N, 1)
    F = np.ones((N, 1))
    G = np.concatenate([g, rs.randn(N, 1)], 1)
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    assert pv[0] < 1e-10
    assert pv[1] > pv[0]
    assert abs(lmm.getBetaSNP()[0, 0] - 2.0) < 0.1


def test_verbose_prints_message(capsys):
    y, F, G, _ = _data(S=4)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 4 variants" in out


def test_quiet_by_default(capsys):
    y, F, G, _ = _data(S=4)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert capsys.readouterr().out == ""
