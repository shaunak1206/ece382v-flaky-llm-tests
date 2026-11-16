import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm.lmm_core import LMMCore


def _data(N=60, S=8, k=2, seed=0):
    rs = np.random.RandomState(seed)
    y = rs.randn(N, 1)
    F = np.concatenate([np.ones((N, 1)), rs.randn(N, k - 1)], 1)
    G = 1.0 * (rs.rand(N, S) < 0.3) + 1.0 * (rs.rand(N, S) < 0.3)
    return y, F, G, rs


def _ols(y, D):
    b = np.linalg.lstsq(D, y[:, 0], rcond=None)[0]
    r = y[:, 0] - D.dot(b)
    return b, float(r.dot(r))


def _wls(y, D, w):
    sw = np.sqrt(w)
    b = np.linalg.lstsq(D * sw[:, None], y[:, 0] * sw, rcond=None)[0]
    r = y[:, 0] - D.dot(b)
    return b, float((w * r * r).sum())


def test_package_exports():
    from limix_lmm.lmm_core import LMMCore as Core
    from limix_lmm.mtlmm import MTLMM as MT

    assert limix_lmm.LMMCore is Core
    assert limix_lmm.MTLMM is MT
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip",
                 "__version__", "test"]:
        assert name in limix_lmm.__all__
        assert hasattr(limix_lmm, name)


def test_version():
    assert limix_lmm.__version__ == "0.1.2"


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


def test_null_model_without_kidot_uses_inputs():
    y, F, _, _ = _data()
    lmm = LMMCore(y, F)
    assert np.allclose(lmm.Kiy, y)
    assert np.allclose(lmm.KiF, F)
    assert np.allclose(lmm.yKiy, y[:, 0].dot(y[:, 0]))
    assert np.allclose(lmm.FKiF, F.T.dot(F))


def test_null_model_matches_ols():
    y, F, _, _ = _data()
    lmm = LMMCore(y, F)
    b, rss = _ols(y, F)
    assert np.allclose(np.ravel(lmm.beta_F0), b)
    assert np.allclose(lmm.s20, rss / lmm.df)


def test_kidot_is_called_on_y_and_F():
    y, F, _, _ = _data()
    calls = []

    def ki(x):
        calls.append(x.shape)
        return x

    LMMCore(y, F, ki)
    assert y.shape in calls
    assert F.shape in calls


def test_process_output_shapes_no_inter():
    y, F, G, _ = _data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (8,)
    assert lmm.getLRT().shape == (8,)
    assert lmm.getBetaSNP().shape == (1, 8)
    assert lmm.getBetaSNPste().shape == (1, 8)


def test_process_matches_ols_no_inter():
    y, F, G, _ = _data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    _, rss0 = _ols(y, F)
    for s in range(G.shape[1]):
        D = np.concatenate([F, G[:, [s]]], 1)
        b, rss1 = _ols(y, D)
        assert np.allclose(lmm.getBetaSNP()[0, s], b[-1])
        lrt = -lmm.df * np.log(rss1 / rss0)
        assert np.allclose(lmm.getLRT()[s], lrt)


def test_pvalues_consistent_with_lrt():
    y, F, G, _ = _data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    assert np.all((pv >= 0) & (pv <= 1))
    assert np.allclose(pv, st.chi2(1).sf(lmm.getLRT()))


def test_process_with_inter_step1():
    y, F, G, rs = _data(S=5)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getPv().shape == (5,)
    assert lmm.getBetaSNP().shape == (2, 5)
    _, rss0 = _ols(y, F)
    for s in range(5):
        X = Inter * G[:, [s]]
        b, rss1 = _ols(y, np.concatenate([F, X], 1))
        assert np.allclose(lmm.getBetaSNP()[:, s], b[2:])
        assert np.allclose(lmm.getLRT()[s], -lmm.df * np.log(rss1 / rss0))
    assert np.allclose(lmm.getPv(), st.chi2(2).sf(lmm.getLRT()))


def test_process_step_shapes():
    y, F, G, _ = _data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)
    assert np.allclose(lmm.getPv(), st.chi2(4).sf(lmm.getLRT()))
    _, rss0 = _ols(y, F)
    _, rss1 = _ols(y, np.concatenate([F, G[:, :4]], 1))
    assert np.allclose(lmm.getLRT()[0], -lmm.df * np.log(rss1 / rss0))


def test_process_step_with_inter_shapes():
    y, F, G, rs = _data(S=8)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=4)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (8, 2)
    assert np.allclose(lmm.getPv(), st.chi2(8).sf(lmm.getLRT()))


def test_scaled_covariance_gives_same_results():
    y, F, G, _ = _data(S=5)
    base = LMMCore(y, F)
    base.process(G)
    scaled = LMMCore(y, F, lambda x: x / 3.0)
    scaled.process(G)
    assert np.allclose(base.getBetaSNP(), scaled.getBetaSNP())
    assert np.allclose(base.getLRT(), scaled.getLRT())
    assert np.allclose(base.getPv(), scaled.getPv())


def test_diagonal_covariance_matches_weighted_least_squares():
    y, F, G, rs = _data(S=4)
    w = rs.uniform(0.5, 2.0, size=y.shape[0])
    lmm = LMMCore(y, F, lambda x: w[:, None] * x)
    lmm.process(G)
    _, rss0 = _wls(y, F, w)
    for s in range(4):
        D = np.concatenate([F, G[:, [s]]], 1)
        b, rss1 = _wls(y, D, w)
        assert np.allclose(lmm.getBetaSNP()[0, s], b[-1])
        assert np.allclose(lmm.getLRT()[s], -lmm.df * np.log(rss1 / rss0))


def test_standard_errors_match_formula_for_single_variant():
    y, F, G, _ = _data(S=5)
    lmm = LMMCore(y, F)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    expected = np.abs(lmm.getBetaSNP()) / np.sqrt(lmm.getLRT())
    assert np.allclose(np.abs(ste), expected, rtol=1e-4)
    assert np.all(ste > 0) or np.all(np.isfinite(ste))


def test_detects_planted_signal():
    rs = np.random.RandomState(3)
    N = 300
    G = 1.0 * (rs.rand(N, 5) < 0.4)
    y = (2.0 * G[:, [2]] + 0.5 * rs.randn(N, 1))
    lmm = LMMCore(y, None)
    lmm.process(G)
    pv = lmm.getPv()
    assert np.argmin(pv) == 2
    assert pv[2] < 1e-10
    assert lmm.getBetaSNP()[0, 2] == pytest.approx(2.0, abs=0.3)


def test_verbose_prints_message(capsys):
    y, F, G, _ = _data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 6 variants" in out
    lmm.process(G, verbose=False)
    assert capsys.readouterr().out == ""
