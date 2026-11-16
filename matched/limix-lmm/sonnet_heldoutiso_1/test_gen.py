import numpy as np
import pytest

import limix_lmm
from limix_lmm import LMMCore, MTLMM
from limix_lmm import lmm_core


def _make_data(N=60, S=8, seed=0):
    rs = np.random.RandomState(seed)
    y = rs.randn(N, 1)
    F = np.concatenate([np.ones((N, 1)), rs.randn(N, 1)], axis=1)
    G = 1.0 * (rs.rand(N, S) < 0.3)
    return y, F, G, rs


# ---------------------------------------------------------------- __init__

def test_init_exports_and_version():
    assert isinstance(limix_lmm.__version__, str)
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip",
                 "__version__", "test"]:
        assert name in limix_lmm.__all__
        assert hasattr(limix_lmm, name)


def test_init_reexports_same_objects():
    assert limix_lmm.LMMCore is lmm_core.LMMCore
    assert isinstance(limix_lmm.MTLMM, type)
    assert MTLMM is limix_lmm.MTLMM
    assert callable(limix_lmm.test)


# ---------------------------------------------------------------- null model

def test_df_computed():
    y, F, _, _ = _make_data(N=50)
    lmm = LMMCore(y, F)
    assert lmm.df == 50 - 2


def test_F_none_defaults_to_ones():
    y, _, _, _ = _make_data(N=30)
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (30, 1)
    assert np.all(lmm.F == 1)
    assert lmm.df == 29


def test_null_beta_matches_ols():
    y, F, _, _ = _make_data()
    lmm = LMMCore(y, F)
    expected = np.linalg.lstsq(F, y, rcond=None)[0]
    np.testing.assert_allclose(np.ravel(lmm.beta_F0), np.ravel(expected),
                               rtol=1e-6, atol=1e-8)


def test_null_s2_matches_ols_residual_variance():
    y, F, _, _ = _make_data()
    lmm = LMMCore(y, F)
    beta = np.linalg.lstsq(F, y, rcond=None)[0]
    res = y - F.dot(beta)
    expected = float(res.T.dot(res)) / (y.shape[0] - F.shape[1])
    assert float(np.squeeze(lmm.s20)) == pytest.approx(expected, rel=1e-6)


def test_null_precomputed_quantities_without_Ki_dot():
    y, F, _, _ = _make_data()
    lmm = LMMCore(y, F)
    np.testing.assert_allclose(lmm.Kiy, y)
    np.testing.assert_allclose(lmm.KiF, F)
    np.testing.assert_allclose(lmm.FKiF, F.T.dot(F))
    assert float(lmm.yKiy) == pytest.approx(float(y[:, 0].dot(y[:, 0])))


# ---------------------------------------------------------------- process

def test_process_shapes_step1():
    y, F, G, _ = _make_data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (8,)
    assert lmm.getLRT().shape == (8,)
    assert lmm.getBetaSNP().shape == (1, 8)


def test_process_shapes_step4_and_remainder():
    y, F, G, _ = _make_data(S=10)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    # 10 / 4 -> 2 complete blocks
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)
    assert lmm.getLRT().shape == (2,)


def test_process_shapes_with_inter():
    y, F, G, rs = _make_data(S=6)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getBetaSNP().shape == (2, 6)
    assert lmm.getPv().shape == (6,)

    lmm.process(G, Inter=Inter, step=3)
    assert lmm.getBetaSNP().shape == (6, 2)
    assert lmm.getPv().shape == (2,)


def test_process_beta_matches_ols_full_model():
    y, F, G, _ = _make_data(S=5)
    lmm = LMMCore(y, F)
    lmm.process(G)
    for s in range(G.shape[1]):
        X = np.concatenate([F, G[:, [s]]], axis=1)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        assert lmm.getBetaSNP()[0, s] == pytest.approx(b[-1, 0], rel=1e-5,
                                                       abs=1e-8)


def test_process_lrt_and_pv_valid_ranges():
    y, F, G, _ = _make_data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert np.all(np.isfinite(lmm.getLRT()))
    assert np.all(lmm.getLRT() >= -1e-10)
    assert np.all(lmm.getPv() >= 0) and np.all(lmm.getPv() <= 1)


def test_process_detects_strong_effect():
    rs = np.random.RandomState(3)
    N = 200
    F = np.ones((N, 1))
    g = rs.randn(N, 1)
    noise = rs.randn(N, 1)
    y = 2.0 * g + 0.1 * noise
    G = np.concatenate([g, rs.randn(N, 1)], axis=1)
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    assert pv[0] < 1e-10
    assert pv[1] > pv[0]
    assert lmm.getBetaSNP()[0, 0] == pytest.approx(2.0, abs=0.1)


def test_inter_single_column_equals_prodcut_input():
    y, F, G, rs = _make_data(S=5)
    inter = rs.randn(y.shape[0], 1)
    lmm1 = LMMCore(y, F)
    lmm1.process(G, Inter=inter)
    lmm2 = LMMCore(y, F)
    lmm2.process(G * inter)
    np.testing.assert_allclose(lmm1.getBetaSNP(), lmm2.getBetaSNP())
    np.testing.assert_allclose(lmm1.getPv(), lmm2.getPv())


# ---------------------------------------------------------------- Ki_dot

def test_identity_Ki_dot_equals_none():
    y, F, G, _ = _make_data(S=6)
    a = LMMCore(y, F)
    a.process(G)
    b = LMMCore(y, F, lambda x: x)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP())
    np.testing.assert_allclose(a.getPv(), b.getPv())
    np.testing.assert_allclose(a.getLRT(), b.getLRT())


def test_scaled_Ki_dot_leaves_results_unchanged():
    # Only sigma^2 is learnt, so scaling K^-1 must not change betas/LRT.
    y, F, G, _ = _make_data(S=6)
    a = LMMCore(y, F)
    a.process(G)
    b = LMMCore(y, F, lambda x: 3.0 * x)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-6)
    np.testing.assert_allclose(a.getLRT(), b.getLRT(), rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------- getters

def test_getBetaSNPste_shape_sign_and_finite():
    y, F, G, _ = _make_data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    assert ste.shape == lmm.getBetaSNP().shape
    assert np.all(np.isfinite(ste))
    assert np.all(ste > 0)


def test_verbose_prints(capsys):
    y, F, G, _ = _make_data(S=4)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 4 variants" in out

    lmm.process(G, verbose=False)
    assert capsys.readouterr().out == ""
