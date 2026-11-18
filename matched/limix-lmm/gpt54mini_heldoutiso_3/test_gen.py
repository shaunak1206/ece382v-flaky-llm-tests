import importlib
import sys
import types

import numpy as np
import pytest


def _fake_calc_Ai_beta_s2(yKiy, FKiF, FKiy, df):
    """Simple deterministic replacement for calc_Ai_beta_s2."""
    A = np.asarray(FKiF, dtype=float)
    b = np.asarray(FKiy, dtype=float)
    beta = np.linalg.solve(A, b)
    resid = float(yKiy) - float(b.T @ beta)
    s2 = abs(resid) / max(df, 1) + 1e-8
    A0i = np.linalg.inv(A)
    return A0i, beta, s2


def _make_lmmcore(monkeypatch, y=None, F=None, Ki_dot=None):
    import limix_lmm.lmm_core as lmm_core

    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)
    if y is None:
        y = np.array([[1.0], [2.0], [3.0], [4.0]])
    if F is None:
        F = np.column_stack([np.ones(y.shape[0]), np.arange(y.shape[0], dtype=float)])
    return lmm_core.LMMCore(y, F, Ki_dot=Ki_dot)


def test_init_default_f_and_df(monkeypatch):
    import limix_lmm.lmm_core as lmm_core

    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)
    y = np.array([[1.0], [2.0], [3.0]])
    core = lmm_core.LMMCore(y, None, Ki_dot=None)
    assert core.F.shape == (3, 1)
    assert np.allclose(core.F, np.ones((3, 1)))
    assert core.df == 2
    assert core.y is y


def test_init_with_ki_dot_calls_transform_and_fits_null(monkeypatch):
    import limix_lmm.lmm_core as lmm_core

    calls = []

    def ki_dot(x):
        calls.append(x.shape)
        return 2.0 * x

    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)
    y = np.array([[1.0], [0.0], [2.0]])
    F = np.column_stack([np.ones(3), np.array([0.0, 1.0, 2.0])])
    core = lmm_core.LMMCore(y, F, Ki_dot=ki_dot)

    assert calls == [(3, 1), (3, 2)]
    assert np.allclose(core.Kiy, 2.0 * y)
    assert np.allclose(core.KiF, 2.0 * F)
    assert core.A0i.shape == (2, 2)
    assert core.beta_F0.shape == (2, 1)
    assert core.s20 > 0


def test_fit_null_without_ki_dot_uses_identity(monkeypatch):
    core = _make_lmmcore(monkeypatch, Ki_dot=None)
    assert np.allclose(core.Kiy, core.y)
    assert np.allclose(core.KiF, core.F)
    assert np.allclose(core.FKiy, core.F.T @ core.y)
    assert np.allclose(core.FKiF, core.F.T @ core.F)


def test_process_without_inter_step1_shapes(monkeypatch):
    core = _make_lmmcore(monkeypatch)

    G = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    core.process(G, step=1)

    assert core.getPv().shape == (4,)
    assert core.getBetaSNP().shape == (1, 4)
    assert core.getLRT().shape == (4,)
    assert np.all(core.getPv() >= 0)
    assert np.all(np.isfinite(core.getLRT()))


def test_process_without_inter_step2_shapes(monkeypatch):
    core = _make_lmmcore(monkeypatch)

    G = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    core.process(G, step=2)

    assert core.getPv().shape == (2,)
    assert core.getBetaSNP().shape == (2, 2)
    assert core.getLRT().shape == (2,)


def test_process_with_inter_step1_shapes(monkeypatch):
    core = _make_lmmcore(monkeypatch)

    G = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    inter = np.column_stack([np.ones(4), np.array([1.0, 2.0, 3.0, 4.0])])
    core.process(G, Inter=inter, step=1)

    assert core.getPv().shape == (4,)
    assert core.getBetaSNP().shape == (2, 4)
    assert core.getLRT().shape == (4,)


def test_process_with_inter_step2_uses_hatodot(monkeypatch):
    import limix_lmm.lmm_core as lmm_core

    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)
    hatodot_calls = []

    def fake_hatodot(A, B):
        hatodot_calls.append((A.shape, B.shape))
        cols = []
        for i in range(A.shape[1]):
            for j in range(B.shape[1]):
                cols.append((A[:, i] * B[:, j])[:, None])
        return np.concatenate(cols, axis=1)

    monkeypatch.setattr(lmm_core, "hatodot", fake_hatodot)

    core = _make_lmmcore(monkeypatch)
    G = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    inter = np.column_stack([np.ones(4), np.array([1.0, 2.0, 3.0, 4.0])])

    core.process(G, Inter=inter, step=2)

    assert hatodot_calls == [((4, 2), (4, 2)), ((4, 2), (4, 2))]
    assert core.getBetaSNP().shape == (4, 2)


def test_process_verbose_prints(monkeypatch, capsys):
    core = _make_lmmcore(monkeypatch)
    G = np.array([[0.0], [1.0], [2.0], [3.0]])

    core.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 1 variants" in out


def test_getters_return_same_objects(monkeypatch):
    core = _make_lmmcore(monkeypatch)
    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    core.process(G)

    assert core.getPv() is core.pv
    assert core.getBetaSNP() is core.beta_g
    assert core.getLRT() is core.lrt


def test_get_beta_cov_is_null_covariate_beta(monkeypatch):
    core = _make_lmmcore(monkeypatch)
    assert core.getBetaCov() is core.beta_F0


def test_beta_snp_standard_errors_match_formula(monkeypatch):
    core = _make_lmmcore(monkeypatch)
    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    core.process(G)

    beta = core.getBetaSNP()
    pv = core.getPv()
    ste = core.getBetaSNPste()

    from scipy.stats import chi2

    expected = beta / (np.sign(beta) * np.sqrt(chi2(1).isf(pv)))
    assert np.allclose(ste, expected, equal_nan=True)


def test_process_different_inputs_change_results(monkeypatch):
    core = _make_lmmcore(monkeypatch)
    G1 = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    G2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

    core.process(G1)
    pv1 = core.getPv().copy()
    beta1 = core.getBetaSNP().copy()

    core.process(G2)
    pv2 = core.getPv().copy()
    beta2 = core.getBetaSNP().copy()

    assert not np.allclose(pv1, pv2)
    assert not np.allclose(beta1, beta2)


def test_package_exports_include_lmmcore_and_mtlmm():
    import limix_lmm

    assert "LMMCore" in limix_lmm.__all__
    assert "MTLMM" in limix_lmm.__all__
    assert limix_lmm.LMMCore.__name__ == "LMMCore"
    assert limix_lmm.MTLMM.__name__ == "MTLMM"


def test_package_version_and_test_callable():
    import limix_lmm

    assert limix_lmm.__version__ == "0.1.2"
    assert callable(limix_lmm.test)


def test_import_from_package_matches_direct_import():
    import limix_lmm
    from limix_lmm.lmm_core import LMMCore

    assert limix_lmm.LMMCore is LMMCore
