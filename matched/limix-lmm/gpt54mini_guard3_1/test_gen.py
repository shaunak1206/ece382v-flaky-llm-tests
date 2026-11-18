import importlib
import sys

import numpy as np
import pytest

import limix_lmm.lmm_core as lmm_core_mod


def _fake_calc_Ai_beta_s2(yKiy, FKiF, FKiy, df):
    """
    Deterministic helper for tests.

    Returns:
      A0i: identity
      beta: least-squares-like vector shaped (p, 1)
      s2: positive scalar depending on inputs
    """
    import numpy as _np

    p = FKiF.shape[0]
    A0i = _np.eye(p)
    # deterministic, simple, and shape-correct
    beta = _np.arange(1, p + 1, dtype=float).reshape(p, 1)
    s2 = float(yKiy + _np.sum(FKiF) + _np.sum(FKiy) + df + 1.0)
    return A0i, beta, s2


def _fake_hatodot(A, B):
    import numpy as _np

    cols = []
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            cols.append((A[:, i] * B[:, j]).reshape(-1, 1))
    return _np.concatenate(cols, axis=1)


def test_init_with_none_F_uses_intercept_and_fits_null(monkeypatch):
    monkeypatch.setattr(lmm_core_mod, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)
    y = np.array([[1.0], [2.0], [3.0]])
    lmm = lmm_core_mod.LMMCore(y, None, Ki_dot=None)

    assert lmm.F.shape == (3, 1)
    assert np.allclose(lmm.F, np.ones((3, 1)))
    assert lmm.df == 2
    assert np.allclose(lmm.Kiy, y)
    assert np.allclose(lmm.KiF, lmm.F)
    assert lmm.A0i.shape == (1, 1)
    assert lmm.beta_F0.shape == (1, 1)
    assert lmm.s20 > 0


def test_fit_null_uses_Ki_dot_when_provided(monkeypatch):
    monkeypatch.setattr(lmm_core_mod, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)

    calls = []

    def ki_dot(x):
        calls.append(x.copy())
        return 2.0 * x

    y = np.array([[1.0], [0.0], [2.0]])
    F = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    lmm = lmm_core_mod.LMMCore(y, F, Ki_dot=ki_dot)

    assert len(calls) == 2
    assert np.allclose(lmm.Kiy, 2.0 * y)
    assert np.allclose(lmm.KiF, 2.0 * F)
    assert np.allclose(lmm.FKiy, F.T @ (2.0 * y))
    assert np.allclose(lmm.FKiF, F.T @ (2.0 * F))


def test_process_without_inter_and_step1_shapes(monkeypatch):
    monkeypatch.setattr(lmm_core_mod, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)

    y = np.array([[1.0], [2.0], [4.0]])
    F = np.array([[1.0], [1.0], [1.0]])
    lmm = lmm_core_mod.LMMCore(y, F, Ki_dot=None)

    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    lmm.process(G, step=1)

    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (1, 2)
    assert lmm.getLRT().shape == (2,)
    assert np.all(np.isfinite(lmm.getPv()))
    assert np.all(np.isfinite(lmm.getLRT()))


def test_process_without_inter_step2_uses_joint_blocks(monkeypatch):
    monkeypatch.setattr(lmm_core_mod, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)

    y = np.array([[1.0], [1.0], [2.0], [3.0]])
    F = np.ones((4, 1))
    lmm = lmm_core_mod.LMMCore(y, F, Ki_dot=None)

    G = np.array(
        [
            [1.0, 0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 2.0],
            [2.0, 0.0, 1.0, 1.0],
        ]
    )
    lmm.process(G, step=2)

    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (2, 2)


def test_process_with_inter_and_step1_uses_elementwise_product(monkeypatch):
    monkeypatch.setattr(lmm_core_mod, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)

    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    lmm = lmm_core_mod.LMMCore(y, F, Ki_dot=None)

    G = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Inter = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    lmm.process(G, Inter=Inter, step=1)

    assert lmm.getBetaSNP().shape == (2, 2)
    assert lmm.getPv().shape == (2,)


def test_process_with_inter_and_step_gt1_uses_hatodot(monkeypatch):
    monkeypatch.setattr(lmm_core_mod, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)
    monkeypatch.setattr(lmm_core_mod, "hatodot", _fake_hatodot)

    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    lmm = lmm_core_mod.LMMCore(y, F, Ki_dot=None)

    G = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Inter = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    lmm.process(G, Inter=Inter, step=2)

    assert lmm.getBetaSNP().shape == (4, 1)
    assert lmm.getPv().shape == (1,)
    assert np.all(np.isfinite(lmm.getBetaSNP()))


def test_getters_return_same_objects_after_process(monkeypatch):
    monkeypatch.setattr(lmm_core_mod, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)

    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    lmm = lmm_core_mod.LMMCore(y, F, Ki_dot=None)
    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    lmm.process(G)

    assert lmm.getPv() is lmm.pv
    assert lmm.getBetaSNP() is lmm.beta_g
    assert lmm.getLRT() is lmm.lrt


def test_getBetaCov_returns_beta_F0_alias(monkeypatch):
    monkeypatch.setattr(lmm_core_mod, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)

    y = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    lmm = lmm_core_mod.LMMCore(y, F, Ki_dot=None)

    assert lmm.getBetaCov() is lmm.beta_F0
    assert np.allclose(lmm.getBetaCov(), np.array([[1.0], [2.0]]))


def test_getBetaSNPste_matches_formula(monkeypatch):
    monkeypatch.setattr(lmm_core_mod, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)

    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    lmm = lmm_core_mod.LMMCore(y, F, Ki_dot=None)

    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    lmm.process(G)

    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    ste = lmm.getBetaSNPste()

    import scipy.stats as st

    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    expected = beta / z
    assert np.allclose(ste, expected, equal_nan=True)


def test_process_verbose_prints_summary(monkeypatch, capsys):
    monkeypatch.setattr(lmm_core_mod, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)
    monkeypatch.setattr(lmm_core_mod, "time", lambda: 10.0)

    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    lmm = lmm_core_mod.LMMCore(y, F, Ki_dot=None)

    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    lmm.process(G, verbose=True)

    out = capsys.readouterr().out
    assert "Tested for 2 variants in" in out
    assert "s" in out


def test_process_with_step_zero_variants_sets_empty_outputs(monkeypatch):
    monkeypatch.setattr(lmm_core_mod, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)

    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    lmm = lmm_core_mod.LMMCore(y, F, Ki_dot=None)

    G = np.empty((3, 0))
    lmm.process(G, step=1)

    assert lmm.getPv().shape == (0,)
    assert lmm.getBetaSNP().shape == (1, 0)
    assert lmm.getLRT().shape == (0,)


def test_package_exports_and_version():
    pkg = importlib.import_module("limix_lmm")

    assert pkg.__version__ == "0.1.2"
    assert "LMMCore" in pkg.__all__
    assert "MTLMM" in pkg.__all__
    assert pkg.LMMCore is lmm_core_mod.LMMCore
    assert hasattr(pkg, "MTLMM")
    assert hasattr(pkg, "LMM")


def test_init_module_imports_required_names():
    pkg = importlib.import_module("limix_lmm")

    for name in ["plot", "download", "unzip", "test", "LMMCore", "MTLMM"]:
        assert hasattr(pkg, name)


def test_mtlmm_can_be_imported_from_package():
    pkg = importlib.import_module("limix_lmm")
    assert pkg.MTLMM is getattr(pkg, "MTLMM")
    assert pkg.MTLMM.__name__ == "MTLMM"
