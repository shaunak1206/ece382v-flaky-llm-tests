import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def lmmcore_module():
    import limix_lmm.lmm_core as lmm_core

    return lmm_core


@pytest.fixture
def fake_calc(monkeypatch):
    calls = []

    def _fake_calc(yKiy, FKiF, F1Kiy, df):
        p = FKiF.shape[0]
        # deterministic but simple outputs
        beta = np.arange(1, p + 1, dtype=float).reshape(p, 1)
        s2 = float(abs(np.sum(FKiF)) + abs(np.sum(F1Kiy)) + abs(yKiy) + df + 1.0)
        A0i = np.eye(p)
        calls.append((yKiy, FKiF.copy(), F1Kiy.copy(), df, beta.copy(), s2))
        return A0i, beta, s2

    monkeypatch.setattr("limix_lmm.lmm_core.calc_Ai_beta_s2", _fake_calc)
    return calls


def test_init_without_F_uses_intercept_and_fits_null(lmmcore_module, fake_calc):
    y = np.array([[1.0], [2.0], [3.0]])
    lmm = lmmcore_module.LMMCore(y, None, Ki_dot=None)

    assert lmm.F.shape == (3, 1)
    assert np.allclose(lmm.F, np.ones((3, 1)))
    assert lmm.df == 2
    assert lmm.Kiy is y
    assert lmm.KiF is lmm.F
    assert lmm.A0i.shape == (1, 1)
    assert lmm.beta_F0.shape == (1, 1)
    assert lmm.s20 > 0
    assert len(fake_calc) == 1


def test_init_with_Ki_dot_applies_it_to_y_and_F(lmmcore_module, fake_calc):
    y = np.array([[1.0], [2.0]])
    F = np.array([[1.0, 0.0], [0.0, 1.0]])

    def ki_dot(x):
        return 2.0 * x + 1.0

    lmm = lmmcore_module.LMMCore(y, F, Ki_dot=ki_dot)

    assert np.allclose(lmm.Kiy, 2.0 * y + 1.0)
    assert np.allclose(lmm.KiF, 2.0 * F + 1.0)
    assert lmm.FKiy.shape == (2, 1)
    assert lmm.FKiF.shape == (2, 2)
    assert len(fake_calc) == 1


def test_fit_null_without_Ki_dot_uses_raw_arrays(lmmcore_module, fake_calc):
    y = np.array([[1.0], [4.0]])
    F = np.array([[1.0], [2.0]])
    lmm = lmmcore_module.LMMCore(y, F, Ki_dot=None)

    assert np.allclose(lmm.Kiy, y)
    assert np.allclose(lmm.KiF, F)
    assert np.isclose(lmm.yKiy, float(y[:, 0] @ y[:, 0]))
    assert np.isclose(lmm.FKiy[0, 0], float(F[:, 0] @ y[:, 0]))
    assert lmm.beta_F0.shape == (1, 1)


def test_process_without_inter_and_step1_computes_expected_shapes(lmmcore_module, fake_calc):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.arange(8, dtype=float).reshape(4, 2)

    lmm = lmmcore_module.LMMCore(y, F)
    lmm.process(G, step=1)

    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (1, 2)
    assert lmm.getLRT().shape == (2,)
    assert np.all(np.isfinite(lmm.getPv()))
    assert np.all(np.isfinite(lmm.getBetaSNP()))
    assert np.all(np.isfinite(lmm.getLRT()))


def test_process_without_inter_step2_batches_variants(lmmcore_module, fake_calc):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.arange(8, dtype=float).reshape(4, 4)

    lmm = lmmcore_module.LMMCore(y, F)
    lmm.process(G, step=2)

    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (2, 2)
    assert lmm.getLRT().shape == (2,)
    assert np.all(np.isfinite(lmm.getBetaSNPste()))


def test_process_with_inter_step1_uses_elementwise_product(lmmcore_module, fake_calc):
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    G = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Inter = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])

    lmm = lmmcore_module.LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=1)

    assert lmm.getBetaSNP().shape == (2, 2)
    assert lmm.getPv().shape == (2,)
    assert np.all(lmm.getPv() >= 0.0)


def test_process_with_inter_step_gt1_uses_hatodot(lmmcore_module, fake_calc, monkeypatch):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.arange(8, dtype=float).reshape(4, 2)
    Inter = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])

    called = {}

    def fake_hatodot(A, B):
        called["args"] = (A.copy(), B.copy())
        # simple deterministic implementation
        cols = []
        for i in range(A.shape[1]):
            for j in range(B.shape[1]):
                cols.append((A[:, i] * B[:, j])[:, None])
        return np.hstack(cols)

    monkeypatch.setattr("limix_lmm.lmm_core.hatodot", fake_hatodot)

    lmm = lmmcore_module.LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)

    assert "args" in called
    assert called["args"][0].shape == (4, 2)
    assert called["args"][1].shape == (4, 2)
    assert lmm.getBetaSNP().shape == (4, 1)
    assert lmm.getPv().shape == (1,)


def test_process_verbose_prints_summary(lmmcore_module, fake_calc, capsys):
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    G = np.array([[1.0], [2.0], [3.0]])

    lmm = lmmcore_module.LMMCore(y, F)
    lmm.process(G, verbose=True)

    out = capsys.readouterr().out
    assert "Tested for 1 variants" in out


def test_getters_return_same_objects_after_process(lmmcore_module, fake_calc):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.arange(4, dtype=float).reshape(4, 1)

    lmm = lmmcore_module.LMMCore(y, F)
    lmm.process(G)

    assert lmm.getPv() is lmm.pv
    assert lmm.getBetaSNP() is lmm.beta_g
    assert lmm.getLRT() is lmm.lrt


def test_getBetaSNPste_matches_manual_formula(lmmcore_module, monkeypatch):
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    lmm = lmmcore_module.LMMCore(y, F)

    lmm.beta_g = np.array([[2.0, -4.0]])
    lmm.pv = np.array([0.5, 0.25])

    import scipy.stats as st

    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    expected = beta / z

    assert np.allclose(lmm.getBetaSNPste(), expected)


def test_process_with_nondivisible_number_of_variants_ignores_remainder(lmmcore_module, fake_calc):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.arange(12, dtype=float).reshape(4, 3)

    lmm = lmmcore_module.LMMCore(y, F)
    lmm.process(G, step=2)

    assert lmm.getPv().shape == (1,)
    assert lmm.getBetaSNP().shape == (2, 1)


def test_process_without_inter_and_with_Ki_dot_affects_results(lmmcore_module, fake_calc):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.arange(8, dtype=float).reshape(4, 2)

    def ki_dot(x):
        return x + 1.0

    lmm = lmmcore_module.LMMCore(y, F, Ki_dot=ki_dot)
    lmm.process(G)

    assert lmm.getPv().shape == (2,)
    assert lmm.beta_F0.shape == (1, 1)
    assert np.all(np.isfinite(lmm.getBetaSNPste()))


def test_package_exposes_expected_symbols_and_version():
    pkg = importlib.import_module("limix_lmm")

    assert pkg.__version__ == "0.1.2"
    assert "LMMCore" in pkg.__all__
    assert "MTLMM" in pkg.__all__
    assert pkg.LMMCore.__name__ == "LMMCore"
    assert hasattr(pkg, "download")
    assert hasattr(pkg, "unzip")
    assert hasattr(pkg, "test")


def test_package_LMMCore_is_same_as_module_class():
    import limix_lmm
    from limix_lmm.lmm_core import LMMCore

    assert limix_lmm.LMMCore is LMMCore


def test_package_MTLMM_import_is_available():
    import limix_lmm
    from limix_lmm.mtlmm import MTLMM

    assert limix_lmm.MTLMM is MTLMM
    assert limix_lmm.MTLMM.__name__ == "MTLMM"
