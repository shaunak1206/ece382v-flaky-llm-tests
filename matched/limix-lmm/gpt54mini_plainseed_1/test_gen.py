import importlib
import sys
import types

import numpy as np
import pytest


def _fake_calc_Ai_beta_s2(yKiy, FKiF, FKiy, df):
    A0i = np.linalg.inv(FKiF)
    beta = A0i.dot(FKiy)
    s2 = float(yKiy) / max(df, 1)
    return A0i, beta, s2


def _install_init_stubs(monkeypatch):
    pkg_names = [
        "limix_lmm.plot",
        "limix_lmm.lmm",
        "limix_lmm.mtlmm",
        "limix_lmm._testit",
        "limix_lmm.sh",
    ]
    for name in pkg_names:
        mod = types.ModuleType(name)
        if name.endswith(".lmm"):
            class LMM:
                pass

            mod.LMM = LMM
        elif name.endswith(".mtlmm"):
            class MTLMM:
                pass

            mod.MTLMM = MTLMM
        elif name.endswith("._testit"):
            def test():
                return "ok"

            mod.test = test
        elif name.endswith(".sh"):
            def download(*args, **kwargs):
                return None

            def unzip(*args, **kwargs):
                return None

            mod.download = download
            mod.unzip = unzip
        sys.modules[name] = mod


def _fresh_lmm_core(monkeypatch):
    import limix_lmm.lmm_core as lmm_core

    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", _fake_calc_Ai_beta_s2)
    return lmm_core


def test_package_init_exports_expected_symbols(monkeypatch):
    _install_init_stubs(monkeypatch)
    sys.modules.pop("limix_lmm", None)
    pkg = importlib.import_module("limix_lmm")

    assert pkg.__version__ == "0.1.2"
    assert "LMMCore" in pkg.__all__
    assert "MTLMM" in pkg.__all__
    assert "LMM" in pkg.__all__
    assert hasattr(pkg, "LMMCore")
    assert hasattr(pkg, "MTLMM")
    assert hasattr(pkg, "download")
    assert hasattr(pkg, "unzip")


def test_package_init_reexports_test_function(monkeypatch):
    _install_init_stubs(monkeypatch)
    sys.modules.pop("limix_lmm", None)
    pkg = importlib.import_module("limix_lmm")

    assert callable(pkg.test)
    assert pkg.test() == "ok"


def test_lmmcore_defaults_to_intercept_when_F_is_none(monkeypatch):
    lmm_core = _fresh_lmm_core(monkeypatch)
    y = np.array([[1.0], [2.0], [3.0]])
    lmm = lmm_core.LMMCore(y, None, Ki_dot=None)

    assert lmm.F.shape == (3, 1)
    assert np.allclose(lmm.F, np.ones((3, 1)))
    assert lmm.df == 2
    assert lmm.Kiy is y
    assert lmm.KiF is lmm.F


def test_fit_null_with_Ki_dot_uses_transformed_arrays(monkeypatch):
    lmm_core = _fresh_lmm_core(monkeypatch)
    calls = []

    def ki_dot(x):
        calls.append(x.copy())
        return 2.0 * x

    y = np.array([[1.0], [0.0], [2.0]])
    F = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    lmm = lmm_core.LMMCore(y, F, Ki_dot=ki_dot)

    assert len(calls) == 2
    assert np.allclose(lmm.Kiy, 2.0 * y)
    assert np.allclose(lmm.KiF, 2.0 * F)
    assert lmm.A0i.shape == (2, 2)
    assert lmm.beta_F0.shape == (2, 1)


def test_process_without_interaction_and_no_Ki_dot(monkeypatch):
    lmm_core = _fresh_lmm_core(monkeypatch)
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
    lmm = lmm_core.LMMCore(y, F, Ki_dot=None)

    lmm.process(G, step=1)

    assert lmm.pv.shape == (2,)
    assert lmm.getPv() is lmm.pv
    assert lmm.getBetaSNP().shape == (1, 2)
    assert lmm.getLRT().shape == (2,)
    assert np.all(np.isfinite(lmm.getPv()))
    assert np.all(np.isfinite(lmm.getBetaSNP()))


def test_process_with_interaction_step_one(monkeypatch):
    lmm_core = _fresh_lmm_core(monkeypatch)
    y = np.array([[1.0], [2.0], [1.5], [3.0]])
    F = np.ones((4, 1))
    G = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    Inter = np.array([[2.0, 3.0], [1.0, 0.5], [0.0, 1.0], [1.0, 2.0]])
    lmm = lmm_core.LMMCore(y, F, Ki_dot=None)

    lmm.process(G, Inter=Inter, step=1)

    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (2, 2)
    assert lmm.lrt.shape == (2,)


def test_process_with_step_two_uses_hatodot(monkeypatch):
    lmm_core = _fresh_lmm_core(monkeypatch)
    hatodot_calls = []

    def fake_hatodot(A, B):
        hatodot_calls.append((A.copy(), B.copy()))
        n = A.shape[0]
        out = np.empty((n, A.shape[1] * B.shape[1]))
        c = 0
        for i in range(A.shape[1]):
            for j in range(B.shape[1]):
                out[:, c] = A[:, i] * B[:, j]
                c += 1
        return out

    monkeypatch.setattr(lmm_core, "hatodot", fake_hatodot)

    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0], [2.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    Inter = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    lmm = lmm_core.LMMCore(y, F, Ki_dot=None)

    lmm.process(G, Inter=Inter, step=2)

    assert len(hatodot_calls) == 2
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_process_with_Ki_dot_is_applied_to_G_and_interaction_terms(monkeypatch):
    lmm_core = _fresh_lmm_core(monkeypatch)
    calls = []

    def ki_dot(x):
        calls.append(x.copy())
        return 3.0 * x

    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    G = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    lmm = lmm_core.LMMCore(y, F, Ki_dot=ki_dot)

    lmm.process(G, step=1)

    assert len(calls) >= 4
    assert np.all(np.isfinite(lmm.getPv()))


def test_process_ignores_incomplete_final_block(monkeypatch):
    lmm_core = _fresh_lmm_core(monkeypatch)
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [2.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    lmm = lmm_core.LMMCore(y, F, Ki_dot=None)

    lmm.process(G, step=2)

    assert lmm.getPv().shape == (1,)
    assert lmm.getBetaSNP().shape == (2, 1)


def test_getBetaSNPste_returns_finite_values(monkeypatch):
    lmm_core = _fresh_lmm_core(monkeypatch)
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
    lmm = lmm_core.LMMCore(y, F, Ki_dot=None)
    lmm.process(G, step=1)

    ste = lmm.getBetaSNPste()
    assert ste.shape == lmm.getBetaSNP().shape
    assert np.all(np.isfinite(ste))


def test_getBetaCov_raises_before_beta_F_is_defined(monkeypatch):
    lmm_core = _fresh_lmm_core(monkeypatch)
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    lmm = lmm_core.LMMCore(y, F, Ki_dot=None)

    with pytest.raises(AttributeError):
        lmm.getBetaCov()


def test_verbose_process_prints_summary(monkeypatch, capsys):
    lmm_core = _fresh_lmm_core(monkeypatch)
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
    lmm = lmm_core.LMMCore(y, F, Ki_dot=None)

    lmm.process(G, step=1, verbose=True)
    captured = capsys.readouterr()

    assert "Tested for 2 variants" in captured.out


def test_getters_return_same_objects_after_process(monkeypatch):
    lmm_core = _fresh_lmm_core(monkeypatch)
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
    lmm = lmm_core.LMMCore(y, F, Ki_dot=None)
    lmm.process(G, step=1)

    assert lmm.getPv() is lmm.pv
    assert lmm.getBetaSNP() is lmm.beta_g
    assert lmm.getLRT() is lmm.lrt
