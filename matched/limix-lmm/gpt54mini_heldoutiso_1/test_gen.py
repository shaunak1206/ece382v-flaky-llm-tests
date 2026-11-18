import importlib
import sys
import types

import numpy as np
import pytest


def _install_fake_submodules(monkeypatch):
    """
    Ensure importing limix_lmm.__init__ does not fail if optional modules
    have heavy import-time dependencies in this environment.
    """
    fake_names = [
        "limix_lmm.plot",
        "limix_lmm.lmm",
        "limix_lmm.mtlmm",
        "limix_lmm._testit",
        "limix_lmm.sh",
    ]
    for name in fake_names:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if name.endswith(".lmm"):
                class LMM:  # noqa: N801
                    pass

                mod.LMM = LMM
            elif name.endswith(".mtlmm"):
                class MTLMM:  # noqa: N801
                    pass

                mod.MTLMM = MTLMM
            elif name.endswith("._testit"):
                mod.test = lambda: None
            elif name.endswith(".sh"):
                mod.download = lambda *args, **kwargs: None
                mod.unzip = lambda *args, **kwargs: None
            sys.modules[name] = mod


def _reload_package(monkeypatch):
    _install_fake_submodules(monkeypatch)
    if "limix_lmm" in sys.modules:
        del sys.modules["limix_lmm"]
    return importlib.import_module("limix_lmm")


def _reload_lmm_core():
    if "limix_lmm.lmm_core" in sys.modules:
        del sys.modules["limix_lmm.lmm_core"]
    return importlib.import_module("limix_lmm.lmm_core")


def _fake_calc_Ai_beta_s2_factory():
    calls = []

    def fake_calc_Ai_beta_s2(yKiy, FKiF, F1Kiy, df):
        n = FKiF.shape[0]
        beta = np.arange(1, n + 1, dtype=float).reshape(-1, 1)
        s2 = float(np.sum(FKiF) + np.sum(F1Kiy) + yKiy + df)
        calls.append((yKiy, FKiF.copy(), F1Kiy.copy(), df))
        return np.eye(n), beta, s2

    return fake_calc_Ai_beta_s2, calls


def test_init_without_F_uses_intercept_and_fits_null(monkeypatch):
    lmm_core = _reload_lmm_core()
    fake_calc, calls = _fake_calc_Ai_beta_s2_factory()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)

    y = np.array([[1.0], [2.0], [3.0]])
    model = lmm_core.LMMCore(y, None, Ki_dot=None)

    assert model.F.shape == (3, 1)
    assert np.allclose(model.F, np.ones((3, 1)))
    assert model.df == 2
    assert model.Kiy is y
    assert model.KiF is model.F
    assert model.FKiy.shape == (1, 1)
    assert model.FKiF.shape == (1, 1)
    assert model.A0i.shape == (1, 1)
    assert model.beta_F0.shape == (1, 1)
    assert model.s20 > 0
    assert len(calls) == 1


def test_fit_null_with_Ki_dot_is_used_for_y_and_F(monkeypatch):
    lmm_core = _reload_lmm_core()
    fake_calc, _ = _fake_calc_Ai_beta_s2_factory()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)

    y = np.array([[1.0], [0.0], [2.0]])
    F = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])

    def ki_dot(x):
        return 2.0 * x + 1.0

    model = lmm_core.LMMCore(y, F, Ki_dot=ki_dot)

    assert np.allclose(model.Kiy, 2.0 * y + 1.0)
    assert np.allclose(model.KiF, 2.0 * F + 1.0)
    assert model.FKiy.shape == (2, 1)
    assert model.FKiF.shape == (2, 2)


def test_process_without_interaction_and_without_Ki_dot(monkeypatch):
    lmm_core = _reload_lmm_core()
    fake_calc, calls = _fake_calc_Ai_beta_s2_factory()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)

    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    model = lmm_core.LMMCore(y, F, Ki_dot=None)

    G = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    model.process(G, step=1)

    assert model.beta_g.shape == (1, 2)
    assert model.pv.shape == (2,)
    assert model.lrt.shape == (2,)
    assert np.allclose(model.beta_g, np.array([[2.0, 2.0]]))
    assert np.all(model.pv >= 0.0)
    assert len(calls) == 3  # 1 null + 2 tests


def test_process_with_interaction_step1_uses_elementwise_product(monkeypatch):
    lmm_core = _reload_lmm_core()
    fake_calc, _ = _fake_calc_Ai_beta_s2_factory()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)

    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    model = lmm_core.LMMCore(y, F, Ki_dot=None)

    G = np.array([[1.0], [2.0], [3.0]])
    Inter = np.array([[10.0], [20.0], [30.0]])
    model.process(G, Inter=Inter, step=1)

    assert model.beta_g.shape == (1, 1)
    assert np.allclose(model.beta_g[:, 0], [2.0])


def test_process_with_interaction_step_gt1_uses_hatodot(monkeypatch):
    lmm_core = _reload_lmm_core()
    fake_calc, _ = _fake_calc_Ai_beta_s2_factory()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)

    hatodot_calls = []

    def fake_hatodot(A, B):
        hatodot_calls.append((A.copy(), B.copy()))
        return np.hstack([A * B[:, [0]], A * B[:, [1]]])

    monkeypatch.setattr(lmm_core, "hatodot", fake_hatodot)

    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    model = lmm_core.LMMCore(y, F, Ki_dot=None)

    G = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
    Inter = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]])
    model.process(G, Inter=Inter, step=2)

    assert len(hatodot_calls) == 1
    assert model.beta_g.shape == (4, 1)


def test_process_with_Ki_dot_applies_to_genotype_matrix(monkeypatch):
    lmm_core = _reload_lmm_core()
    fake_calc, _ = _fake_calc_Ai_beta_s2_factory()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)

    ki_calls = []

    def ki_dot(x):
        ki_calls.append(x.shape)
        return 3.0 * x

    y = np.array([[1.0], [2.0], [3.0]])
    F = np.ones((3, 1))
    model = lmm_core.LMMCore(y, F, Ki_dot=ki_dot)

    G = np.array([[1.0], [2.0], [3.0]])
    model.process(G, step=1)

    assert (3, 1) in ki_calls  # for y or F
    assert ki_calls.count((3, 1)) >= 3
    assert model.beta_g.shape == (1, 1)


def test_getters_return_values_after_process(monkeypatch):
    lmm_core = _reload_lmm_core()
    fake_calc, _ = _fake_calc_Ai_beta_s2_factory()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)

    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    model = lmm_core.LMMCore(y, F, Ki_dot=None)
    G = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    model.process(G, step=1)

    assert np.allclose(model.getPv(), model.pv)
    assert np.allclose(model.getBetaSNP(), model.beta_g)
    assert np.allclose(model.getLRT(), model.lrt)


def test_getBetaSNPste_matches_formula(monkeypatch):
    lmm_core = _reload_lmm_core()
    fake_calc, _ = _fake_calc_Ai_beta_s2_factory()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)

    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    model = lmm_core.LMMCore(y, F, Ki_dot=None)
    G = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    model.process(G, step=1)

    ste = model.getBetaSNPste()
    beta = model.getBetaSNP()
    pv = model.getPv()

    import scipy.stats as st

    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    expected = beta / z
    assert np.allclose(ste, expected, equal_nan=True)


def test_getBetaCov_returns_manual_attribute():
    lmm_core = _reload_lmm_core()
    model = object.__new__(lmm_core.LMMCore)
    model.beta_F = np.array([[1.5], [2.5]])
    assert np.allclose(model.getBetaCov(), np.array([[1.5], [2.5]]))


def test_process_verbose_prints_message(monkeypatch, capsys):
    lmm_core = _reload_lmm_core()
    fake_calc, _ = _fake_calc_Ai_beta_s2_factory()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)
    monkeypatch.setattr(lmm_core, "time", lambda: 0.0)

    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    model = lmm_core.LMMCore(y, F, Ki_dot=None)
    G = np.array([[1.0], [0.0], [1.0], [0.0]])
    model.process(G, step=1, verbose=True)

    out = capsys.readouterr().out
    assert "Tested for 1 variants" in out


def test_process_handles_non_divisible_number_of_variants_by_truncation(monkeypatch):
    lmm_core = _reload_lmm_core()
    fake_calc, _ = _fake_calc_Ai_beta_s2_factory()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)

    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    model = lmm_core.LMMCore(y, F, Ki_dot=None)

    G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    model.process(G, step=2)

    assert model.pv.shape == (1,)
    assert model.beta_g.shape == (2, 1)


def test_init_sets_degrees_of_freedom_correctly(monkeypatch):
    lmm_core = _reload_lmm_core()
    fake_calc, _ = _fake_calc_Ai_beta_s2_factory()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)

    y = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    F = np.ones((5, 2))
    model = lmm_core.LMMCore(y, F, Ki_dot=None)
    assert model.df == 3


def test_package_exports_include_LMMCore_MTLMM_and_version(monkeypatch):
    pkg = _reload_package(monkeypatch)

    assert hasattr(pkg, "LMMCore")
    assert hasattr(pkg, "MTLMM")
    assert hasattr(pkg, "__version__")
    assert pkg.__version__ == "0.1.2"
    assert "LMMCore" in pkg.__all__
    assert "MTLMM" in pkg.__all__
    assert "test" in pkg.__all__


def test_package_LMMCore_matches_direct_import(monkeypatch):
    pkg = _reload_package(monkeypatch)
    lmm_core = _reload_lmm_core()

    assert pkg.LMMCore is lmm_core.LMMCore


def test_package_MTLMM_is_importable(monkeypatch):
    pkg = _reload_package(monkeypatch)
    assert pkg.MTLMM is not None
