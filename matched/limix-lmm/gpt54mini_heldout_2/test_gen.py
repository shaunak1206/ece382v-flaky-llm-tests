import importlib
import sys

import numpy as np
import pytest


def make_fake_calc():
    calls = {"n": 0}

    def fake_calc_Ai_beta_s2(yKiy, FKiF, FKiy, df):
        calls["n"] += 1
        n = FKiF.shape[0]
        # Make later calls (alternative models) have smaller residual variance
        # so LRT is positive and p-values are in (0, 1].
        s2 = 1.0 / (n + 1.0)
        beta = np.arange(n, dtype=float).reshape(n, 1)
        A0i = np.eye(n)
        return A0i, beta, s2

    return fake_calc_Ai_beta_s2, calls


@pytest.fixture
def lmm_core_module(monkeypatch):
    import limix_lmm.lmm_core as lmm_core

    fake_calc, calls = make_fake_calc()
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", fake_calc)
    monkeypatch.setattr(lmm_core, "hatodot", lambda A, B: np.hstack([A * B[:, [j]] for j in range(B.shape[1])]))
    return lmm_core, calls


def test_init_default_F_and_df(lmm_core_module):
    lmm_core, calls = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    model = lmm_core.LMMCore(y, None, Ki_dot=None)

    assert model.F.shape == (4, 1)
    assert np.all(model.F == 1.0)
    assert model.df == 3
    assert calls["n"] == 1
    assert model.Kiy is y
    assert model.KiF is model.F


def test_init_with_custom_F_and_no_Ki_dot(lmm_core_module):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [0.0], [2.0], [1.0]])
    F = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]])
    model = lmm_core.LMMCore(y, F)

    assert model.F is F
    assert model.Kiy is y
    assert model.KiF is F
    assert model.FKiy.shape == (2, 1)
    assert model.FKiF.shape == (2, 2)


def test_init_uses_Ki_dot(lmm_core_module):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[1.0], [2.0], [3.0]])

    def ki_dot(x):
        return 2.0 * x

    model = lmm_core.LMMCore(y, F, Ki_dot=ki_dot)

    assert np.allclose(model.Kiy, 2.0 * y)
    assert np.allclose(model.KiF, 2.0 * F)
    assert np.allclose(model.FKiy, 2.0 * (F.T @ y))
    assert np.allclose(model.FKiF, 2.0 * (F.T @ F))


def test_process_without_inter_and_step_one(lmm_core_module):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])

    model = lmm_core.LMMCore(y, F)
    model.process(G)

    assert model.pv.shape == (2,)
    assert model.beta_g.shape == (1, 2)
    assert np.all(model.pv >= 0.0)
    assert np.all(model.pv <= 1.0)
    assert np.all(model.lrt >= 0.0)


def test_process_with_inter_and_step_one(lmm_core_module):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
    Inter = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

    model = lmm_core.LMMCore(y, F)
    model.process(G, Inter=Inter, step=1)

    assert model.beta_g.shape == (2, 2)
    assert model.pv.shape == (2,)
    assert np.all(np.isfinite(model.beta_g))


def test_process_with_step_gt_one_uses_hatodot(lmm_core_module):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
    )
    Inter = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

    model = lmm_core.LMMCore(y, F)
    model.process(G, Inter=Inter, step=2)

    assert model.beta_g.shape == (4, 2)
    assert model.pv.shape == (2,)
    assert np.all(model.lrt >= 0.0)


def test_process_respects_Ki_dot_for_genotypes(lmm_core_module):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]])

    def ki_dot(x):
        return 3.0 * x

    model = lmm_core.LMMCore(y, F, Ki_dot=ki_dot)
    model.process(G)

    assert model.beta_g.shape == (1, 2)
    assert np.all(np.isfinite(model.pv))
    assert np.all(np.isfinite(model.lrt))


def test_getPv_and_getLRT_return_cached_arrays(lmm_core_module):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])

    model = lmm_core.LMMCore(y, F)
    model.process(G)

    assert model.getPv() is model.pv
    assert model.getLRT() is model.lrt


def test_getBetaSNP_returns_cached_beta_matrix(lmm_core_module):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])

    model = lmm_core.LMMCore(y, F)
    model.process(G)

    beta = model.getBetaSNP()
    assert beta is model.beta_g
    assert beta.shape == (1, 2)


def test_getBetaSNPste_computes_finite_values(lmm_core_module):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])

    model = lmm_core.LMMCore(y, F)
    model.process(G)

    ste = model.getBetaSNPste()
    assert ste.shape == model.beta_g.shape
    assert np.all(np.isfinite(ste))


def test_getBetaCov_raises_attribute_error_when_not_set(lmm_core_module):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    model = lmm_core.LMMCore(y, F)

    with pytest.raises(AttributeError):
        model.getBetaCov()


def test_process_with_verbose_prints_message(lmm_core_module, capsys):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])

    model = lmm_core.LMMCore(y, F)
    model.process(G, verbose=True)

    out = capsys.readouterr().out
    assert "Tested for 2 variants" in out


def test_process_with_non_multiple_of_step_ignores_remainder(lmm_core_module):
    lmm_core, _ = lmm_core_module
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [2.0, 1.0, 1.0],
        ]
    )

    model = lmm_core.LMMCore(y, F)
    model.process(G, step=2)

    assert model.pv.shape == (1,)
    assert model.beta_g.shape == (2, 1)


def test_import_package_exposes_expected_symbols():
    import limix_lmm

    assert limix_lmm.__version__ == "0.1.2"
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "test"]:
        assert hasattr(limix_lmm, name)
    assert "LMMCore" in limix_lmm.__all__


def test_from_package_import_lmmcore_works():
    from limix_lmm import LMMCore

    assert LMMCore.__name__ == "LMMCore"


def test_package_import_does_not_override_existing_module_state(monkeypatch):
    # Re-importing the package should still leave the public version constant intact.
    if "limix_lmm" in sys.modules:
        del sys.modules["limix_lmm"]
    mod = importlib.import_module("limix_lmm")
    assert mod.__version__ == "0.1.2"
