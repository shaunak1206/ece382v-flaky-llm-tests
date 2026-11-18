import importlib
import sys

import numpy as np
import pytest


@pytest.fixture
def lmm_core_module():
    # Ensure a fresh import for each test so monkeypatching is isolated.
    if "limix_lmm.lmm_core" in sys.modules:
        del sys.modules["limix_lmm.lmm_core"]
    import limix_lmm.lmm_core as lmm_core

    return lmm_core


@pytest.fixture
def patched_core(monkeypatch, lmm_core_module):
    def fake_calc_Ai_beta_s2(yKiy, FKiF, FKiy, df):
        n = FKiF.shape[0]
        A0i = np.eye(n)
        beta = np.arange(n, dtype=float).reshape(n, 1) + 1.0
        s2 = 1.0 / float(n)
        return A0i, beta, s2

    monkeypatch.setattr(lmm_core_module, "calc_Ai_beta_s2", fake_calc_Ai_beta_s2)
    return lmm_core_module


def test_init_sets_default_F_to_ones_and_computes_null(patched_core):
    y = np.array([[1.0], [2.0], [3.0]])
    lmm = patched_core.LMMCore(y, None, Ki_dot=None)

    assert lmm.F.shape == (3, 1)
    assert np.allclose(lmm.F, np.ones((3, 1)))
    assert lmm.df == 2
    assert lmm.A0i.shape == (1, 1)
    assert lmm.beta_F0.shape == (1, 1)
    assert lmm.s20 == pytest.approx(1.0)


def test_init_uses_provided_F_and_stores_inputs(patched_core):
    y = np.array([[1.0], [0.0], [2.0], [1.0]])
    F = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

    lmm = patched_core.LMMCore(y, F)

    assert lmm.y is y
    assert lmm.F is F
    assert lmm.df == 2
    assert lmm.FKiy.shape == (2, 1)
    assert lmm.FKiF.shape == (2, 2)


def test_fit_null_without_ki_dot_uses_raw_y_and_F(patched_core):
    y = np.array([[1.0], [2.0]])
    F = np.array([[1.0], [3.0]])
    lmm = patched_core.LMMCore(y, F, Ki_dot=None)

    assert np.allclose(lmm.Kiy, y)
    assert np.allclose(lmm.KiF, F)
    assert np.allclose(lmm.FKiy, F.T @ y)
    assert np.allclose(lmm.FKiF, F.T @ F)


def test_fit_null_with_ki_dot_applies_operator_to_y_and_F(patched_core):
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[1.0], [2.0], [4.0]])

    calls = []

    def ki_dot(x):
        calls.append(x.copy())
        return 2.0 * x

    lmm = patched_core.LMMCore(y, F, Ki_dot=ki_dot)

    assert len(calls) == 2
    assert np.allclose(lmm.Kiy, 2.0 * y)
    assert np.allclose(lmm.KiF, 2.0 * F)
    assert np.allclose(lmm.FKiy, F.T @ (2.0 * y))
    assert np.allclose(lmm.FKiF, F.T @ (2.0 * F))


def test_process_without_inter_step_one_sets_shapes_and_outputs(patched_core):
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[1.0], [1.0], [1.0]])
    G = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 0.0], [2.0, 1.0, 0.0, 1.0]])

    lmm = patched_core.LMMCore(y, F)
    lmm.process(G, step=1)

    assert lmm.pv.shape == (4,)
    assert lmm.lrt.shape == (4,)
    assert lmm.beta_g.shape == (1, 4)
    assert np.all(np.isfinite(lmm.pv))
    assert np.all(np.isfinite(lmm.lrt))


def test_process_without_inter_step_two_groups_variants(patched_core):
    y = np.array([[1.0], [0.0], [2.0], [1.0]])
    F = np.array([[1.0], [1.0], [1.0], [1.0]])
    G = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )

    lmm = patched_core.LMMCore(y, F)
    lmm.process(G, step=2)

    assert lmm.pv.shape == (2,)
    assert lmm.beta_g.shape == (2, 2)
    assert np.all(np.isfinite(lmm.beta_g))


def test_process_with_inter_step_one_uses_elementwise_product(patched_core, monkeypatch):
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[1.0], [1.0], [1.0]])
    G = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0]])
    Inter = np.array([[10.0], [20.0], [30.0]])

    lmm = patched_core.LMMCore(y, F)

    captured = {}

    def fake_calc_Ai_beta_s2(yKiy, FKiF, FKiy, df):
        captured["FKiy"] = FKiy.copy()
        captured["FKiF"] = FKiF.copy()
        n = FKiF.shape[0]
        return np.eye(n), np.arange(n, dtype=float).reshape(n, 1), 1.0 / n

    monkeypatch.setattr(patched_core, "calc_Ai_beta_s2", fake_calc_Ai_beta_s2)
    lmm.process(G, Inter=Inter, step=1)

    expected_x0 = Inter * G[:, 0:1]
    assert np.allclose(captured["FKiy"][1:, 0], expected_x0.T @ lmm.Kiy[:, 0])
    assert lmm.beta_g.shape == (1, 2)


def test_process_with_inter_step_gt_one_uses_hatodot(patched_core, monkeypatch):
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[1.0], [1.0], [1.0]])
    G = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0]])
    Inter = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

    called = {}

    def fake_hatodot(A, B):
        called["args"] = (A.copy(), B.copy())
        return np.full((A.shape[0], A.shape[1] * B.shape[1]), 7.0)

    monkeypatch.setattr(patched_core, "hatodot", fake_hatodot)

    lmm = patched_core.LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)

    assert "args" in called
    assert np.array_equal(called["args"][0], Inter)
    assert np.array_equal(called["args"][1], G[:, 0:2])
    assert lmm.beta_g.shape == (4, 1)


def test_getters_return_same_objects_after_process(patched_core):
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[1.0], [1.0], [1.0]])
    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])

    lmm = patched_core.LMMCore(y, F)
    lmm.process(G, step=1)

    assert lmm.getPv() is lmm.pv
    assert lmm.getBetaSNP() is lmm.beta_g
    assert lmm.getLRT() is lmm.lrt


def test_get_beta_snp_ste_computes_finite_values(patched_core):
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[1.0], [1.0], [1.0]])
    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])

    lmm = patched_core.LMMCore(y, F)
    lmm.process(G, step=1)

    ste = lmm.getBetaSNPste()
    assert ste.shape == lmm.beta_g.shape
    assert np.all(np.isfinite(ste))


def test_get_beta_cov_raises_attribute_error_due_to_missing_assignment(patched_core):
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[1.0], [1.0], [1.0]])
    lmm = patched_core.LMMCore(y, F)

    with pytest.raises(AttributeError):
        _ = lmm.getBetaCov()


def test_process_verbose_prints_summary(patched_core, capsys):
    y = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[1.0], [1.0], [1.0]])
    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])

    lmm = patched_core.LMMCore(y, F)
    lmm.process(G, step=1, verbose=True)

    out = capsys.readouterr().out
    assert "Tested for 2 variants" in out


def test_package_exports_expected_symbols():
    import limix_lmm

    assert limix_lmm.__version__ == "0.1.2"
    assert "LMMCore" in limix_lmm.__all__
    assert "MTLMM" in limix_lmm.__all__
    assert limix_lmm.LMMCore.__name__ == "LMMCore"
    assert limix_lmm.MTLMM.__name__ == "MTLMM"


def test_package_imports_are_available():
    import limix_lmm

    assert hasattr(limix_lmm, "plot")
    assert hasattr(limix_lmm, "download")
    assert hasattr(limix_lmm, "unzip")
    assert callable(limix_lmm.test)


def test_lmmcore_can_be_imported_from_package():
    from limix_lmm import LMMCore

    assert LMMCore.__name__ == "LMMCore"
