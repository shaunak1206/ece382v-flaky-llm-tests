import importlib
import io
import sys
from contextlib import redirect_stdout

import numpy as np
import pytest

import limix_lmm
import limix_lmm.lmm_core as lmm_core


def _stub_calc_Ai_beta_s2(yKiy, FKiF, F1Kiy, df):
    """
    Deterministic stub:
    - returns an identity-like A0i
    - returns beta equal to F1Kiy flattened
    - returns s2 as the mean of absolute beta plus a small df-based offset
    """
    beta = np.array(F1Kiy, dtype=float, copy=True)
    if beta.ndim == 1:
        beta = beta[:, None]
    a0i = np.eye(FKiF.shape[0])
    s2 = float(np.mean(np.abs(beta)) + 0.1 * df + 1.0)
    return a0i, beta, s2


def _make_core(monkeypatch, n=4, p=2, use_ki=True):
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", _stub_calc_Ai_beta_s2)
    y = np.arange(1, n + 1, dtype=float).reshape(-1, 1)
    F = np.column_stack([np.ones(n), np.linspace(0.0, 1.0, n)])
    if use_ki:
        Ki_dot = lambda x: 2.0 * np.asarray(x, dtype=float)
    else:
        Ki_dot = None
    return lmm_core.LMMCore(y, F, Ki_dot=Ki_dot)


def test_init_default_F_sets_intercept_and_df(monkeypatch):
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", _stub_calc_Ai_beta_s2)
    y = np.arange(1, 5, dtype=float).reshape(-1, 1)
    core = lmm_core.LMMCore(y, None, Ki_dot=None)
    assert core.F.shape == (4, 1)
    assert np.allclose(core.F, np.ones((4, 1)))
    assert core.df == 3
    assert core.Kiy is y
    assert core.KiF is core.F


def test_init_with_kidot_computes_transformed_null_terms(monkeypatch):
    core = _make_core(monkeypatch, n=4, use_ki=True)
    assert np.allclose(core.Kiy, 2.0 * core.y)
    assert np.allclose(core.KiF, 2.0 * core.F)
    assert core.FKiy.shape == (2, 1)
    assert core.FKiF.shape == (2, 2)
    assert core.yKiy == pytest.approx(float(np.dot(core.y[:, 0], core.Kiy[:, 0])))


def test_process_without_interaction_uses_genotypes_directly(monkeypatch):
    core = _make_core(monkeypatch, n=4, use_ki=True)
    G = np.arange(12, dtype=float).reshape(4, 3)
    core.process(G, step=1)
    assert core.beta_g.shape == (1, 3)
    assert core.pv.shape == (3,)
    assert core.lrt.shape == (3,)
    assert np.all(core.pv >= 0.0)
    assert np.all(core.pv <= 1.0)


def test_process_with_interaction_step1_multiplies_inter_and_g(monkeypatch):
    core = _make_core(monkeypatch, n=4, use_ki=False)
    G = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    inter = np.array([[10.0], [20.0], [30.0], [40.0]])
    core.process(G, Inter=inter, step=1)
    assert core.beta_g.shape == (1, 2)
    assert core.pv.shape == (2,)
    assert np.isfinite(core.getLRT()).all()


def test_process_with_interaction_step_gt1_uses_hatodot(monkeypatch):
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", _stub_calc_Ai_beta_s2)

    called = {"args": None}

    def fake_hatodot(A, B):
        called["args"] = (A.copy(), B.copy())
        # simple but shape-correct interaction expansion: concatenate elementwise products
        out = []
        for j in range(B.shape[1]):
            out.append(A * B[:, [j]])
        return np.concatenate(out, axis=1)

    monkeypatch.setattr(lmm_core, "hatodot", fake_hatodot)

    y = np.arange(1, 5, dtype=float).reshape(-1, 1)
    F = np.ones((4, 1))
    core = lmm_core.LMMCore(y, F, Ki_dot=None)

    G = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    inter = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]])

    core.process(G, Inter=inter, step=2)
    assert called["args"] is not None
    assert called["args"][0].shape == inter.shape
    assert called["args"][1].shape == (4, 2)
    assert core.beta_g.shape == (4, 2)
    assert core.pv.shape == (2,)


def test_process_ignores_remainder_variants_when_not_divisible_by_step(monkeypatch):
    core = _make_core(monkeypatch, n=4, use_ki=True)
    G = np.arange(16, dtype=float).reshape(4, 4)
    core.process(G, step=3)
    # int(4/3) == 1, so only one test is run
    assert core.beta_g.shape == (3, 1)
    assert core.pv.shape == (1,)


def test_getters_return_processed_attributes(monkeypatch):
    core = _make_core(monkeypatch, n=4, use_ki=True)
    G = np.arange(8, dtype=float).reshape(4, 2)
    core.process(G, step=1)
    assert core.getPv() is core.pv
    assert core.getBetaSNP() is core.beta_g
    assert core.getLRT() is core.lrt


def test_get_beta_cov_before_set_raises_attribute_error(monkeypatch):
    core = _make_core(monkeypatch, n=4, use_ki=True)
    with pytest.raises(AttributeError):
        core.getBetaCov()


def test_process_verbose_prints_summary(monkeypatch):
    core = _make_core(monkeypatch, n=4, use_ki=True)
    G = np.arange(8, dtype=float).reshape(4, 2)
    buf = io.StringIO()
    with redirect_stdout(buf):
        core.process(G, step=1, verbose=True)
    out = buf.getvalue()
    assert "Tested for 2 variants" in out


def test_get_beta_snpste_matches_shape_and_is_finite(monkeypatch):
    core = _make_core(monkeypatch, n=4, use_ki=True)
    G = np.arange(12, dtype=float).reshape(4, 3)
    core.process(G, step=1)
    ste = core.getBetaSNPste()
    assert ste.shape == core.getBetaSNP().shape
    assert np.isfinite(ste).all()


def test_get_beta_snpste_sign_consistency(monkeypatch):
    core = _make_core(monkeypatch, n=4, use_ki=True)
    G = np.arange(8, dtype=float).reshape(4, 2)
    core.process(G, step=1)
    beta = core.getBetaSNP()
    ste = core.getBetaSNPste()
    # For non-zero beta, ste should have the same sign as beta because z has sign(beta)
    nz = beta != 0
    assert np.all(np.sign(ste[nz]) == np.sign(beta[nz]))


def test_pvalues_and_lrt_are_consistent_with_stubbed_s2(monkeypatch):
    core = _make_core(monkeypatch, n=4, use_ki=True)
    G = np.arange(8, dtype=float).reshape(4, 2)
    core.process(G, step=1)
    assert np.all(core.lrt >= 0.0)
    assert np.all(core.pv <= 1.0)
    assert np.all(core.pv >= 0.0)


def test_init_without_kidot_uses_raw_inputs(monkeypatch):
    monkeypatch.setattr(lmm_core, "calc_Ai_beta_s2", _stub_calc_Ai_beta_s2)
    y = np.arange(1, 5, dtype=float).reshape(-1, 1)
    F = np.column_stack([np.ones(4), np.arange(4, dtype=float)])
    core = lmm_core.LMMCore(y, F, Ki_dot=None)
    assert np.allclose(core.Kiy, y)
    assert np.allclose(core.KiF, F)
    assert core.A0i.shape == (2, 2)


def test_init_and_process_do_not_mutate_input_arrays(monkeypatch):
    core = _make_core(monkeypatch, n=4, use_ki=True)
    G = np.arange(8, dtype=float).reshape(4, 2)
    G_copy = G.copy()
    F_copy = core.F.copy()
    y_copy = core.y.copy()
    core.process(G, step=1)
    assert np.array_equal(G, G_copy)
    assert np.array_equal(core.F, F_copy)
    assert np.array_equal(core.y, y_copy)


def test_package_exports_expected_symbols():
    expected = {"LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "__version__", "test"}
    assert expected.issubset(set(limix_lmm.__all__))
    assert limix_lmm.__version__ == "0.1.2"
    assert limix_lmm.LMMCore is lmm_core.LMMCore
    assert limix_lmm.MTLMM.__name__ == "MTLMM"


def test_package_imported_modules_are_accessible():
    # __init__ should expose imported submodules/objects
    assert hasattr(limix_lmm, "plot")
    assert hasattr(limix_lmm, "download")
    assert hasattr(limix_lmm, "unzip")
    assert callable(limix_lmm.test)


def test_reload_package_keeps_version_and_exports():
    mod = importlib.reload(limix_lmm)
    assert mod.__version__ == "0.1.2"
    assert "LMMCore" in mod.__all__
    assert mod.LMMCore is lmm_core.LMMCore
