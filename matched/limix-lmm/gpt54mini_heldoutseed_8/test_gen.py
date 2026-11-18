import importlib
import sys

import numpy as np
import pytest


def _make_stub_calc():
    def stub_calc_Ai_beta_s2(yKiy, FKiF, FKiy, df):
        # Stable, deterministic stand-in for the internal quadratic-form solver.
        A = np.asarray(FKiF, dtype=float)
        b = np.asarray(FKiy, dtype=float)
        beta = np.linalg.pinv(A).dot(b)
        resid = float(yKiy - (b.T.dot(beta))[0, 0])
        s2 = max(resid / max(df, 1), 1e-12)
        A0i = np.linalg.pinv(A)
        return A0i, beta, s2

    return stub_calc_Ai_beta_s2


def _make_data(n=6, s=4, m=2, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randn(n, 1)
    F = np.column_stack([np.ones(n), rng.randn(n)])
    G = rng.randn(n, s)
    Inter = rng.randn(n, m)
    return y, F, G, Inter


def test_package_exports_expected_symbols():
    import limix_lmm

    assert limix_lmm.__version__ == "0.1.2"
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "test"]:
        assert name in limix_lmm.__all__
        assert hasattr(limix_lmm, name)


def test_package_imported_classes_match_submodules():
    import limix_lmm
    from limix_lmm.lmm_core import LMMCore as CoreClass

    assert limix_lmm.LMMCore is CoreClass


def test_lmmcore_initializes_with_default_intercept(monkeypatch):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    y = np.array([[1.0], [2.0], [3.0]])
    lmm = core.LMMCore(y, None, Ki_dot=None)

    assert lmm.F.shape == (3, 1)
    assert np.allclose(lmm.F, np.ones((3, 1)))
    assert lmm.df == 2
    assert lmm.Kiy is y
    assert lmm.KiF is lmm.F
    assert lmm.s20 > 0


def test_lmmcore_initializes_with_custom_F_and_no_Ki_dot(monkeypatch):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    y, F, _, _ = _make_data(n=5, s=2, m=2)
    lmm = core.LMMCore(y, F, Ki_dot=None)

    assert lmm.df == 3
    assert np.allclose(lmm.Kiy, y)
    assert np.allclose(lmm.KiF, F)
    assert lmm.beta_F0.shape == (F.shape[1], 1)
    assert lmm.A0i.shape == (F.shape[1], F.shape[1])


def test_lmmcore_initializes_with_Ki_dot_and_calls_it(monkeypatch):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    y, F, _, _ = _make_data(n=4, s=2, m=2)
    calls = []

    def ki_dot(x):
        calls.append(x.shape)
        return 2.0 * x

    lmm = core.LMMCore(y, F, Ki_dot=ki_dot)

    assert calls[:2] == [y.shape, F.shape]
    assert np.allclose(lmm.Kiy, 2.0 * y)
    assert np.allclose(lmm.KiF, 2.0 * F)


def test_process_without_intermediate_matrix(monkeypatch):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    y, F, G, _ = _make_data(n=6, s=4, m=2)
    lmm = core.LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=None, step=1)

    assert lmm.getPv().shape == (4,)
    assert lmm.getBetaSNP().shape == (1, 4)
    assert lmm.getLRT().shape == (4,)
    assert np.all(lmm.getPv() >= 0)
    assert np.all(np.isfinite(lmm.getBetaSNP()))


def test_process_with_intermediate_matrix_step_one(monkeypatch):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    y, F, G, Inter = _make_data(n=6, s=3, m=2)
    lmm = core.LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=Inter, step=1)

    assert lmm.getPv().shape == (3,)
    assert lmm.getBetaSNP().shape == (2, 3)
    assert np.all(np.isfinite(lmm.getBetaSNPste()))


def test_process_step_greater_than_one_uses_hatodot(monkeypatch):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    y, F, G, Inter = _make_data(n=5, s=4, m=2)
    calls = []

    def fake_hatodot(A, B):
        calls.append((A.shape, B.shape))
        cols = []
        for i in range(B.shape[1]):
            cols.append(A * B[:, [i]])
        return np.hstack(cols)

    monkeypatch.setattr(core, "hatodot", fake_hatodot)
    lmm = core.LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=Inter, step=2)

    assert calls == [((5, 2), (5, 2)), ((5, 2), (5, 2))]
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_process_with_step_one_and_no_inter_equals_genotype_columns(monkeypatch):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    y, F, G, _ = _make_data(n=6, s=2, m=2)
    lmm = core.LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=None, step=1)

    assert lmm.beta_g.shape == (1, 2)
    assert np.all(np.isfinite(lmm.beta_g))


def test_getters_return_same_objects_after_process(monkeypatch):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    y, F, G, _ = _make_data(n=6, s=3, m=2)
    lmm = core.LMMCore(y, F, Ki_dot=None)
    lmm.process(G)

    assert lmm.getPv() is lmm.pv
    assert lmm.getBetaSNP() is lmm.beta_g
    assert lmm.getLRT() is lmm.lrt


def test_get_betacov_is_missing_when_not_set(monkeypatch):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    y, F, _, _ = _make_data(n=5, s=2, m=2)
    lmm = core.LMMCore(y, F, Ki_dot=None)

    with pytest.raises(AttributeError):
        lmm.getBetaCov()


def test_get_beta_snp_standard_errors_shape_and_finiteness(monkeypatch):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    y, F, G, _ = _make_data(n=6, s=4, m=2)
    lmm = core.LMMCore(y, F, Ki_dot=None)
    lmm.process(G, step=1)
    ste = lmm.getBetaSNPste()

    assert ste.shape == lmm.getBetaSNP().shape
    assert np.all(np.isfinite(ste))


def test_process_verbose_prints_message(monkeypatch, capsys):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    monkeypatch.setattr(core, "time", lambda: 0.0)
    y, F, G, _ = _make_data(n=5, s=2, m=2)
    lmm = core.LMMCore(y, F, Ki_dot=None)
    lmm.process(G, verbose=True)

    out = capsys.readouterr().out
    assert "Tested for 2 variants" in out


def test_step_with_nondivisible_number_of_variants_uses_integer_quotient(monkeypatch):
    import limix_lmm.lmm_core as core

    monkeypatch.setattr(core, "calc_Ai_beta_s2", _make_stub_calc())
    y, F, G, _ = _make_data(n=6, s=5, m=2)
    lmm = core.LMMCore(y, F, Ki_dot=None)
    lmm.process(G, step=2)

    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (2, 2)


def test_package_can_be_reimported_after_clearing_modules(monkeypatch):
    import limix_lmm

    for name in ["limix_lmm", "limix_lmm.lmm_core"]:
        sys.modules.pop(name, None)
    mod = importlib.import_module("limix_lmm")
    assert mod.__version__ == "0.1.2"
    assert hasattr(mod, "LMMCore")
