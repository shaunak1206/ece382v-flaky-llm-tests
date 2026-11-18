import importlib
import sys
import types

import numpy as np
import pytest

from limix_lmm.lmm_core import LMMCore


def make_identity_core(n=4, p=1):
    y = np.arange(1, n + 1, dtype=float).reshape(n, 1)
    F = np.ones((n, p), dtype=float)
    return LMMCore(y, F, Ki_dot=lambda x: x)


def test_init_exports_expected_names():
    import limix_lmm

    assert limix_lmm.__version__ == "0.1.2"
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "test"]:
        assert name in limix_lmm.__all__


def test_init_can_import_with_stubbed_optional_submodules(monkeypatch):
    # Ensure package import works even if optional submodules are missing or heavy.
    dummy_plot = types.ModuleType("limix_lmm.plot")
    dummy_lmm = types.ModuleType("limix_lmm.lmm")
    dummy_lmm.LMM = object()
    dummy_mtlmm = types.ModuleType("limix_lmm.mtlmm")
    dummy_mtlmm.MTLMM = object()
    dummy_testit = types.ModuleType("limix_lmm._testit")
    dummy_testit.test = lambda: "ok"
    dummy_sh = types.ModuleType("limix_lmm.sh")
    dummy_sh.download = lambda *args, **kwargs: None
    dummy_sh.unzip = lambda *args, **kwargs: None

    for mod in [dummy_plot, dummy_lmm, dummy_mtlmm, dummy_testit, dummy_sh]:
        monkeypatch.setitem(sys.modules, mod.__name__, mod)

    pkg = importlib.reload(importlib.import_module("limix_lmm"))
    assert pkg.__version__ == "0.1.2"
    assert pkg.test() == "ok"


def test_lmmcore_uses_intercept_when_F_is_none():
    y = np.array([[1.0], [2.0], [3.0]])
    core = LMMCore(y, None, Ki_dot=lambda x: x)

    assert core.F.shape == (3, 1)
    assert np.allclose(core.F, np.ones((3, 1)))
    assert core.df == 2


def test_fit_null_with_identity_ki_matches_basic_products():
    core = make_identity_core(n=3, p=1)

    assert np.allclose(core.Kiy, core.y)
    assert np.allclose(core.KiF, core.F)
    assert np.allclose(core.FKiy, core.F.T @ core.y)
    assert np.allclose(core.FKiF, core.F.T @ core.F)
    assert np.isfinite(core.s20)
    assert core.A0i.shape == (1, 1)
    assert core.beta_F0.shape == (1, 1)


def test_process_without_interaction_produces_expected_shapes():
    core = make_identity_core(n=4, p=1)
    G = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    core.process(G, step=1)

    assert core.getPv().shape == (4,)
    assert core.getBetaSNP().shape == (1, 4)
    assert core.getLRT().shape == (4,)
    assert np.all(np.isfinite(core.getPv()))
    assert np.all((core.getPv() >= 0.0) & (core.getPv() <= 1.0))


def test_process_without_interaction_is_deterministic():
    core1 = make_identity_core(n=5, p=1)
    core2 = make_identity_core(n=5, p=1)
    G = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    core1.process(G, step=1)
    core2.process(G, step=1)

    assert np.allclose(core1.getPv(), core2.getPv())
    assert np.allclose(core1.getBetaSNP(), core2.getBetaSNP())
    assert np.allclose(core1.getLRT(), core2.getLRT())


def test_process_with_interaction_and_step_one_has_expected_beta_shape():
    core = make_identity_core(n=4, p=1)
    G = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    Inter = np.array([[1.0, 2.0], [0.5, 1.5], [2.0, 0.0], [1.0, 1.0]])

    core.process(G, Inter=Inter, step=1)

    assert core.getBetaSNP().shape == (2, 2)
    assert core.getPv().shape == (2,)
    assert np.all(np.isfinite(core.getBetaSNP()))


def test_process_with_interaction_and_step_two_has_expected_beta_shape():
    core = make_identity_core(n=4, p=1)
    G = np.array(
        [
            [1.0, 0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ]
    )
    Inter = np.array([[1.0, 2.0], [0.5, 1.5], [2.0, 0.0], [1.0, 1.0]])

    core.process(G, Inter=Inter, step=2)

    assert core.getBetaSNP().shape == (4, 2)
    assert core.getPv().shape == (2,)
    assert np.all(np.isfinite(core.getPv()))


def test_process_truncates_to_integer_number_of_tests():
    core = make_identity_core(n=4, p=1)
    G = np.array(
        [
            [1.0, 0.0, 2.0, 1.0, 3.0],
            [0.0, 1.0, 1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 1.0, 0.0, 2.0],
        ]
    )

    core.process(G, step=2)

    # floor(5 / 2) = 2 tests, the last column is ignored
    assert core.getPv().shape == (2,)
    assert core.getBetaSNP().shape == (2, 2)


def test_get_beta_snpste_has_same_shape_as_betas_and_is_finite():
    core = make_identity_core(n=5, p=1)
    G = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    core.process(G, step=1)
    ste = core.getBetaSNPste()

    assert ste.shape == core.getBetaSNP().shape
    assert np.all(np.isfinite(ste))


def test_get_beta_cov_raises_attribute_error_before_setting_beta_F():
    core = make_identity_core(n=4, p=1)
    with pytest.raises(AttributeError):
        core.getBetaCov()


def test_process_verbose_prints_expected_message(monkeypatch, capsys):
    core = make_identity_core(n=4, p=1)
    G = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]])

    times = iter([10.0, 12.5])
    monkeypatch.setattr("limix_lmm.lmm_core.time", lambda: next(times))

    core.process(G, step=1, verbose=True)
    out = capsys.readouterr().out

    assert "Tested for 2 variants in 2.50 s" in out


def test_custom_ki_dot_is_called_for_y_f_and_each_genotype_block():
    calls = []

    def ki_dot(x):
        calls.append(x.shape)
        return x

    y = np.arange(1, 5, dtype=float).reshape(4, 1)
    F = np.ones((4, 2), dtype=float)
    core = LMMCore(y, F, Ki_dot=ki_dot)

    G = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    core.process(G, step=1)

    assert calls[0] == (4, 1)  # y
    assert calls[1] == (4, 2)  # F
    assert calls[2] == (4, 1)  # first genotype block
    assert calls[3] == (4, 1)  # second genotype block


def test_lrt_and_pvalues_are_nonnegative_and_finite():
    core = make_identity_core(n=4, p=1)
    G = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    core.process(G, step=1)

    assert np.all(np.isfinite(core.getLRT()))
    assert np.all(core.getLRT() >= 0.0)
    assert np.all(np.isfinite(core.getPv()))
    assert np.all((core.getPv() >= 0.0) & (core.getPv() <= 1.0))
