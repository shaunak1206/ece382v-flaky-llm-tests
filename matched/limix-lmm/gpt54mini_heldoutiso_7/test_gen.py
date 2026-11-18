import importlib
import sys
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def lmm_core_module():
    return importlib.import_module("limix_lmm.lmm_core")


@pytest.fixture
def simple_data():
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    return y, F


def test_package_exports_expected_names():
    pkg = importlib.import_module("limix_lmm")
    assert pkg.__version__ == "0.1.2"
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "test"]:
        assert hasattr(pkg, name), name
    assert isinstance(pkg.__all__, list)
    assert "LMMCore" in pkg.__all__
    assert "MTLMM" in pkg.__all__


def test_package_lmmcore_is_same_class_as_module():
    pkg = importlib.import_module("limix_lmm")
    mod = importlib.import_module("limix_lmm.lmm_core")
    assert pkg.LMMCore is mod.LMMCore


def test_package_mtlmm_is_importable_and_exposed():
    pkg = importlib.import_module("limix_lmm")
    assert hasattr(pkg, "MTLMM")
    # We only verify exposure here because the implementation lives elsewhere.
    assert pkg.MTLMM is not None


def test_lmmcore_initializes_with_default_f(simple_data, lmm_core_module):
    y, _ = simple_data
    core = lmm_core_module.LMMCore(y, None, Ki_dot=None)
    assert core.F.shape == (4, 1)
    assert np.allclose(core.F, np.ones((4, 1)))
    assert core.df == 3
    assert core.Ki_dot is None
    assert core.Kiy.shape == y.shape
    assert core.KiF.shape == core.F.shape


def test_lmmcore_initializes_and_fits_null_without_ki_dot(simple_data, lmm_core_module):
    y, F = simple_data
    core = lmm_core_module.LMMCore(y, F, Ki_dot=None)
    assert np.allclose(core.Kiy, y)
    assert np.allclose(core.KiF, F)
    assert np.allclose(core.FKiy, F.T @ y)
    assert np.allclose(core.FKiF, F.T @ F)
    assert core.A0i.shape == (2, 2)
    assert core.beta_F0.shape == (2, 1)
    assert np.isscalar(core.s20)


def test_lmmcore_initializes_and_fits_null_with_ki_dot(simple_data, lmm_core_module):
    y, F = simple_data

    def ki_dot(x):
        return 2.0 * x

    core = lmm_core_module.LMMCore(y, F, Ki_dot=ki_dot)
    assert np.allclose(core.Kiy, 2.0 * y)
    assert np.allclose(core.KiF, 2.0 * F)
    assert np.allclose(core.FKiy, F.T @ (2.0 * y))
    assert np.allclose(core.FKiF, F.T @ (2.0 * F))


def test_lmmcore_process_without_interaction_returns_expected_shapes(lmm_core_module):
    y = np.array([[1.0], [0.0], [1.0], [0.0]])
    F = np.ones((4, 1))
    G = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 0.0], [2.0, 1.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0]])
    core = lmm_core_module.LMMCore(y, F, Ki_dot=None)
    core.process(G, Inter=None, step=1)
    assert core.pv.shape == (4,)
    assert core.beta_g.shape == (1, 4)
    assert core.lrt.shape == (4,)
    assert np.all(np.isfinite(core.pv))
    assert np.all(np.isfinite(core.lrt))


def test_lmmcore_process_with_interaction_step1_changes_beta_shape(lmm_core_module):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    Inter = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    core = lmm_core_module.LMMCore(y, F, Ki_dot=None)
    core.process(G, Inter=Inter, step=1)
    assert core.beta_g.shape == (2, 2)
    assert core.pv.shape == (2,)
    assert core.lrt.shape == (2,)


def test_lmmcore_process_with_interaction_step_gt1_uses_hatodot_like_expansion(lmm_core_module):
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
    core = lmm_core_module.LMMCore(y, F, Ki_dot=None)
    core.process(G, Inter=Inter, step=2)
    assert core.beta_g.shape == (4, 2)
    assert core.pv.shape == (2,)
    assert np.all(np.isfinite(core.beta_g))


def test_lmmcore_getters_return_same_objects_after_process(lmm_core_module):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    core = lmm_core_module.LMMCore(y, F, Ki_dot=None)
    core.process(G, step=1)
    assert np.array_equal(core.getPv(), core.pv)
    assert np.array_equal(core.getBetaSNP(), core.beta_g)
    assert np.array_equal(core.getLRT(), core.lrt)


def test_lmmcore_get_beta_cov_raises_attribute_error_before_assignment(simple_data, lmm_core_module):
    y, F = simple_data
    core = lmm_core_module.LMMCore(y, F, Ki_dot=None)
    with pytest.raises(AttributeError):
        core.getBetaCov()


def test_lmmcore_get_betasnpste_matches_manual_formula(lmm_core_module):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [3.0, 2.0]])
    core = lmm_core_module.LMMCore(y, F, Ki_dot=None)
    core.process(G, step=1)
    ste = core.getBetaSNPste()
    import scipy.stats as st

    beta = core.getBetaSNP()
    pv = core.getPv()
    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    expected = beta / z
    assert ste.shape == beta.shape
    assert np.allclose(ste, expected, equal_nan=True)


def test_lmmcore_process_verbose_prints_message(capsys, lmm_core_module):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [3.0, 2.0]])
    core = lmm_core_module.LMMCore(y, F, Ki_dot=None)
    core.process(G, step=1, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 2 variants" in out


def test_lmmcore_process_step_truncates_incomplete_block(lmm_core_module):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 4.0], [2.0, 1.0, 5.0], [3.0, 2.0, 6.0]])
    core = lmm_core_module.LMMCore(y, F, Ki_dot=None)
    core.process(G, step=2)
    assert core.pv.shape == (1,)
    assert core.beta_g.shape == (2, 1)
    assert core.lrt.shape == (1,)


def test_lmmcore_with_custom_ki_dot_affects_statistics(lmm_core_module):
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    F = np.ones((4, 1))
    G = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [3.0, 2.0]])

    core1 = lmm_core_module.LMMCore(y, F, Ki_dot=None)
    core1.process(G, step=1)

    def ki_dot(x):
        return 3.0 * x

    core2 = lmm_core_module.LMMCore(y, F, Ki_dot=ki_dot)
    core2.process(G, step=1)

    assert not np.allclose(core1.getLRT(), core2.getLRT())
    assert not np.allclose(core1.getPv(), core2.getPv())
