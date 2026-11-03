import importlib
import sys
import types

import numpy as np
import pytest


def _install_dummy_submodules():
    """
    limix_lmm.__init__ imports several submodules that may pull optional
    dependencies. Install lightweight stand-ins so importing limix_lmm
    remains focused on the package's public namespace.
    """
    dummy_names = [
        "limix_lmm.plot",
        "limix_lmm.lmm",
        "limix_lmm.mtlmm",
        "limix_lmm._testit",
        "limix_lmm.sh",
    ]
    for name in dummy_names:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if name.endswith(".lmm"):
                class LMM:  # noqa: N801 - mimic exported class name
                    pass

                mod.LMM = LMM
            elif name.endswith(".mtlmm"):
                class MTLMM:  # noqa: N801 - mimic exported class name
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


def _reload_limix_lmm():
    _install_dummy_submodules()
    if "limix_lmm" in sys.modules:
        del sys.modules["limix_lmm"]
    import limix_lmm  # noqa: F401

    return importlib.reload(sys.modules["limix_lmm"])


def _make_identity_kidot():
    def kidot(x):
        return np.array(x, dtype=float, copy=True)

    return kidot


def _make_simple_data(n=4):
    y = np.arange(1, n + 1, dtype=float).reshape(-1, 1)
    F = np.column_stack([np.ones(n), np.linspace(0.0, 1.0, n)]).astype(float)
    return y, F


def test_init_with_none_F_uses_intercept_only():
    from limix_lmm.lmm_core import LMMCore

    y = np.array([[1.0], [2.0], [3.0]])
    lmm = LMMCore(y, None, Ki_dot=_make_identity_kidot())
    assert lmm.F.shape == (3, 1)
    assert np.allclose(lmm.F, np.ones((3, 1)))
    assert lmm.df == 2


def test_init_computes_null_quantities_with_identity_kidot():
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data()
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    assert lmm.Kiy.shape == y.shape
    assert lmm.KiF.shape == F.shape
    assert lmm.FKiy.shape == (F.shape[1], 1)
    assert lmm.FKiF.shape == (F.shape[1], F.shape[1])
    assert lmm.yKiy == pytest.approx(float(np.dot(y[:, 0], y[:, 0])))
    assert lmm.s20 > 0


def test_process_without_inter_returns_expected_shapes():
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data()
    G = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    lmm.process(G, step=1)
    assert lmm.getPv().shape == (4,)
    assert lmm.getBetaSNP().shape == (1, 4)
    assert lmm.getLRT().shape == (4,)


def test_process_with_inter_and_step1_returns_expected_beta_shape():
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data()
    G = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.5],
        ]
    )
    Inter = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [1.5, 0.5],
            [0.5, 1.5],
        ]
    )
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    lmm.process(G, Inter=Inter, step=1)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (2, 2)


def test_process_with_inter_and_step2_uses_hadodot_path():
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data()
    G = np.array(
        [
            [1.0, 0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
            [1.0, 1.0, 0.5, 0.0],
            [2.0, 0.5, 1.5, 1.0],
        ]
    )
    Inter = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [1.5, 0.5],
            [0.5, 1.5],
        ]
    )
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    lmm.process(G, Inter=Inter, step=2)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_process_works_when_kidot_is_none():
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data()
    G = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5]])
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, step=1)
    assert np.isfinite(lmm.getLRT()).all()
    assert np.isfinite(lmm.getPv()).all()


def test_process_pvalues_are_between_zero_and_one():
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data()
    G = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5]])
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    lmm.process(G, step=1)
    pv = lmm.getPv()
    assert np.all(pv >= 0.0)
    assert np.all(pv <= 1.0)


def test_getters_return_same_object_references():
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data()
    G = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5]])
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    lmm.process(G, step=1)
    assert lmm.getPv() is lmm.pv
    assert lmm.getBetaSNP() is lmm.beta_g
    assert lmm.getLRT() is lmm.lrt


def test_get_beta_snpste_has_same_shape_as_betas_and_is_finite():
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data()
    G = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5]])
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    lmm.process(G, step=1)
    ste = lmm.getBetaSNPste()
    assert ste.shape == lmm.getBetaSNP().shape
    assert np.isfinite(ste).all()


def test_process_with_verbose_true_prints_message(capsys):
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data()
    G = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5]])
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    lmm.process(G, step=1, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 2 variants" in out


def test_process_overwrites_previous_results_on_second_call():
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data()
    G1 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5]])
    G2 = np.array([[0.5, 1.5, 2.5, 3.5], [1.0, 0.0, 1.0, 0.0], [2.0, 1.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0]])
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    lmm.process(G1, step=1)
    pv1 = lmm.getPv().copy()
    lmm.process(G2, step=1)
    assert lmm.getPv().shape == (4,)
    assert not np.array_equal(pv1, lmm.getPv())


def test_get_beta_cov_raises_attribute_error():
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data()
    G = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5]])
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    lmm.process(G, step=1)
    with pytest.raises(AttributeError):
        lmm.getBetaCov()


def test_init_and_process_with_float_inputs_produces_float_outputs():
    from limix_lmm.lmm_core import LMMCore

    y = np.array([[1], [2], [4], [8]], dtype=np.int64)
    F = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=np.int64)
    G = np.array([[1, 0], [0, 1], [1, 1], [2, 0]], dtype=np.int64)
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    lmm.process(G, step=1)
    assert lmm.getPv().dtype.kind == "f"
    assert lmm.getBetaSNP().dtype.kind == "f"


def test_init_sets_degrees_of_freedom_correctly():
    from limix_lmm.lmm_core import LMMCore

    y, F = _make_simple_data(n=6)
    lmm = LMMCore(y, F, Ki_dot=_make_identity_kidot())
    assert lmm.df == 4


def test_limix_lmm_init_exports_expected_public_names():
    limix_lmm = _reload_limix_lmm()
    assert hasattr(limix_lmm, "LMM")
    assert hasattr(limix_lmm, "LMMCore")
    assert hasattr(limix_lmm, "MTLMM")
    assert hasattr(limix_lmm, "plot")
    assert hasattr(limix_lmm, "download")
    assert hasattr(limix_lmm, "unzip")
    assert hasattr(limix_lmm, "test")
    assert limix_lmm.__version__ == "0.1.2"


def test_limix_lmm_all_contains_expected_symbols():
    limix_lmm = _reload_limix_lmm()
    expected = {"LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "__version__", "test"}
    assert expected.issubset(set(limix_lmm.__all__))


def test_limix_lmm_init_imports_dummy_test_callable():
    limix_lmm = _reload_limix_lmm()
    assert callable(limix_lmm.test)
    assert limix_lmm.test() == "ok"
