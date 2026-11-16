"""Unit tests for LMMCore and MTLMM (and the package exports in limix_lmm/__init__.py).

Note: the __init__.py imports MTLMM from ``.mtlmm``, so MTLMM is defined in
``limix_lmm/mtlmm.py`` rather than in ``lmm_core.py``. Its internals were not
provided, so the MTLMM tests below only check that it is importable and exported.
"""
import numpy as np
import pytest
from scipy import stats as st

import limix_lmm
from limix_lmm import LMMCore, MTLMM
from limix_lmm.lmm_core import LMMCore as LMMCoreFromModule


def _make_data(N=40, L=2, S=6, seed=0):
    rs = np.random.RandomState(seed)
    y = rs.randn(N, 1)
    F = np.column_stack([np.ones(N), rs.randn(N, L - 1)]) if L > 1 else np.ones((N, 1))
    G = rs.randn(N, S)
    return y, F, G


def _ls_rss(design, y):
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design.dot(coef)
    return coef, float(resid.T.dot(resid))


def _manual_alt(F, y, X):
    """Least-squares fit of the alternative model; returns coefs, rss0, rss1."""
    _, rss0 = _ls_rss(F, y)
    coef, rss1 = _ls_rss(np.hstack([F, X]), y)
    return coef, rss0, rss1


def test_init_defaults_F_to_intercept():
    y, _, _ = _make_data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (y.shape[0], 1)
    np.testing.assert_allclose(lmm.F, np.ones((y.shape[0], 1)))


def test_init_sets_degrees_of_freedom():
    y, F, _ = _make_data(N=40, L=3)
    lmm = LMMCore(y, F)
    assert lmm.df == 40 - 3


def test_init_without_ki_dot_uses_raw_y_and_F():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F, None)
    np.testing.assert_allclose(lmm.Kiy, y)
    np.testing.assert_allclose(lmm.KiF, F)


def test_init_with_ki_dot_applies_function_to_y_and_F():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F, lambda x: 2.0 * x)
    np.testing.assert_allclose(lmm.Kiy, 2.0 * y)
    np.testing.assert_allclose(lmm.KiF, 2.0 * F)
    np.testing.assert_allclose(lmm.yKiy, 2.0 * float(y.T.dot(y)))


def test_null_fit_sufficient_statistics_identity_covariance():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F)
    np.testing.assert_allclose(lmm.FKiF, F.T.dot(F), rtol=1e-10)
    np.testing.assert_allclose(lmm.FKiy, F.T.dot(y), rtol=1e-10)
    np.testing.assert_allclose(lmm.yKiy, float(y.T.dot(y)), rtol=1e-10)


def test_null_fit_beta_and_sigma_match_least_squares():
    y, F, _ = _make_data(N=40, L=2)
    lmm = LMMCore(y, F)
    coef, rss0 = _ls_rss(F, y)
    np.testing.assert_allclose(np.ravel(lmm.beta_F0), np.ravel(coef), rtol=1e-6, atol=1e-10)
    assert lmm.s20 == pytest.approx(rss0 / lmm.df, rel=1e-6)


def test_process_default_output_shapes():
    y, F, G = _make_data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (6,)
    assert lmm.getLRT().shape == (6,)
    assert lmm.getBetaSNP().shape == (1, 6)
    assert lmm.getBetaSNPste().shape == (1, 6)


def test_process_step_output_shapes():
    y, F, G = _make_data(S=10)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    # int(10 / 4) == 2 blocks of 4 variants each
    assert lmm.getPv().shape == (2,)
    assert lmm.getLRT().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_process_with_inter_output_shapes():
    y, F, G = _make_data(N=40, S=5)
    Inter = np.random.RandomState(3).randn(40, 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getPv().shape == (5,)
    assert lmm.getBetaSNP().shape == (2, 5)


def test_process_with_inter_and_step_output_shapes():
    y, F, G = _make_data(N=40, S=8)
    Inter = np.random.RandomState(4).randn(40, 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)
    # m = mi * step = 4 tested coefficients per block, 4 blocks
    assert lmm.getPv().shape == (4,)
    assert lmm.getBetaSNP().shape == (4, 4)


def test_lrt_matches_manual_nested_regression():
    y, F, G = _make_data(N=40, L=2, S=5)
    lmm = LMMCore(y, F)
    lmm.process(G)
    df = 40 - 2
    for s in range(G.shape[1]):
        _, rss0, rss1 = _manual_alt(F, y, G[:, s:s + 1])
        expected = -df * np.log(rss1 / rss0)
        assert lmm.getLRT()[s] == pytest.approx(expected, rel=1e-6)


def test_beta_snp_matches_least_squares_coefficient():
    y, F, G = _make_data(N=40, L=2, S=5)
    lmm = LMMCore(y, F)
    lmm.process(G)
    for s in range(G.shape[1]):
        coef, _, _ = _manual_alt(F, y, G[:, s:s + 1])
        assert lmm.getBetaSNP()[0, s] == pytest.approx(coef[-1, 0], rel=1e-6, abs=1e-10)


def test_inter_lrt_and_beta_match_manual_regression():
    y, F, G = _make_data(N=40, L=2, S=4)
    Inter = np.random.RandomState(5).randn(40, 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    df = 40 - 2
    for s in range(G.shape[1]):
        X = Inter * G[:, s:s + 1]
        coef, rss0, rss1 = _manual_alt(F, y, X)
        assert lmm.getLRT()[s] == pytest.approx(-df * np.log(rss1 / rss0), rel=1e-6)
        np.testing.assert_allclose(
            lmm.getBetaSNP()[:, s], coef[-2:, 0], rtol=1e-6, atol=1e-10
        )


def test_pvalues_are_chi2_survival_of_lrt():
    y, F, G = _make_data(N=40, S=5)
    lmm = LMMCore(y, F)
    lmm.process(G)
    expected = st.chi2(1).sf(lmm.getLRT())
    np.testing.assert_allclose(lmm.getPv(), expected, rtol=1e-10, atol=1e-300)


def test_pvalues_lie_in_unit_interval():
    y, F, G = _make_data(N=60, S=20, seed=7)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    pv = lmm.getPv()
    assert np.all(pv >= 0.0) and np.all(pv <= 1.0)


def test_scaled_covariance_gives_same_test_statistics():
    # Scaling K by a constant is absorbed by sigma^2, so LRTs are unchanged.
    y, F, G = _make_data(N=40, S=5)
    ref = LMMCore(y, F)
    ref.process(G)
    scaled = LMMCore(y, F, lambda x: x / 3.0)
    scaled.process(G)
    np.testing.assert_allclose(scaled.getLRT(), ref.getLRT(), rtol=1e-6)
    np.testing.assert_allclose(scaled.getPv(), ref.getPv(), rtol=1e-6, atol=1e-12)


def test_getBetaSNPste_consistent_with_beta_and_pvalues():
    y, F, G = _make_data(N=40, S=5)
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    ste = lmm.getBetaSNPste()
    assert ste.shape == beta.shape
    z_abs = np.abs(beta / ste)
    np.testing.assert_allclose(z_abs, np.sqrt(st.chi2(1).isf(lmm.getPv())), rtol=1e-6)


def test_process_verbose_prints_summary(capsys):
    y, F, G = _make_data(N=40, S=3)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 3 variants" in out


def test_process_is_silent_by_default(capsys):
    y, F, G = _make_data(N=40, S=3)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert capsys.readouterr().out == ""


def test_getters_return_stored_attributes():
    y, F, G = _make_data(N=40, S=4)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv() is lmm.pv
    assert lmm.getLRT() is lmm.lrt
    assert lmm.getBetaSNP() is lmm.beta_g


def test_package_exports_lmmcore_and_mtlmm():
    assert "LMMCore" in limix_lmm.__all__
    assert "MTLMM" in limix_lmm.__all__
    assert LMMCore is LMMCoreFromModule
    assert isinstance(MTLMM, type)


def test_mtlmm_is_importable_from_its_module():
    from limix_lmm.mtlmm import MTLMM as MTLMMFromModule

    assert MTLMM is MTLMMFromModule


def test_package_version_is_string():
    assert isinstance(limix_lmm.__version__, str)
    assert limix_lmm.__version__ in limix_lmm.__all__
