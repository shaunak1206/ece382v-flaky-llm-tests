"""Unit tests for limix_lmm.lmm_core.LMMCore, MTLMM, and the limix_lmm package __init__.

Note: the lmm_core.py source provided with this task does not define MTLMM.
limix_lmm/__init__.py imports it from the .mtlmm module, so the MTLMM tests
check the package export and its origin rather than its internals.
"""

import numpy as np
import pytest
from scipy import stats

import limix_lmm
from limix_lmm import LMMCore, MTLMM
from limix_lmm import lmm_core

N = 60
L = 2
S = 12


def _make_data(n=N, n_cov=L, s=S, seed=3):
    rs = np.random.RandomState(seed)
    y = rs.randn(n, 1)
    F = np.column_stack([np.ones(n), rs.randn(n, n_cov - 1)])
    G = rs.randn(n, s)
    return y, F, G


def _dense_ki(n, seed=7):
    rs = np.random.RandomState(seed)
    A = rs.randn(n, n)
    K = A.dot(A.T) / n + np.eye(n)
    return (lambda x: np.linalg.solve(K, x)), K


def _scalar(x):
    return float(np.asarray(x).item())


def test_init_default_F_is_intercept():
    y, _, _ = _make_data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (y.shape[0], 1)
    assert np.all(lmm.F == 1.0)


def test_init_sets_residual_degrees_of_freedom():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F)
    assert lmm.df == N - L


def test_init_without_ki_dot_uses_identity_covariance():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F)
    assert lmm.Ki_dot is None
    np.testing.assert_allclose(lmm.Kiy, y)
    np.testing.assert_allclose(lmm.KiF, F)


def test_null_sufficient_statistics_identity_covariance():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F)
    np.testing.assert_allclose(lmm.FKiy, F.T.dot(y))
    np.testing.assert_allclose(lmm.FKiF, F.T.dot(F))
    np.testing.assert_allclose(_scalar(lmm.yKiy), _scalar(y.T.dot(y)))


def test_ki_dot_is_applied_to_null_statistics():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F, Ki_dot=lambda x: 2.0 * x)
    np.testing.assert_allclose(lmm.Kiy, 2.0 * y)
    np.testing.assert_allclose(lmm.KiF, 2.0 * F)
    np.testing.assert_allclose(lmm.FKiy, 2.0 * F.T.dot(y))
    np.testing.assert_allclose(lmm.FKiF, 2.0 * F.T.dot(F))
    np.testing.assert_allclose(_scalar(lmm.yKiy), 2.0 * _scalar(y.T.dot(y)))


def test_null_fit_matches_ols_under_identity_covariance():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F)
    beta = np.linalg.lstsq(F, y, rcond=None)[0]
    resid = y - F.dot(beta)
    np.testing.assert_allclose(np.ravel(lmm.beta_F0), np.ravel(beta),
                               rtol=1e-8, atol=1e-10)
    expected_s20 = _scalar(resid.T.dot(resid)) / (N - L)
    np.testing.assert_allclose(_scalar(lmm.s20), expected_s20, rtol=1e-8)


def test_null_fit_matches_gls_closed_form_with_dense_covariance():
    y, F, _ = _make_data()
    ki_dot, K = _dense_ki(N)
    lmm = LMMCore(y, F, Ki_dot=ki_dot)
    Ki = np.linalg.inv(K)
    beta_expected = np.linalg.solve(F.T @ Ki @ F, F.T @ Ki @ y)
    resid = y - F.dot(beta_expected)
    s2_expected = _scalar(resid.T @ Ki @ resid) / (N - L)
    np.testing.assert_allclose(np.ravel(lmm.beta_F0), np.ravel(beta_expected),
                               rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(_scalar(lmm.s20), s2_expected, rtol=1e-6)


def test_process_default_output_shapes():
    y, F, G = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (S,)
    assert lmm.getLRT().shape == (S,)
    assert lmm.getBetaSNP().shape == (1, S)
    assert lmm.getBetaSNPste().shape == (1, S)


def test_process_step_drops_incomplete_trailing_block():
    y, F, G = _make_data(s=10)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_process_with_inter_single_step_shapes():
    y, F, G = _make_data()
    Inter = np.random.RandomState(11).randn(N, 3)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getPv().shape == (S,)
    assert lmm.getBetaSNP().shape == (3, S)


def test_process_with_inter_and_step_shapes():
    y, F, G = _make_data()
    Inter = np.random.RandomState(11).randn(N, 3)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)
    assert lmm.getPv().shape == (S // 2,)
    assert lmm.getBetaSNP().shape == (3 * 2, S // 2)


def test_getters_return_fitted_attributes():
    y, F, G = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv() is lmm.pv
    assert lmm.getBetaSNP() is lmm.beta_g
    assert lmm.getLRT() is lmm.lrt


@pytest.mark.parametrize("use_inter, step", [(False, 1), (False, 3), (True, 1), (True, 2)])
def test_pvalues_are_chi2_survival_of_lrt(use_inter, step):
    y, F, G = _make_data()
    Inter = np.random.RandomState(5).randn(N, 2) if use_inter else None
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=step)
    m = (2 if use_inter else 1) * step
    np.testing.assert_allclose(lmm.getPv(), stats.chi2(m).sf(lmm.getLRT()),
                               rtol=1e-8, atol=0)


def test_single_variant_beta_and_lrt_match_ols():
    y, F, G = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)

    beta0 = np.linalg.lstsq(F, y, rcond=None)[0]
    rss_null = _scalar((y - F.dot(beta0)).T.dot(y - F.dot(beta0)))

    for s in range(S):
        X = np.column_stack([F, G[:, s]])
        coef = np.linalg.lstsq(X, y, rcond=None)[0].ravel()
        rss_alt = _scalar((y - X.dot(coef.reshape(-1, 1))).T.dot(
            y - X.dot(coef.reshape(-1, 1))))
        np.testing.assert_allclose(lmm.getBetaSNP()[0, s], coef[-1],
                                   rtol=1e-6, atol=1e-9)
        expected_lrt = -(N - L) * np.log(rss_alt / rss_null)
        np.testing.assert_allclose(lmm.getLRT()[s], expected_lrt,
                                   rtol=1e-6, atol=1e-9)


def test_lrt_nonnegative_and_pvalues_in_unit_interval():
    y, F, G = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert np.all(lmm.getLRT() >= -1e-8)
    assert np.all((lmm.getPv() >= 0.0) & (lmm.getPv() <= 1.0))


def test_process_again_overwrites_previous_results():
    y, F, G = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    lmm.process(G[:, :4], step=1)
    assert lmm.getPv().shape == (4,)
    assert lmm.getBetaSNP().shape == (1, 4)


def test_process_verbose_prints_summary(capsys):
    y, F, G = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for %d variants" % S in out


def test_process_is_silent_by_default(capsys):
    y, F, G = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert capsys.readouterr().out == ""


def test_beta_ste_equals_beta_over_signed_z():
    y, F, G = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    z = np.sign(beta) * np.sqrt(stats.chi2(1).isf(pv))
    np.testing.assert_allclose(lmm.getBetaSNPste(), beta / z, rtol=1e-10)


def test_lmmcore_export_is_defined_in_lmm_core_module():
    assert limix_lmm.LMMCore is lmm_core.LMMCore
    assert LMMCore is lmm_core.LMMCore


def test_mtlmm_is_exported_and_is_a_class():
    # MTLMM is re-exported by limix_lmm/__init__.py from the .mtlmm module.
    from limix_lmm import mtlmm as mtlmm_module
    assert MTLMM is mtlmm_module.MTLMM
    assert isinstance(MTLMM, type)
    assert "MTLMM" in limix_lmm.__all__


def test_version_string():
    assert limix_lmm.__version__ == "0.1.2"


def test_every_name_in_all_is_an_attribute():
    for name in limix_lmm.__all__:
        assert hasattr(limix_lmm, name), name


def test_all_has_no_duplicate_names():
    assert len(limix_lmm.__all__) == len(set(limix_lmm.__all__))
