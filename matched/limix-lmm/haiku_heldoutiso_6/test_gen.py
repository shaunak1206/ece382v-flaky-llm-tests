"""
Unit tests for limix_lmm.lmm_core.LMMCore and the limix_lmm package __init__.

Note: the provided lmm_core.py does not define MTLMM. __init__.py imports it
from limix_lmm.mtlmm, so the MTLMM tests below only check the package export.
"""
import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm import LMMCore, MTLMM
from limix_lmm import lmm_core as lmm_core_module
from limix_lmm import mtlmm as mtlmm_module


def _make_data(seed=0, N=40, L=2, S=12, M=3):
    rs = np.random.RandomState(seed)
    y = rs.randn(N, 1)
    F = rs.randn(N, L)
    F[:, 0] = 1.0  # intercept column
    G = rs.randn(N, S)
    Inter = rs.randn(N, M)
    return y, F, G, Inter


def _spd_inverse(N, seed=1):
    rs = np.random.RandomState(seed)
    A = rs.randn(N, N)
    K = A.dot(A.T) / N + np.eye(N)
    return np.linalg.inv(K)


def _ols_rss(X, y):
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X.dot(coef)
    return coef, float(resid.T.dot(resid))


def test_default_F_is_intercept_column():
    y = np.random.RandomState(0).randn(20, 1)
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (20, 1)
    assert np.all(lmm.F == 1.0)
    assert lmm.df == 19


def test_df_is_n_minus_number_of_covariates():
    y, F, _, _ = _make_data(N=40, L=3)
    lmm = LMMCore(y, F)
    assert lmm.df == 40 - 3


def test_without_ki_dot_Kiy_and_KiF_are_identity_transforms():
    y, F, _, _ = _make_data()
    lmm = LMMCore(y, F, None)
    assert np.array_equal(lmm.Kiy, y)
    assert np.array_equal(lmm.KiF, F)


def test_ki_dot_is_applied_to_y_and_F():
    y, F, _, _ = _make_data()
    lmm = LMMCore(y, F, lambda x: 2.0 * x)
    assert np.allclose(lmm.Kiy, 2.0 * y)
    assert np.allclose(lmm.KiF, 2.0 * F)


def test_sufficient_statistics_without_covariance():
    y, F, _, _ = _make_data()
    lmm = LMMCore(y, F)
    assert np.allclose(lmm.FKiy, F.T.dot(y))
    assert np.allclose(lmm.FKiF, F.T.dot(F))
    assert np.isclose(lmm.yKiy, float(y.T.dot(y)))


def test_null_fit_matches_ols_when_no_covariance():
    y, F, _, _ = _make_data()
    lmm = LMMCore(y, F)
    beta_ols, rss = _ols_rss(F, y)
    assert np.allclose(lmm.beta_F0.ravel(), beta_ols.ravel())
    assert np.isclose(lmm.s20, rss / lmm.df)


def test_null_fit_matches_gls_with_ki_dot():
    y, F, _, _ = _make_data(N=30)
    Ki = _spd_inverse(30)
    lmm = LMMCore(y, F, lambda x: Ki.dot(x))
    beta_expected = np.linalg.solve(F.T.dot(Ki).dot(F), F.T.dot(Ki).dot(y))
    s2_expected = (y.T.dot(Ki).dot(y) - beta_expected.T.dot(F.T.dot(Ki).dot(y))) / lmm.df
    assert np.allclose(lmm.beta_F0.ravel(), beta_expected.ravel())
    assert np.isclose(float(np.squeeze(lmm.s20)), float(np.squeeze(s2_expected)))


def test_process_output_shapes_single_variant_steps():
    y, F, G, _ = _make_data(S=12)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (12,)
    assert lmm.getLRT().shape == (12,)
    assert lmm.getBetaSNP().shape == (1, 12)


def test_process_step_uses_floor_of_variants_over_step():
    y, F, G, _ = _make_data(S=13)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (3,)
    assert lmm.getBetaSNP().shape == (4, 3)


def test_process_with_inter_single_variant_shape():
    y, F, G, Inter = _make_data(S=12, M=3)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getPv().shape == (12,)
    assert lmm.getBetaSNP().shape == (3, 12)


def test_process_with_inter_and_step_shape():
    y, F, G, Inter = _make_data(S=12, M=3)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=4)
    assert lmm.getPv().shape == (3,)
    assert lmm.getBetaSNP().shape == (12, 3)


def test_single_variant_beta_matches_ols_joint_regression():
    y, F, G, _ = _make_data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    for s in range(G.shape[1]):
        W = np.hstack([F, G[:, [s]]])
        coef, _ = _ols_rss(W, y)
        assert np.isclose(beta[0, s], coef[-1, 0])


def test_single_variant_beta_matches_gls_joint_regression_with_ki_dot():
    y, F, G, _ = _make_data(N=30, S=5)
    Ki = _spd_inverse(30)
    lmm = LMMCore(y, F, lambda x: Ki.dot(x))
    lmm.process(G)
    beta = lmm.getBetaSNP()
    for s in range(G.shape[1]):
        W = np.hstack([F, G[:, [s]]])
        coef = np.linalg.solve(W.T.dot(Ki).dot(W), W.T.dot(Ki).dot(y))
        assert np.isclose(beta[0, s], coef[-1, 0])


def test_lrt_matches_formula_from_residual_variance():
    y, F, G, _ = _make_data(S=6)
    _, rss0 = _ols_rss(F, y)
    df = y.shape[0] - F.shape[1]
    s20 = rss0 / df
    lmm = LMMCore(y, F)
    lmm.process(G)
    lrt_expected = []
    for s in range(G.shape[1]):
        W = np.hstack([F, G[:, [s]]])
        _, rss1 = _ols_rss(W, y)
        lrt_expected.append(-df * np.log((rss1 / df) / s20))
    assert np.allclose(lmm.getLRT(), np.array(lrt_expected))


def test_lrt_is_nonnegative_and_pvalues_in_unit_interval():
    y, F, G, _ = _make_data(S=20)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert np.all(lmm.getLRT() >= -1e-10)
    pv = lmm.getPv()
    assert np.all(np.isfinite(pv))
    assert np.all((pv >= 0) & (pv <= 1))


def test_pvalues_use_chi2_with_degrees_of_freedom_equal_to_block_size():
    y, F, G, _ = _make_data(S=12)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    assert np.allclose(lmm.getPv(), st.chi2(2).sf(lmm.getLRT()))


def test_getters_return_stored_attributes():
    y, F, G, _ = _make_data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv() is lmm.pv
    assert lmm.getBetaSNP() is lmm.beta_g
    assert lmm.getLRT() is lmm.lrt


def test_getBetaSNPste_matches_formula():
    y, F, G, _ = _make_data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    ste = lmm.getBetaSNPste()
    assert ste.shape == beta.shape
    assert np.allclose(ste, beta / z)


def test_process_verbose_prints_summary(capsys):
    y, F, G, _ = _make_data(S=12)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 12 variants" in out


def test_process_does_not_modify_inputs():
    y, F, G, Inter = _make_data(S=8, M=2)
    y0, F0, G0, Inter0 = y.copy(), F.copy(), G.copy(), Inter.copy()
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)
    assert np.array_equal(y, y0)
    assert np.array_equal(F, F0)
    assert np.array_equal(G, G0)
    assert np.array_equal(Inter, Inter0)


@pytest.mark.xfail(
    reason="getBetaCov returns self.beta_F, which LMMCore never assigns",
    raises=AttributeError,
    strict=False,
)
def test_getBetaCov_returns_covariate_betas():
    y, F, _, _ = _make_data()
    lmm = LMMCore(y, F)
    assert lmm.getBetaCov().shape == (F.shape[1], 1)


def test_package_exports_all_public_names():
    for name in limix_lmm.__all__:
        assert hasattr(limix_lmm, name), name


def test_LMMCore_is_reexported_from_lmm_core_module():
    assert limix_lmm.LMMCore is lmm_core_module.LMMCore


def test_MTLMM_is_reexported_from_mtlmm_module():
    assert isinstance(MTLMM, type)
    assert limix_lmm.MTLMM is mtlmm_module.MTLMM


def test_version_string():
    assert limix_lmm.__version__ == "0.1.2"
