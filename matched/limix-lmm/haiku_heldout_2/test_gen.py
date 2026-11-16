"""Unit tests for LMMCore (limix_lmm/lmm_core.py) and the package-level
exports in limix_lmm/__init__.py.

Note: the provided source of lmm_core.py does not define MTLMM. MTLMM is
imported into the package from limix_lmm/mtlmm.py, so the MTLMM tests below
only check its package-level export and identity. They make no assumptions
about its interface.
"""

import numpy as np
import pytest
from scipy import stats

import limix_lmm
from limix_lmm import LMMCore, MTLMM
from limix_lmm.lmm_core import LMMCore as LMMCoreFromModule


N = 60
L = 2
S = 6


def _make_data(seed=0, n=N, l=L, s=S):
    rng = np.random.RandomState(seed)
    y = rng.randn(n, 1)
    F = np.hstack([np.ones((n, 1)), rng.randn(n, l - 1)]) if l > 1 else np.ones((n, 1))
    G = rng.randn(n, s)
    return y, F, G


def _rss(X, y):
    """Residual sum of squares of an OLS fit of y (1-D) on X (2-D)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X.dot(beta)
    return float(resid.dot(resid)), beta


# ---------------------------------------------------------------------------
# LMMCore: initialisation and null model
# ---------------------------------------------------------------------------

def test_init_default_F_is_intercept_when_none():
    y, _, _ = _make_data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (N, 1)
    assert np.all(lmm.F == 1.0)
    assert lmm.df == N - 1


def test_init_df_is_n_minus_number_of_covariates():
    y, F, _ = _make_data(l=3)
    lmm = LMMCore(y, F, None)
    assert lmm.df == N - 3


def test_init_without_ki_dot_uses_identity_covariance():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F, None)
    np.testing.assert_allclose(lmm.Kiy, y)
    np.testing.assert_allclose(lmm.KiF, F)
    np.testing.assert_allclose(lmm.FKiy, F.T.dot(y))
    np.testing.assert_allclose(lmm.FKiF, F.T.dot(F))
    np.testing.assert_allclose(lmm.yKiy, float(y[:, 0].dot(y[:, 0])))


def test_init_calls_ki_dot_on_y_and_F():
    y, F, _ = _make_data()
    calls = []

    def ki_dot(x):
        calls.append(np.array(x, copy=True))
        return x

    LMMCore(y, F, ki_dot)
    assert len(calls) == 2
    np.testing.assert_allclose(calls[0], y)
    np.testing.assert_allclose(calls[1], F)


def test_init_null_variance_matches_rss_over_df():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F, None)
    rss0, _ = _rss(F, y[:, 0])
    np.testing.assert_allclose(lmm.s20, rss0 / (N - F.shape[1]), rtol=1e-8)


def test_ki_dot_scaling_leaves_lrt_and_beta_invariant():
    y, F, G = _make_data()
    lmm_a = LMMCore(y, F, None)
    lmm_a.process(G)
    lmm_b = LMMCore(y, F, lambda x: 2.0 * x)
    lmm_b.process(G)
    np.testing.assert_allclose(lmm_a.getLRT(), lmm_b.getLRT(), rtol=1e-8)
    np.testing.assert_allclose(lmm_a.getPv(), lmm_b.getPv(), rtol=1e-8)
    np.testing.assert_allclose(lmm_a.getBetaSNP(), lmm_b.getBetaSNP(), rtol=1e-8)


# ---------------------------------------------------------------------------
# LMMCore.process: shapes
# ---------------------------------------------------------------------------

def test_process_default_step_output_shapes():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    assert lmm.getPv().shape == (S,)
    assert lmm.getLRT().shape == (S,)
    assert lmm.getBetaSNP().shape == (1, S)


def test_process_step_groups_variants_into_blocks():
    s_total = 10
    y, F, G = _make_data(s=s_total)
    lmm = LMMCore(y, F, None)
    lmm.process(G, step=4)
    # int(10 / 4) == 2 blocks; the trailing 2 variants are not tested.
    assert lmm.getPv().shape == (2,)
    assert lmm.getLRT().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_process_with_inter_step1_output_shapes():
    y, F, G = _make_data()
    Inter = np.random.RandomState(3).randn(N, 3)
    lmm = LMMCore(y, F, None)
    lmm.process(G, Inter=Inter)
    assert lmm.getPv().shape == (S,)
    assert lmm.getBetaSNP().shape == (3, S)


def test_process_with_inter_and_step_output_shapes():
    y, F, G = _make_data()
    Inter = np.random.RandomState(4).randn(N, 3)
    lmm = LMMCore(y, F, None)
    lmm.process(G, Inter=Inter, step=2)
    # m = mi * step = 3 * 2 = 6 interaction coefficients per block, 3 blocks
    assert lmm.getPv().shape == (3,)
    assert lmm.getBetaSNP().shape == (6, 3)


# ---------------------------------------------------------------------------
# LMMCore.process: numerical correctness against closed-form OLS
# ---------------------------------------------------------------------------

def test_lrt_matches_closed_form_rss_ratio():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    y1 = y[:, 0]
    rss0, _ = _rss(F, y1)
    expected = []
    for j in range(S):
        X = np.hstack([F, G[:, j:j + 1]])
        rss1, _ = _rss(X, y1)
        expected.append(-(N - L) * np.log(rss1 / rss0))
    np.testing.assert_allclose(lmm.getLRT(), np.array(expected), rtol=1e-6)


def test_beta_matches_ols_coefficient_single_variant():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    y1 = y[:, 0]
    for j in range(S):
        X = np.hstack([F, G[:, j:j + 1]])
        _, beta_ols = _rss(X, y1)
        np.testing.assert_allclose(
            lmm.getBetaSNP()[0, j], beta_ols[-1, 0], rtol=1e-6, atol=1e-10
        )


def test_beta_matches_ols_with_interaction_step1():
    y, F, G = _make_data()
    Inter = np.random.RandomState(5).randn(N, 2)
    lmm = LMMCore(y, F, None)
    lmm.process(G, Inter=Inter)
    y1 = y[:, 0]
    for j in range(S):
        X_int = Inter * G[:, j:j + 1]
        X = np.hstack([F, X_int])
        _, beta_ols = _rss(X, y1)
        np.testing.assert_allclose(
            lmm.getBetaSNP()[:, j], beta_ols[L:, 0], rtol=1e-6, atol=1e-10
        )


def test_pv_is_chi2_survival_of_lrt():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    expected = stats.chi2(1).sf(lmm.getLRT())
    np.testing.assert_allclose(lmm.getPv(), expected, rtol=1e-10)


def test_pv_values_lie_in_unit_interval():
    y, F, G = _make_data(s=40)
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    pv = lmm.getPv()
    assert np.all(np.isfinite(pv))
    assert np.all(pv >= 0.0) and np.all(pv <= 1.0)


# ---------------------------------------------------------------------------
# LMMCore: getters, standard errors, verbose output
# ---------------------------------------------------------------------------

def test_getters_return_stored_results():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    assert lmm.getPv() is lmm.pv
    assert lmm.getLRT() is lmm.lrt
    assert lmm.getBetaSNP() is lmm.beta_g


def test_getBetaSNPste_matches_beta_over_z_score():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    z = np.sign(beta) * np.sqrt(stats.chi2(1).isf(pv))
    ste = lmm.getBetaSNPste()
    assert ste.shape == beta.shape
    np.testing.assert_allclose(ste, beta / z, rtol=1e-8)


def test_process_verbose_prints_timing(capsys):
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for %d variants" % S in out


@pytest.mark.xfail(
    strict=False,
    raises=AttributeError,
    reason="LMMCore.getBetaCov returns self.beta_F, which is never assigned",
)
def test_getBetaCov_returns_covariate_effects():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    beta_cov = lmm.getBetaCov()
    assert beta_cov is not None


# ---------------------------------------------------------------------------
# Package exports (limix_lmm/__init__.py)
# ---------------------------------------------------------------------------

def test_lmmcore_exported_from_package_is_module_class():
    assert limix_lmm.LMMCore is LMMCoreFromModule


def test_mtlmm_exported_from_package():
    # MTLMM is re-exported by the package; its class lives in limix_lmm/mtlmm.py.
    from limix_lmm.mtlmm import MTLMM as MTLMMFromModule

    assert limix_lmm.MTLMM is MTLMMFromModule
    assert MTLMM is MTLMMFromModule


def test_all_entries_are_present_on_package():
    for name in limix_lmm.__all__:
        assert hasattr(limix_lmm, name), "missing attribute: %s" % name


def test_version_string_and_all_contents():
    assert limix_lmm.__version__ == "0.1.2"
    assert "__version__" in limix_lmm.__all__
    assert "LMMCore" in limix_lmm.__all__
    assert "MTLMM" in limix_lmm.__all__
