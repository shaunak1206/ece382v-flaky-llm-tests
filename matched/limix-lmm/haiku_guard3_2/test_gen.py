"""Unit tests for limix_lmm.lmm_core.LMMCore and the limix_lmm package exports.

Note: MTLMM is not defined in lmm_core.py. Its source lives in limix_lmm/mtlmm.py,
and it is re-exported from limix_lmm/__init__.py. The MTLMM tests below therefore
check the export and its module of origin only.
"""

import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm import mtlmm as mtlmm_module
from limix_lmm import lmm_core as lmm_core_module
from limix_lmm.lmm_core import LMMCore


def _ols(X, y):
    """Least-squares coefficients and residual sum of squares, using numpy only."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X.dot(beta)
    return beta, float(resid.dot(resid))


def test_default_F_is_intercept_only():
    random = np.random.RandomState(1)
    N = 30
    y = random.randn(N, 1)

    lmm = LMMCore(y, None)

    assert lmm.F.shape == (N, 1)
    np.testing.assert_array_equal(lmm.F, np.ones((N, 1)))
    assert lmm.df == N - 1


def test_df_is_N_minus_number_of_covariates():
    random = np.random.RandomState(2)
    N = 40
    F = np.column_stack([np.ones(N), random.randn(N, 2)])
    y = random.randn(N, 1)

    lmm = LMMCore(y, F)

    assert lmm.df == N - 3


def test_without_Ki_dot_null_fit_is_ordinary_least_squares():
    random = np.random.RandomState(3)
    N = 50
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)

    lmm = LMMCore(y, F)

    # With no covariance function, Kiy and KiF are the inputs themselves.
    assert lmm.Kiy is y
    assert lmm.KiF is F

    beta_ols, rss = _ols(F, y[:, 0])
    np.testing.assert_allclose(np.ravel(lmm.beta_F0), beta_ols, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(float(np.ravel(lmm.s20)[0]), rss / lmm.df, rtol=1e-6)
    np.testing.assert_allclose(lmm.A0i, np.linalg.inv(F.T.dot(F)), rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(float(lmm.yKiy), float(y[:, 0].dot(y[:, 0])), rtol=1e-10)


def test_Ki_dot_is_applied_to_y_then_F():
    random = np.random.RandomState(4)
    N = 25
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    calls = []

    def Ki_dot(v):
        calls.append(v.shape)
        return 2.0 * v

    lmm = LMMCore(y, F, Ki_dot)

    np.testing.assert_allclose(lmm.Kiy, 2.0 * y, rtol=1e-12)
    np.testing.assert_allclose(lmm.KiF, 2.0 * F, rtol=1e-12)
    assert calls == [y.shape, F.shape]


def test_process_output_shapes_single_variant_steps():
    random = np.random.RandomState(5)
    N, S = 50, 20
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = 1.0 * (random.rand(N, S) < 0.3)

    lmm = LMMCore(y, F)
    lmm.process(G)

    assert lmm.getPv().shape == (S,)
    assert lmm.getBetaSNP().shape == (1, S)
    assert lmm.getLRT().shape == (S,)


def test_process_output_shapes_with_step_drops_remainder():
    random = np.random.RandomState(6)
    N, S, step = 50, 10, 3
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = 1.0 * (random.rand(N, S) < 0.3)

    lmm = LMMCore(y, F)
    lmm.process(G, step=step)

    ntests = S // step  # 3 blocks; the last variant is not tested
    assert lmm.getPv().shape == (ntests,)
    assert lmm.getBetaSNP().shape == (step, ntests)
    assert lmm.getLRT().shape == (ntests,)


def test_process_output_shapes_with_inter():
    random = np.random.RandomState(7)
    N, S, M = 50, 7, 2
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    Inter = random.randn(N, M)
    G = random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)

    assert lmm.getPv().shape == (S,)
    assert lmm.getBetaSNP().shape == (M, S)


def test_process_output_shapes_with_inter_and_step():
    random = np.random.RandomState(8)
    N, S, M, step = 50, 9, 2, 3
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    Inter = random.randn(N, M)
    G = random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=step)

    ntests = S // step
    assert lmm.getPv().shape == (ntests,)
    assert lmm.getBetaSNP().shape == (M * step, ntests)


def test_pvalues_and_lrt_are_valid_for_any_data():
    # Holds for every outcome: with identity covariance, adding regressors
    # cannot increase the residual sum of squares, so LRT >= 0 and 0 <= p <= 1.
    random = np.random.RandomState(9)
    N, S = 60, 40
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = 1.0 * (random.rand(N, S) < 0.3)

    lmm = LMMCore(y, F)
    lmm.process(G)

    pv = lmm.getPv()
    lrt = lmm.getLRT()
    assert np.all(np.isfinite(pv))
    assert np.all(np.isfinite(lrt))
    assert np.all(pv >= 0.0) and np.all(pv <= 1.0)
    assert np.all(lrt >= -1e-8)


def test_single_variant_beta_matches_ols_on_augmented_design():
    random = np.random.RandomState(10)
    N, S = 50, 5
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G)

    for s in range(S):
        X_full = np.column_stack([F, G[:, s]])
        beta_full, _ = _ols(X_full, y[:, 0])
        np.testing.assert_allclose(lmm.getBetaSNP()[0, s], beta_full[-1], rtol=1e-6, atol=1e-8)


def test_block_beta_matches_ols_on_augmented_design():
    random = np.random.RandomState(11)
    N, S, step = 50, 6, 2
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=step)

    for s in range(S // step):
        block = G[:, step * s:step * (s + 1)]
        X_full = np.column_stack([F, block])
        beta_full, _ = _ols(X_full, y[:, 0])
        np.testing.assert_allclose(
            lmm.getBetaSNP()[:, s], beta_full[-step:], rtol=1e-6, atol=1e-8
        )


def test_inter_beta_matches_ols_on_interaction_design():
    random = np.random.RandomState(12)
    N, S, M = 50, 4, 2
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    Inter = random.randn(N, M)
    G = random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)

    for s in range(S):
        X = Inter * G[:, s:s + 1]
        beta_full, _ = _ols(np.column_stack([F, X]), y[:, 0])
        np.testing.assert_allclose(lmm.getBetaSNP()[:, s], beta_full[-M:], rtol=1e-6, atol=1e-8)


def test_lrt_equals_minus_df_log_rss_ratio():
    random = np.random.RandomState(13)
    N, S = 50, 5
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G)

    _, rss_null = _ols(F, y[:, 0])
    for s in range(S):
        _, rss_full = _ols(np.column_stack([F, G[:, s]]), y[:, 0])
        expected = -lmm.df * np.log(rss_full / rss_null)
        np.testing.assert_allclose(lmm.getLRT()[s], expected, rtol=1e-6, atol=1e-8)


def test_pvalues_are_chi2_survival_of_lrt():
    random = np.random.RandomState(14)
    N, S = 50, 8
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G)

    expected = st.chi2(1).sf(lmm.getLRT())
    np.testing.assert_allclose(lmm.getPv(), expected, rtol=1e-8, atol=1e-12)


def test_lrt_and_beta_invariant_to_scaling_of_covariance():
    # Scaling K by a constant c leaves beta and the likelihood ratio unchanged.
    random = np.random.RandomState(15)
    N, S = 50, 5
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = random.randn(N, S)

    def scaled(v):
        return 3.0 * v

    lmm_plain = LMMCore(y, F)
    lmm_scaled = LMMCore(y, F, scaled)
    lmm_plain.process(G)
    lmm_scaled.process(G)

    np.testing.assert_allclose(lmm_scaled.getLRT(), lmm_plain.getLRT(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        lmm_scaled.getBetaSNP(), lmm_plain.getBetaSNP(), rtol=1e-6, atol=1e-8
    )


def test_repeated_process_gives_identical_results():
    random = np.random.RandomState(16)
    N, S = 50, 6
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G)
    first_pv = np.array(lmm.getPv(), copy=True)
    first_beta = np.array(lmm.getBetaSNP(), copy=True)

    lmm.process(G)

    np.testing.assert_allclose(lmm.getPv(), first_pv, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(lmm.getBetaSNP(), first_beta, rtol=0.0, atol=1e-12)


def test_getters_return_stored_attributes():
    random = np.random.RandomState(17)
    N, S = 40, 4
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G)

    assert lmm.getPv() is lmm.pv
    assert lmm.getBetaSNP() is lmm.beta_g
    assert lmm.getLRT() is lmm.lrt


def test_getBetaSNPste_matches_wald_formula():
    random = np.random.RandomState(18)
    N, S = 50, 6
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G)

    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    expected = beta / z

    ste = lmm.getBetaSNPste()
    assert ste.shape == beta.shape
    np.testing.assert_allclose(ste, expected, rtol=1e-8, atol=1e-12)


def test_verbose_flag_controls_printed_summary(capsys):
    random = np.random.RandomState(19)
    N, S = 30, 12
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = random.randn(N, S)

    lmm = LMMCore(y, F)

    lmm.process(G, verbose=False)
    assert capsys.readouterr().out == ""

    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    # Only the prefix and variant count are checked; the elapsed time is not.
    assert out.startswith("Tested for 12 variants in ")


def test_null_fit_and_each_variant_fit_use_calc_helper_once(monkeypatch):
    random = np.random.RandomState(20)
    N, S = 30, 5
    F = np.column_stack([np.ones(N), random.randn(N, 1)])
    y = random.randn(N, 1)
    G = random.randn(N, S)

    original = lmm_core_module.calc_Ai_beta_s2
    calls = []

    def spy(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(lmm_core_module, "calc_Ai_beta_s2", spy)

    lmm = LMMCore(y, F)
    assert len(calls) == 1  # one null-model fit in __init__

    lmm.process(G)
    assert len(calls) == 1 + S  # one fit per tested variant


def test_package_exports_lmmcore_and_version():
    assert limix_lmm.LMMCore is LMMCore
    assert "LMMCore" in limix_lmm.__all__
    assert limix_lmm.__version__ == "0.1.2"


def test_package_exports_MTLMM_from_its_module():
    assert isinstance(limix_lmm.MTLMM, type)
    assert limix_lmm.MTLMM is mtlmm_module.MTLMM
    assert "MTLMM" in limix_lmm.__all__
