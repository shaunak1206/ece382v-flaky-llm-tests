"""Unit tests for limix_lmm.lmm_core.LMMCore and the limix_lmm package exports.

Note: the provided source for limix_lmm/lmm_core.py does not define MTLMM.
MTLMM lives in limix_lmm/mtlmm.py and is re-exported by limix_lmm/__init__.py,
so MTLMM is covered here only through its package-level export.
"""
import numpy as np
import scipy.stats as st

import limix_lmm
import limix_lmm.lmm_core as lmm_core_module
import limix_lmm.mtlmm as mtlmm_module
from limix_lmm.lmm_core import LMMCore

SEED = 1234
N = 30  # number of samples
S = 8  # number of variants used in process tests


def _covariates(n_extra):
    """Intercept plus n_extra random covariates (uses the globally seeded RNG)."""
    return np.hstack([np.ones((N, 1)), np.random.randn(N, n_extra)])


def _ols(X, y):
    """Ordinary least squares: returns coefficients and residual sum of squares."""
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    return coef, float(np.sum(resid ** 2))


def _gls(X, y, Ki):
    """Generalised least squares with inverse covariance Ki."""
    XtKi = X.T @ Ki
    b = np.linalg.solve(XtKi @ X, XtKi @ y)
    r = y - X @ b
    return b, float((r.T @ Ki @ r).item())


def _lrt_reference(y, F, X_alt, Ki=None):
    """Brute-force reference: returns (genetic betas, LRT) for the alternative
    design [F, X_alt] against the null design F."""
    n = y.shape[0]
    if Ki is None:
        Ki = np.eye(n)
    df = n - F.shape[1]
    _, rss0 = _gls(F, y, Ki)
    b1, rss1 = _gls(np.hstack([F, X_alt]), y, Ki)
    lrt = -df * np.log(rss1 / rss0)
    return b1[F.shape[1]:], lrt


def _random_spd_inverse(n):
    """Inverse of a well-conditioned random SPD matrix (uses the seeded RNG)."""
    M = np.random.randn(n, n)
    K = M @ M.T + n * np.eye(n)
    return np.linalg.inv(K)


def test_init_defaults_to_intercept_when_F_is_none():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)

    lmm = LMMCore(y, None)

    assert lmm.F.shape == (N, 1)
    assert np.all(lmm.F == 1.0)
    assert lmm.df == N - 1
    assert lmm.Ki_dot is None


def test_init_stores_inputs_and_degrees_of_freedom():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)

    lmm = LMMCore(y, F)

    assert lmm.y is y
    assert lmm.F is F
    assert lmm.df == N - 3


def test_null_model_without_covariance_matches_ols():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)

    lmm = LMMCore(y, F)
    coef, rss = _ols(F, y)

    np.testing.assert_allclose(lmm.Kiy, y)
    np.testing.assert_allclose(lmm.KiF, F)
    np.testing.assert_allclose(lmm.yKiy, float((y.T @ y).item()), rtol=1e-10)
    np.testing.assert_allclose(lmm.beta_F0.ravel(), coef.ravel(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(lmm.s20, rss / (N - F.shape[1]), rtol=1e-6, atol=1e-8)


def test_null_model_with_covariance_matches_gls():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    Ki = _random_spd_inverse(N)

    lmm = LMMCore(y, F, lambda A: Ki @ A)
    b, rss = _gls(F, y, Ki)

    np.testing.assert_allclose(lmm.beta_F0.ravel(), b.ravel(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(lmm.s20, rss / (N - F.shape[1]), rtol=1e-6, atol=1e-8)


def test_ki_dot_is_applied_to_phenotype_and_covariates():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    calls = []

    def ki_dot(A):
        calls.append(A.shape)
        return 2.0 * A

    lmm = LMMCore(y, F, ki_dot)

    assert calls == [y.shape, F.shape]
    np.testing.assert_allclose(lmm.Kiy, 2.0 * y)
    np.testing.assert_allclose(lmm.KiF, 2.0 * F)


def test_process_output_shapes_without_inter():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F)

    lmm.process(G)

    assert lmm.getPv().shape == (S,)
    assert lmm.getLRT().shape == (S,)
    assert lmm.getBetaSNP().shape == (1, S)
    assert lmm.getBetaSNPste().shape == (1, S)


def test_process_matches_per_variant_ols():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F)

    lmm.process(G)

    for s in range(S):
        beta_ref, lrt_ref = _lrt_reference(y, F, G[:, [s]])
        np.testing.assert_allclose(lmm.getBetaSNP()[:, s], beta_ref.ravel(),
                                   rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(lmm.getLRT()[s], lrt_ref, rtol=1e-6, atol=1e-8)


def test_process_with_covariance_matches_gls():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    Ki = _random_spd_inverse(N)
    lmm = LMMCore(y, F, lambda A: Ki @ A)

    lmm.process(G)

    for s in range(S):
        beta_ref, lrt_ref = _lrt_reference(y, F, G[:, [s]], Ki)
        np.testing.assert_allclose(lmm.getBetaSNP()[:, s], beta_ref.ravel(),
                                   rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(lmm.getLRT()[s], lrt_ref, rtol=1e-6, atol=1e-8)


def test_process_step_tests_blocks_of_variants():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, 12)
    step = 3
    lmm = LMMCore(y, F)

    lmm.process(G, step=step)

    assert lmm.getPv().shape == (4,)
    assert lmm.getLRT().shape == (4,)
    assert lmm.getBetaSNP().shape == (step, 4)
    for s in range(4):
        block = G[:, step * s:step * (s + 1)]
        beta_ref, lrt_ref = _lrt_reference(y, F, block)
        np.testing.assert_allclose(lmm.getBetaSNP()[:, s], beta_ref.ravel(),
                                   rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(lmm.getLRT()[s], lrt_ref, rtol=1e-6, atol=1e-8)


def test_process_with_inter_step1_matches_brute_force():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, 2)
    lmm = LMMCore(y, F)

    lmm.process(G, Inter=Inter)

    assert lmm.getBetaSNP().shape == (2, S)
    for s in range(S):
        X_alt = Inter * G[:, [s]]
        beta_ref, lrt_ref = _lrt_reference(y, F, X_alt)
        np.testing.assert_allclose(lmm.getBetaSNP()[:, s], beta_ref.ravel(),
                                   rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(lmm.getLRT()[s], lrt_ref, rtol=1e-6, atol=1e-8)


def test_process_with_inter_and_step_lrt_matches_brute_force():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, 2)
    step = 2
    lmm = LMMCore(y, F)

    lmm.process(G, Inter=Inter, step=step)

    ntests = S // step
    assert lmm.getPv().shape == (ntests,)
    assert lmm.getBetaSNP().shape == (2 * step, ntests)
    for s in range(ntests):
        block = G[:, step * s:step * (s + 1)]
        # The LRT depends only on the span of the alternative design, so the
        # column ordering of the interaction products does not matter here.
        X_alt = np.column_stack(
            [block[:, i] * Inter[:, j] for i in range(step) for j in range(2)]
        )
        _, lrt_ref = _lrt_reference(y, F, X_alt)
        np.testing.assert_allclose(lmm.getLRT()[s], lrt_ref, rtol=1e-6, atol=1e-8)


def test_pvalues_are_chi2_survival_of_lrt():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F)

    lmm.process(G, step=2)

    lrt = lmm.getLRT()
    pv = lmm.getPv()
    np.testing.assert_allclose(pv, st.chi2(2).sf(lrt), rtol=1e-10, atol=1e-12)
    assert np.all((pv >= 0.0) & (pv <= 1.0))


def test_getters_return_fitted_arrays():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F)

    lmm.process(G)

    assert lmm.getPv() is lmm.pv
    assert lmm.getLRT() is lmm.lrt
    assert lmm.getBetaSNP() is lmm.beta_g


def test_getBetaSNPste_matches_definition():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F)

    lmm.process(G)

    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    ste = lmm.getBetaSNPste()
    np.testing.assert_allclose(ste, beta / z, rtol=1e-10, atol=1e-12)
    assert np.all(np.isfinite(ste))


def test_process_repeated_call_replaces_results():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F)

    lmm.process(G)
    assert lmm.getPv().shape == (S,)

    lmm.process(G, step=2)
    assert lmm.getPv().shape == (S // 2,)
    assert lmm.getBetaSNP().shape == (2, S // 2)


def test_process_does_not_mutate_inputs():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, 2)
    y_before, F_before, G_before, Inter_before = y.copy(), F.copy(), G.copy(), Inter.copy()
    lmm = LMMCore(y, F)

    lmm.process(G, Inter=Inter)

    assert np.array_equal(y, y_before)
    assert np.array_equal(F, F_before)
    assert np.array_equal(G, G_before)
    assert np.array_equal(Inter, Inter_before)


def test_process_verbose_reports_elapsed_time_with_patched_clock(monkeypatch, capsys):
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F)
    ticks = iter([10.0, 11.5])
    # Replace the module-level clock so the printed duration is fixed.
    monkeypatch.setattr(lmm_core_module, "time", lambda: next(ticks))

    lmm.process(G, verbose=True)

    out = capsys.readouterr().out
    assert out == "Tested for %d variants in 1.50 s\n" % S


def test_process_is_silent_by_default(capsys):
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F)

    lmm.process(G)

    assert capsys.readouterr().out == ""


def test_process_with_fewer_variants_than_step_gives_empty_results():
    np.random.seed(SEED)
    y = np.random.randn(N, 1)
    F = _covariates(2)
    G = np.random.randn(N, 2)
    lmm = LMMCore(y, F)

    lmm.process(G, step=3)

    assert lmm.getPv().shape == (0,)
    assert lmm.getLRT().shape == (0,)
    assert lmm.getBetaSNP().shape == (3, 0)


def test_package_exports_lmmcore_and_version():
    assert limix_lmm.LMMCore is LMMCore
    assert limix_lmm.__version__ == "0.1.2"


def test_package_all_lists_public_names():
    expected = {"LMM", "LMMCore", "MTLMM", "plot", "download", "unzip",
                "__version__", "test"}
    assert set(limix_lmm.__all__) == expected
    for name in limix_lmm.__all__:
        assert hasattr(limix_lmm, name)


def test_MTLMM_is_exported_from_mtlmm_module():
    assert limix_lmm.MTLMM is mtlmm_module.MTLMM
    assert isinstance(limix_lmm.MTLMM, type)
