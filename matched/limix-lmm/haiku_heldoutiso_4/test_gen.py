import numpy as np
import pytest
from scipy.stats import chi2

import limix_lmm
from limix_lmm import LMMCore, MTLMM


N = 50
S = 12
L = 2


def _make_data(seed=0, n=N, s=S, n_cov=L):
    rs = np.random.RandomState(seed)
    y = rs.randn(n, 1)
    F = np.hstack([np.ones((n, 1)), rs.randn(n, n_cov - 1)])
    G = (rs.rand(n, s) < 0.3).astype(float)
    return y, F, G


def _scalar(x):
    return float(np.squeeze(np.asarray(x)))


# ---------------------------------------------------------------------------
# LMMCore: construction and null model
# ---------------------------------------------------------------------------

def test_default_F_is_intercept_column():
    y, _, _ = _make_data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (N, 1)
    assert np.all(lmm.F == 1.0)
    assert lmm.df == N - 1


def test_df_is_n_minus_number_of_covariates():
    y, F, _ = _make_data(n_cov=3)
    lmm = LMMCore(y, F, None)
    assert lmm.df == N - 3


def test_null_fit_without_ki_dot_uses_identity_covariance():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F, None)
    assert np.allclose(lmm.Kiy, y)
    assert np.allclose(lmm.KiF, F)
    assert np.isclose(_scalar(lmm.yKiy), float(y[:, 0] @ y[:, 0]))


def test_null_fit_matches_least_squares_beta_and_sigma():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F, None)
    beta_ls = np.linalg.lstsq(F, y, rcond=None)[0]
    rss = float(np.sum((y - F @ beta_ls) ** 2))
    assert np.allclose(np.asarray(lmm.beta_F0).reshape(-1), beta_ls.reshape(-1))
    assert np.isclose(_scalar(lmm.s20), rss / lmm.df)


def test_ki_dot_is_applied_to_y_and_F():
    y, F, _ = _make_data()
    lmm = LMMCore(y, F, lambda x: 2.0 * x)
    assert np.allclose(lmm.Kiy, 2.0 * y)
    assert np.allclose(lmm.KiF, 2.0 * F)
    assert np.isclose(_scalar(lmm.yKiy), 2.0 * float(y[:, 0] @ y[:, 0]))


def test_identity_ki_dot_matches_no_ki_dot():
    y, F, G = _make_data()
    lmm_none = LMMCore(y, F, None)
    lmm_id = LMMCore(y, F, lambda x: x)
    lmm_none.process(G)
    lmm_id.process(G)
    assert np.allclose(lmm_none.getPv(), lmm_id.getPv())
    assert np.allclose(lmm_none.getBetaSNP(), lmm_id.getBetaSNP())


def test_global_covariance_scaling_leaves_pvalues_unchanged():
    y, F, G = _make_data()
    lmm_a = LMMCore(y, F, None)
    lmm_b = LMMCore(y, F, lambda x: 2.0 * x)
    lmm_a.process(G)
    lmm_b.process(G)
    assert np.allclose(lmm_a.getPv(), lmm_b.getPv())


# ---------------------------------------------------------------------------
# LMMCore: process() output shapes
# ---------------------------------------------------------------------------

def test_process_default_shapes_without_interaction():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    assert lmm.getPv().shape == (S,)
    assert lmm.getBetaSNP().shape == (1, S)
    assert lmm.getLRT().shape == (S,)


def test_process_with_step_reduces_number_of_tests():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (S // 4,)
    assert lmm.getBetaSNP().shape == (4, S // 4)


def test_process_with_non_divisible_step_drops_trailing_variants():
    y, F, G = _make_data(s=10)
    lmm = LMMCore(y, F, None)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_process_with_interaction_shapes_step1():
    y, F, G = _make_data()
    Inter = np.random.RandomState(3).randn(N, 2)
    lmm = LMMCore(y, F, None)
    lmm.process(G, Inter=Inter)
    assert lmm.getPv().shape == (S,)
    assert lmm.getBetaSNP().shape == (2, S)


def test_process_with_interaction_and_step_shapes():
    y, F, G = _make_data()
    Inter = np.random.RandomState(3).randn(N, 2)
    lmm = LMMCore(y, F, None)
    lmm.process(G, Inter=Inter, step=3)
    assert lmm.getPv().shape == (S // 3,)
    assert lmm.getBetaSNP().shape == (6, S // 3)


def test_process_can_be_rerun_with_different_step():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G, step=1)
    assert lmm.getPv().shape == (S,)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (S // 4,)
    assert lmm.getBetaSNP().shape == (4, S // 4)


# ---------------------------------------------------------------------------
# LMMCore: statistical properties of outputs
# ---------------------------------------------------------------------------

def test_pvalues_lie_in_unit_interval_and_are_finite():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    pv = lmm.getPv()
    assert np.all(np.isfinite(pv))
    assert np.all(pv >= 0.0) and np.all(pv <= 1.0)


def test_lrt_is_nonnegative_for_nested_models():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    assert np.all(lmm.getLRT() >= -1e-8)


def test_pvalues_consistent_with_chi2_survival_of_lrt():
    y, F, G = _make_data()
    Inter = np.random.RandomState(5).randn(N, 2)
    lmm = LMMCore(y, F, None)
    lmm.process(G, Inter=Inter, step=3)
    m = 3 * 2
    assert np.allclose(lmm.getPv(), chi2(m).sf(lmm.getLRT()))


def test_strong_signal_gives_small_pvalue():
    rs = np.random.RandomState(7)
    G = rs.randn(N, S)
    y = 3.0 * G[:, [0]] + 0.01 * rs.randn(N, 1)
    F = np.ones((N, 1))
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    assert lmm.getPv()[0] < 1e-6


def test_beta_standard_errors_match_formula():
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    ste = lmm.getBetaSNPste()
    assert ste.shape == beta.shape
    expected = np.abs(beta) / np.sqrt(chi2(1).isf(pv))
    assert np.allclose(ste, expected)


# ---------------------------------------------------------------------------
# LMMCore: interaction handling
# ---------------------------------------------------------------------------

def test_all_ones_interaction_equals_no_interaction_step1():
    y, F, G = _make_data()
    ones = np.ones((N, 1))
    lmm_plain = LMMCore(y, F, None)
    lmm_inter = LMMCore(y, F, None)
    lmm_plain.process(G)
    lmm_inter.process(G, Inter=ones)
    assert np.allclose(lmm_plain.getPv(), lmm_inter.getPv())
    assert np.allclose(lmm_plain.getBetaSNP(), lmm_inter.getBetaSNP())


def test_all_ones_interaction_equals_no_interaction_with_step():
    y, F, G = _make_data()
    ones = np.ones((N, 1))
    lmm_plain = LMMCore(y, F, None)
    lmm_inter = LMMCore(y, F, None)
    lmm_plain.process(G, step=4)
    lmm_inter.process(G, Inter=ones, step=4)
    assert np.allclose(lmm_plain.getPv(), lmm_inter.getPv())


def test_verbose_flag_reports_number_of_variants(capsys):
    y, F, G = _make_data()
    lmm = LMMCore(y, F, None)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for %d variants" % S in out


# ---------------------------------------------------------------------------
# Package exports (limix_lmm/__init__.py)
# ---------------------------------------------------------------------------

def test_lmmcore_and_mtlmm_are_exported_classes():
    assert isinstance(LMMCore, type)
    assert isinstance(MTLMM, type)
    assert LMMCore is not MTLMM


def test_all_names_in_package_all_resolve():
    for name in limix_lmm.__all__:
        assert hasattr(limix_lmm, name), name


def test_version_is_nonempty_string():
    assert isinstance(limix_lmm.__version__, str)
    assert limix_lmm.__version__.strip() != ""
