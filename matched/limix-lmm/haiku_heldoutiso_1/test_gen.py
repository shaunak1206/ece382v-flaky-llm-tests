"""
Unit tests for limix_lmm.lmm_core.LMMCore and the public exports in
limix_lmm/__init__.py.

Note: in the provided sources, MTLMM is not defined in lmm_core.py. It is
imported into limix_lmm/__init__.py from limix_lmm/mtlmm.py, so the MTLMM
tests check that export and its origin rather than a definition in lmm_core.py.
"""

import inspect

import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm import LMMCore, MTLMM, __version__
from limix_lmm.lmm_core import LMMCore as LMMCoreFromModule


N = 50
S = 6


@pytest.fixture
def data():
    rs = np.random.RandomState(0)
    y = rs.randn(N, 1)
    F = np.hstack([np.ones((N, 1)), rs.randn(N, 2)])
    G = rs.randn(N, S)
    return {"y": y, "F": F, "G": G, "rs": rs}


def _rss(X, y):
    """Residual sum of squares and coefficients from ordinary least squares."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X.dot(beta)
    return float(r.T.dot(r)), beta


# ---------------------------------------------------------------------------
# Null model fitting
# ---------------------------------------------------------------------------

def test_default_F_is_intercept_and_df(data):
    y = data["y"]
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (N, 1)
    assert np.all(lmm.F == 1.0)
    assert lmm.df == N - 1


def test_df_is_n_minus_number_of_covariates(data):
    lmm = LMMCore(data["y"], data["F"])
    assert lmm.df == N - data["F"].shape[1]


def test_no_ki_dot_uses_identity_covariance(data):
    y, F = data["y"], data["F"]
    lmm = LMMCore(y, F)
    assert np.allclose(lmm.Kiy, y)
    assert np.allclose(lmm.KiF, F)


def test_intercept_only_null_fit_matches_sample_statistics(data):
    y = data["y"]
    lmm = LMMCore(y, None)
    assert np.isclose(np.ravel(lmm.beta_F0)[0], y.mean())
    assert np.isclose(lmm.s20, np.var(y, ddof=1))


def test_yKiy_is_scaled_quadratic_form(data):
    y = data["y"]
    lmm = LMMCore(y, data["F"], Ki_dot=lambda x: 2.0 * x)
    assert np.isclose(float(lmm.yKiy), 2.0 * float(y.T.dot(y)))


def test_ki_dot_is_called_on_y_and_F(data):
    y, F = data["y"], data["F"]
    calls = []

    def ki(x):
        calls.append(np.array(x))
        return x

    LMMCore(y, F, Ki_dot=ki)
    assert len(calls) == 2
    assert np.array_equal(calls[0], y)
    assert np.array_equal(calls[1], F)


# ---------------------------------------------------------------------------
# process(): output shapes and values
# ---------------------------------------------------------------------------

def test_process_single_variant_output_shapes(data):
    lmm = LMMCore(data["y"], data["F"])
    lmm.process(data["G"])
    assert lmm.getPv().shape == (S,)
    assert lmm.getLRT().shape == (S,)
    assert lmm.getBetaSNP().shape == (1, S)
    assert lmm.getBetaSNPste().shape == (1, S)


def test_process_step_groups_variants(data):
    lmm = LMMCore(data["y"], data["F"])
    lmm.process(data["G"], step=2)
    assert lmm.getPv().shape == (S // 2,)
    assert lmm.getBetaSNP().shape == (2, S // 2)


def test_process_with_inter_step1_shape(data):
    Inter = data["rs"].randn(N, 2)
    lmm = LMMCore(data["y"], data["F"])
    lmm.process(data["G"], Inter=Inter)
    assert lmm.getPv().shape == (S,)
    assert lmm.getBetaSNP().shape == (2, S)


def test_process_with_inter_and_step_shape(data):
    Inter = data["rs"].randn(N, 2)
    lmm = LMMCore(data["y"], data["F"])
    lmm.process(data["G"], Inter=Inter, step=3)
    assert lmm.getPv().shape == (S // 3,)
    assert lmm.getBetaSNP().shape == (2 * 3, S // 3)


def test_process_reprocessing_overwrites_results(data):
    lmm = LMMCore(data["y"], data["F"])
    lmm.process(data["G"], step=1)
    lmm.process(data["G"], step=3)
    assert lmm.getPv().shape == (S // 3,)
    assert lmm.getBetaSNP().shape == (3, S // 3)


def test_pvalues_lie_in_unit_interval(data):
    lmm = LMMCore(data["y"], data["F"])
    lmm.process(data["G"])
    pv = lmm.getPv()
    assert np.all(pv >= 0.0) and np.all(pv <= 1.0)


def test_lrt_is_nonnegative(data):
    # The alternative model nests the null, so its residual sum of squares
    # can not be larger; hence the LRT statistic must be non-negative.
    lmm = LMMCore(data["y"], data["F"])
    lmm.process(data["G"])
    assert np.all(lmm.getLRT() >= -1e-10)


def test_single_variant_matches_ordinary_least_squares(data):
    y, F, G = data["y"], data["F"], data["G"]
    lmm = LMMCore(y, F)
    lmm.process(G)

    rss0, _ = _rss(F, y)
    df = lmm.df
    for s in range(S):
        rss1, b = _rss(np.hstack([F, G[:, [s]]]), y)
        assert np.isclose(lmm.getLRT()[s], -df * np.log(rss1 / rss0))
        assert np.isclose(lmm.getBetaSNP()[0, s], b[-1, 0])
        assert np.isclose(
            lmm.getPv()[s], st.chi2(1).sf(-df * np.log(rss1 / rss0))
        )


def test_interaction_matches_ordinary_least_squares(data):
    y, F, G = data["y"], data["F"], data["G"]
    Inter = data["rs"].randn(N, 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)

    for s in range(S):
        X_s = Inter * G[:, [s]]
        _, b = _rss(np.hstack([F, X_s]), y)
        assert np.allclose(lmm.getBetaSNP()[:, s], b[-2:, 0])


def test_scaled_covariance_does_not_change_results(data):
    y, F, G = data["y"], data["F"], data["G"]
    base = LMMCore(y, F)
    base.process(G)
    scaled = LMMCore(y, F, Ki_dot=lambda x: 3.0 * x)
    scaled.process(G)
    assert np.allclose(base.getPv(), scaled.getPv())
    assert np.allclose(base.getBetaSNP(), scaled.getBetaSNP())


def test_generalized_least_squares_with_diagonal_covariance(data):
    y, F, G = data["y"], data["F"], data["G"]
    d = data["rs"].uniform(0.5, 2.0, N)
    Ki = np.diag(1.0 / d)
    lmm = LMMCore(y, F, Ki_dot=lambda x: x / d[:, None])
    lmm.process(G)

    for s in range(S):
        X = np.hstack([F, G[:, [s]]])
        beta = np.linalg.solve(X.T.dot(Ki).dot(X), X.T.dot(Ki).dot(y))
        assert np.isclose(lmm.getBetaSNP()[0, s], beta[-1, 0])


def test_beta_ste_consistent_with_pvalues(data):
    lmm = LMMCore(data["y"], data["F"])
    lmm.process(data["G"])
    beta = lmm.getBetaSNP()
    ste = lmm.getBetaSNPste()
    z_abs = np.sqrt(st.chi2(1).isf(lmm.getPv()))
    assert np.allclose(np.abs(beta / ste), z_abs)


def test_verbose_prints_timing_message(data, capsys):
    lmm = LMMCore(data["y"], data["F"])
    lmm.process(data["G"], verbose=True)
    out = capsys.readouterr().out
    assert "Tested for %d variants" % S in out.replace("Tested for %d" % S, "Tested for %d" % S)
    assert out.startswith("Tested for %d variants" % S)


def test_getters_before_process_raise(data):
    lmm = LMMCore(data["y"], data["F"])
    with pytest.raises(AttributeError):
        lmm.getPv()
    with pytest.raises(AttributeError):
        lmm.getBetaSNP()


# ---------------------------------------------------------------------------
# Package exports (limix_lmm/__init__.py)
# ---------------------------------------------------------------------------

def test_version_string():
    assert __version__ == "0.1.2"
    assert limix_lmm.__version__ == "0.1.2"


def test_all_exported_names_exist():
    for name in limix_lmm.__all__:
        assert hasattr(limix_lmm, name), name


def test_exported_lmmcore_is_module_class():
    assert LMMCore is LMMCoreFromModule
    assert inspect.isclass(LMMCore)


def test_exported_mtlmm_comes_from_mtlmm_module():
    from limix_lmm.mtlmm import MTLMM as MTLMMFromModule

    assert MTLMM is MTLMMFromModule
    assert inspect.isclass(MTLMM)
    assert "MTLMM" in limix_lmm.__all__
