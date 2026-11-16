"""Unit tests for LMMCore (limix_lmm/lmm_core.py) and the MTLMM export
from the top-level limix_lmm package (limix_lmm/__init__.py).

Note: MTLMM is defined in limix_lmm/mtlmm.py, not in lmm_core.py, and its
source was not provided. These tests therefore only check that MTLMM is
exported correctly and does not exercise its behaviour.
"""

import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm import LMMCore, MTLMM
from limix_lmm import lmm_core as lmm_core_module


N = 50
S = 6


@pytest.fixture
def data():
    rs = np.random.RandomState(0)
    y = rs.randn(N, 1)
    F = np.hstack([np.ones((N, 1)), rs.randn(N, 1)])
    G = (rs.rand(N, S) < 0.3).astype(float)
    # ensure no constant genotype columns
    G[0, :] = 1.0
    G[1, :] = 0.0
    Inter = rs.randn(N, 2)
    return y, F, G, Inter


def _ols(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X.dot(beta)
    return float(resid.T.dot(resid)), beta


class _CountingKi(object):
    """Identity-like covariance inverse that records what it was called on."""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.calls = []

    def __call__(self, x):
        self.calls.append(np.array(x, copy=True))
        return self.scale * x


def test_init_defaults_F_to_intercept_when_none(data):
    y, _, _, _ = data
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (N, 1)
    np.testing.assert_allclose(lmm.F, np.ones((N, 1)))
    assert lmm.df == N - 1


def test_init_stores_inputs_and_degrees_of_freedom(data):
    y, F, _, _ = data
    lmm = LMMCore(y, F)
    assert lmm.y is y
    assert lmm.F is F
    assert lmm.Ki_dot is None
    assert lmm.df == N - F.shape[1]


def test_null_fit_without_covariance_inverse_uses_identity(data):
    y, F, _, _ = data
    lmm = LMMCore(y, F)
    np.testing.assert_allclose(lmm.Kiy, y)
    np.testing.assert_allclose(lmm.KiF, F)
    np.testing.assert_allclose(lmm.FKiy, F.T.dot(y))
    np.testing.assert_allclose(lmm.FKiF, F.T.dot(F))
    np.testing.assert_allclose(lmm.yKiy, float(y.T.dot(y)))


def test_null_fit_matches_ols_under_identity_covariance(data):
    y, F, _, _ = data
    lmm = LMMCore(y, F)
    rss0, beta_ols = _ols(F, y)
    np.testing.assert_allclose(np.ravel(lmm.beta_F0), np.ravel(beta_ols), rtol=1e-6)
    np.testing.assert_allclose(lmm.s20, rss0 / lmm.df, rtol=1e-6)


def test_ki_dot_is_applied_to_y_and_F_during_init(data):
    y, F, _, _ = data
    ki = _CountingKi(scale=0.5)
    lmm = LMMCore(y, F, ki)
    assert len(ki.calls) == 2
    np.testing.assert_allclose(ki.calls[0], y)
    np.testing.assert_allclose(ki.calls[1], F)
    np.testing.assert_allclose(lmm.Kiy, 0.5 * y)
    np.testing.assert_allclose(lmm.yKiy, 0.5 * float(y.T.dot(y)))


def test_process_output_shapes_default_step(data):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (S,)
    assert lmm.getLRT().shape == (S,)
    assert lmm.getBetaSNP().shape == (1, S)


def test_process_output_shapes_with_step_and_truncation():
    rs = np.random.RandomState(1)
    n, s = 40, 10
    y = rs.randn(n, 1)
    F = np.ones((n, 1))
    G = rs.randn(n, s)
    lmm = LMMCore(y, F)
    lmm.process(G, step=3)  # int(10 / 3) == 3 tests, last variant ignored
    assert lmm.getPv().shape == (3,)
    assert lmm.getLRT().shape == (3,)
    assert lmm.getBetaSNP().shape == (3, 3)


def test_process_with_interaction_shapes(data):
    y, F, G, Inter = data
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getPv().shape == (S,)
    assert lmm.getBetaSNP().shape == (Inter.shape[1], S)

    lmm.process(G, Inter=Inter, step=2)
    ntests = S // 2
    assert lmm.getPv().shape == (ntests,)
    assert lmm.getBetaSNP().shape == (2 * Inter.shape[1], ntests)


def test_lrt_matches_manual_ols_single_variant(data):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G)
    rss0, _ = _ols(F, y)
    expected = np.empty(S)
    for s in range(S):
        x = G[:, s:s + 1]
        rss1, _ = _ols(np.hstack([F, x]), y)
        expected[s] = -lmm.df * np.log(rss1 / rss0)
    np.testing.assert_allclose(lmm.getLRT(), expected, rtol=1e-6, atol=1e-10)


def test_beta_matches_manual_ols_single_variant(data):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G)
    for s in range(S):
        x = G[:, s:s + 1]
        _, beta = _ols(np.hstack([F, x]), y)
        np.testing.assert_allclose(
            lmm.getBetaSNP()[0, s], beta[-1, 0], rtol=1e-6, atol=1e-10
        )


def test_lrt_matches_manual_ols_block_of_variants(data):
    y, F, G, _ = data
    step = 2
    lmm = LMMCore(y, F)
    lmm.process(G, step=step)
    rss0, _ = _ols(F, y)
    ntests = S // step
    expected = np.empty(ntests)
    for s in range(ntests):
        block = G[:, step * s:step * (s + 1)]
        rss1, _ = _ols(np.hstack([F, block]), y)
        expected[s] = -lmm.df * np.log(rss1 / rss0)
    np.testing.assert_allclose(lmm.getLRT(), expected, rtol=1e-6, atol=1e-10)


def test_lrt_matches_manual_ols_with_interaction_step1(data):
    y, F, G, Inter = data
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    rss0, _ = _ols(F, y)
    expected = np.empty(S)
    for s in range(S):
        g = G[:, s:s + 1]
        X = np.hstack([F, Inter * g])
        rss1, _ = _ols(X, y)
        expected[s] = -lmm.df * np.log(rss1 / rss0)
    np.testing.assert_allclose(lmm.getLRT(), expected, rtol=1e-6, atol=1e-10)


def test_pv_is_chi2_survival_of_lrt(data):
    y, F, G, Inter = data
    lmm = LMMCore(y, F)
    lmm.process(G)
    np.testing.assert_allclose(
        lmm.getPv(), st.chi2(1).sf(lmm.getLRT()), rtol=1e-10, atol=1e-300
    )
    lmm.process(G, Inter=Inter, step=2)
    np.testing.assert_allclose(
        lmm.getPv(), st.chi2(4).sf(lmm.getLRT()), rtol=1e-10, atol=1e-300
    )
    assert np.all((lmm.getPv() >= 0) & (lmm.getPv() <= 1))


def test_lrt_and_pv_invariant_to_covariance_scaling(data):
    y, F, G, _ = data
    base = LMMCore(y, F)
    base.process(G)
    scaled = LMMCore(y, F, lambda x: 0.5 * x)
    scaled.process(G)
    np.testing.assert_allclose(scaled.getLRT(), base.getLRT(), rtol=1e-6)
    np.testing.assert_allclose(scaled.getPv(), base.getPv(), rtol=1e-6)
    np.testing.assert_allclose(scaled.getBetaSNP(), base.getBetaSNP(), rtol=1e-6)


def test_ki_dot_is_applied_to_genotype_blocks_during_process(data):
    y, F, G, _ = data
    ki = _CountingKi(scale=1.0)
    lmm = LMMCore(y, F, ki)
    calls_after_init = len(ki.calls)
    lmm.process(G)
    # one Ki_dot call per tested block
    assert len(ki.calls) - calls_after_init == S


def test_getters_return_fitted_attributes(data):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv() is lmm.pv
    assert lmm.getLRT() is lmm.lrt
    assert lmm.getBetaSNP() is lmm.beta_g


def test_getBetaSNPste_matches_formula(data):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    ste = lmm.getBetaSNPste()
    assert ste.shape == beta.shape
    np.testing.assert_allclose(ste, beta / z, rtol=1e-10)


def test_process_verbose_prints_timing_message(data, capsys):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for %d variants" % S in out or "Tested for {} variants".format(S) in out


def test_process_silent_by_default(data, capsys):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert capsys.readouterr().out == ""


def test_mtlmm_is_exported_from_package_root():
    assert isinstance(MTLMM, type)
    from limix_lmm import mtlmm as mtlmm_module
    assert limix_lmm.MTLMM is mtlmm_module.MTLMM
    assert "MTLMM" in limix_lmm.__all__


def test_lmmcore_exported_from_package_root_is_same_class():
    assert limix_lmm.LMMCore is lmm_core_module.LMMCore
    assert LMMCore is lmm_core_module.LMMCore


def test_package_all_entries_are_attributes():
    for name in limix_lmm.__all__:
        assert hasattr(limix_lmm, name), name


def test_package_version_string():
    assert limix_lmm.__version__ == "0.1.2"
