import inspect

import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm import LMMCore, MTLMM
from limix_lmm import lmm_core as lmm_core_module

# Note: the provided sources do not include MTLMM's implementation.
# limix_lmm/__init__.py imports it from limix_lmm/mtlmm.py, so these tests
# only check how MTLMM is exposed and do not exercise its behaviour.

N = 60
S = 12


def _make_data(seed=0, n=N, s=S, n_inter=None):
    rs = np.random.RandomState(seed)
    y = rs.randn(n, 1)
    F = np.hstack([np.ones((n, 1)), rs.randn(n, 1)])
    G = rs.randn(n, s)
    Inter = rs.randn(n, n_inter) if n_inter is not None else None
    return y, F, G, Inter


def _rss(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X.dot(beta)
    return float(r.T.dot(r)[0, 0])


def _reference_lrt(y, F, g):
    """Least-squares LRT for a single variant under K = I."""
    rss0 = _rss(y, F)
    rss1 = _rss(y, np.hstack([F, g]))
    df = y.shape[0] - F.shape[1]
    return -df * np.log(rss1 / rss0)


@pytest.fixture
def data():
    return _make_data()


# ---------------------------------------------------------------- package


def test_package_exports_expected_names():
    for name in limix_lmm.__all__:
        assert hasattr(limix_lmm, name), name
    for name in ["LMMCore", "MTLMM", "__version__", "test"]:
        assert name in limix_lmm.__all__


def test_version_string():
    assert limix_lmm.__version__ == "0.1.2"


def test_package_lmmcore_is_module_lmmcore():
    assert limix_lmm.LMMCore is lmm_core_module.LMMCore
    assert inspect.isclass(LMMCore)


def test_package_mtlmm_is_exported_class():
    assert inspect.isclass(MTLMM)
    from limix_lmm.mtlmm import MTLMM as direct

    assert MTLMM is direct


# ---------------------------------------------------------------- __init__


def test_init_defaults_F_to_intercept(data):
    y, _, _, _ = data
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (N, 1)
    assert np.all(lmm.F == 1.0)
    assert lmm.df == N - 1


def test_init_df_is_n_minus_covariates(data):
    y, F, _, _ = data
    lmm = LMMCore(y, F)
    assert lmm.df == N - F.shape[1]


def test_init_without_ki_dot_uses_identity(data):
    y, F, _, _ = data
    lmm = LMMCore(y, F, None)
    assert np.allclose(lmm.Kiy, y)
    assert np.allclose(lmm.KiF, F)
    assert np.allclose(lmm.yKiy, float(y.T.dot(y)))
    assert np.allclose(lmm.FKiy, F.T.dot(y))


def test_init_with_ki_dot_applies_operator(data):
    y, F, _, _ = data
    lmm = LMMCore(y, F, lambda x: 2.0 * x)
    assert np.allclose(lmm.Kiy, 2.0 * y)
    assert np.allclose(lmm.KiF, 2.0 * F)
    assert np.allclose(lmm.yKiy, 2.0 * float(y.T.dot(y)))


def test_null_fit_beta_matches_least_squares(data):
    y, F, _, _ = data
    lmm = LMMCore(y, F)
    beta_ls = np.linalg.lstsq(F, y, rcond=None)[0]
    assert np.allclose(np.ravel(lmm.beta_F0), np.ravel(beta_ls))


# ---------------------------------------------------------------- process


def test_process_output_shapes_step1(data):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (S,)
    assert lmm.getLRT().shape == (S,)
    assert lmm.getBetaSNP().shape == (1, S)
    assert lmm.getBetaSNPste().shape == (1, S)


def test_process_step_groups_variants(data):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (S // 4,)
    assert lmm.getBetaSNP().shape == (4, S // 4)


def test_process_step_drops_incomplete_block():
    y, F, G, _ = _make_data(s=10)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_process_with_inter_step1_shapes():
    y, F, G, Inter = _make_data(n_inter=2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getPv().shape == (S,)
    assert lmm.getBetaSNP().shape == (2, S)


def test_process_with_inter_and_step_shapes():
    y, F, G, Inter = _make_data(s=12, n_inter=2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=3)
    # 12 variants in blocks of 3 -> 4 tests, 2 interactions * 3 variants = 6 betas
    assert lmm.getPv().shape == (4,)
    assert lmm.getBetaSNP().shape == (6, 4)


def test_pvalues_are_chi2_survival_of_lrt(data):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    m = lmm.getBetaSNP().shape[0]
    expected = st.chi2(m).sf(lmm.getLRT())
    assert np.allclose(lmm.getPv(), expected)
    assert np.all(lmm.getPv() >= 0) and np.all(lmm.getPv() <= 1)


def test_lrt_matches_least_squares_reference(data):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G)
    expected = np.array([_reference_lrt(y, F, G[:, [s]]) for s in range(S)])
    assert np.allclose(lmm.getLRT(), expected, rtol=1e-6, atol=1e-8)


def test_lrt_invariant_to_covariance_scale(data):
    # Only sigma^2 is learnt, so scaling K by a constant leaves the LRT unchanged.
    y, F, G, _ = data
    lmm_a = LMMCore(y, F, None)
    lmm_a.process(G)
    lmm_b = LMMCore(y, F, lambda x: 2.0 * x)
    lmm_b.process(G)
    assert np.allclose(lmm_a.getLRT(), lmm_b.getLRT(), rtol=1e-8)
    assert np.allclose(lmm_a.getPv(), lmm_b.getPv(), rtol=1e-8)


def test_identity_ki_dot_equivalent_to_none(data):
    y, F, G, _ = data
    lmm_none = LMMCore(y, F, None)
    lmm_none.process(G)
    lmm_id = LMMCore(y, F, lambda x: x)
    lmm_id.process(G)
    assert np.allclose(lmm_none.getBetaSNP(), lmm_id.getBetaSNP())
    assert np.allclose(lmm_none.getLRT(), lmm_id.getLRT())


def test_process_is_repeatable(data):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G)
    first = lmm.getPv().copy()
    lmm.process(G)
    assert np.allclose(first, lmm.getPv())


def test_process_verbose_prints_summary(data, capsys):
    y, F, G, _ = data
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for %d variants" % S in out


# ---------------------------------------------------------------- getters


def test_getters_return_stored_attributes(data):
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
    assert np.allclose(lmm.getBetaSNPste(), beta / z)
</br>
