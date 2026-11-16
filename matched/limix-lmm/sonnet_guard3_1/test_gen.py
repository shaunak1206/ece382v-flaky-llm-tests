import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm.lmm_core import LMMCore


def _make_data(N=50, S=8, seed=0):
    np.random.seed(seed)
    y = np.random.randn(N, 1)
    F = np.concatenate([np.ones((N, 1)), np.random.randn(N, 1)], 1)
    G = np.random.randn(N, S)
    return y, F, G


def _ols(X, y):
    return np.linalg.lstsq(X, y[:, 0], rcond=None)[0]


def test_package_exports():
    np.random.seed(0)
    from limix_lmm import MTLMM

    assert "LMMCore" in limix_lmm.__all__
    assert "MTLMM" in limix_lmm.__all__
    assert "LMM" in limix_lmm.__all__
    assert limix_lmm.LMMCore is LMMCore
    assert isinstance(MTLMM, type)
    assert limix_lmm.__version__ == "0.1.2"


def test_all_names_resolvable():
    np.random.seed(0)
    for name in limix_lmm.__all__:
        assert hasattr(limix_lmm, name)


def test_default_covariates_and_df():
    np.random.seed(0)
    y, _, _ = _make_data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (y.shape[0], 1)
    assert np.allclose(lmm.F, 1.0)
    assert lmm.df == y.shape[0] - 1


def test_df_with_covariates():
    np.random.seed(0)
    y, F, _ = _make_data()
    lmm = LMMCore(y, F)
    assert lmm.df == y.shape[0] - F.shape[1]


def test_null_beta_matches_ols():
    np.random.seed(0)
    y, F, _ = _make_data()
    lmm = LMMCore(y, F)
    expected = _ols(F, y)
    assert np.allclose(np.asarray(lmm.beta_F0).ravel(), expected, atol=1e-8)


def test_null_s20_matches_residual_variance():
    np.random.seed(0)
    y, F, _ = _make_data()
    lmm = LMMCore(y, F)
    beta = _ols(F, y)
    rss = np.sum((y[:, 0] - F.dot(beta)) ** 2)
    assert np.allclose(np.ravel(lmm.s20)[0], rss / lmm.df, rtol=1e-6)


def test_process_shapes_step1_no_inter():
    np.random.seed(0)
    y, F, G = _make_data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (8,)
    assert lmm.getBetaSNP().shape == (1, 8)
    assert lmm.getLRT().shape == (8,)
    assert lmm.getBetaSNPste().shape == (1, 8)


def test_process_shapes_with_step_and_leftover():
    np.random.seed(0)
    y, F, G = _make_data(S=10)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    # 10 // 4 == 2 complete blocks
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_process_shapes_with_inter_and_step():
    np.random.seed(0)
    y, F, G = _make_data(S=8)
    Inter = np.random.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)
    assert lmm.getPv().shape == (4,)
    assert lmm.getBetaSNP().shape == (4, 4)


def test_beta_matches_joint_ols():
    np.random.seed(0)
    y, F, G = _make_data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    for s in range(G.shape[1]):
        X = np.hstack([F, G[:, s:s + 1]])
        expected = _ols(X, y)[-1]
        assert np.allclose(beta[0, s], expected, atol=1e-7)


def test_beta_with_interaction_matches_ols():
    np.random.seed(0)
    y, F, G = _make_data(S=4)
    Inter = np.random.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=1)
    beta = lmm.getBetaSNP()
    assert beta.shape == (2, 4)
    for s in range(G.shape[1]):
        X = np.hstack([F, Inter * G[:, s:s + 1]])
        expected = _ols(X, y)[F.shape[1]:]
        assert np.allclose(beta[:, s], expected, atol=1e-7)


def test_pvalue_consistent_with_lrt_chi2():
    np.random.seed(0)
    y, F, G = _make_data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    expected = st.chi2(1).sf(lmm.getLRT())
    assert np.allclose(lmm.getPv(), expected, atol=1e-12)

    Inter = np.random.randn(y.shape[0], 3)
    lmm.process(G, Inter=Inter)
    expected = st.chi2(3).sf(lmm.getLRT())
    assert np.allclose(lmm.getPv(), expected, atol=1e-12)


def test_pvalues_in_unit_interval_and_lrt_nonnegative():
    np.random.seed(0)
    y, F, G = _make_data(S=12)
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    assert np.all(pv >= 0.0) and np.all(pv <= 1.0)
    assert np.all(lmm.getLRT() >= -1e-8)


def test_identity_ki_dot_equals_none():
    np.random.seed(0)
    y, F, G = _make_data(S=5)
    a = LMMCore(y, F)
    a.process(G)
    b = LMMCore(y, F, lambda x: x.copy())
    b.process(G)
    assert np.allclose(a.getPv(), b.getPv(), atol=1e-10)
    assert np.allclose(a.getBetaSNP(), b.getBetaSNP(), atol=1e-10)
    assert np.allclose(a.getLRT(), b.getLRT(), atol=1e-8)


def test_scaled_covariance_does_not_change_inference():
    np.random.seed(0)
    y, F, G = _make_data(S=5)
    a = LMMCore(y, F)
    a.process(G)
    b = LMMCore(y, F, lambda x: x / 2.0)
    b.process(G)
    assert np.allclose(a.getBetaSNP(), b.getBetaSNP(), atol=1e-8)
    assert np.allclose(a.getPv(), b.getPv(), atol=1e-8)


def test_strong_signal_gives_small_pvalue():
    np.random.seed(0)
    N = 100
    F = np.ones((N, 1))
    g = np.random.randn(N, 1)
    noise = 0.01 * np.random.randn(N, 1)
    y = 5.0 * g + noise
    lmm = LMMCore(y, F)
    lmm.process(g)
    assert lmm.getPv()[0] < 1e-10
    assert abs(lmm.getBetaSNP()[0, 0] - 5.0) < 0.1


def test_beta_ste_consistent_with_pvalue():
    np.random.seed(0)
    y, F, G = _make_data(S=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    ste = lmm.getBetaSNPste()
    z2 = (beta / ste) ** 2
    assert np.allclose(z2[0], st.chi2(1).isf(lmm.getPv()), rtol=1e-6, atol=1e-8)
    assert np.all(ste > 0)


def test_process_twice_overwrites_results():
    np.random.seed(0)
    y, F, G = _make_data(S=8)
    lmm = LMMCore(y, F)
    lmm.process(G)
    first = lmm.getPv().copy()
    lmm.process(G, step=2)
    assert lmm.getPv().shape == (4,)
    lmm.process(G)
    assert np.allclose(lmm.getPv(), first, atol=1e-12)


def test_verbose_output(capsys):
    np.random.seed(0)
    y, F, G = _make_data(S=7)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 7 variants" in out


def test_quiet_by_default(capsys):
    np.random.seed(0)
    y, F, G = _make_data(S=3)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert capsys.readouterr().out == ""
