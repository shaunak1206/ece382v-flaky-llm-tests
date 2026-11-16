import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm.lmm_core import LMMCore


def _make_data(n=50, s=6, k=2, seed=0):
    rs = np.random.RandomState(seed)
    y = rs.randn(n, 1)
    F = np.concatenate([np.ones((n, 1)), rs.randn(n, k - 1)], 1)
    G = 1.0 * (rs.rand(n, s) < 0.3) + 0.1 * rs.randn(n, s)
    return y, F, G, rs


def _rss(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X.dot(beta)
    return float((r ** 2).sum()), beta


def test_package_exports():
    np.random.seed(0)
    assert limix_lmm.LMMCore is LMMCore
    assert isinstance(limix_lmm.MTLMM, type)
    assert isinstance(limix_lmm.LMM, type)
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip",
                 "__version__", "test"]:
        assert name in limix_lmm.__all__
    assert limix_lmm.__version__ == "0.1.2"


def test_init_default_covariates():
    np.random.seed(1)
    y, _, _, _ = _make_data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (50, 1)
    assert np.allclose(lmm.F, 1.0)
    assert lmm.df == 49


def test_init_df_with_covariates():
    np.random.seed(2)
    y, F, _, _ = _make_data(n=40, k=3)
    lmm = LMMCore(y, F)
    assert lmm.df == 37
    assert lmm.Ki_dot is None


def test_null_model_matches_ols():
    np.random.seed(3)
    y, F, _, _ = _make_data()
    lmm = LMMCore(y, F)
    rss, beta = _rss(y, F)
    assert np.allclose(np.ravel(lmm.beta_F0), np.ravel(beta), rtol=1e-6, atol=1e-8)
    assert np.isclose(float(np.ravel(lmm.s20)[0]), rss / lmm.df, rtol=1e-6)


def test_identity_ki_dot_equals_none():
    np.random.seed(4)
    y, F, G, _ = _make_data()
    a = LMMCore(y, F)
    b = LMMCore(y, F, lambda x: x.copy())
    a.process(G)
    b.process(G)
    assert np.allclose(a.getPv(), b.getPv(), rtol=1e-8, atol=1e-12)
    assert np.allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-8, atol=1e-12)
    assert np.allclose(a.getLRT(), b.getLRT(), rtol=1e-8, atol=1e-12)


def test_process_shapes_step1():
    np.random.seed(5)
    y, F, G, _ = _make_data(s=7)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (7,)
    assert lmm.getBetaSNP().shape == (1, 7)
    assert lmm.getLRT().shape == (7,)


def test_single_variant_matches_ols():
    np.random.seed(6)
    y, F, G, _ = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    rss0, _ = _rss(y, F)
    for s in range(G.shape[1]):
        X = np.concatenate([F, G[:, [s]]], 1)
        rss1, beta = _rss(y, X)
        assert np.isclose(lmm.getBetaSNP()[0, s], beta[-1, 0], rtol=1e-6, atol=1e-8)
        assert np.isclose(lmm.getLRT()[s], -lmm.df * np.log(rss1 / rss0),
                          rtol=1e-6, atol=1e-8)


def test_pv_consistent_with_lrt_chi2():
    np.random.seed(7)
    y, F, G, _ = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    assert np.all(pv >= 0) and np.all(pv <= 1)
    assert np.allclose(pv, st.chi2(1).sf(lmm.getLRT()), rtol=1e-8, atol=1e-12)


def test_step_shapes_and_pvalue_df():
    np.random.seed(8)
    y, F, G, _ = _make_data(s=8)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)
    assert np.allclose(lmm.getPv(), st.chi2(4).sf(lmm.getLRT()),
                       rtol=1e-8, atol=1e-12)


def test_step_joint_matches_ols():
    np.random.seed(9)
    y, F, G, _ = _make_data(s=8)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    rss0, _ = _rss(y, F)
    X = np.concatenate([F, G[:, 0:2]], 1)
    rss1, beta = _rss(y, X)
    k = F.shape[1]
    assert np.allclose(lmm.getBetaSNP()[:, 0], beta[k:, 0], rtol=1e-6, atol=1e-8)
    assert np.isclose(lmm.getLRT()[0], -lmm.df * np.log(rss1 / rss0),
                      rtol=1e-6, atol=1e-8)


def test_non_divisible_columns_drop_remainder():
    np.random.seed(10)
    y, F, G, _ = _make_data(s=10)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)


def test_interaction_step1_matches_ols():
    np.random.seed(11)
    y, F, G, rs = _make_data(s=5)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getBetaSNP().shape == (2, 5)
    assert lmm.getPv().shape == (5,)
    rss0, _ = _rss(y, F)
    k = F.shape[1]
    for s in range(5):
        X = np.concatenate([F, Inter * G[:, [s]]], 1)
        rss1, beta = _rss(y, X)
        assert np.allclose(lmm.getBetaSNP()[:, s], beta[k:, 0], rtol=1e-6, atol=1e-8)
        assert np.isclose(lmm.getLRT()[s], -lmm.df * np.log(rss1 / rss0),
                          rtol=1e-6, atol=1e-8)
    assert np.allclose(lmm.getPv(), st.chi2(2).sf(lmm.getLRT()),
                       rtol=1e-8, atol=1e-12)


def test_interaction_with_step_shapes():
    np.random.seed(12)
    y, F, G, rs = _make_data(s=8)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)
    assert lmm.getBetaSNP().shape == (4, 4)
    assert lmm.getPv().shape == (4,)
    assert np.allclose(lmm.getPv(), st.chi2(4).sf(lmm.getLRT()),
                       rtol=1e-8, atol=1e-12)


def test_scaled_covariance_inverse_leaves_betas_and_lrt_unchanged():
    np.random.seed(13)
    y, F, G, _ = _make_data()
    plain = LMMCore(y, F)
    scaled = LMMCore(y, F, lambda x: 2.0 * x)
    plain.process(G)
    scaled.process(G)
    assert np.allclose(plain.getBetaSNP(), scaled.getBetaSNP(), rtol=1e-6, atol=1e-8)
    assert np.allclose(plain.getLRT(), scaled.getLRT(), rtol=1e-6, atol=1e-8)
    assert np.allclose(plain.getPv(), scaled.getPv(), rtol=1e-6, atol=1e-10)


def test_strong_signal_gives_small_pvalue():
    np.random.seed(14)
    n = 100
    rs = np.random.RandomState(14)
    g = rs.randn(n, 1)
    F = np.ones((n, 1))
    y = 5.0 * g + 0.1 * rs.randn(n, 1)
    lmm = LMMCore(y, F)
    lmm.process(g)
    assert lmm.getPv()[0] < 1e-10
    assert abs(lmm.getBetaSNP()[0, 0] - 5.0) < 0.1


def test_beta_ste_consistent_with_beta_and_pv():
    np.random.seed(15)
    y, F, G, _ = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    assert ste.shape == beta.shape
    assert np.all(np.isfinite(ste))
    assert np.all(ste > 0)
    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    assert np.allclose(ste * z, beta, rtol=1e-6, atol=1e-10)


def test_getters_return_attributes():
    np.random.seed(16)
    y, F, G, _ = _make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv() is lmm.pv
    assert lmm.getBetaSNP() is lmm.beta_g
    assert lmm.getLRT() is lmm.lrt


def test_verbose_prints_summary(capsys):
    np.random.seed(17)
    y, F, G, _ = _make_data(s=6)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 6 variants" in out
    lmm.process(G, verbose=False)
    assert capsys.readouterr().out == ""


def test_getbetacov_unset_raises():
    np.random.seed(18)
    y, F, G, _ = _make_data()
    lmm = LMMCore(y, F)
    if hasattr(lmm, "beta_F"):
        pytest.skip("beta_F is defined")
    with pytest.raises(AttributeError):
        lmm.getBetaCov()
