import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
from limix_lmm.lmm_core import LMMCore
from limix_lmm.mtlmm import MTLMM
from limix_lmm.mtlmm_helper import MTLMMHelper2KronSum


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def ols(X, y):
    """Least squares fit returning (beta, residual sum of squares)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X.dot(beta)
    return beta, float(r.dot(r))


def make_data(seed=0, N=60, L=2, S=6):
    rs = np.random.RandomState(seed)
    F = np.concatenate([np.ones((N, 1)), rs.randn(N, L - 1)], axis=1)
    G = rs.randn(N, S)
    y = (F.dot(rs.randn(L, 1)) + 0.5 * G[:, [0]] + rs.randn(N, 1)).reshape(N, 1)
    return y, F, G


class FakeCovar(object):
    """Identity covariance exposing the interface used by the 2KronSum helper."""

    def __init__(self, N, P):
        self._N = N
        self._P = P

    def Lr(self):
        return np.eye(self._N)

    def Lc(self):
        return np.eye(self._P)

    def D(self):
        return np.ones((self._N, self._P))


@pytest.fixture
def patched_helper(monkeypatch):
    """Make MTLMM use the 2KronSum helper with an identity fake covariance."""

    def fake_define_helper(Y, F, A, Asnp, covar=None):
        return MTLMMHelper2KronSum(Y, F, A, Asnp, FakeCovar(Y.shape[0], Y.shape[1]))

    monkeypatch.setattr("limix_lmm.mtlmm.define_helper", fake_define_helper)


def make_mt_data(seed=1, N=50, P=3, S=4, L=1):
    rs = np.random.RandomState(seed)
    if L == 1:
        F = np.ones((N, 1))
    else:
        F = np.concatenate([np.ones((N, 1)), rs.randn(N, L - 1)], axis=1)
    G = rs.randn(N, S)
    Y = rs.randn(N, P) + 0.7 * G[:, [0]]
    return Y, F, G


# ---------------------------------------------------------------------------
# package level
# ---------------------------------------------------------------------------


def test_package_exports():
    assert limix_lmm.__version__ == "0.1.2"
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "test"]:
        assert name in limix_lmm.__all__
        assert hasattr(limix_lmm, name)
    assert limix_lmm.LMMCore is LMMCore
    assert limix_lmm.MTLMM is MTLMM
    assert issubclass(MTLMM, LMMCore)


# ---------------------------------------------------------------------------
# LMMCore
# ---------------------------------------------------------------------------


def test_lmmcore_default_covariates_is_intercept():
    y, _, _ = make_data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (y.shape[0], 1)
    np.testing.assert_array_equal(lmm.F, np.ones((y.shape[0], 1)))
    assert lmm.df == y.shape[0] - 1
    np.testing.assert_allclose(lmm.beta_F0[0, 0], y.mean())


def test_lmmcore_null_model_matches_ols():
    y, F, _ = make_data(L=3)
    lmm = LMMCore(y, F)
    beta, rss = ols(F, y[:, 0])
    assert lmm.df == y.shape[0] - F.shape[1]
    np.testing.assert_allclose(lmm.beta_F0[:, 0], beta, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(lmm.s20, rss / lmm.df, rtol=1e-6)
    np.testing.assert_allclose(lmm.yKiy, y[:, 0].dot(y[:, 0]))
    np.testing.assert_allclose(lmm.FKiF, F.T.dot(F))


def test_lmmcore_process_single_variant_matches_ols():
    y, F, G = make_data()
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getBetaSNP().shape == (1, G.shape[1])
    assert lmm.getPv().shape == (G.shape[1],)
    assert lmm.getLRT().shape == (G.shape[1],)
    for s in range(G.shape[1]):
        X = np.concatenate([F, G[:, [s]]], axis=1)
        beta, _ = ols(X, y[:, 0])
        np.testing.assert_allclose(
            lmm.getBetaSNP()[0, s], beta[-1], rtol=1e-6, atol=1e-8
        )


def test_lmmcore_lrt_and_pvalue_formula():
    y, F, G = make_data(seed=3)
    lmm = LMMCore(y, F)
    lmm.process(G)
    _, rss0 = ols(F, y[:, 0])
    for s in range(G.shape[1]):
        X = np.concatenate([F, G[:, [s]]], axis=1)
        _, rss1 = ols(X, y[:, 0])
        expected_lrt = -lmm.df * np.log(rss1 / rss0)
        np.testing.assert_allclose(lmm.getLRT()[s], expected_lrt, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(
            lmm.getPv()[s], st.chi2(1).sf(expected_lrt), rtol=1e-5, atol=1e-12
        )
    assert np.all(lmm.getPv() >= 0) and np.all(lmm.getPv() <= 1)


def test_lmmcore_strong_effect_gives_small_pvalue():
    rs = np.random.RandomState(5)
    N = 200
    G = rs.randn(N, 5)
    y = (3.0 * G[:, [0]] + 0.1 * rs.randn(N, 1)).reshape(N, 1)
    lmm = LMMCore(y, None)
    lmm.process(G)
    pv = lmm.getPv()
    assert pv[0] < 1e-10
    assert np.argmin(pv) == 0
    assert lmm.getBetaSNP()[0, 0] == pytest.approx(3.0, abs=0.1)


def test_lmmcore_step_tests_variants_jointly():
    y, F, G = make_data(seed=7, S=6)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    assert lmm.getBetaSNP().shape == (2, 3)
    assert lmm.getPv().shape == (3,)
    _, rss0 = ols(F, y[:, 0])
    for s in range(3):
        X = np.concatenate([F, G[:, 2 * s : 2 * s + 2]], axis=1)
        beta, rss1 = ols(X, y[:, 0])
        np.testing.assert_allclose(
            lmm.getBetaSNP()[:, s], beta[-2:], rtol=1e-6, atol=1e-8
        )
        lrt = -lmm.df * np.log(rss1 / rss0)
        np.testing.assert_allclose(lmm.getLRT()[s], lrt, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(
            lmm.getPv()[s], st.chi2(2).sf(lrt), rtol=1e-5, atol=1e-12
        )


def test_lmmcore_step_drops_incomplete_trailing_block():
    y, F, G = make_data(seed=8, S=5)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (2, 2)


def test_lmmcore_interaction_step_one():
    y, F, G = make_data(seed=9, S=4)
    rs = np.random.RandomState(10)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getBetaSNP().shape == (2, 4)
    for s in range(4):
        X = np.concatenate([F, Inter * G[:, [s]]], axis=1)
        beta, _ = ols(X, y[:, 0])
        np.testing.assert_allclose(
            lmm.getBetaSNP()[:, s], beta[-2:], rtol=1e-6, atol=1e-8
        )


def test_lmmcore_interaction_with_step_uses_pairwise_products():
    y, F, G = make_data(seed=11, S=4)
    rs = np.random.RandomState(12)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)
    assert lmm.getBetaSNP().shape == (4, 2)
    assert lmm.getPv().shape == (2,)
    _, rss0 = ols(F, y[:, 0])
    for s in range(2):
        g = G[:, 2 * s : 2 * s + 2]
        cols = []
        for i in range(Inter.shape[1]):
            for j in range(g.shape[1]):
                cols.append(Inter[:, i] * g[:, j])
        X = np.concatenate([F, np.stack(cols, axis=1)], axis=1)
        beta, rss1 = ols(X, y[:, 0])
        np.testing.assert_allclose(
            lmm.getBetaSNP()[:, s], beta[-4:], rtol=1e-5, atol=1e-7
        )
        lrt = -lmm.df * np.log(rss1 / rss0)
        np.testing.assert_allclose(
            lmm.getPv()[s], st.chi2(4).sf(lrt), rtol=1e-4, atol=1e-12
        )


def test_lmmcore_identity_Ki_dot_equals_no_Ki_dot():
    y, F, G = make_data(seed=13)
    a = LMMCore(y, F)
    a.process(G)
    b = LMMCore(y, F, lambda x: x)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP())
    np.testing.assert_allclose(a.getLRT(), b.getLRT())
    np.testing.assert_allclose(a.getPv(), b.getPv())
    np.testing.assert_allclose(a.s20, b.s20)


def test_lmmcore_scaled_Ki_dot_keeps_betas_and_lrt():
    y, F, G = make_data(seed=14)
    a = LMMCore(y, F)
    a.process(G)
    b = LMMCore(y, F, lambda x: 2.0 * x)
    b.process(G)
    np.testing.assert_allclose(a.getBetaSNP(), b.getBetaSNP(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(a.getLRT(), b.getLRT(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(b.s20, 2.0 * a.s20, rtol=1e-6)


def test_lmmcore_diagonal_Ki_dot_matches_weighted_least_squares():
    y, F, G = make_data(seed=15)
    rs = np.random.RandomState(16)
    w = rs.uniform(0.5, 2.0, size=y.shape[0])

    def Ki_dot(x):
        return w[:, None] * x

    lmm = LMMCore(y, F, Ki_dot)
    lmm.process(G)
    sw = np.sqrt(w)
    _, wrss0 = ols(sw[:, None] * F, sw * y[:, 0])
    np.testing.assert_allclose(lmm.s20, wrss0 / lmm.df, rtol=1e-6)
    for s in range(G.shape[1]):
        X = np.concatenate([F, G[:, [s]]], axis=1)
        beta, wrss1 = ols(sw[:, None] * X, sw * y[:, 0])
        np.testing.assert_allclose(
            lmm.getBetaSNP()[0, s], beta[-1], rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            lmm.getLRT()[s], -lmm.df * np.log(wrss1 / wrss0), rtol=1e-6, atol=1e-9
        )


def test_lmmcore_zero_genotype_column_has_no_effect():
    y, F, G = make_data(seed=17)
    G = G.copy()
    G[:, 1] = 0.0
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getBetaSNP()[0, 1] == pytest.approx(0.0, abs=1e-8)
    assert lmm.getLRT()[1] == pytest.approx(0.0, abs=1e-8)
    assert lmm.getPv()[1] == pytest.approx(1.0, abs=1e-6)


def test_lmmcore_beta_standard_errors_single_variant():
    y, F, G = make_data(seed=18)
    lmm = LMMCore(y, F)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    assert ste.shape == lmm.getBetaSNP().shape
    expected = np.abs(lmm.getBetaSNP()) / np.sqrt(lmm.getLRT())
    np.testing.assert_allclose(ste, expected, rtol=1e-4)
    assert np.all(ste > 0)


def test_lmmcore_process_verbose_prints_summary(capsys):
    y, F, G = make_data(seed=19, S=3)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 3 variants" in out
    lmm.process(G, verbose=False)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# MTLMM (identity covariance through the 2KronSum helper)
# ---------------------------------------------------------------------------


def test_mtlmm_defaults_and_degrees_of_freedom(patched_helper):
    Y, _, _ = make_mt_data(N=40, P=3)
    mt = MTLMM(Y)
    np.testing.assert_array_equal(mt.F, np.ones((40, 1)))
    np.testing.assert_array_equal(mt.A, np.eye(3))
    np.testing.assert_array_equal(mt.Asnp, np.eye(3))
    assert mt.df == Y.size - 3 * 1
    assert mt.Y is Y


def test_mtlmm_null_model_matches_per_trait_ols(patched_helper):
    Y, F, _ = make_mt_data(N=45, P=3, L=2)
    mt = MTLMM(Y, F=F)
    P, L = Y.shape[1], F.shape[1]
    assert mt.df == Y.size - P * L
    assert mt.beta_F0.shape == (P * L, 1)
    rss = 0.0
    for p in range(P):
        beta, r = ols(F, Y[:, p])
        np.testing.assert_allclose(
            mt.beta_F0[:, 0].reshape(P, L)[p], beta, rtol=1e-6, atol=1e-8
        )
        rss += r
    np.testing.assert_allclose(mt.s20, rss / mt.df, rtol=1e-6)
    np.testing.assert_allclose(mt.yKiy, (Y ** 2).sum())


def test_mtlmm_process_returns_pv_and_beta(patched_helper):
    Y, F, G = make_mt_data(N=50, P=3, S=4)
    mt = MTLMM(Y, F=F)
    pv, beta = mt.process(G)
    assert pv.shape == (4,)
    assert beta.shape == (3, 4)
    assert pv is mt.getPv()
    assert beta is mt.getBetaSNP()
    assert mt.getLRT().shape == (4,)
    assert np.all((pv >= 0) & (pv <= 1))


def test_mtlmm_independent_traits_match_per_trait_ols(patched_helper):
    Y, F, G = make_mt_data(seed=21, N=50, P=3, S=4, L=2)
    mt = MTLMM(Y, F=F)
    pv, beta = mt.process(G)
    P = Y.shape[1]
    _, rss0 = zip(*[ols(F, Y[:, p]) for p in range(P)])
    for s in range(G.shape[1]):
        rss1 = 0.0
        for p in range(P):
            X = np.concatenate([F, G[:, [s]]], axis=1)
            b, r = ols(X, Y[:, p])
            rss1 += r
            np.testing.assert_allclose(beta[p, s], b[-1], rtol=1e-6, atol=1e-8)
        lrt = -mt.df * np.log(rss1 / (sum(rss0)))
        np.testing.assert_allclose(mt.getLRT()[s], lrt, rtol=1e-5, atol=1e-9)
        np.testing.assert_allclose(pv[s], st.chi2(P).sf(lrt), rtol=1e-4, atol=1e-12)


def test_mtlmm_shared_genetic_effect_with_Asnp(patched_helper):
    Y, F, G = make_mt_data(seed=22, N=40, P=3, S=3)
    N, P = Y.shape
    Asnp = np.ones((P, 1))
    mt = MTLMM(Y, F=F, Asnp=Asnp)
    pv, beta = mt.process(G)
    assert beta.shape == (1, 3)
    yvec = Y.T.reshape(-1)  # column stacking: index p*N + i
    W = np.kron(np.eye(P), np.ones((N, 1)))
    for s in range(3):
        X = np.concatenate([W, np.kron(Asnp, G[:, [s]])], axis=1)
        b, _ = ols(X, yvec)
        np.testing.assert_allclose(beta[0, s], b[-1], rtol=1e-6, atol=1e-8)
        assert 0.0 <= pv[s] <= 1.0


def test_mtlmm_zero_genotype_gives_null_result(patched_helper):
    Y, F, G = make_mt_data(seed=23, N=40, P=2, S=3)
    G = G.copy()
    G[:, 2] = 0.0
    mt = MTLMM(Y, F=F)
    pv, beta = mt.process(G)
    np.testing.assert_allclose(beta[:, 2], 0.0, atol=1e-8)
    assert mt.getLRT()[2] == pytest.approx(0.0, abs=1e-8)
    assert pv[2] == pytest.approx(1.0, abs=1e-6)


def test_mtlmm_process_verbose_prints_summary(patched_helper, capsys):
    Y, F, G = make_mt_data(seed=24, N=30, P=2, S=3)
    mt = MTLMM(Y, F=F)
    mt.process(G, verbose=True)
    assert "Tested for 3 variants" in capsys.readouterr().out
    mt.process(G)
    assert capsys.readouterr().out == ""


def test_mtlmm_rejects_unsupported_covariance_type():
    pytest.importorskip("limix_core")
    Y, F, _ = make_mt_data(N=20, P=2)
    with pytest.raises(AssertionError):
        MTLMM(Y, F=F, covar=object())
