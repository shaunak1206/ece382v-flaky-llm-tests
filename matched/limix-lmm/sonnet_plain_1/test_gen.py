import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
import limix_lmm.mtlmm as mtlmm_mod
from limix_lmm import LMMCore, MTLMM
from limix_lmm.util import christof_trick


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def ols(X, y):
    """Least squares coefficients and residual sum of squares."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X.dot(beta)
    return beta, float((res ** 2).sum())


def make_data(n=60, s=6, k=2, seed=0):
    rs = np.random.RandomState(seed)
    y = rs.randn(n, 1)
    F = np.concatenate([np.ones((n, 1)), rs.randn(n, k - 1)], 1)
    G = rs.randn(n, s)
    return y, F, G


class IdentityHelper(object):
    """Stand-in for the covariance helper with K = I (independent noise)."""

    def __init__(self, Y, F, A, Asnp, covar=None):
        self.Y = Y
        self.F = F
        self.A = A
        self.Asnp = Asnp
        self.D = np.ones(Y.shape)
        self.g = None

    def set_g(self, g):
        self.g = g

    def get_WKiy(self):
        rv = self.F.T.dot(self.Y.dot(self.A))
        return np.reshape(rv, [rv.size, 1], order="F")

    def get_yKiy(self):
        return np.einsum("ip,ip->", self.Y, self.Y)

    def get_WKiW(self):
        return christof_trick(self.A, self.F, self.D)

    def get_W1Kiy_2(self):
        return self.Y.dot(self.Asnp).T.dot(self.g)

    def get_W1KiW1_12(self):
        return christof_trick(self.A, self.F, self.D, A2=self.Asnp, F2=self.g)

    def get_W1KiW1_22(self):
        return christof_trick(self.Asnp, self.g, self.D)


@pytest.fixture
def identity_helper(monkeypatch):
    monkeypatch.setattr(mtlmm_mod, "define_helper", IdentityHelper)


# ---------------------------------------------------------------------------
# package level
# ---------------------------------------------------------------------------


def test_package_exports():
    assert limix_lmm.__version__ == "0.1.2"
    for name in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "test"]:
        assert name in limix_lmm.__all__
        assert hasattr(limix_lmm, name)


def test_package_classes_are_the_core_classes():
    from limix_lmm.lmm_core import LMMCore as CoreDirect

    assert limix_lmm.LMMCore is CoreDirect
    assert limix_lmm.MTLMM is MTLMM
    assert issubclass(MTLMM, LMMCore)


# ---------------------------------------------------------------------------
# LMMCore: null model
# ---------------------------------------------------------------------------


def test_lmmcore_default_intercept_covariate():
    y, _, _ = make_data()
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (y.shape[0], 1)
    assert np.all(lmm.F == 1)
    assert lmm.df == y.shape[0] - 1


def test_lmmcore_null_fit_matches_ols():
    y, F, _ = make_data(seed=1)
    lmm = LMMCore(y, F)
    beta, rss = ols(F, y)
    assert lmm.df == y.shape[0] - F.shape[1]
    np.testing.assert_allclose(lmm.beta_F0, beta, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(lmm.s20, rss / lmm.df, rtol=1e-6)
    np.testing.assert_allclose(lmm.yKiy, float(y.T.dot(y)), rtol=1e-8)


def test_lmmcore_identity_ki_dot_equals_no_ki_dot():
    y, F, G = make_data(seed=2)
    a = LMMCore(y, F)
    b = LMMCore(y, F, lambda x: x)
    a.process(G)
    b.process(G)
    np.testing.assert_allclose(a.s20, b.s20)
    np.testing.assert_allclose(a.pv, b.pv)
    np.testing.assert_allclose(a.beta_g, b.beta_g)


def test_lmmcore_scaled_inverse_covariance_scales_s20():
    y, F, _ = make_data(seed=3)
    base = LMMCore(y, F)
    scaled = LMMCore(y, F, lambda x: 2.0 * x)
    np.testing.assert_allclose(scaled.beta_F0, base.beta_F0, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(scaled.s20, 2.0 * base.s20, rtol=1e-6)


# ---------------------------------------------------------------------------
# LMMCore: process
# ---------------------------------------------------------------------------


def test_lmmcore_process_shapes_single_variants():
    y, F, G = make_data(s=7, seed=5)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert lmm.getPv().shape == (7,)
    assert lmm.getLRT().shape == (7,)
    assert lmm.getBetaSNP().shape == (1, 7)


def test_lmmcore_process_matches_ols_likelihood_ratio():
    y, F, G = make_data(seed=6)
    lmm = LMMCore(y, F)
    lmm.process(G)
    _, rss0 = ols(F, y)
    expected_lrt = []
    expected_beta = []
    for s in range(G.shape[1]):
        X = np.concatenate([F, G[:, [s]]], 1)
        beta, rss1 = ols(X, y)
        expected_lrt.append(-lmm.df * np.log(rss1 / rss0))
        expected_beta.append(beta[-1, 0])
    np.testing.assert_allclose(lmm.getLRT(), expected_lrt, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(lmm.getBetaSNP()[0], expected_beta, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(lmm.getPv(), st.chi2(1).sf(lmm.getLRT()), rtol=1e-8)


def test_lmmcore_pvalues_are_probabilities_and_lrt_nonnegative():
    y, F, G = make_data(s=20, seed=7)
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert np.all(lmm.getPv() >= 0)
    assert np.all(lmm.getPv() <= 1)
    assert np.all(lmm.getLRT() > -1e-10)


def test_lmmcore_strong_effect_gives_tiny_pvalue():
    rs = np.random.RandomState(8)
    n = 100
    g = rs.randn(n, 1)
    F = np.ones((n, 1))
    y = 3.0 * g + 0.1 * rs.randn(n, 1)
    lmm = LMMCore(y, F)
    lmm.process(g)
    assert lmm.getPv()[0] < 1e-10
    assert abs(lmm.getBetaSNP()[0, 0] - 3.0) < 0.1


def test_lmmcore_variant_in_covariate_span_has_no_effect():
    y, F, G = make_data(seed=9)
    G = G.copy()
    G[:, 0] = F[:, 1]  # exactly a covariate
    lmm = LMMCore(y, F)
    lmm.process(G)
    assert abs(lmm.getLRT()[0]) < 1e-8
    assert lmm.getPv()[0] > 0.999


def test_lmmcore_process_with_interaction():
    y, F, G = make_data(s=5, seed=10)
    rs = np.random.RandomState(11)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    assert lmm.getBetaSNP().shape == (2, 5)
    assert lmm.getPv().shape == (5,)
    _, rss0 = ols(F, y)
    for s in range(5):
        X = np.concatenate([F, Inter * G[:, [s]]], 1)
        beta, rss1 = ols(X, y)
        np.testing.assert_allclose(lmm.getBetaSNP()[:, s], beta[F.shape[1]:, 0], rtol=1e-6, atol=1e-10)
        expected_lrt = -lmm.df * np.log(rss1 / rss0)
        np.testing.assert_allclose(lmm.getLRT()[s], expected_lrt, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(lmm.getPv()[s], st.chi2(2).sf(expected_lrt), rtol=1e-8)


def test_lmmcore_process_with_step_tests_blocks_jointly():
    y, F, G = make_data(s=8, seed=12)
    lmm = LMMCore(y, F)
    lmm.process(G, step=4)
    assert lmm.getPv().shape == (2,)
    assert lmm.getBetaSNP().shape == (4, 2)
    _, rss0 = ols(F, y)
    for b in range(2):
        X = np.concatenate([F, G[:, 4 * b:4 * (b + 1)]], 1)
        beta, rss1 = ols(X, y)
        np.testing.assert_allclose(lmm.getBetaSNP()[:, b], beta[F.shape[1]:, 0], rtol=1e-6, atol=1e-10)
        expected_lrt = -lmm.df * np.log(rss1 / rss0)
        np.testing.assert_allclose(lmm.getPv()[b], st.chi2(4).sf(expected_lrt), rtol=1e-8)


def test_lmmcore_step_with_interaction_uses_hatodot_ordering():
    y, F, G = make_data(s=6, seed=14)
    rs = np.random.RandomState(15)
    Inter = rs.randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)
    assert lmm.getBetaSNP().shape == (4, 3)
    block = G[:, 0:2]
    X = np.column_stack([Inter[:, i] * block[:, j] for i in range(2) for j in range(2)])
    beta, _ = ols(np.concatenate([F, X], 1), y)
    np.testing.assert_allclose(lmm.getBetaSNP()[:, 0], beta[F.shape[1]:, 0], rtol=1e-6, atol=1e-10)


def test_lmmcore_process_with_general_inverse_covariance():
    n = 40
    y, F, G = make_data(n=n, s=3, seed=16)
    rs = np.random.RandomState(17)
    M = rs.randn(n, n)
    Ki = M.dot(M.T) / n + np.eye(n)

    def ki_dot(x):
        return Ki.dot(x)

    lmm = LMMCore(y, F, ki_dot)
    lmm.process(G)

    def gls(X):
        XKiX = X.T.dot(Ki).dot(X)
        XKiy = X.T.dot(Ki).dot(y)
        beta = np.linalg.solve(XKiX, XKiy)
        return beta, float(y.T.dot(Ki).dot(y) - XKiy.T.dot(beta))

    _, q0 = gls(F)
    for s in range(3):
        beta, q1 = gls(np.concatenate([F, G[:, [s]]], 1))
        np.testing.assert_allclose(lmm.getBetaSNP()[0, s], beta[-1, 0], rtol=1e-6)
        np.testing.assert_allclose(lmm.getLRT()[s], -lmm.df * np.log(q1 / q0), rtol=1e-6, atol=1e-10)


def test_lmmcore_beta_standard_errors_consistent_with_lrt():
    y, F, G = make_data(s=5, seed=18)
    lmm = LMMCore(y, F)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    assert ste.shape == lmm.getBetaSNP().shape
    assert np.all(ste > 0)
    np.testing.assert_allclose(
        ste, np.abs(lmm.getBetaSNP()) / np.sqrt(lmm.getLRT()), rtol=1e-4
    )


def test_lmmcore_verbose_reports_variant_count(capsys):
    y, F, G = make_data(s=6, seed=19)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for 6 variants" in out
    lmm.process(G, verbose=False)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# MTLMM (with an identity-covariance helper)
# ---------------------------------------------------------------------------


def test_mtlmm_defaults_and_degrees_of_freedom(identity_helper):
    rs = np.random.RandomState(20)
    Y = rs.randn(30, 3)
    mt = MTLMM(Y)
    assert mt.F.shape == (30, 1)
    np.testing.assert_array_equal(mt.A, np.eye(3))
    np.testing.assert_array_equal(mt.Asnp, np.eye(3))
    assert mt.df == Y.size - 3 * 1
    assert isinstance(mt.helper, IdentityHelper)


def test_mtlmm_null_fit_matches_per_trait_ols(identity_helper):
    rs = np.random.RandomState(21)
    n, p = 50, 2
    Y = rs.randn(n, p)
    F = np.concatenate([np.ones((n, 1)), rs.randn(n, 2)], 1)
    mt = MTLMM(Y, F=F)
    beta_ols = np.linalg.lstsq(F, Y, rcond=None)[0]
    beta0 = np.reshape(mt.beta_F0, [F.shape[1], p], order="F")
    np.testing.assert_allclose(beta0, beta_ols, rtol=1e-6, atol=1e-8)
    rss = float(((Y - F.dot(beta_ols)) ** 2).sum())
    np.testing.assert_allclose(mt.s20, rss / mt.df, rtol=1e-6)


def test_mtlmm_single_trait_matches_lmmcore(identity_helper):
    y, F, G = make_data(s=5, seed=22)
    core = LMMCore(y, F)
    core.process(G)
    mt = MTLMM(y, F=F)
    pv, beta = mt.process(G)
    np.testing.assert_allclose(mt.s20, core.s20, rtol=1e-6)
    np.testing.assert_allclose(pv, core.getPv(), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(beta, core.getBetaSNP(), rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(mt.getLRT(), core.getLRT(), rtol=1e-6, atol=1e-10)


def test_mtlmm_process_returns_results_stored_on_object(identity_helper):
    rs = np.random.RandomState(23)
    Y = rs.randn(40, 2)
    G = rs.randn(40, 4)
    mt = MTLMM(Y)
    pv, beta = mt.process(G)
    assert pv is mt.getPv()
    assert beta is mt.getBetaSNP()
    assert pv.shape == (4,)
    assert beta.shape == (2, 4)


def test_mtlmm_multi_trait_identity_design_matches_trait_wise_regression(identity_helper):
    rs = np.random.RandomState(24)
    n, p, s = 60, 2, 4
    Y = rs.randn(n, p)
    F = np.concatenate([np.ones((n, 1)), rs.randn(n, 1)], 1)
    G = rs.randn(n, s)
    mt = MTLMM(Y, F=F)
    pv, beta = mt.process(G)
    rss0 = sum(ols(F, Y[:, [j]])[1] for j in range(p))
    for i in range(s):
        X = np.concatenate([F, G[:, [i]]], 1)
        rss1 = 0.0
        for j in range(p):
            b, r = ols(X, Y[:, [j]])
            rss1 += r
            np.testing.assert_allclose(beta[j, i], b[-1, 0], rtol=1e-6, atol=1e-10)
        expected_lrt = -mt.df * np.log(rss1 / rss0)
        np.testing.assert_allclose(mt.getLRT()[i], expected_lrt, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(pv[i], st.chi2(p).sf(expected_lrt), rtol=1e-8)


def test_mtlmm_shared_genetic_effect_design(identity_helper):
    rs = np.random.RandomState(25)
    n, p, s = 50, 3, 3
    Y = rs.randn(n, p)
    G = rs.randn(n, s)
    mt = MTLMM(Y, Asnp=np.ones((p, 1)))
    pv, beta = mt.process(G)
    assert beta.shape == (1, s)
    assert pv.shape == (s,)
    assert np.all((pv >= 0) & (pv <= 1))
    np.testing.assert_allclose(pv, st.chi2(1).sf(mt.getLRT()), rtol=1e-8)


def test_mtlmm_verbose_reports_variant_count(identity_helper, capsys):
    rs = np.random.RandomState(26)
    Y = rs.randn(30, 2)
    G = rs.randn(30, 3)
    mt = MTLMM(Y)
    mt.process(G, verbose=True)
    assert "Tested for 3 variants" in capsys.readouterr().out
