"""Unit tests for limix_lmm.lmm_core.LMMCore, limix_lmm.mtlmm.MTLMM and the package __init__."""

import numpy as np
import pytest
import scipy.stats as st

import limix_lmm
import limix_lmm.mtlmm as mtlmm_mod
from limix_lmm.lmm_core import LMMCore
from limix_lmm.mtlmm import MTLMM


def _ols_rss(X, y):
    """Least-squares fit; returns (residual sum of squares, coefficients)."""
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X.dot(coef)
    return float(resid.T.dot(resid)), coef


def _vec(M):
    """Column-major vectorisation as a column vector."""
    return M.reshape(-1, order="F")[:, None]


def _make_fixture(seed=0, n=60, n_snps=6):
    rs = np.random.RandomState(seed)
    F = np.column_stack([np.ones(n), rs.randn(n)])
    y = rs.randn(n, 1) + 0.5 * F[:, [1]]
    G = rs.randn(n, n_snps)
    return y, F, G


def _spd(n, seed):
    rs = np.random.RandomState(seed)
    A = rs.randn(n, n)
    return A.dot(A.T) + n * np.eye(n)


def _expected_lrt(y, F, X_snp):
    """LRT for adding X_snp to the null design F, with LMMCore dof df = N - L."""
    df = y.shape[0] - F.shape[1]
    rss0, _ = _ols_rss(F, y)
    rss1, _ = _ols_rss(np.hstack([F, X_snp]), y)
    return -df * np.log(rss1 / rss0)


class _IdentityCovarHelper(object):
    """Stand-in for the MTLMM covariance helper with K = I.

    It builds the stacked design explicitly, so the MTLMM tests check the
    block assembly and the chi2 test against a plain OLS reference.
    """

    def __init__(self, Y, F, A, Asnp):
        self._y = _vec(Y)
        self._W = np.kron(A, F)
        self._Asnp = Asnp
        self._g = None

    def set_g(self, g):
        self._g = g

    def _G1(self):
        return np.kron(self._Asnp, self._g)

    def get_WKiy(self):
        return self._W.T.dot(self._y)

    def get_yKiy(self):
        return float(self._y.T.dot(self._y))

    def get_WKiW(self):
        return self._W.T.dot(self._W)

    def get_W1Kiy_2(self):
        return self._G1().T.dot(self._y)

    def get_W1KiW1_12(self):
        return self._W.T.dot(self._G1())

    def get_W1KiW1_22(self):
        G1 = self._G1()
        return G1.T.dot(G1)


@pytest.fixture
def identity_helper(monkeypatch):
    def _fake_define_helper(Y, F, A, Asnp, covar=None):
        return _IdentityCovarHelper(Y, F, A, Asnp)

    monkeypatch.setattr(mtlmm_mod, "define_helper", _fake_define_helper)


# ---------------------------------------------------------------- LMMCore


def test_null_fit_defaults_to_intercept_when_F_is_none():
    y, _, _ = _make_fixture()
    n = y.shape[0]
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (n, 1)
    assert np.all(lmm.F == 1.0)
    assert lmm.df == n - 1
    assert lmm.beta_F0[0, 0] == pytest.approx(y.mean())
    assert lmm.s20 == pytest.approx(((y - y.mean()) ** 2).sum() / (n - 1))


def test_null_fit_matches_ols_residual_variance():
    y, F, _ = _make_fixture()
    lmm = LMMCore(y, F)
    rss0, coef = _ols_rss(F, y)
    assert lmm.df == y.shape[0] - F.shape[1]
    assert lmm.yKiy == pytest.approx(float(y.T.dot(y)))
    assert lmm.s20 == pytest.approx(rss0 / lmm.df, rel=1e-8)
    assert np.allclose(lmm.beta_F0, coef, atol=1e-8)


def test_null_fit_with_scaled_identity_covariance_scales_variance():
    y, F, _ = _make_fixture()
    c = 4.0
    base = LMMCore(y, F)
    scaled = LMMCore(y, F, Ki_dot=lambda x: x / c)
    assert scaled.yKiy == pytest.approx(base.yKiy / c)
    assert scaled.s20 == pytest.approx(base.s20 / c)
    assert np.allclose(scaled.beta_F0, base.beta_F0, atol=1e-10)


def test_null_fit_with_general_covariance_matches_whitened_ols():
    y, F, _ = _make_fixture()
    K = _spd(y.shape[0], seed=11)
    lmm = LMMCore(y, F, Ki_dot=lambda x: np.linalg.solve(K, x))
    L = np.linalg.cholesky(K)
    rss0, coef = _ols_rss(np.linalg.solve(L, F), np.linalg.solve(L, y))
    assert lmm.df == y.shape[0] - F.shape[1]
    assert lmm.s20 == pytest.approx(rss0 / lmm.df, rel=1e-8)
    assert np.allclose(lmm.beta_F0, coef, rtol=1e-6, atol=1e-8)


def test_process_lrt_and_pvalues_match_ols_reference():
    y, F, G = _make_fixture()
    lmm = LMMCore(y, F)
    lmm.process(G)
    expected_lrt = np.array(
        [_expected_lrt(y, F, G[:, [s]]) for s in range(G.shape[1])]
    )
    assert lmm.getPv().shape == (G.shape[1],)
    assert lmm.getBetaSNP().shape == (1, G.shape[1])
    assert lmm.getLRT() is lmm.lrt
    assert lmm.getPv() is lmm.pv
    assert np.allclose(lmm.getLRT(), expected_lrt, rtol=1e-6, atol=1e-8)
    assert np.allclose(
        lmm.getPv(), st.chi2(1).sf(expected_lrt), rtol=1e-5, atol=1e-12
    )


def test_process_beta_matches_ols_genotype_coefficient():
    y, F, G = _make_fixture()
    lmm = LMMCore(y, F)
    lmm.process(G)
    expected = np.array(
        [
            [_ols_rss(np.hstack([F, G[:, [s]]]), y)[1][-1, 0]
             for s in range(G.shape[1])]
        ]
    )
    assert np.allclose(lmm.getBetaSNP(), expected, atol=1e-8)


def test_process_zero_genotype_is_exactly_null():
    y, F, _ = _make_fixture()
    lmm = LMMCore(y, F)
    lmm.process(np.zeros((y.shape[0], 1)))
    assert lmm.getBetaSNP()[0, 0] == pytest.approx(0.0, abs=1e-10)
    assert lmm.getLRT()[0] == pytest.approx(0.0, abs=1e-8)
    assert lmm.getPv()[0] == pytest.approx(1.0)


def test_process_step_tests_consecutive_blocks_jointly():
    y, F, G = _make_fixture(n_snps=7)
    step = 3
    n_tests = G.shape[1] // step  # trailing column is not tested
    lmm = LMMCore(y, F)
    lmm.process(G, step=step)
    assert lmm.getPv().shape == (n_tests,)
    assert lmm.getBetaSNP().shape == (step, n_tests)
    blocks = [G[:, s * step:(s + 1) * step] for s in range(n_tests)]
    expected_lrt = np.array([_expected_lrt(y, F, b) for b in blocks])
    expected_beta = np.column_stack(
        [_ols_rss(np.hstack([F, b]), y)[1][-step:, 0] for b in blocks]
    )
    assert np.allclose(lmm.getLRT(), expected_lrt, rtol=1e-6, atol=1e-8)
    assert np.allclose(lmm.getBetaSNP(), expected_beta, atol=1e-7)
    assert np.allclose(
        lmm.getPv(), st.chi2(step).sf(expected_lrt), rtol=1e-5, atol=1e-12
    )


def test_process_interaction_matches_ols_reference():
    y, F, G = _make_fixture(n_snps=4)
    Inter = np.random.RandomState(7).randn(y.shape[0], 2)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter)
    expected_lrt = []
    expected_beta = []
    for s in range(G.shape[1]):
        X = Inter * G[:, [s]]
        expected_lrt.append(_expected_lrt(y, F, X))
        expected_beta.append(_ols_rss(np.hstack([F, X]), y)[1][-2:, 0])
    expected_lrt = np.array(expected_lrt)
    assert lmm.getPv().shape == (G.shape[1],)
    assert lmm.getBetaSNP().shape == (2, G.shape[1])
    assert np.allclose(lmm.getLRT(), expected_lrt, rtol=1e-6, atol=1e-8)
    assert np.allclose(lmm.getBetaSNP(), np.column_stack(expected_beta), atol=1e-7)
    assert np.allclose(
        lmm.getPv(), st.chi2(2).sf(expected_lrt), rtol=1e-5, atol=1e-12
    )


def test_process_interaction_block_column_order_matches_hatodot():
    y, F, G = _make_fixture(n_snps=4)
    M, step = 2, 2
    Inter = np.random.RandomState(3).randn(y.shape[0], M)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=step)
    n_tests = G.shape[1] // step
    assert lmm.getBetaSNP().shape == (M * step, n_tests)
    for s in range(n_tests):
        block = G[:, s * step:(s + 1) * step]
        # hatodot order: interaction index outer, variant index inner
        design = np.column_stack(
            [Inter[:, i] * block[:, j] for i in range(M) for j in range(step)]
        )
        expected = _ols_rss(np.hstack([F, design]), y)[1][-M * step:, 0]
        assert np.allclose(lmm.getBetaSNP()[:, s], expected, atol=1e-7)


def test_process_with_general_covariance_matches_whitened_ols():
    y, F, G = _make_fixture()
    K = _spd(y.shape[0], seed=5)
    lmm = LMMCore(y, F, Ki_dot=lambda x: np.linalg.solve(K, x))
    lmm.process(G)
    L = np.linalg.cholesky(K)
    y_w = np.linalg.solve(L, y)
    F_w = np.linalg.solve(L, F)
    G_w = np.linalg.solve(L, G)
    rss0, _ = _ols_rss(F_w, y_w)
    expected_lrt = []
    expected_beta = []
    for s in range(G.shape[1]):
        rss1, coef = _ols_rss(np.hstack([F_w, G_w[:, [s]]]), y_w)
        expected_lrt.append(-lmm.df * np.log(rss1 / rss0))
        expected_beta.append(coef[-1, 0])
    assert np.allclose(lmm.getLRT(), expected_lrt, rtol=1e-6, atol=1e-8)
    assert np.allclose(lmm.getBetaSNP()[0], expected_beta, atol=1e-6)


def test_process_verbose_reports_timing(capsys):
    y, F, G = _make_fixture()
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=False)
    assert capsys.readouterr().out == ""
    lmm.process(G, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for %d variants" % G.shape[1] in out


def test_beta_snp_standard_error_matches_wald_formula():
    y, F, G = _make_fixture()
    lmm = LMMCore(y, F)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    expected = np.abs(beta) / np.sqrt(st.chi2(1).isf(pv))
    assert np.all(np.isfinite(expected))
    assert np.allclose(lmm.getBetaSNPste(), expected, rtol=1e-6)


@pytest.mark.xfail(
    raises=AttributeError,
    strict=False,
    reason="getBetaCov reads self.beta_F, which LMMCore never sets (only beta_F0)",
)
def test_get_beta_cov_returns_covariate_effects():
    y, F, _ = _make_fixture()
    lmm = LMMCore(y, F)
    assert np.allclose(lmm.getBetaCov(), lmm.beta_F0)


# ----------------------------------------------------------------- MTLMM


@pytest.mark.usefixtures("identity_helper")
def test_mtlmm_defaults_to_intercept_and_identity_designs():
    rs = np.random.RandomState(1)
    n, p = 20, 3
    Y = rs.randn(n, p)
    lmm = MTLMM(Y)
    assert lmm.F.shape == (n, 1)
    assert np.all(lmm.F == 1.0)
    assert np.array_equal(lmm.A, np.eye(p))
    assert np.array_equal(lmm.Asnp, np.eye(p))
    assert lmm.df == n * p - p
    assert np.allclose(lmm.beta_F0[:, 0], Y.mean(axis=0), atol=1e-8)


@pytest.mark.usefixtures("identity_helper")
def test_mtlmm_null_fit_matches_stacked_ols():
    rs = np.random.RandomState(0)
    n, p = 25, 3
    F = np.column_stack([np.ones(n), rs.randn(n)])
    Y = rs.randn(n, p) + 0.3 * F[:, [1]]
    A = rs.randn(p, 2)
    lmm = MTLMM(Y, F=F, A=A)
    rss0, coef = _ols_rss(np.kron(A, F), _vec(Y))
    assert lmm.df == n * p - A.shape[1] * F.shape[1]
    assert lmm.s20 == pytest.approx(rss0 / lmm.df, rel=1e-8)
    assert np.allclose(lmm.beta_F0, coef, atol=1e-7)


@pytest.mark.usefixtures("identity_helper")
def test_mtlmm_process_matches_stacked_ols_reference():
    rs = np.random.RandomState(0)
    n, p, S = 25, 3, 4
    F = np.column_stack([np.ones(n), rs.randn(n)])
    Y = rs.randn(n, p) + 0.3 * F[:, [1]]
    A = rs.randn(p, 2)
    G = rs.randn(n, S)
    lmm = MTLMM(Y, F=F, A=A)
    pv, beta = lmm.process(G)

    X0 = np.kron(A, F)
    rss0, _ = _ols_rss(X0, _vec(Y))
    expected_lrt = []
    expected_beta = []
    for s in range(S):
        X1 = np.hstack([X0, np.kron(np.eye(p), G[:, [s]])])
        rss1, coef = _ols_rss(X1, _vec(Y))
        expected_lrt.append(-lmm.df * np.log(rss1 / rss0))
        expected_beta.append(coef[-p:, 0])
    expected_lrt = np.array(expected_lrt)
    expected_beta = np.column_stack(expected_beta)

    assert pv.shape == (S,)
    assert beta.shape == (p, S)
    assert beta is lmm.beta_g
    assert pv is lmm.pv
    assert np.allclose(lmm.lrt, expected_lrt, rtol=1e-6, atol=1e-8)
    assert np.allclose(beta, expected_beta, atol=1e-7)
    assert np.allclose(pv, st.chi2(p).sf(expected_lrt), rtol=1e-5, atol=1e-12)


@pytest.mark.usefixtures("identity_helper")
def test_mtlmm_shared_genetic_effect_has_one_coefficient():
    rs = np.random.RandomState(0)
    n, p, S = 25, 3, 4
    F = np.column_stack([np.ones(n), rs.randn(n)])
    Y = rs.randn(n, p) + 0.3 * F[:, [1]]
    A = rs.randn(p, 2)
    G = rs.randn(n, S)
    Asnp = np.ones((p, 1))
    lmm = MTLMM(Y, F=F, A=A, Asnp=Asnp)
    pv, beta = lmm.process(G)
    assert beta.shape == (1, S)
    assert pv.shape == (S,)

    X0 = np.kron(A, F)
    rss0, _ = _ols_rss(X0, _vec(Y))
    for s in range(S):
        X1 = np.hstack([X0, np.kron(Asnp, G[:, [s]])])
        rss1, coef = _ols_rss(X1, _vec(Y))
        assert beta[0, s] == pytest.approx(coef[-1, 0], abs=1e-7)
        expected_lrt = -lmm.df * np.log(rss1 / rss0)
        assert lmm.lrt[s] == pytest.approx(expected_lrt, rel=1e-6, abs=1e-8)
        assert pv[s] == pytest.approx(
            st.chi2(1).sf(expected_lrt), rel=1e-5, abs=1e-12
        )


def test_mtlmm_rejects_unsupported_covariance_type():
    pytest.importorskip("limix_core.covar")
    Y = np.random.RandomState(0).randn(5, 2)
    with pytest.raises(AssertionError):
        MTLMM(Y, covar=object())


@pytest.mark.xfail(
    raises=AttributeError,
    strict=False,
    reason="MTLMMHelperBase reads self.LY and self.Y, which it never sets",
)
def test_mtlmm_without_covariance_fits_null_model():
    pytest.importorskip("limix_core")
    Y = np.random.RandomState(2).randn(10, 2)
    lmm = MTLMM(Y)
    assert lmm.df == 10 * 2 - 2


# ------------------------------------------------------------- __init__


def test_package_exports_declared_public_names():
    expected = {
        "LMM",
        "LMMCore",
        "MTLMM",
        "plot",
        "download",
        "unzip",
        "__version__",
        "test",
    }
    assert set(limix_lmm.__all__) == expected
    for name in limix_lmm.__all__:
        assert hasattr(limix_lmm, name)
    assert limix_lmm.__version__ == "0.1.2"


def test_package_reexports_core_classes_from_their_modules():
    from limix_lmm import lmm_core

    assert limix_lmm.LMMCore is lmm_core.LMMCore
    assert limix_lmm.MTLMM is mtlmm_mod.MTLMM
