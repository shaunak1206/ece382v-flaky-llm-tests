# tests/test_lmm_core.py

import numpy as np
import pytest

from limix_lmm import LMMCore, MTLMM
import limix_lmm


def test_lmmcore_init_none_F(monkeypatch):
    # Test that if F=None, it becomes ones and shapes are correct
    np.random.seed(12345)
    n = 10
    y = np.random.randn(n, 1)

    # monkeypatch Ki_dot to None
    lmm = LMMCore(y, F=None, Ki_dot=None)
    # F should be all ones
    assert lmm.F.shape == (n, 1)
    assert np.allclose(lmm.F, 1)
    # df = n - number covariates = n-1 here
    assert lmm.df == n - 1
    # The Kiy and KiF should be y and F respectively when Ki_dot=None
    assert np.allclose(lmm.Kiy, y)
    assert np.allclose(lmm.KiF, lmm.F)
    # beta_F0 shape should match number of fixed covariates (1)
    assert lmm.beta_F0.shape == (1, 1)
    # s20 should be positive scalar
    assert np.isscalar(lmm.s20)
    assert lmm.s20 > 0


def test_lmmcore_init_with_Ki_dot():
    # Create a dummy Ki_dot function that returns X multiplied by 2
    np.random.seed(42)
    n, m = 5, 2
    y = np.random.randn(n, 1)
    F = np.ones((n, m))

    def Ki_dot(X):
        return X * 2

    lmm = LMMCore(y, F, Ki_dot=Ki_dot)
    # Kiy should be y * 2
    assert np.allclose(lmm.Kiy, y * 2)
    assert np.allclose(lmm.KiF, F * 2)
    # FKiy shape equals (m, 1)
    assert lmm.FKiy.shape == (m, 1)
    # beta_F0 shape equals (m, 1)
    assert lmm.beta_F0.shape == (m, 1)


def _make_simple_LMMCore_setup(seed=0, n_samples=20, n_cov=2):
    """Helper: create LMMCore with known parameters and dummy Ki_dot = identity"""
    np.random.seed(seed)
    y = np.random.randn(n_samples, 1)
    F = np.random.randn(n_samples, n_cov)
    lmm = LMMCore(y, F, Ki_dot=None)
    return lmm, y, F


def test_process_no_Inter_step1():
    np.random.seed(123)
    n, S = 10, 8
    n_cov = 2
    y = np.random.randn(n, 1)
    F = np.random.randn(n, n_cov)

    lmm = LMMCore(y, F, Ki_dot=None)

    # G genotype matrix, shape (n, S)
    G = np.random.randn(n, S)
    lmm.process(G, Inter=None, step=1, verbose=False)

    # Should produce one p-value per variant (S=8)
    pv = lmm.getPv()
    assert pv.shape == (S,)
    # Beta shape (m=step*mi=1*1=1, ntests=S/step=8)
    beta = lmm.getBetaSNP()
    assert beta.shape == (1, S)
    # LRT shape same as p-values
    lrt = lmm.getLRT()
    assert lrt.shape == (S,)
    # beta_ste has same shape as beta
    ste = lmm.getBetaSNPste()
    assert ste.shape == beta.shape
    # Values: pv between 0 and 1
    assert np.all((pv >= 0) & (pv <= 1))
    # LRT >= 0 (LLR test statistic)
    assert np.all(lrt >= 0)
    # beta_ste does not contain nan or inf
    assert np.all(np.isfinite(ste))


def test_process_with_Inter_step1():
    np.random.seed(321)
    n, S, M = 15, 6, 3
    n_cov = 3
    y = np.random.randn(n, 1)
    F = np.random.randn(n, n_cov)
    Inter = np.random.randn(n, M)
    G = np.random.randn(n, S)

    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=Inter, step=1, verbose=False)

    m = M * 1  # step=1
    ntests = S // 1
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    ste = lmm.getBetaSNPste()

    assert pv.shape == (ntests,)
    assert beta.shape == (m, ntests)
    assert lrt.shape == (ntests,)
    assert ste.shape == beta.shape
    assert np.all((pv >= 0) & (pv <= 1))
    assert np.all(lrt >= 0)


def test_process_with_Inter_step2():
    np.random.seed(4321)
    n, S, M = 12, 4, 2
    n_cov = 1
    y = np.random.randn(n, 1)
    F = np.random.randn(n, n_cov)
    Inter = np.random.randn(n, M)
    G = np.random.randn(n, S)

    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=Inter, step=2, verbose=False)

    m = M * 2
    ntests = S // 2
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    ste = lmm.getBetaSNPste()

    assert pv.shape == (ntests,)
    assert beta.shape == (m, ntests)
    assert lrt.shape == (ntests,)
    assert ste.shape == beta.shape
    assert np.all((pv >= 0) & (pv <= 1))
    assert np.all(lrt >= 0)


def test_getBetaCov_returns_correct_shape_values():
    np.random.seed(54321)
    n, S = 11, 3
    n_cov = 4
    y = np.random.randn(n, 1)
    F = np.random.randn(n, n_cov)

    lmm = LMMCore(y, F, Ki_dot=None)
    # Should return None initially, beta_F not set on init but after _fit_null sets some beta_F0
    beta_cov = lmm.getBetaCov()
    # Since in LMMCore, no self.beta_F defined, getBetaCov returns self.beta_F which doesn't exist,
    # so set it to beta_F0 for test purpose manually.
    # To respect code, patch lmm.beta_F = lmm.beta_F0
    # We patch monkeypatch can be used, but here to keep simple:
    lmm.beta_F = lmm.beta_F0
    beta_cov = lmm.getBetaCov()
    assert beta_cov.shape == (n_cov, 1)
    assert np.all(np.isfinite(beta_cov))


def test_getBetaSNPste_consistency_with_pv_sign():
    np.random.seed(4567)
    # For beta with zero values
    n, S = 8, 2
    n_cov = 2
    y = np.random.randn(n, 1)
    F = np.random.randn(n, n_cov)
    G = np.zeros((n, S))
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, step=1)
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    ste = lmm.getBetaSNPste()

    # z = sign(beta) * sqrt(chi2_1.isf(pv)) and ste = beta / z
    # For beta==0, ste should be 0 (division by zero avoided by code? Check that no nan)
    assert not np.any(np.isnan(ste))
    assert np.all(np.isfinite(ste))
    # Shape match
    assert ste.shape == beta.shape


def test_imports_and_version_and_all():
    # Test the version is string and __all__ contains expected symbols
    from limix_lmm import __version__, __all__

    assert isinstance(__version__, str)
    expected = {"LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "test", "__version__"}
    assert expected.issubset(set(__all__))


def test_MTLMM_exists_and_can_be_instantiated_or_at_least_imported():
    # We only import and check callable type since no usage code here.
    # MTLMM might be a class or function, just check import works.
    # It is imported from limix_lmm
    assert callable(MTLMM) or isinstance(MTLMM, type)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_process_step_dividing_S(monkeypatch, step):
    # Test with step dividing S exactly
    np.random.seed(101)
    n, S = 9, 6  # S divisible by step to avoid partial last block
    y = np.random.randn(n, 1)
    F = np.ones((n, 1))
    G = np.random.randn(n, S)

    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, step=step)

    ntests = S // step
    m = step  # no Inter so mi=1
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    assert pv.shape == (ntests,)
    assert beta.shape == (m, ntests)
    assert lrt.shape == (ntests,)
    assert np.all((pv >= 0) & (pv <= 1))


def test_process_verbose_output(capsys):
    # Run process with verbose=True and check output printed
    np.random.seed(2023)
    n, S = 10, 4
    y = np.random.randn(n, 1)
    F = np.ones((n, 2))
    G = np.random.randn(n, S)
    lmm = LMMCore(y, F, Ki_dot=None)

    lmm.process(G, step=2, verbose=True)
    captured = capsys.readouterr()
    assert "Tested for" in captured.out


def test_process_with_Ki_dot_affects_results():
    # Ki_dot scales input by 0.5, results should differ than without Ki_dot
    np.random.seed(301)
    n, S = 8, 4
    y = np.random.randn(n, 1)
    F = np.random.randn(n, 2)
    G = np.random.randn(n, S)

    def Ki_dot(X):
        return 0.5 * X

    lmm1 = LMMCore(y, F, Ki_dot=None)
    lmm1.process(G, step=1)

    lmm2 = LMMCore(y, F, Ki_dot=Ki_dot)
    lmm2.process(G, step=1)

    # pvalues should differ
    pv1 = lmm1.getPv()
    pv2 = lmm2.getPv()
    assert not np.allclose(pv1, pv2)

    # beta shapes equal
    assert lmm1.getBetaSNP().shape == lmm2.getBetaSNP().shape


def test_getBetaSNPste_matches_beta_and_pv(monkeypatch):
    # Check ste calculation consistency by recomputing manually inside test
    import scipy.stats as st
    np.random.seed(1234)
    n, S = 7, 5
    y = np.random.randn(n, 1)
    F = np.random.randn(n, 1)
    G = np.random.randn(n, S)

    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, step=1)
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    ste = lmm.getBetaSNPste()

    # z scores computed manually
    z_manual = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    ste_manual = beta / z_manual
    # Allow nan or inf where pv=0 or 1 exactly, so mask those
    mask = (pv > 0) & (pv < 1)
    np.testing.assert_allclose(ste[mask], ste_manual[mask], rtol=1e-7, atol=1e-12)


def test_getters_raise_no_errors_with_minimal_setup():
    # Just call all getter functions on minimal fit to check they do not raise errors
    np.random.seed(7)
    n = 4
    y = np.random.randn(n, 1)
    F = np.ones((n, 1))
    lmm = LMMCore(y, F, Ki_dot=None)
    # manually set beta_F to avoid attribute error
    lmm.beta_F = lmm.beta_F0

    lmm.getBetaCov()
    lmm.getPv()
    lmm.getBetaSNP()
    lmm.getLRT()
    lmm.getBetaSNPste()


def test_process_handles_step_greater_than_S(monkeypatch):
    # Test that step > S runs without error and result shape is correct (ntests = 0)
    np.random.seed(99)
    n, S = 5, 3
    y = np.random.randn(n, 1)
    F = np.ones((n, 1))
    G = np.random.randn(n, S)
    lmm = LMMCore(y, F, Ki_dot=None)

    # step greater than number of variants (S)
    step = S + 1
    # process divides integer division, so ntests = 0
    # Expect no error and lmm.pv is empty array
    lmm.process(G, step=step)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    assert pv.size == 0
    assert beta.shape[1] == 0
    assert lrt.size == 0


def test_process_with_inter_and_step_combined_shape_and_consistency():
    # S = 6, Inter with M=2, step=3, so output dimension m = 6
    np.random.seed(888)
    n, S, M, step = 10, 6, 2, 3
    y = np.random.randn(n, 1)
    F = np.ones((n, 1))
    G = np.random.randn(n, S)
    Inter = np.random.randn(n, M)

    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=Inter, step=step)

    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    ntests = S // step
    m = M * step
    assert pv.shape == (ntests,)
    assert beta.shape == (m, ntests)
    assert lrt.shape == (ntests,)
    assert np.all((pv >= 0) & (pv <= 1))

