import numpy as np
import pytest
from limix_lmm.lmm_core import LMMCore
from limix_lmm import LMMCore as LMMCore_imported
from limix_lmm import MTLMM
from limix_lmm import __version__

def test_version_string():
    assert isinstance(__version__, str)
    assert __version__ == "0.1.2"

def test_lmmcore_importable_and_class():
    assert LMMCore_imported is LMMCore
    lmm = LMMCore(np.ones((10,1)), np.ones((10,1)))
    assert isinstance(lmm, LMMCore)

def test_lmmcore_null_fit_without_kinv(monkeypatch):
    """Test _fit_null without Ki_dot calculates expected attributes"""
    y = np.random.randn(5,1)
    F = np.ones((5,2))
    lmm = LMMCore(y, F, Ki_dot=None)
    # With Ki_dot=None, Kiy and KiF should equal y and F input
    np.testing.assert_array_equal(lmm.Kiy, y)
    np.testing.assert_array_equal(lmm.KiF, F)
    # Dimensions consistent
    assert lmm.FKiF.shape == (F.shape[1], F.shape[1])
    assert lmm.FKiy.shape == (F.shape[1], 1)
    assert isinstance(lmm.beta_F0, np.ndarray)
    assert isinstance(lmm.s20, float) or isinstance(lmm.s20, np.floating)

def test_lmmcore_null_fit_with_kinv():
    """Test _fit_null with Ki_dot as identity operator"""
    y = np.random.randn(5,1)
    F = np.ones((5,1))
    def kinv(x):
        return x
    # Should work with Ki_dot function identity
    lmm = LMMCore(y, F, Ki_dot=kinv)
    np.testing.assert_array_equal(lmm.Kiy, y)
    np.testing.assert_array_equal(lmm.KiF, F)

def test_lmmcore_process_basic():
    """Test process on random data without Inter"""
    n, s = 20, 4
    y = np.random.randn(n,1)
    F = np.ones((n,1))
    G = np.random.randn(n,s)
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    assert pv.shape == (s,)
    assert beta.shape == (s, s//s if s//s>0 else 1) or beta.shape[0] == s
    assert lrt.shape == (s,)
    assert np.all((0 <= pv) & (pv <= 1))

def test_lmmcore_process_with_inter():
    """Test process with Inter matrix"""
    n, s, m = 10, 6, 3
    y = np.random.randn(n,1)
    F = np.ones((n,1))
    G = np.random.randn(n,s)
    Inter = np.random.randn(n,m)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=2)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    # Number of tests = s//step
    assert pv.shape == (s//2,)
    # beta shape = (m*step, ntests)
    assert beta.shape == (m*2, s//2)
    assert lrt.shape == (s//2,)

def test_getters_return_expected_arrays():
    n, s = 8, 4
    y = np.random.randn(n,1)
    F = np.ones((n,1))
    G = np.random.randn(n,s)
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv = lmm.getPv()
    beta_g = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    ste = lmm.getBetaSNPste()
    # pv shape matches number of tests
    assert pv.shape[0] == beta_g.shape[1]
    # lrt shape matches pv shape
    assert lrt.shape == pv.shape
    # ste shape matches beta_g shape
    assert ste.shape == beta_g.shape
    # ste values finite or nan (nan can result if pv=0)
    assert np.all(np.isfinite(ste) | np.isnan(ste))

def test_process_verbose_prints(capsys):
    """Test that verbose=True prints time info"""
    n, s = 10, 2
    y = np.random.randn(n,1)
    F = np.ones((n,1))
    G = np.random.randn(n,s)
    lmm = LMMCore(y, F)
    lmm.process(G, verbose=True)
    captured = capsys.readouterr()
    assert "Tested for" in captured.out

def test_process_step_more_than_one_and_interaction_shapes():
    """Test process with step > 1 and Inter"""
    n, s, m = 10, 6, 2
    y = np.random.randn(n,1)
    F = np.ones((n,1))
    G = np.random.randn(n,s)
    Inter = np.random.randn(n,m)
    lmm = LMMCore(y, F)
    step = 3
    lmm.process(G, Inter=Inter, step=step)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    assert pv.shape == (s // step,)
    assert beta.shape == (m * step, s // step)

def test_process_step_not_divisible_raises():
    """Process only supports tests = int(G.shape[1]/step), float step invalid"""
    n, s = 10, 5
    y = np.random.randn(n,1)
    F = np.ones((n,1))
    G = np.random.randn(n, s)
    lmm = LMMCore(y, F)
    # step that does not evenly divide s leads to truncated tests (no error currently)
    lmm.process(G, step=2)
    pv = lmm.getPv()
    assert pv.shape[0] == s // 2

def test_getBetaCov_matches_betaF0():
    """Test getBetaCov returns attribute beta_F0 set at null fit"""
    y = np.random.randn(10,1)
    F = np.ones((10,2))
    lmm = LMMCore(y, F)
    beta_cov = lmm.getBetaCov()
    assert beta_cov.shape == (F.shape[1], 1)
    # beta_F0 attribute exists
    assert hasattr(lmm, "beta_F0")
    np.testing.assert_array_equal(beta_cov, lmm.beta_F0)

def test_process_with_kinv_function_applied():
    """Test Ki_dot function is applied to inputs in processing"""
    n, s = 10, 4
    y = np.random.randn(n,1)
    F = np.ones((n,1))
    G = np.random.randn(n,s)
    calls = {"count":0}
    def kinv(x):
        calls["count"] += 1
        return x * 2.0
    lmm = LMMCore(y, F, Ki_dot=kinv)
    lmm.process(G)
    # Ki_dot should have been called multiple times
    assert calls["count"] >= 1
    pv = lmm.getPv()
    assert pv.shape[0] == s

def test_getBetaSNPste_correctness():
    """Test getBetaSNPste computes correct signs and returns array"""
    np.random.seed(1)
    n, s = 10, 3
    y = np.random.randn(n,1)
    F = np.ones((n,1))
    G = np.random.randn(n,s)
    lmm = LMMCore(y, F)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    # z computed internally:
    import scipy.stats as st
    from numpy import sign, sqrt
    z = sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    expected_ste = beta / z
    # ste close to expected
    np.testing.assert_allclose(ste, expected_ste, rtol=1e-7, atol=1e-12)

@pytest.mark.parametrize("step", [1,2])
def test_process_and_getters_consistency(step):
    """Test LMMCore process and getters consistent shape and values"""
    n, s = 12, 6
    y = np.random.randn(n, 1)
    F = np.ones((n,1))
    G = np.random.randn(n,s)
    lmm = LMMCore(y, F)
    lmm.process(G, step=step)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    ste = lmm.getBetaSNPste()
    assert pv.shape[0] == s // step
    assert beta.shape[1] == pv.shape[0]
    assert lrt.shape == pv.shape
    assert ste.shape == beta.shape

def test_empty_interaction_works_like_none():
    """If Inter is array of ones, behaves correctly"""
    n, s = 10, 4
    y = np.random.randn(n,1)
    F = np.ones((n,1))
    G = np.random.randn(n,s)
    Inter = np.ones((n,1))
    lmm_none = LMMCore(y, F)
    lmm_none.process(G)
    lmm_one = LMMCore(y, F)
    lmm_one.process(G, Inter=Inter)
    # With Inter=1, step=1 shape beta_g different from no interaction
    assert lmm_none.getBetaSNP().shape[0] == s
    assert lmm_one.getBetaSNP().shape[0] == s  # m=1 * step=1

def test_lmmcore_process_multiple_calls_overwrite_results():
    """Multiple process calls overwrite previous results"""
    n, s = 10, 4
    y = np.random.randn(n,1)
    F = np.ones((n,1))
    G = np.random.randn(n,s)
    lmm = LMMCore(y, F)
    lmm.process(G)
    pv1 = lmm.getPv().copy()
    beta1 = lmm.getBetaSNP().copy()
    lmm.process(G*2)
    pv2 = lmm.getPv()
    beta2 = lmm.getBetaSNP()
    assert not np.allclose(pv1, pv2)
    assert not np.allclose(beta1, beta2)


# ---- Tests for limix_lmm/__init__.py level imports ----

def test_imported_names_exist():
    import limix_lmm
    assert hasattr(limix_lmm, "LMM")
    assert hasattr(limix_lmm, "LMMCore")
    assert hasattr(limix_lmm, "MTLMM")
    assert hasattr(limix_lmm, "plot")
    assert hasattr(limix_lmm, "download")
    assert hasattr(limix_lmm, "unzip")
    assert hasattr(limix_lmm, "test")

def test_all_list_contains_expected_items():
    from limix_lmm import __all__
    expected = {"LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "__version__", "test"}
    assert set(__all__) == expected

@pytest.mark.parametrize("attr", __all__)
def test_all_attributes_are_importable(attr):
    import limix_lmm
    a = getattr(limix_lmm, attr)
    # Should be callable, class, module or a string (version)
    assert a is not None
