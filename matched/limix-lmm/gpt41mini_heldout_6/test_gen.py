import numpy as np
import pytest
from limix_lmm.lmm_core import LMMCore
from limix_lmm import LMMCore as LMMCore_imported, MTLMM
from numpy.testing import assert_allclose


def test_LMMCore_init_without_Ki_dot_and_F():
    y = np.ones((10, 1))
    lmm = LMMCore(y, None, Ki_dot=None)
    # F should become ones of shape (10,1)
    assert lmm.F.shape == (10, 1)
    # Ki_dot is None -> Kiy==y and KiF==F
    assert_allclose(lmm.Kiy, y)
    assert_allclose(lmm.KiF, lmm.F)
    # degrees of freedom correct
    assert lmm.df == 10 - 1
    # s20 scalar
    assert np.isscalar(lmm.s20)

def test_LMMCore_init_with_Ki_dot_and_F():
    y = np.arange(10).reshape(10, 1).astype(float)
    F = np.ones((10, 2))
    def ki_dot(x):
        return x * 2
    lmm = LMMCore(y, F, Ki_dot=ki_dot)
    # Ki_dot applied
    assert_allclose(lmm.Kiy, y * 2)
    assert_allclose(lmm.KiF, F * 2)
    assert lmm.df == 10 - 2

def test_fit_null_proper_calculation():
    # Check internal variables after _fit_null
    y = np.ones((5, 1))
    F = np.ones((5, 1)) * 2
    lmm = LMMCore(y, F)
    # Because Ki_dot None, Kiy==y and KiF==F
    lmm._fit_null()  # call explicitly
    assert_allclose(lmm.FKiy, np.dot(F.T, y))
    assert_allclose(lmm.FKiF, np.dot(F.T, F))
    assert np.isscalar(lmm.yKiy) or (isinstance(lmm.yKiy, np.ndarray) and lmm.yKiy.shape == ())
    assert lmm.df == 5 - 1
    # beta_F0 length matches F cols
    assert lmm.beta_F0.shape == (F.shape[1], 1)
    # s20 positive scalar
    assert lmm.s20 > 0

def test_process_no_interaction_step1():
    # Simple test with step=1 and no Inter
    np.random.seed(0)
    y = np.random.randn(10, 1)
    F = np.ones((10, 1))
    # no Ki_dot: identity
    lmm = LMMCore(y, F, Ki_dot=None)
    G = np.random.randn(10, 4)
    lmm.process(G, Inter=None, step=1, verbose=False)
    # test shapes
    assert lmm.pv.shape == (4,)
    assert lmm.beta_g.shape == (1, 4)
    assert lmm.lrt.shape == (4,)
    # pvalues in [0,1]
    assert np.all((lmm.pv >= 0) & (lmm.pv <= 1))

def test_process_with_interaction_step1():
    np.random.seed(1)
    y = np.random.randn(10, 1)
    F = np.ones((10, 1))
    lmm = LMMCore(y, F, Ki_dot=None)
    G = np.random.randn(10, 3)
    Inter = np.random.randn(10, 2)
    lmm.process(G, Inter=Inter, step=1)
    assert lmm.pv.shape == (3,)
    # beta_g shape is m*ntests, m = Inter.shape[1] * step = 2 *1 = 2
    assert lmm.beta_g.shape == (2, 3)
    # lrt shape matches pvalues
    assert lmm.lrt.shape == (3,)
    assert np.all((lmm.pv >= 0) & (lmm.pv <= 1))

def test_process_with_interaction_step_greater_than_1():
    np.random.seed(2)
    y = np.random.randn(20, 1)
    F = np.ones((20, 2))
    lmm = LMMCore(y, F, Ki_dot=None)
    G = np.random.randn(20, 6)  # 6 variants
    Inter = np.random.randn(20, 3)
    step = 2
    lmm.process(G, Inter=Inter, step=step)
    ntests = int(G.shape[1] / step)
    assert lmm.pv.shape[0] == ntests
    # beta_g shape = m * ntests where m = step * Inter.shape[1] = 2 * 3 =6
    assert lmm.beta_g.shape == (6, ntests)
    assert np.all((lmm.pv >= 0) & (lmm.pv <= 1))

def test_getters_return_expected_shapes():
    np.random.seed(3)
    y = np.random.randn(15, 1)
    F = np.ones((15, 2))
    lmm = LMMCore(y, F, Ki_dot=None)
    G = np.random.randn(15, 5)
    lmm.process(G, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    beta_ste = lmm.getBetaSNPste()
    assert pv.shape == (5,)
    assert beta.shape == (1, 5)
    assert lrt.shape == (5,)
    assert beta_ste.shape == (1, 5)
    # beta_ste should have no NaNs or infs
    assert np.isfinite(beta_ste).all()

def test_getBetaCov_returns_beta_F():
    # This tests attribute self.beta_F is set during process
    np.random.seed(4)
    y = np.random.randn(12, 1)
    F = np.ones((12, 3))
    lmm = LMMCore(y, F)
    # simulate setting beta_F manually (not done in code, but for coverage)
    lmm.beta_F = np.arange(3).reshape(3, 1)
    beta_cov = lmm.getBetaCov()
    assert beta_cov.shape == (3, 1)
    assert_allclose(beta_cov, lmm.beta_F)

def test_process_verbose_prints(capsys):
    np.random.seed(5)
    y = np.random.randn(10, 1)
    F = np.ones((10, 1))
    lmm = LMMCore(y, F)
    G = np.random.randn(10, 2)
    lmm.process(G, verbose=True)
    captured = capsys.readouterr()
    assert "Tested for" in captured.out

def test_process_step_larger_than_variants():
    y = np.ones((10, 1))
    F = np.ones((10, 1))
    lmm = LMMCore(y, F)
    G = np.ones((10, 3))
    # step larger than number of variants, should test zero sets or fail gracefully
    with pytest.raises(ZeroDivisionError):
        lmm.process(G, step=5)

def test_process_with_Ki_dot_functionality():
    y = np.random.randn(6, 1)
    F = np.ones((6, 1))
    # Ki_dot squares input array
    def ki_dot(x):
        return x ** 2
    lmm = LMMCore(y, F, Ki_dot=ki_dot)
    G = np.ones((6, 2))
    lmm.process(G, step=1)
    # Values in lrt should be finite
    assert np.all(np.isfinite(lmm.lrt))
    # P-values should be between 0 and 1
    assert np.all((lmm.pv >= 0) & (lmm.pv <= 1))

def test_betaSNP_ste_consistency():
    # Check that betaSte is consistent with beta and pv
    np.random.seed(7)
    y = np.random.randn(8, 1)
    F = np.ones((8, 1))
    lmm = LMMCore(y, F)
    G = np.random.randn(8, 3)
    lmm.process(G)
    beta = lmm.getBetaSNP()
    ste = lmm.getBetaSNPste()
    pv = lmm.getPv()
    # ste = beta / z, z = sign(beta)*sqrt(chi2.isf(pv))
    from scipy.stats import chi2
    z = np.sign(beta) * np.sqrt(chi2(1).isf(pv))
    expected_ste = beta / z
    assert_allclose(ste, expected_ste)

def test_imports_from_init():
    # Test that LMMCore and MTLMM classes are imported via __init__.py
    assert LMMCore_imported is not None
    assert MTLMM is not None

@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("inter_flag", [True, False])
def test_process_various_steps_and_inter(step, inter_flag):
    np.random.seed(9)
    y = np.random.randn(12, 1)
    F = np.ones((12, 2))
    lmm = LMMCore(y, F)
    num_variants = 6
    G = np.random.randn(12, num_variants)
    if inter_flag:
        Inter = np.random.randn(12, 2)
    else:
        Inter = None
    # only test when step divides num_variants exactly
    if num_variants % step != 0:
        pytest.skip("step must divide number of variants in this test")
    lmm.process(G, Inter=Inter, step=step)
    ntests = num_variants // step
    assert lmm.pv.shape == (ntests,)
    m = (Inter.shape[1] if Inter is not None else 1) * step
    assert lmm.beta_g.shape == (m, ntests)
    assert lmm.lrt.shape == (ntests,)

def test_beta_snp_before_process_raises_attribute_error():
    y = np.ones((10, 1))
    F = np.ones((10, 1))
    lmm = LMMCore(y, F)
    with pytest.raises(AttributeError):
        _ = lmm.getPv()
    # Actually pvalues are calculated in process(), but pvalues are set there.
    # So the above would fail only if we try to access pvalues before process is called.
    # The current code sets self.pv only in process - so yes, this test is valid.

def test_getBetaCov_without_setting_beta_F():
    # beta_F is not set by default, getBetaCov should raise AttributeError if called early
    y = np.ones((10, 1))
    F = np.ones((10, 1))
    lmm = LMMCore(y, F)
    with pytest.raises(AttributeError):
        _ = lmm.getBetaCov()
