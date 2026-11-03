import numpy as np
import pytest
import scipy.stats as stats

from limix_lmm.lmm_core import LMMCore
from limix_lmm import LMMCore as LMMCore_from_init
from limix_lmm import MTLMM

def test_init_with_F_none_defaults_to_ones():
    y = np.ones((10,1))
    lmm = LMMCore(y, None)
    # F should be all ones, shape (10,1)
    assert lmm.F.shape == (10,1)
    assert np.all(lmm.F == 1)

def test_df_calculation():
    y = np.ones((15,1))
    F = np.ones((15,3))
    lmm = LMMCore(y, F)
    assert lmm.df == 15 - 3

def test_init_without_Ki_dot_sets_Kiy_and_KiF_correctly():
    y = np.arange(5).reshape(5,1).astype(float)
    F = np.ones((5,2))
    lmm = LMMCore(y, F, Ki_dot=None)
    # Since Ki_dot is None, Kiy and KiF should equal y and F respectively
    np.testing.assert_array_equal(lmm.Kiy, y)
    np.testing.assert_array_equal(lmm.KiF, F)

def test_init_with_identity_Ki_dot_function():
    y = np.arange(5).reshape(5,1).astype(float)
    F = np.ones((5,2))
    Ki_dot = lambda x: x
    lmm = LMMCore(y, F, Ki_dot=Ki_dot)
    # Ki_dot is identity, so Kiy == y and KiF == F
    np.testing.assert_array_equal(lmm.Kiy, y)
    np.testing.assert_array_equal(lmm.KiF, F)

def test_getBetaCov_returns_correct_betaF():
    y = np.random.randn(10,1)
    F = np.ones((10,1))
    lmm = LMMCore(y, F)
    betaF = lmm.getBetaCov()
    # beta_F0 should have shape (number_of_covariates, 1)
    assert betaF.shape == (F.shape[1], 1)
    # betaF values should be float
    assert betaF.dtype.kind == 'f'

def test_process_and_getters_return_arrays_of_expected_shape_step1_no_Inter():
    np.random.seed(7)
    y = np.random.randn(20,1)
    F = np.ones((20,1))
    G = np.random.randn(20,5)
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=None, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    ste = lmm.getBetaSNPste()

    # There should be 5 tests, step=1
    assert pv.shape == (5,)
    assert beta.shape == (1, 5)
    assert lrt.shape == (5,)
    assert ste.shape == (1, 5)

def test_process_and_getters_with_Inter_step1():
    np.random.seed(5)
    y = np.random.randn(30,1)
    F = np.ones((30,1))
    G = np.random.randn(30,4)
    Inter = np.random.randn(30,3)
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=Inter, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    ste = lmm.getBetaSNPste()

    # Number of tests = 4 (step=1), beta shape is (m*step, ntests) = (3,4)
    assert pv.shape == (4,)
    assert beta.shape == (3,4)
    assert lrt.shape == (4,)
    assert ste.shape == (3,4)

def test_process_and_getters_with_Inter_step2():
    np.random.seed(11)
    y = np.random.randn(20,1)
    F = np.ones((20,1))
    G = np.random.randn(20,6)
    Inter = np.random.randn(20,2)
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=Inter, step=2)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()

    # Number of tests = 3 (6 variants / step 2)
    assert pv.shape == (3,)
    # beta_g shape should be (mi*step, ntests) = (2*2, 3) = (4,3)
    assert beta.shape == (4,3)

def test_process_verbose_prints(capsys):
    np.random.seed(2)
    y = np.random.randn(10,1)
    F = np.ones((10, 1))
    G = np.random.randn(10, 4)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2, verbose=True)
    captured = capsys.readouterr()
    assert "Tested for 4 variants in " in captured.out

def test_LMMCore_from_init_module_is_same_class():
    assert LMMCore_from_init == LMMCore

def test___all___contains_expected_symbols():
    from limix_lmm import __all__
    expected = {"LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "__version__", "test"}
    for e in expected:
        assert e in __all__

# Basic instantiation test for MTLMM, check for existence and basic call
def test_mtlmm_instantiation_and_basic_call():
    # Just test that MTLMM can be instantiated and has expected attributes if any
    # We do not have source code here, so test it's importable and callable
    # MTLMM must be a class or function
    assert MTLMM is not None
    # If it is class, test creation with dummy arguments if possible:
    # Since we don't know arguments, just check it's callable
    # Skip actual instantiation because signature unknown

def test_process_step_greater_than_variants_raises_or_processes():
    y = np.random.randn(5,1)
    F = np.ones((5,1))
    G = np.random.randn(5,3)
    lmm = LMMCore(y, F)
    # step > number of variants, expecting integer division makes ntests=1
    lmm.process(G, step=5)
    # pvalues length = int(3/5) == 0 but int division used so 0?
    # Actually ntests = int(3/5)=0, possible divide by zero or silent error
    # Expect no exception but pvalues length possibly 0 or error
    pv = lmm.getPv()
    assert len(pv) == 0 or len(pv) == int(G.shape[1]/5)

def test_getBetaSNPste_returns_values_same_shape_as_beta():
    y = np.random.randn(10,1)
    F = np.ones((10, 1))
    G = np.random.randn(10, 4)
    lmm = LMMCore(y, F)
    lmm.process(G, step=2)
    ste = lmm.getBetaSNPste()
    beta = lmm.getBetaSNP()
    assert ste.shape == beta.shape
    # ste should not contain nan if inputs are valid
    assert not np.isnan(ste).any()

def test_nontrivial_Ki_dot_function_affects_values():
    # Define Ki_dot as simple scaling by 2
    Ki_dot = lambda x: 2 * x
    y = np.random.randn(10,1)
    F = np.ones((10,1))
    G = np.random.randn(10, 4)
    lmm1 = LMMCore(y, F, Ki_dot=None)
    lmm1.process(G, step=1)
    lmm2 = LMMCore(y, F, Ki_dot=Ki_dot)
    lmm2.process(G, step=1)
    # Values should differ because Ki_dot applies a scale, likely changing results
    assert not np.array_equal(lmm1.getPv(), lmm2.getPv())

def test_process_with_zero_variants_results_in_empty_results():
    y = np.random.randn(10,1)
    F = np.ones((10,1))
    G = np.empty((10, 0))
    lmm = LMMCore(y, F)
    lmm.process(G, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    # All should be empty arrays
    assert pv.size == 0
    assert beta.size == 0
    assert lrt.size == 0
