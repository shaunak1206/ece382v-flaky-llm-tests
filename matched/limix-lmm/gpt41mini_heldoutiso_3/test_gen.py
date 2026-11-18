import numpy as np
import pytest
import scipy.stats as stats
from limix_lmm.lmm_core import LMMCore
from limix_lmm import LMMCore as LMMCore_imported
from limix_lmm import MTLMM
from limix_lmm import __version__


def _dummy_ki_dot_identity(x):
    # Dummy Ki_dot function that returns the input (identity)
    return x


def _dummy_ki_dot_scale(x):
    # Dummy Ki_dot scales input by 2
    return 2 * x


def test_lmmcore_init_none_F():
    y = np.ones((5, 1))
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (5, 1)
    assert np.allclose(lmm.F, 1)
    # Without Ki_dot, KiF and Kiy should equal F and y resp.
    assert lmm.Kiy.shape == y.shape
    assert lmm.KiF.shape == lmm.F.shape


def test_lmmcore_init_with_ki_dot():
    y = np.ones((5, 1))
    F = np.ones((5, 2))
    lmm = LMMCore(y, F, Ki_dot=_dummy_ki_dot_scale)
    # Ki_dot scales by 2, check values
    assert np.allclose(lmm.Kiy, 2 * y)
    assert np.allclose(lmm.KiF, 2 * F)


def test_fit_null_sets_attributes_correctly():
    y = np.arange(1, 6).reshape(5, 1).astype(float)
    F = np.ones((5, 2))
    lmm = LMMCore(y, F)
    # A0i should be scalar
    assert np.isscalar(lmm.A0i)
    # beta_F0 should be correct shape (number of covariates, 1)
    assert lmm.beta_F0.shape == (2, 1)
    # s20 should be scalar and positive
    assert isinstance(lmm.s20, float)
    assert lmm.s20 > 0


@pytest.fixture
def example_data():
    np.random.seed(0)
    N = 10
    S = 8
    k = 2
    y = np.random.randn(N, 1)
    F = np.random.randn(N, k)
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, 3)
    return y, F, G, Inter


def test_process_basic(example_data):
    y, F, G, _ = example_data
    lmm = LMMCore(y, F, Ki_dot=_dummy_ki_dot_identity)
    lmm.process(G)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    # We expect pv of length S because step=1 by default
    assert pv.shape == (G.shape[1] // 1,)
    # Beta shape should be (step,) x (# tests), here step=1 so (1, S)
    assert beta.shape == (1, G.shape[1])
    # LRT shape matches pv shape
    assert lrt.shape == pv.shape
    # p-values between 0 and 1
    assert np.all((pv >= 0) & (pv <= 1))


def test_process_with_step_and_inter(example_data):
    y, F, G, Inter = example_data
    lmm = LMMCore(y, F, Ki_dot=_dummy_ki_dot_identity)
    step = 2
    lmm.process(G, Inter=Inter, step=step)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    # shape of pv should be S/step
    assert pv.shape == (G.shape[1] // step,)
    # beta shape is (step * Inter.shape[1], ntests)
    expected_beta_rows = step * Inter.shape[1]
    assert beta.shape == (expected_beta_rows, pv.shape[0])
    # p-values between 0 and 1
    assert np.all((pv >= 0) & (pv <= 1))


def test_process_without_ki_dot(example_data):
    y, F, G, Inter = example_data
    lmm = LMMCore(y, F, Ki_dot=None)
    step = 2
    lmm.process(G, Inter=Inter, step=step)
    # Check output sizes
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    assert pv.shape == (G.shape[1] // step,)
    assert beta.shape == (step * Inter.shape[1], pv.shape[0])


def test_getBetaSNPste_output(example_data):
    y, F, G, _ = example_data
    lmm = LMMCore(y, F, Ki_dot=_dummy_ki_dot_identity)
    lmm.process(G)
    ste = lmm.getBetaSNPste()
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    # shapes consistent with beta
    assert ste.shape == beta.shape
    # Check ste mostly finite values
    assert np.all(np.isfinite(ste))
    # When beta very close to zero, ste can be nan or inf but mostly finite


def test_getBetaCov_correct_after_init(example_data):
    y, F, _, _ = example_data
    lmm = LMMCore(y, F)
    # _fit_null sets beta_F0 to attribute beta_F0 but getBetaCov returns beta_F, check we add it
    # getBetaCov returns self.beta_F, but it is never set in __init__, expect AttributeError
    with pytest.raises(AttributeError):
        _ = lmm.getBetaCov()
    # Set beta_F for test and check returned values
    lmm.beta_F = np.ones((F.shape[1], 1))
    beta_cov = lmm.getBetaCov()
    assert np.all(beta_cov == 1)


def test_process_verbose_prints(capsys, example_data):
    y, F, G, _ = example_data
    lmm = LMMCore(y, F, Ki_dot=_dummy_ki_dot_identity)
    lmm.process(G, verbose=True)
    captured = capsys.readouterr()
    assert "Tested for" in captured.out


def test_lmmcore_constructor_invalid_shapes():
    # y shape mismatch
    y = np.ones((5, 2))  # Should be (N,1)
    F = np.ones((5, 1))
    with pytest.raises(IndexError):
        LMMCore(y, F)
    # F rows mismatch y
    y = np.ones((5, 1))
    F = np.ones((4, 1))
    with pytest.raises(ValueError):
        LMMCore(y, F)


def test_module_level_imports():
    # Test that LMMCore imported properly from limix_lmm package
    assert LMMCore_imported is LMMCore
    # Check version string is set and is string
    assert isinstance(__version__, str)
    # Check MTLMM is importable (exists)
    assert MTLMM is not None


def test_process_zero_step_error(example_data):
    y, F, G, _ = example_data
    lmm = LMMCore(y, F)
    with pytest.raises(ZeroDivisionError):
        lmm.process(G, step=0)


def test_process_one_variant_with_inter(example_data):
    y, F, G, Inter = example_data
    lmm = LMMCore(y, F)
    # Use step=1 with Inter, shapes should match multiplication
    lmm.process(G, Inter=Inter, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    # With step=1 and Inter shape (N, M), beta rows = M
    assert beta.shape[0] == Inter.shape[1]
    assert pv.shape[0] == G.shape[1]


def test_process_step_larger_than_variants(example_data):
    y, F, G, _ = example_data
    lmm = LMMCore(y, F)
    step = G.shape[1] + 10
    # This means ntests = 0, which leads to zero-sized arrays; should not error but return empty
    lmm.process(G, step=step)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    # pv and beta should be empty arrays with 0 in shape at test dimension
    assert pv.size == 0
    assert beta.shape[1] == 0

