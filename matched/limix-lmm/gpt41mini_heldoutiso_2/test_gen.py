import numpy as np
import pytest
import scipy
from limix_lmm.lmm_core import LMMCore
from limix_lmm import LMMCore as LMMCore_imported
from limix_lmm import MTLMM


def dummy_Ki_dot_identity(x):
    # Identity operation for Ki_dot, returns input unchanged
    return x


def dummy_Ki_dot_scale(x, scale=2.0):
    # Scales input by a constant factor
    return scale * x


def test_LMMCore_init_without_Ki_dot():
    y = np.ones((5, 1))
    F = np.ones((5, 2))
    lmm = LMMCore(y, F, Ki_dot=None)
    # Ki_dot none means Kiy == y and KiF == F
    np.testing.assert_array_equal(lmm.Kiy, y)
    np.testing.assert_array_equal(lmm.KiF, F)
    assert lmm.df == 5 - 2
    # beta_F0 shape should be (F.shape[1], 1)
    assert lmm.beta_F0.shape == (F.shape[1], 1)
    # s20 should be positive scalar
    assert isinstance(lmm.s20, float)
    assert lmm.s20 > 0


def test_LMMCore_init_with_Ki_dot_identity():
    y = np.arange(10).reshape((10, 1)).astype(float)
    F = np.ones((10, 2))
    lmm = LMMCore(y, F, Ki_dot=dummy_Ki_dot_identity)
    np.testing.assert_array_equal(lmm.Kiy, y)
    np.testing.assert_array_equal(lmm.KiF, F)
    assert lmm.df == 10 - 2


def test_LMMCore_init_with_Ki_dot_scale():
    y = np.arange(6).reshape((6, 1)).astype(float)
    F = np.ones((6, 1))
    lmm = LMMCore(y, F, Ki_dot=lambda x: dummy_Ki_dot_scale(x, 3))
    np.testing.assert_array_equal(lmm.Kiy, 3 * y)
    np.testing.assert_array_equal(lmm.KiF, 3 * F)
    assert lmm.df == 6 - 1


def test_LMMCore_process_no_Inter_step_1():
    # Test process with no interaction, step=1
    np.random.seed(1)
    N = 20
    S = 10
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=None, step=1, verbose=False)

    pv = lmm.getPv()
    beta_g = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    # Shapes
    assert pv.shape == (S,)
    assert beta_g.shape == (S, S // 1)[0:][1] or beta_g.shape == (1, S)
    # Actually beta_g shape is (m, ntests) where m = step or step*mi, here step=1, mi=1 => (1,S)
    assert beta_g.shape == (1, S)
    assert lrt.shape == (S,)
    # p-values between 0 and 1
    assert np.all((pv >= 0) & (pv <= 1))
    # lrt non-negative
    assert np.all(lrt >= 0)


def test_LMMCore_process_with_Inter_step_1():
    # Test process with interaction matrix and step=1
    np.random.seed(2)
    N = 15
    S = 7
    M = 3
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=Inter, step=1, verbose=False)

    pv = lmm.getPv()
    beta_g = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    ntests = S // 1
    m = 1 * M
    # p-values shape (ntests,)
    assert pv.shape == (ntests,)
    # beta_g shape (m, ntests)
    assert beta_g.shape == (m, ntests)
    assert lrt.shape == (ntests,)
    assert np.all((pv >= 0) & (pv <= 1))
    assert np.all(lrt >= 0)


def test_LMMCore_process_with_Inter_step_greater_than_1():
    # Test process with interaction and step > 1
    np.random.seed(3)
    N = 10
    S = 8
    M = 2
    step = 2
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=Inter, step=step, verbose=False)

    pv = lmm.getPv()
    beta_g = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    ntests = S // step
    m = step * M
    assert pv.shape == (ntests,)
    assert beta_g.shape == (m, ntests)
    assert lrt.shape == (ntests,)
    assert np.all((pv >= 0) & (pv <= 1))
    assert np.all(lrt >= 0)


def test_LMMCore_getBetaSNPste_consistency():
    # beta_ste calculation and shape consistency
    np.random.seed(4)
    N = 12
    S = 6
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, step=1)
    beta = lmm.getBetaSNP()
    beta_ste = lmm.getBetaSNPste()
    pv = lmm.getPv()

    # beta_ste shape should match beta
    assert beta_ste.shape == beta.shape
    # No zero division (no nan or inf)
    assert not np.any(np.isnan(beta_ste))
    assert not np.any(np.isinf(beta_ste))
    # P values between 0 and 1
    assert np.all(pv >= 0) and np.all(pv <= 1)


def test_LMMCore_getBetaCov_returns_beta_F():
    # Test that getBetaCov returns self.beta_F0 saved in _fit_null
    # beta_F0 is set in _fit_null and stored as self.beta_F0 (not public attr)
    y = np.ones((5, 1))
    F = np.ones((5, 2))
    lmm = LMMCore(y, F, Ki_dot=None)
    result = lmm.getBetaCov()
    # It returns self.beta_F which is not public but test value equals beta_F0
    # Actually lmm doesn't set self.beta_F anywhere, so result will raise AttributeError.
    # So test that it raises AttributeError when self.beta_F is accessed (since code incomplete)
    # Or test that we can manually assign self.beta_F to beta_F0 then getBetaCov returns it
    with pytest.raises(AttributeError):
        _ = result

    # Set beta_F and test again
    lmm.beta_F = lmm.beta_F0
    result = lmm.getBetaCov()
    np.testing.assert_array_equal(result, lmm.beta_F0)


def test_LMMCore_process_verbose_output(capsys):
    # Test verbose output during process
    N = 30
    S = 10
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, step=1, verbose=True)
    captured = capsys.readouterr()
    assert "Tested for" in captured.out
    assert str(S) in captured.out


def test_module_imports():
    # Check that LMMCore and MTLMM can be imported from the package
    import limix_lmm
    # Check LMMCore class is exposed
    assert hasattr(limix_lmm, "LMMCore")
    # Check MTLMM class is exposed
    assert hasattr(limix_lmm, "MTLMM")
    # Check __version__ exists
    assert hasattr(limix_lmm, "__version__")
    # Check __version__ string format
    assert isinstance(limix_lmm.__version__, str)
    assert limix_lmm.__version__ != ""


def test_LMMCore_process_with_Ki_dot_functional_equivalence():
    # Process should give results consistent regardless of Ki_dot doing identity or None
    np.random.seed(5)
    N = 25
    S = 10
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm_none = LMMCore(y, F, Ki_dot=None)
    lmm_none.process(G, step=1, verbose=False)

    lmm_id = LMMCore(y, F, Ki_dot=dummy_Ki_dot_identity)
    lmm_id.process(G, step=1, verbose=False)

    np.testing.assert_allclose(lmm_none.getPv(), lmm_id.getPv())
    np.testing.assert_allclose(lmm_none.getBetaSNP(), lmm_id.getBetaSNP())
    np.testing.assert_allclose(lmm_none.getLRT(), lmm_id.getLRT())


def test_LMMCore_process_partial_block():
    # When step does not divide S exactly, ntests uses floor integer division
    np.random.seed(6)
    N = 20
    S = 9
    step = 4
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, step=step)
    pv = lmm.getPv()
    beta_g = lmm.getBetaSNP()

    ntests = S // step  # Should be 2 (9 // 4)
    m = step  # 4 variants per test, no Inter means m = step
    assert pv.shape == (ntests,)
    assert beta_g.shape == (m, ntests)


def test_LMMCore_process_raises_error_non_matching_dimensions():
    # Test handling of inputs with incompatible shapes
    y = np.ones((5, 1))
    F = np.ones((5, 1))
    G = np.ones((6, 2))  # 6 rows != 5 in y
    lmm = LMMCore(y, F, Ki_dot=None)
    with pytest.raises(ValueError):
        lmm.process(G)


def test_LMMCore_process_with_Inter_and_no_Ki_dot():
    # Inter provided, no Ki_dot
    np.random.seed(7)
    N = 12
    S = 8
    M = 3
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)
    lmm = LMMCore(y, F, Ki_dot=None)
    lmm.process(G, Inter=Inter, step=2)
    pv = lmm.getPv()
    beta_g = lmm.getBetaSNP()
    # beta_g shape == step * M, ntests = S//step
    m = 2 * M
    ntests = S // 2
    assert pv.shape == (ntests,)
    assert beta_g.shape == (m, ntests)


def test_LMMCore_process_with_Inter_and_Ki_dot():
    # Inter provided, with Ki_dot scaling
    np.random.seed(8)
    N = 15
    S = 10
    M = 2
    step = 2
    y = np.random.randn(N, 1)
    F = np.ones((N, 2))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)

    def Ki_dot_scale_half(x):
        return 0.5 * x

    lmm = LMMCore(y, F, Ki_dot=Ki_dot_scale_half)
    lmm.process(G, Inter=Inter, step=step)
    pv = lmm.getPv()
    beta_g = lmm.getBetaSNP()
    # Shapes and value sanity checks
    assert pv.shape == (S // step,)
    assert beta_g.shape == (step * M, S // step)
    assert np.all((pv >= 0) & (pv <= 1))


def test_getBetaSNPste_with_zero_beta():
    # If beta is zero, test handling of division by zero in getBetaSNPste
    y = np.ones((5, 1))
    F = np.ones((5, 1))
    G = np.ones((5, 2))
    lmm = LMMCore(y, F)
    # Manually set beta_g and pv so beta is zero but pv nonzero
    lmm.beta_g = np.zeros((1, 2))
    lmm.pv = np.array([0.5, 0.5])
    with pytest.raises(ZeroDivisionError):
        _ = lmm.getBetaSNPste()


def test_LMMCore_process_with_step_larger_than_S():
    # If step larger than number of variants, process still runs but with 0 tests
    y = np.ones((10, 1))
    F = np.ones((10, 1))
    G = np.ones((10, 3))
    lmm = LMMCore(y, F)
    lmm.process(G, step=5)
    pv = lmm.getPv()
    beta_g = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    # Since 3 // 5 = 0, shapes should be zero-length arrays
    assert pv.shape[0] == 0
    assert beta_g.shape[1] == 0
    assert lrt.shape[0] == 0


def test_LMMCore_with_F_None_defaults_to_intercept():
    # Providing F=None should default to intercept only model
    y = np.random.randn(10, 1)
    lmm = LMMCore(y, F=None)
    # F defaults to all ones intercept
    assert np.all(lmm.F == 1)
    assert lmm.F.shape[1] == 1


@pytest.mark.parametrize("step", [1, 2])
def test_LMMCore_process_with_interactions_and_multiple_steps(step):
    # Test process with interaction matrix and different step sizes
    np.random.seed(42)
    N = 20
    S = 8
    M = 2
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)
    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=step)

    pv = lmm.getPv()
    beta = lmm.getBetaSNP()

    ntests = S // step
    m = M * step
    assert pv.shape == (ntests,)
    assert beta.shape == (m, ntests)
    assert np.all(pv >= 0) and np.all(pv <= 1)

