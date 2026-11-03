import numpy as np
import pytest
import scipy.stats as st

from limix_lmm import LMMCore
from limix_lmm.lmm_core import hatodot, calc_Ai_beta_s2
from limix_lmm import MTLMM


def test_LMMCore_init_without_Ki_dot():
    # simple y and F, no Ki_dot function
    y = np.ones((5, 1))
    F = np.eye(5, 2)
    lmm = LMMCore(y, F)
    assert lmm.y.shape == (5, 1)
    assert lmm.F.shape == (5, 2)
    assert lmm.df == 3
    # Kiy and KiF should equal y and F respectively
    np.testing.assert_array_equal(lmm.Kiy, y)
    np.testing.assert_array_equal(lmm.KiF, F)
    assert isinstance(lmm.beta_F0, np.ndarray)
    assert lmm.s20 > 0


def test_LMMCore_init_with_Ki_dot_identity():
    y = np.ones((4, 1)) * 2
    F = np.ones((4, 1))
    # Ki_dot returns the input unchanged
    def Ki_dot(x):
        return x

    lmm = LMMCore(y, F, Ki_dot=Ki_dot)
    # Kiy, KiF should be same as input
    np.testing.assert_array_equal(lmm.Kiy, y)
    np.testing.assert_array_equal(lmm.KiF, F)
    assert lmm.df == 3


def test_calc_Ai_beta_s2_basic():
    # Using simple 1D inputs to check output shapes and types
    yKiy = 10.0
    FKiF = np.array([[2.0]])
    FKiy = np.array([[4.0]])
    df = 2
    Ai, beta, s2 = calc_Ai_beta_s2(yKiy, FKiF, FKiy, df)
    assert isinstance(Ai, np.ndarray)
    assert isinstance(beta, np.ndarray)
    assert isinstance(s2, float)
    assert beta.shape == (1, 1)
    assert s2 > 0


def test_hatodot_basic():
    # Test elementwise product of columns expanded by hatodot
    A = np.array([[1, 2], [3, 4]])  # shape (2, 2)
    B = np.array([[10, 20], [30, 40]])  # shape (2, 2)
    result = hatodot(A, B)
    # result shape should be (2, 4) (2*2)
    assert result.shape == (2, 4)
    # manually compute expected
    expected = np.array([
        [1*10, 1*20, 2*10, 2*20],
        [3*30, 3*40, 4*30, 4*40],
    ])
    np.testing.assert_array_equal(result, expected)


def test_process_no_interaction_step1():
    # Create simple data where we can run process with step=1, no Inter
    N = 10
    S = 4
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=1, verbose=False)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    assert pv.shape == (S,)
    assert beta.shape == (S, S)[::-1] or beta.shape[0] == 1  # since no Inter, beta rows = step = 1
    assert lrt.shape == (S,)


def test_process_with_interaction_step1():
    N = 8
    S = 3
    M = 2
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    # pv length should be S // step = S (because step=1)
    assert pv.shape == (S,)
    # beta rows = M * step = M since step=1
    assert beta.shape == (M, S)
    assert lrt.shape == (S,)


def test_process_with_interaction_step2():
    N = 12
    S = 6
    M = 3
    step = 2
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=step)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    expected_tests = S // step
    assert pv.shape == (expected_tests,)
    assert beta.shape == (M * step, expected_tests)
    assert lrt.shape == (expected_tests,)


def test_getBetaSNPste_shape_and_values():
    N = 15
    S = 5
    M = 2
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=1)
    beta_ste = lmm.getBetaSNPste()

    # shape matches beta shape
    beta = lmm.getBetaSNP()
    assert beta_ste.shape == beta.shape

    # values should be float and finite
    assert np.all(np.isfinite(beta_ste))


def test_getBetaCov_equal_beta_F0():
    y = np.random.randn(7, 1)
    F = np.ones((7, 1))
    lmm = LMMCore(y, F)
    np.testing.assert_array_equal(lmm.getBetaCov(), lmm.beta_F0)


def test_repr_and_imports_in_init():
    # test that import statements and docstring do not raise errors
    y = np.random.randn(4, 1)
    F = np.ones((4, 1))
    lmm = LMMCore(y, F)
    # simply verify attributes exist
    assert hasattr(lmm, "process")
    assert hasattr(lmm, "getPv")
    assert hasattr(lmm, "getBetaSNP")
    assert hasattr(lmm, "getLRT")


def test_LMMCore_process_verbose_print(capsys):
    y = np.random.randn(9, 1)
    F = np.ones((9, 1))
    G = np.random.randn(9, 3)
    lmm = LMMCore(y, F)
    lmm.process(G, step=1, verbose=True)
    captured = capsys.readouterr()
    assert "Tested for" in captured.out


def test_LMMCore_process_with_G_not_divisible_by_step():
    y = np.random.randn(10, 1)
    F = np.ones((10, 1))
    G = np.random.randn(10, 7)
    lmm = LMMCore(y, F)
    with pytest.raises(IndexError):
        # 7 / 2 = 3.5 -> int(3.5) = 3 no out of range for 6 columns, but 7th causes error
        lmm.process(G, step=2)


def test___init___default_F_is_ones():
    y = np.ones((3, 1))
    lmm = LMMCore(y, None)
    assert np.all(lmm.F == 1.0)
    assert lmm.F.shape == (3, 1)


def test_imports_from_init_module():
    from limix_lmm import LMMCore, MTLMM, __version__, plot, download, unzip, test
    # Basic import tests:
    assert isinstance(__version__, str)
    assert callable(download)
    assert callable(unzip)
    assert callable(test)
    assert isinstance(LMMCore, type)
    assert isinstance(MTLMM, type)


@pytest.mark.parametrize("step,val", [(1, 3), (2, 4), (4, 3)])
def test_process_multiple_steps_and_pvalues_shape(step, val):
    N = 20
    S = step * val
    M = 2
    y = np.random.randn(N, 1)
    F = np.ones((N, 2))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=step)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()

    assert pv.shape == (val,)
    assert beta.shape == (M * step, val)


def test_process_without_Ki_dot_and_with_Ki_dot():
    N = 10
    S = 4
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm_no_ki = LMMCore(y, F, Ki_dot=None)
    lmm_with_ki = LMMCore(y, F, Ki_dot=lambda x: x * 2)

    lmm_no_ki.process(G)
    lmm_with_ki.process(G)

    # Check that outputs differ due to Ki_dot
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                             lmm_no_ki.getPv(), lmm_with_ki.getPv())

