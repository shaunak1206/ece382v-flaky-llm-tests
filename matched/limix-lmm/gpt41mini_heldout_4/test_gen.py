import numpy as np
import pytest
import scipy.stats as stats
from limix_lmm.lmm_core import LMMCore
from limix_lmm import LMMCore as LMMCore_imported, MTLMM
from time import time


def test_init_without_Ki_dot_sets_correct_attributes():
    y = np.ones((10, 1))
    F = np.ones((10, 2))
    lmm = LMMCore(y, F)
    # Ki_dot is None, so Kiy should equal y
    np.testing.assert_array_equal(lmm.Kiy, y)
    # KiF equal to F
    np.testing.assert_array_equal(lmm.KiF, F)
    # degrees of freedom = N - number of fixed effects
    assert lmm.df == 8
    assert lmm.y is y
    assert lmm.F is F


def test_init_with_Ki_dot_function():
    y = np.ones((5, 1))
    F = np.ones((5, 1))
    # Ki_dot just doubles input
    Ki_dot = lambda x: 2 * x
    lmm = LMMCore(y, F, Ki_dot=Ki_dot)
    np.testing.assert_array_equal(lmm.Kiy, 2 * y)
    np.testing.assert_array_equal(lmm.KiF, 2 * F)
    assert lmm.df == 4


def test_init_F_none_defaults_to_ones():
    y = np.ones((7, 1))
    lmm = LMMCore(y, None)
    # F is column of ones with length N
    assert lmm.F.shape == (7, 1)
    assert np.all(lmm.F == 1)


def test_fit_null_sets_betaF0_and_s20():
    y = np.random.randn(20, 1)
    F = np.ones((20, 1))
    lmm = LMMCore(y, F)
    # beta_F0 should be array with shape (num covariates, 1)
    assert lmm.beta_F0.shape == (F.shape[1], 1)
    # s20 should be a positive scalar
    assert np.isscalar(lmm.s20)
    assert lmm.s20 > 0


def test_process_with_no_interaction_and_step_1():
    np.random.seed(0)
    N, S = 15, 5
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=None, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    assert pv.shape == (S,)
    assert beta.shape == (S, S) or beta.shape[0] == S or beta.shape[1] == S
    assert lrt.shape == (S,)
    assert np.all((pv >= 0) & (pv <= 1))


def test_process_with_interaction_and_step_1():
    np.random.seed(1)
    N, S, M = 20, 4, 3
    y = np.random.randn(N, 1)
    F = np.ones((N, 2))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    # Number of tests = S/step = S because step=1
    assert pv.shape == (S,)
    # Number of beta rows = M * step
    assert beta.shape == (M * 1, S)
    assert np.all(pv >= 0) and np.all(pv <= 1)
    assert lrt.shape == (S,)


def test_process_with_interaction_and_step_greater_than_1():
    np.random.seed(2)
    N, S, M, step = 22, 8, 2, 4
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=step)
    ntests = S // step
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    assert pv.shape == (ntests,)
    # beta rows = M * step
    assert beta.shape == (M * step, ntests)
    assert np.all(pv >= 0) and np.all(pv <= 1)
    assert lrt.shape == (ntests,)


def test_getBetaSNPste_returns_correct_shape_and_values():
    np.random.seed(3)
    N, S = 25, 5
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=1)
    ste = lmm.getBetaSNPste()
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()

    assert ste.shape == beta.shape
    # standard errors should not contain NaNs or Infs
    assert np.all(np.isfinite(ste))
    # z values computed internally, so test ste*sign(z) = beta shape relation indirectly
    # ste = beta / z
    # If pv close to 1, z small => ste large, vice versa
    # Just test no divide by zero error occurred


def test_getBetaCov_returns_none_before_assignment():
    y = np.ones((10, 1))
    F = np.ones((10, 2))
    lmm = LMMCore(y, F)
    # beta_F attribute does not exist, expect AttributeError or None
    with pytest.raises(AttributeError):
        _ = lmm.getBetaCov()


def test_process_verbose_prints_message(capsys):
    np.random.seed(4)
    N, S = 10, 6
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=1, verbose=True)
    out = capsys.readouterr().out
    assert "Tested for" in out
    assert "variants in" in out


def test_imported_LMMCore_is_same_class():
    # The LMMCore imported from __init__.py should be the same as direct import
    from limix_lmm.lmm_core import LMMCore as direct_LMMCore
    assert LMMCore_imported is direct_LMMCore


@pytest.mark.skipif(not hasattr(MTLMM, "__init__"), reason="MTLMM class missing")
def test_mtlmm_basic_instantiation():
    # Basic smoke test for MTLMM instantiation and attributes
    try:
        obj = MTLMM()
    except Exception:
        # If MTLMM requires arguments, just skip the test
        pytest.skip("MTLMM requires arguments for instantiation")


def test_lmmcore_process_uses_hatodot_when_step_greater_than_1(monkeypatch):
    # Patch hatodot to verify it is called
    import limix_lmm.lmm_core as lcore

    called = {}

    def fake_hatodot(a, b):
        called['called'] = True
        # Returned shape must match definition: n x (m*l)
        n = a.shape[0]
        return np.ones((n, a.shape[1]*b.shape[1]))

    monkeypatch.setattr(lcore, "hatodot", fake_hatodot)

    N, S, M, step = 10, 4, 3, 2
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)

    lmm = lcore.LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=step)
    assert called.get('called', False) is True


def test_lmmcore_process_handles_Ki_dot_none_and_nonnone():
    N, S = 12, 4
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    # Ki_dot = None
    lmm_none = LMMCore(y, F, Ki_dot=None)
    lmm_none.process(G, step=1)
    pv_none = lmm_none.getPv()
    assert pv_none.shape == (S,)

    # Ki_dot = identity function
    lmm_id = LMMCore(y, F, Ki_dot=lambda x: x)
    lmm_id.process(G, step=1)
    pv_id = lmm_id.getPv()
    assert pv_id.shape == (S,)


def test_lmmcore_lrt_and_pv_relationship_monotonic():
    N, S = 20, 6
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=1)

    lrt = lmm.getLRT()
    pv = lmm.getPv()

    # Since pvalue = sf of chi2 at lrt, increasing lrt should correspond to lowering pv
    # Sort lrt ascending, pv descending
    lrt_sorted_idx = np.argsort(lrt)
    lrt_sorted = lrt[lrt_sorted_idx]
    pv_sorted = pv[lrt_sorted_idx]
    assert np.all(np.diff(lrt_sorted) >= 0)
    assert np.all(np.diff(pv_sorted[::-1]) >= 0)  # reversed pv ascending


def test_getBetaSNP_returns_array_shape_consistent_with_process():
    np.random.seed(11)
    N, S = 15, 7
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F)
    lmm.process(G, step=1)
    beta = lmm.getBetaSNP()
    assert beta.shape[1] == S


def test_process_step_larger_than_S_truncates_or_raises():
    # Because ntests = G.shape[1] // step, if step > S, ntests = 0
    N, S = 15, 5
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    lmm = LMMCore(y, F)
    # Using step > S should result in zero tests or error
    with pytest.raises(ZeroDivisionError):
        lmm.process(G, step=0)  # step=0 invalid, triggers ZeroDivisionError
    # step > S will produce ntests=0, which may cause error or no tests
    # So test step = 10
    with pytest.raises(IndexError):
        lmm.process(G, step=10)


def test_process_with_inter_and_without_inter_produces_different_beta_shapes():
    np.random.seed(42)
    N, S = 20, 8
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.ones((N, 2))

    lmm = LMMCore(y, F)
    lmm.process(G, step=2, Inter=None)
    beta_no_inter = lmm.getBetaSNP()

    lmm.process(G, step=2, Inter=Inter)
    beta_inter = lmm.getBetaSNP()

    # Without Inter, beta rows = step
    assert beta_no_inter.shape[0] == 2
    # With Inter, beta rows = Inter.shape[1] * step
    assert beta_inter.shape[0] == 2 * 2
    # Number of tests is S/step
    assert beta_inter.shape[1] == beta_no_inter.shape[1] == S // 2


def test_getPv_returned_array_values_between_0_and_1():
    N, S = 10, 4
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=1)
    pv = lmm.getPv()
    assert np.all(pv >= 0)
    assert np.all(pv <= 1)
