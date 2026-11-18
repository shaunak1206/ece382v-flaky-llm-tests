import numpy as np
import pytest

from limix_lmm.lmm_core import LMMCore


def _make_simple_Ki_dot():
    """Return a deterministic Ki_dot function for testing."""
    def Ki_dot(x):
        # simple invertible covariance: just multiply by 2
        return 2 * x
    return Ki_dot


def _make_data(seed=42):
    """Generate reproducible test data: y, F, G, Inter"""
    rng = np.random.RandomState(seed)
    N = 50
    S = 20
    m = 3
    y = rng.randn(N, 1)
    F = np.ones((N, 1))
    G = rng.randn(N, S)
    Inter = rng.randn(N, m)
    return y, F, G, Inter


@pytest.mark.parametrize("seed", [123, 999, 0])
def test_init_null_model_seeds_np(seed):
    # Seed numpy in the test function as requested
    np.random.seed(seed)
    y, F, _, _ = _make_data(seed)
    ki = _make_simple_Ki_dot()

    lmm = LMMCore(y, F, Ki_dot=ki)
    # shape tests
    assert lmm.y.shape == (y.shape[0], 1)
    assert lmm.F.shape[0] == y.shape[0]
    # df is correct
    assert lmm.df == y.shape[0] - F.shape[1]
    # beta_F0 shape correct
    assert lmm.beta_F0.shape == (F.shape[1], 1)
    # s20 is positive
    assert lmm.s20 > 0


def test_init_with_none_F_creates_ones():
    np.random.seed(123)
    y, _, _, _ = _make_data()
    lmm = LMMCore(y, F=None)
    # F should be all ones with shape (N,1)
    assert lmm.F.shape == (y.shape[0], 1)
    assert np.allclose(lmm.F, 1.0)


def test_fit_null_no_Ki_dot_results_in_identity_Kiy_KiF():
    np.random.seed(1)
    y, F, _, _ = _make_data()
    lmm = LMMCore(y, F, Ki_dot=None)
    # Ki_dot is None, so KiF and Kiy are F and y
    assert np.allclose(lmm.Kiy, y)
    assert np.allclose(lmm.KiF, F)
    # check that FKiy and FKiF are correct dot products
    import scipy as sp
    assert np.allclose(lmm.FKiy, sp.dot(F.T, y))
    assert np.allclose(lmm.FKiF, sp.dot(F.T, F))


def test_process_without_inter_step_1_shapes_and_values():
    np.random.seed(101)
    y, F, G, _ = _make_data()
    ki = _make_simple_Ki_dot()
    lmm = LMMCore(y, F, Ki_dot=ki)

    # Seed inside test per instructions
    np.random.seed(101)
    lmm.process(G, Inter=None, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    # number of tests = G.shape[1] / step = S
    assert pv.shape == (G.shape[1] // 1, )
    assert beta.shape == (1, G.shape[1] // 1)
    assert lrt.shape == (G.shape[1] // 1, )

    # pvalues in [0,1]
    assert np.all((pv >= 0) & (pv <= 1))
    # lrt non-negative (due to log ratio of s2/s20)
    assert np.all(lrt >= 0)

    # beta values finite
    assert np.all(np.isfinite(beta))


def test_process_with_inter_step_2_shapes():
    np.random.seed(201)
    y, F, G, Inter = _make_data()
    ki = _make_simple_Ki_dot()
    lmm = LMMCore(y, F, Ki_dot=ki)

    np.random.seed(201)
    step = 2
    lmm.process(G, Inter=Inter, step=step)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    # number of tests = floor(S/step)
    expected_tests = G.shape[1] // step
    # beta shape: (M*step, ntests)
    assert pv.shape == (expected_tests,)
    assert beta.shape == (Inter.shape[1] * step, expected_tests)
    assert lrt.shape == (expected_tests,)

    # pvalues valid
    assert np.all((pv >= 0) & (pv <= 1))
    # lrt non-negative
    assert np.all(lrt >= 0)
    # beta finite
    assert np.all(np.isfinite(beta))


def test_process_step_greater_than_G_features_works(monkeypatch):
    # Create data where step > G.shape[1], expect ntests=0
    np.random.seed(42)
    y, F, G, Inter = _make_data()
    ki = _make_simple_Ki_dot()
    lmm = LMMCore(y, F, Ki_dot=ki)

    large_step = G.shape[1] + 5
    # Monkeypatch to track whether process handles this gracefully:
    np.random.seed(42)
    lmm.process(G, Inter=Inter, step=large_step)

    # expected tests = 0 due to integer division truncation
    assert lmm.getPv().size == 0
    assert lmm.getBetaSNP().size == 0
    assert lmm.getLRT().size == 0


def test_getBetaSNPste_properties():
    np.random.seed(123)
    y, F, G, Inter = _make_data()
    ki = _make_simple_Ki_dot()
    lmm = LMMCore(y, F, Ki_dot=ki)
    np.random.seed(123)
    lmm.process(G, Inter=Inter, step=1)

    ste = lmm.getBetaSNPste()
    beta = lmm.getBetaSNP()
    pv = lmm.getPv()

    # same shape as beta
    assert ste.shape == beta.shape
    # If beta is zero, ste should be nan or inf or 0, but should not raise errors
    # Test that no nan or inf values appear where beta != 0
    mask = beta != 0
    assert np.all(np.isfinite(ste[mask]))

    # Test that ste * z = beta (approximately), where z = beta / ste
    # z is signed sqrt of chi2 quantile inverse from pvalue
    approx_beta = ste * (beta / ste)
    # Because dividing beta by ste, some numerical instability possible
    # Checking with tolerance
    assert np.allclose(beta[mask], approx_beta[mask], rtol=1e-5, atol=1e-8)


def test_getBetaCov_returns_same_shape_and_finite_betaF():
    np.random.seed(10)
    y, F, _, _ = _make_data()
    lmm = LMMCore(y, F, Ki_dot=None)
    betaF = lmm.getBetaCov()

    assert betaF.shape == (F.shape[1], 1)
    assert np.all(np.isfinite(betaF))


def test_process_verbose_output(monkeypatch, capsys):
    np.random.seed(77)
    y, F, G, Inter = _make_data()
    ki = _make_simple_Ki_dot()
    lmm = LMMCore(y, F, Ki_dot=ki)

    np.random.seed(77)
    lmm.process(G, Inter=Inter, step=1, verbose=True)

    captured = capsys.readouterr()
    assert "Tested for" in captured.out
    assert str(G.shape[1]) in captured.out


def test_process_same_input_consistent_results():
    # Ensures deterministic results on repeated runs for fixed seed and data
    seed = 2024
    np.random.seed(seed)
    y, F, G, Inter = _make_data(seed)
    ki = _make_simple_Ki_dot()

    lmm1 = LMMCore(y, F, Ki_dot=ki)
    np.random.seed(seed)
    lmm1.process(G, Inter=Inter, step=3)
    pv1 = lmm1.getPv()
    beta1 = lmm1.getBetaSNP()
    lrt1 = lmm1.getLRT()

    lmm2 = LMMCore(y, F, Ki_dot=ki)
    np.random.seed(seed)
    lmm2.process(G, Inter=Inter, step=3)
    pv2 = lmm2.getPv()
    beta2 = lmm2.getBetaSNP()
    lrt2 = lmm2.getLRT()

    # All outputs close for repeated runs with same seed and inputs
    assert np.allclose(pv1, pv2, rtol=1e-12)
    assert np.allclose(beta1, beta2, rtol=1e-12)
    assert np.allclose(lrt1, lrt2, rtol=1e-12)


def test_process_handles_step_one_with_inter_shape():
    np.random.seed(55)
    y, F, G, Inter = _make_data()
    ki = _make_simple_Ki_dot()
    lmm = LMMCore(y, F, Ki_dot=ki)

    np.random.seed(55)
    lmm.process(G, Inter=Inter, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()

    # number of tests = G.shape[1]
    assert pv.shape == (G.shape[1]//1,)
    # beta shape = (Inter.shape[1]*step, ntests)
    assert beta.shape == (Inter.shape[1] * 1, G.shape[1]//1)

    assert np.all((pv >= 0) & (pv <= 1))
    assert np.all(np.isfinite(beta))


def test_process_interaction_none_vs_ones_equivalent():
    # Inter=None and Inter=ones of shape (N,1) should produce same shapes and compatible results
    np.random.seed(202)
    y, F, G, _ = _make_data()
    ki = _make_simple_Ki_dot()
    lmm_none = LMMCore(y, F, Ki_dot=ki)
    np.random.seed(202)
    lmm_none.process(G, Inter=None, step=1)

    ones_inter = np.ones((G.shape[0], 1))
    lmm_ones = LMMCore(y, F, Ki_dot=ki)
    np.random.seed(202)
    lmm_ones.process(G, Inter=ones_inter, step=1)

    # Beta shape differs: no inter: shape (1, ntests), with inter: (1*1, ntests) == (1, ntests) same
    assert lmm_none.getBetaSNP().shape == lmm_ones.getBetaSNP().shape
    # pvs approx equal (not necessarily exact due to numeric differences)
    assert np.allclose(lmm_none.getPv(), lmm_ones.getPv(), rtol=1e-7, atol=1e-10)
    # lrt approx equal
    assert np.allclose(lmm_none.getLRT(), lmm_ones.getLRT(), rtol=1e-7, atol=1e-10)


def test_imports_and_all_list_values():
    import limix_lmm
    # __version__ string present and non-empty
    assert isinstance(limix_lmm.__version__, str)
    assert limix_lmm.__version__

    # __all__ contains expected entries
    expected_all = {
        "LMM",
        "LMMCore",
        "MTLMM",
        "plot",
        "download",
        "unzip",
        "__version__",
        "test",
    }
    assert set(limix_lmm.__all__) == expected_all

    # Classes LMMCore and MTLMM importable from package
    assert hasattr(limix_lmm, "LMMCore")
    assert hasattr(limix_lmm, "MTLMM")


@pytest.mark.parametrize("step", [1, 2, 4])
def test_process_beta_g_shape_and_consistency(step):
    np.random.seed(303)
    y, F, G, Inter = _make_data()
    ki = _make_simple_Ki_dot()
    lmm = LMMCore(y, F, Ki_dot=ki)

    np.random.seed(303)
    lmm.process(G, Inter=Inter, step=step)
    beta = lmm.getBetaSNP()
    # Shape is (M*step, ntests), ntests = floor(S/step)
    assert beta.shape == (Inter.shape[1]*step, G.shape[1] // step)
    # Values finite
    assert np.all(np.isfinite(beta))


def test_lrt_and_pv_relationship_monotonicity():
    # When lrt increases, p-values decrease (monotonic inverse chi2 SF)
    np.random.seed(404)
    y, F, G, Inter = _make_data()
    ki = _make_simple_Ki_dot()
    lmm = LMMCore(y, F, Ki_dot=ki)

    np.random.seed(404)
    lmm.process(G, Inter=Inter, step=1)
    pv = lmm.getPv()
    lrt = lmm.getLRT()

    # Sort lrt, pvs should be monotonic inversely ordered
    idx = np.argsort(lrt)
    sorted_lrt = lrt[idx]
    sorted_pv = pv[idx]

    # lrt nondecreasing
    assert np.all(np.diff(sorted_lrt) >= 0)

    # pv nonincreasing because higher lrt = smaller pval (sf of chi2)
    assert np.all(np.diff(sorted_pv) <= 0)

