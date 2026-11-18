import numpy as np
import pytest
import scipy.stats as stats
from limix_lmm.lmm_core import LMMCore
from limix_lmm import LMMCore as LMMCore_imported
from limix_lmm import MTLMM


@pytest.fixture
def simple_data():
    # Simple phenotype vector (N=5)
    y = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    # Fixed effects design matrix (N=5, L=2)
    F = np.ones((5, 2))
    F[:, 1] = np.arange(5)
    return y, F


@pytest.fixture
def Ki_dot_identity():
    # identity operation simulating Ki_dot that returns input unchanged
    return lambda x: x


@pytest.fixture
def random_phenotype_and_covariates():
    np.random.seed(0)
    N = 10
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    return y, F


def test_module_imports():
    # Test that import from __init__ works for LMMCore and MTLMM
    assert hasattr(LMMCore_imported, "__init__")
    assert MTLMM is not None


def test_LMMCore_init_without_Ki_dot(simple_data):
    y, F = simple_data
    lmm = LMMCore(y, F)
    # expected df = N - L
    assert lmm.df == y.shape[0] - F.shape[1]
    # Ki_dot None means Kiy = y, KiF = F
    np.testing.assert_array_equal(lmm.Kiy, y)
    np.testing.assert_array_equal(lmm.KiF, F)


def test_LMMCore_init_with_Ki_dot(simple_data, Ki_dot_identity):
    y, F = simple_data
    lmm = LMMCore(y, F, Ki_dot=Ki_dot_identity)
    np.testing.assert_array_equal(lmm.Kiy, y)
    np.testing.assert_array_equal(lmm.KiF, F)


def test_process_no_Inter_step1(simple_data, Ki_dot_identity):
    y, F = simple_data
    lmm = LMMCore(y, F, Ki_dot=Ki_dot_identity)
    N, _ = y.shape
    S = 4  # number of variants
    G = np.ones((N, S))
    # set G so each variant differs for testing
    G[::2, :] = 0
    lmm.process(G, Inter=None, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    # pv length must be number of tests = S / step = 4
    assert pv.shape[0] == S
    # beta shape = (m, number of tests), m=step=1 here
    assert beta.shape == (1, S)
    assert lrt.shape == (S,)
    # p-values between 0 and 1
    assert np.all((pv >= 0) & (pv <= 1))
    # lrt all >=0 (because log of ratio of estimated variance)
    assert np.all(lrt >= 0)


def test_process_with_Inter_step1(simple_data, Ki_dot_identity):
    y, F = simple_data
    N, _ = y.shape
    lmm = LMMCore(y, F, Ki_dot=Ki_dot_identity)
    S = 3
    M = 2
    G = np.ones((N, S))
    Inter = np.arange(N).reshape(-1, 1) * np.ones((1, M))
    lmm.process(G, Inter=Inter, step=1)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    expected_tests = S
    expected_m = M * 1
    assert pv.shape[0] == expected_tests
    assert beta.shape == (expected_m, expected_tests)
    assert lrt.shape == (expected_tests,)
    assert np.all((pv >= 0) & (pv <= 1))


def test_process_with_Inter_and_step_gt1(simple_data, Ki_dot_identity):
    y, F = simple_data
    N, _ = y.shape
    lmm = LMMCore(y, F, Ki_dot=Ki_dot_identity)
    S = 4
    step = 2
    M = 2
    G = np.tile(np.arange(S), (N, 1)).astype(float)
    Inter = np.linspace(1, 2, N).reshape(-1, 1).repeat(M, axis=1)
    lmm.process(G, Inter=Inter, step=step)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    expected_tests = S // step
    expected_m = M * step
    assert pv.shape[0] == expected_tests
    assert beta.shape == (expected_m, expected_tests)
    assert lrt.shape == (expected_tests,)
    assert np.all((pv >= 0) & (pv <= 1))


def test_getBetaSNPste_returns_correct_shape(simple_data):
    y, F = simple_data
    N, _ = y.shape
    S = 3
    lmm = LMMCore(y, F)
    G = np.ones((N, S))
    # process to populate beta_g and pv
    lmm.process(G, step=1)
    beta_ste = lmm.getBetaSNPste()
    # beta_ste shape matches beta shape
    beta = lmm.getBetaSNP()
    assert beta_ste.shape == beta.shape
    # beta_ste elements are finite and not NaN
    assert np.all(np.isfinite(beta_ste))


def test_getBetaCov_returns_none_if_not_set(simple_data):
    # Initially, beta_F is never set, so getBetaCov returns None
    y, F = simple_data
    lmm = LMMCore(y, F)
    # Attribute beta_F does not exist/set to None
    assert not hasattr(lmm, "beta_F") or lmm.getBetaCov() is None


def test_process_verbose_output(simple_data, capsys):
    y, F = simple_data
    lmm = LMMCore(y, F)
    N, _ = y.shape
    S = 4
    G = np.ones((N, S))
    lmm.process(G, step=2, verbose=True)
    captured = capsys.readouterr()
    assert "Tested for" in captured.out


def test_process_zero_variants(simple_data):
    y, F = simple_data
    lmm = LMMCore(y, F)
    N = y.shape[0]
    G = np.empty((N, 0))
    # Should not error, but produce zero length arrays
    lmm.process(G, step=1)
    assert lmm.getPv().size == 0
    assert lmm.getBetaSNP().size == 0
    assert lmm.getLRT().size == 0


def test_calc_Ai_beta_s2_consistency(simple_data):
    # Test internal consistency of _fit_null and calc_Ai_beta_s2 usage
    y, F = simple_data
    lmm = LMMCore(y, F)
    # Check s20 > 0
    assert lmm.s20 > 0, "Estimated variance s20 should be positive"


def test_hatodot_output_shape():
    # Test hatodot returns expected shape and values
    from limix_lmm.lmm_core import hatodot
    N = 4
    mi = 3
    ml = 2
    A = np.arange(N*mi).reshape(N, mi).astype(float)
    B = np.arange(N*ml).reshape(N, ml).astype(float) + 1
    result = hatodot(A, B)
    # Result shape = (N, mi*ml)
    assert result.shape == (N, mi*ml)
    # Check vectorization correctness for first column pair
    expected_first_column = A[:, 0] * B[:, 0]
    np.testing.assert_array_equal(result[:, 0], expected_first_column)


def test_import_version_and_all():
    import limix_lmm
    # Verify __version__ attribute
    assert hasattr(limix_lmm, "__version__")
    assert isinstance(limix_lmm.__version__, str)
    # Verify __all__ membership
    keys = set(limix_lmm.__all__)
    for expected in ["LMM", "LMMCore", "MTLMM", "plot", "download", "unzip", "test"]:
        assert expected in keys


@pytest.mark.parametrize("step", [1, 2, 3])
def test_process_step_sizes(simple_data, step):
    y, F = simple_data
    N = y.shape[0]
    S = 12  # number of variants divisible by step to test exact groups
    G = np.arange(N * S).reshape(N, S).astype(float)
    lmm = LMMCore(y, F)
    lmm.process(G, step=step)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    expected_tests = S // step
    expected_m = step
    assert pv.shape[0] == expected_tests
    assert beta.shape == (expected_m, expected_tests)
    assert np.all((pv >= 0) & (pv <= 1))


def test_process_with_Inter_and_unbalanced_step(simple_data):
    # Test step that doesn't evenly divide number of variants
    y, F = simple_data
    N = y.shape[0]
    S = 5
    step = 3
    G = np.ones((N, S))
    Inter = np.ones((N, 2))
    lmm = LMMCore(y, F)
    # Should ignore extra variant or raise meaningful error
    with pytest.raises(IndexError):
        lmm.process(G, Inter=Inter, step=step)


def test_process_with_Ki_dot_none_works(simple_data):
    # When Ki_dot is None, internally uses y and F directly as KiF, Kiy
    y, F = simple_data
    N = y.shape[0]
    S = 6
    G = np.ones((N, S))
    lmm = LMMCore(y, F, Ki_dot=None)
    # This should not raise exception
    lmm.process(G, step=2)
    assert lmm.getPv().size == S // 2


def test_getters_return_without_process(simple_data):
    y, F = simple_data
    lmm = LMMCore(y, F)
    with pytest.raises(AttributeError):
        _ = lmm.getPv()
    with pytest.raises(AttributeError):
        _ = lmm.getBetaSNP()
    with pytest.raises(AttributeError):
        _ = lmm.getLRT()
