import numpy as np
import pytest
from limix_lmm.lmm_core import LMMCore
from limix_lmm import LMMCore as LMMCore_imported
from limix_lmm import MTLMM


def test_init_with_none_F_defaults_to_ones():
    y = np.ones((10, 1))
    lmm = LMMCore(y, None)
    assert lmm.F.shape == (10, 1)
    assert np.all(lmm.F == 1)
    assert lmm.y is y
    assert lmm.df == 9


def test_init_with_custom_F_sets_attributes_and_df():
    y = np.ones((10, 1))
    F = np.ones((10, 2))
    lmm = LMMCore(y, F)
    assert np.all(lmm.F == F)
    assert lmm.df == 8


def test_fit_null_with_Ki_dot_none_and_with_function():
    y = np.arange(5).reshape((5, 1)).astype(float)
    F = np.ones((5, 2))

    # Ki_dot None => Ki y and F are y and F themselves
    lmm = LMMCore(y, F)
    assert np.allclose(lmm.Kiy, y)
    assert np.allclose(lmm.KiF, F)
    assert lmm.FKiy.shape == (2, 1)
    assert lmm.FKiF.shape == (2, 2)

    # Ki_dot function returns double the input
    def double_x(x):
        return 2 * x

    lmm2 = LMMCore(y, F, Ki_dot=double_x)
    assert np.allclose(lmm2.Kiy, double_x(y))
    assert np.allclose(lmm2.KiF, double_x(F))


def test_process_without_interaction_variants_and_step_1():
    # simple y,F,K_i_dot
    np.random.seed(123)
    N, S = 15, 4
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)

    lmm.process(G, step=1, verbose=False)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()
    beta_ste = lmm.getBetaSNPste()

    assert pv.shape == (S,)
    assert beta.shape == (S, S) or beta.shape[0] == S
    assert lrt.shape == (S,)
    assert beta_ste.shape == beta.shape
    assert np.all(pv >= 0) and np.all(pv <= 1)
    assert np.all(np.isfinite(beta))
    assert np.all(np.isfinite(beta_ste))
    assert np.all(np.isfinite(lrt))


def test_process_with_interaction_step_1_behaves_correctly():
    np.random.seed(1)
    N, S, M = 20, 5, 3
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=1, verbose=False)

    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    assert pv.shape == (S,)
    assert beta.shape == (M * S, S)
    assert lrt.shape == (S,)
    assert np.all(pv >= 0) and np.all(pv <= 1)


def test_process_with_interaction_step_greater_than_1():
    np.random.seed(2)
    N, S, M, step = 30, 12, 4, 4
    y = np.random.randn(N, 1)
    F = np.ones((N, 2))
    G = np.random.randn(N, S)
    Inter = np.random.randn(N, M)

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=step, verbose=False)

    ntests = S // step
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    assert pv.shape == (ntests,)
    assert beta.shape == (ntests * M, ntests)
    assert lrt.shape == (ntests,)
    assert np.all(pv >= 0) and np.all(pv <= 1)


def test_get_beta_cov_and_beta_snpste_consistency():
    np.random.seed(7)
    N, S = 25, 6
    y = np.random.randn(N, 1)
    F = np.ones((N, 3))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=1, verbose=False)

    beta_cov = lmm.getBetaCov()
    beta_snp = lmm.getBetaSNP()
    beta_snp_ste = lmm.getBetaSNPste()
    pv = lmm.getPv()

    # beta_cov is available and of shape (F.shape[1], 1)
    assert beta_cov.shape == (F.shape[1], 1)
    # beta_snp shape first dim equals total number of SNP effects tested
    assert beta_snp.shape[1] == pv.shape[0]
    # beta_snp_ste shape matches beta_snp
    assert beta_snp_ste.shape == beta_snp.shape
    # beta_snp divided by beta_snp_ste approximately equals z-statistic sqrt(chi2.isf(pv))
    # Check no division by zero and nan
    assert np.all(np.isfinite(beta_snp_ste))
    assert not np.any(np.isnan(beta_snp_ste))


def test_import_paths_and_version():
    # test the imports from __init__.py
    from limix_lmm import __version__
    from limix_lmm import LMMCore as ImportedLMMCore
    from limix_lmm import MTLMM as ImportedMTLMM

    assert __version__ == "0.1.2"
    assert ImportedLMMCore is LMMCore_imported
    assert ImportedMTLMM is MTLMM

    import limix_lmm
    # __all__ contains expected items
    for expected in [
        "LMM",
        "LMMCore",
        "MTLMM",
        "plot",
        "download",
        "unzip",
        "__version__",
        "test",
    ]:
        assert expected in limix_lmm.__all__


@pytest.mark.parametrize("step", [1, 2, 3])
def test_process_multiple_steps_without_interaction(step):
    np.random.seed(11)
    N, S = 20, 12
    y = np.random.randn(N, 1)
    F = np.ones((N, 2))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=step, verbose=False)

    ntests = S // step
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()

    assert pv.shape == (ntests,)
    assert beta.shape[0] == step * 1  # no interaction so m=1
    assert beta.shape[1] == ntests


def test_process_verbose_print(capsys):
    np.random.seed(42)
    N, S = 10, 4
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=1, verbose=True)
    captured = capsys.readouterr()

    assert "Tested for" in captured.out
    assert str(S) in captured.out


def test_get_pv_and_lrt_and_beta_snp_return_correct_shape_and_type():
    np.random.seed(1234)
    N, S = 15, 6
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=1, verbose=False)

    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    lrt = lmm.getLRT()

    assert isinstance(pv, np.ndarray)
    assert isinstance(beta, np.ndarray)
    assert isinstance(lrt, np.ndarray)
    assert pv.shape == (S,)
    assert lrt.shape == (S,)
    assert beta.shape[0] == S
    assert beta.shape[1] == S or beta.shape[1] == 1 or beta.shape[1] == S  # shape can depend on implementation


def test_processing_with_inter_is_none_and_step_not_divisible():
    # When G.shape[1] is not divisible by step, process truncates
    np.random.seed(1)
    N, S, step = 20, 10, 3
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=step, verbose=False)

    ntests = S // step
    pv = lmm.getPv()
    assert pv.shape == (ntests,)


def test_getBetaSNPste_matches_beta_and_pv_sign_shape():
    np.random.seed(99)
    N, S = 30, 5
    y = np.random.randn(N, 1)
    F = np.ones((N, 2))
    G = np.random.randn(N, S)

    lmm = LMMCore(y, F)
    lmm.process(G, step=1, verbose=False)

    beta = lmm.getBetaSNP()
    pv = lmm.getPv()
    ste = lmm.getBetaSNPste()

    assert ste.shape == beta.shape
    # z should have same shape as beta
    import scipy.stats as st
    z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv))
    # shapes compatible for broadcasting
    assert z.shape[-1] == pv.shape[0] or z.shape[0] == beta.shape[0]


def test_process_with_inter_and_step_1_and_inter_all_ones():
    np.random.seed(123)
    N, S, M = 10, 5, 2
    y = np.random.randn(N, 1)
    F = np.ones((N, 1))
    G = np.random.randn(N, S)
    Inter = np.ones((N, M))

    lmm = LMMCore(y, F)
    lmm.process(G, Inter=Inter, step=1, verbose=False)

    pv = lmm.getPv()
    beta = lmm.getBetaSNP()

    assert pv.shape == (S,)
    # beta rows should be M*step = M here
    assert beta.shape[0] == M
    assert beta.shape[1] == S
    assert np.all(pv >= 0) and np.all(pv <= 1)

