import sys

import numpy as np
import pytest

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver


@pytest.fixture(autouse=True)
def _seed_rng():
    np.random.seed(12345)


def _is_orthonormal(X, atol=1e-6):
    return np.allclose(X.T @ X, np.eye(X.shape[1]), atol=atol)


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_non_square_sizes_rejected():
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((6, 8, 2, 2), problemnumber=1)


def test_m_not_divisible_by_p_rejected():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 4, 4), problemnumber=1)


def test_sizes_problemnumber_and_stats_stored():
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=2)
    assert problem.sizes == (6, 6, 2, 2)
    assert problem.problemnumber == 2
    assert problem.stats == {}


def test_problemnumber_1_singular_values_in_range():
    problem = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert sv.shape == (8,)
    assert sv.min() >= 10.0 - 1e-8
    assert sv.max() <= 12.0 + 1e-8


def test_problemnumber_1_builds_consistent_matrices():
    problem = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)
    assert problem.A.shape == (8, 8)
    assert problem.B.shape == (8, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (8, 2)
    assert np.array_equal(problem.C, np.eye(2))
    assert _is_orthonormal(problem.Xsol)
    assert np.allclose(problem.B, problem.A @ problem.Xsol @ problem.C)


def test_problemnumber_2_shapes_and_known_solution():
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=2)
    assert problem.A.shape == (6, 6)
    assert problem.B.shape == (6, 2)
    assert problem.Xsol.shape == (6, 2)
    assert np.all(np.isfinite(problem.A))
    assert _is_orthonormal(problem.Xsol)
    assert np.allclose(problem.B, problem.A @ problem.Xsol @ problem.C)


def test_problemnumber_3_singular_value_structure():
    problem = ProcrustesProblem((50, 50, 2, 2), problemnumber=3)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(sv[:15] >= 10.0 - 1e-8)
    assert np.all((sv[15:30] >= 5.0 - 1e-8) & (sv[15:30] < 6.0 + 1e-8))
    assert np.all(sv[-8:] < 1e-3)


def test_problemnumber_3_unsupported_n_rejected():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_no_problemnumber_and_no_matrices_rejected():
    with pytest.raises(Exception):
        ProcrustesProblem((4, 4, 2, 2))
    with pytest.raises(Exception):
        ProcrustesProblem((4, 4, 2, 2), problemnumber=4)


def test_user_matrices_stored_by_reference():
    A = np.eye(6)
    B = np.ones((6, 2))
    C = np.eye(2)
    problem = ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C))
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert not hasattr(problem, "Xsol")
    assert problem.problemnumber is None

    Xs = np.eye(6, 2)
    problem4 = ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C, Xs])
    assert problem4.Xsol is Xs


def test_wrong_A_shape_rejected():
    with pytest.raises(Exception, match="A must be"):
        ProcrustesProblem((6, 6, 2, 2), matrices=(np.ones((6, 5)),
                                                  np.ones((6, 2)),
                                                  np.eye(2)))


def test_wrong_B_shape_rejected():
    with pytest.raises(Exception, match="B must be"):
        ProcrustesProblem((6, 6, 2, 2), matrices=(np.eye(6),
                                                  np.ones((6, 3)),
                                                  np.eye(2)))


def test_wrong_C_shape_rejected():
    with pytest.raises(Exception, match="C must be"):
        ProcrustesProblem((6, 6, 2, 2), matrices=(np.eye(6),
                                                  np.ones((6, 2)),
                                                  np.eye(3)))


def test_wrong_Xsol_shape_rejected():
    with pytest.raises(Exception, match="Xsol must be"):
        ProcrustesProblem((6, 6, 2, 2), matrices=(np.eye(6),
                                                  np.ones((6, 2)),
                                                  np.eye(2),
                                                  np.ones((6, 3))))


def test_too_few_matrices_rejected():
    with pytest.raises(Exception, match="matrices must be"):
        ProcrustesProblem((6, 6, 2, 2), matrices=[np.eye(6), np.ones((6, 2))])


# ---------------------------------------------------------------------------
# GBBSolver
# ---------------------------------------------------------------------------

def test_default_options_and_solvername():
    solver = GBBSolver()
    assert solver.solvername == "gbb"
    assert solver.options == {
        "full_results": False,
        "xtol": 1e-6,
        "gtol": 1e-6,
        "ftol": 1e-12,
        "zeta": 1e-4,
        "kappa": 0.1,
        "eta": 0.85,
        "etavar": False,
        "tau": 1e-3,
        "projected_gradient": 1,
        "maxiter": 5000,
        "verbose": 1,
        "filename": None,
        "timer": False,
    }


def test_user_options_override_defaults():
    solver = GBBSolver(gtol=1e-4, maxiter=7, verbose=0)
    assert solver.options["gtol"] == 1e-4
    assert solver.options["maxiter"] == 7
    assert solver.options["verbose"] == 0
    assert solver.options["xtol"] == 1e-6


@pytest.mark.parametrize("kwargs", [
    {"gtol": 1},                  # int instead of float
    {"full_results": 1},          # int instead of bool
    {"verbose": 5},               # out of allowed set
    {"projected_gradient": 3},    # out of allowed set
    {"maxiter": 10.0},            # float instead of int
    {"filename": 3},              # not a string
    {"timer": "yes"},             # not a bool
])
def test_invalid_option_values_rejected(kwargs):
    with pytest.raises(Exception):
        GBBSolver(**kwargs)


def test_open_file_none_uses_stdout():
    solver = GBBSolver(verbose=0)
    solver.open_file()
    assert solver.file is sys.stdout
    solver.close_file()  # must be a no-op for stdout
    assert not sys.stdout.closed


def test_open_and_close_file_with_path(tmp_path):
    path = tmp_path / "gbb.log"
    solver = GBBSolver(verbose=0, filename=str(path))
    solver.open_file()
    assert not solver.file.closed
    solver.file.write("probe")
    solver.close_file()
    assert solver.file.closed
    assert path.exists()
    assert path.read_text() == "probe"


def test_solve_returns_expected_result_fields():
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    solver = GBBSolver(verbose=0)
    result = solver.solve(problem)

    expected = {"success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"}
    assert expected.issubset(set(result.keys()))
    assert result.solution.shape == (6, 2)
    assert isinstance(result.success, bool)
    assert result.success == (result.status == 0)
    assert result.nbiter == problem.stats["nbiter"]
    assert result.nbiter >= 0
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf
    assert np.isfinite(result.fun)
    assert result.fun >= 0
    assert np.isfinite(result.error)


def test_solve_solution_is_orthonormal():
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0).solve(problem)
    assert _is_orthonormal(result.solution)


def test_solve_maxiter_exceeded_reports_failure():
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=1).solve(problem)
    assert result.status == 1
    assert result.success is False
    assert result.message == ("Maximum number of iterations has been "
                              "exceeded.")


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="gbb_solver passes fileobj positionally to print(), "
                          "which becomes the 'sep' argument when verbose > 0")
def test_solve_with_verbose_1_known_print_bug():
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    GBBSolver(verbose=1).solve(problem)


@pytest.mark.xfail(strict=True, raises=Exception,
                   reason="gbb_solver never fills problem.stats['total_crit'], "
                          "so solve() with full_results=True raises")
def test_solve_full_results_known_stats_bug():
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, full_results=True).solve(problem)
    assert len(result.total_fun) > 0
