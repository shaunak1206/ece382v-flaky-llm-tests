import sys

import numpy as np
import pytest

from skprocrustes.skprocrustes import (ProcrustesProblem, GBBSolver,
                                       gbb_solver, OptimizeResult)


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def _user_matrices(m=4, n=4, p=2, q=2, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.randn(m, n)
    B = rng.randn(m, q)
    C = np.eye(p, q)
    return A, B, C


def test_problem_non_square_sizes_raise():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 8, 2, 2), problemnumber=1)
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_problem_m_not_divisible_by_p_raises():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_problem_without_matrices_or_number_raises():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2))


def test_problem_default_1_shapes_and_consistency():
    np.random.seed(1)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert problem.sizes == (10, 10, 2, 2)
    assert problem.problemnumber == 1
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    # B is built from the known solution
    np.testing.assert_allclose(problem.B,
                               problem.A @ problem.Xsol @ problem.C,
                               atol=1e-10)
    # Xsol has orthonormal columns
    np.testing.assert_allclose(problem.Xsol.T @ problem.Xsol, np.eye(2),
                               atol=1e-12)
    np.testing.assert_allclose(problem.C, np.eye(2))


def test_problem_default_1_singular_values_in_range():
    np.random.seed(2)
    problem = ProcrustesProblem((12, 12, 3, 3), problemnumber=1)
    s = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(s >= 10.0 - 1e-8)
    assert np.all(s <= 12.0 + 1e-8)


def test_problem_default_2_shapes():
    np.random.seed(3)
    problem = ProcrustesProblem((8, 8, 2, 2), problemnumber=2)
    assert problem.A.shape == (8, 8)
    assert problem.B.shape == (8, 2)
    np.testing.assert_allclose(problem.B,
                               problem.A @ problem.Xsol @ problem.C,
                               atol=1e-9)


def test_problem_default_3_requires_specific_n():
    np.random.seed(4)
    problem = ProcrustesProblem((50, 50, 2, 2), problemnumber=3)
    assert problem.A.shape == (50, 50)
    assert problem.Xsol.shape == (50, 2)
    with pytest.raises(Exception):
        ProcrustesProblem((40, 40, 2, 2), problemnumber=3)


def test_problem_stats_initially_empty_dict():
    problem = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    assert problem.stats == {}


def test_problem_user_matrices_with_and_without_xsol():
    A, B, C = _user_matrices()
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert not hasattr(problem, "Xsol")

    Xsol = np.eye(4, 2)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C, Xsol])
    assert problem.Xsol is Xsol


def test_problem_user_matrices_wrong_shapes_raise():
    A, B, C = _user_matrices()
    with pytest.raises(Exception):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A[:, :3], B, C))
    with pytest.raises(Exception):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B[:, :1], C))
    with pytest.raises(Exception):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, np.eye(3, 2)))
    with pytest.raises(Exception):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, np.eye(3, 2)))


def test_problem_too_few_matrices_raises():
    A, B, C = _user_matrices()
    with pytest.raises(Exception):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B))


# ---------------------------------------------------------------------------
# GBBSolver options
# ---------------------------------------------------------------------------

def test_gbb_default_options():
    solver = GBBSolver()
    assert solver.solvername == "gbb"
    opts = solver.options
    assert opts["full_results"] is False
    assert opts["xtol"] == 1e-6
    assert opts["gtol"] == 1e-6
    assert opts["ftol"] == 1e-12
    assert opts["zeta"] == 1e-4
    assert opts["kappa"] == 0.1
    assert opts["eta"] == 0.85
    assert opts["etavar"] is False
    assert opts["tau"] == 1e-3
    assert opts["projected_gradient"] == 1
    assert opts["maxiter"] == 5000
    assert opts["verbose"] == 1
    assert opts["filename"] is None
    assert opts["timer"] is False


def test_gbb_custom_valid_options_are_kept():
    solver = GBBSolver(xtol=1e-4, gtol=1e-5, maxiter=10, verbose=0,
                       projected_gradient=2, etavar=True, timer=True,
                       full_results=True, filename="out.txt")
    opts = solver.options
    assert opts["xtol"] == 1e-4
    assert opts["gtol"] == 1e-5
    assert opts["maxiter"] == 10
    assert opts["verbose"] == 0
    assert opts["projected_gradient"] == 2
    assert opts["etavar"] is True
    assert opts["timer"] is True
    assert opts["full_results"] is True
    assert opts["filename"] == "out.txt"


@pytest.mark.parametrize("kwargs", [
    {"full_results": 1},
    {"xtol": 1},
    {"gtol": 1},
    {"ftol": "small"},
    {"zeta": 1},
    {"kappa": 1},
    {"eta": 1},
    {"etavar": 0},
    {"tau": 1},
    {"projected_gradient": 3},
    {"maxiter": 10.0},
    {"verbose": 2},
    {"filename": 5},
    {"timer": "yes"},
])
def test_gbb_invalid_options_raise(kwargs):
    with pytest.raises(Exception):
        GBBSolver(**kwargs)


def test_gbb_open_and_close_file(tmp_path):
    solver = GBBSolver()
    solver.open_file()
    assert solver.file is sys.stdout
    solver.close_file()
    assert not sys.stdout.closed

    fname = str(tmp_path / "gbb_out.txt")
    solver = GBBSolver(filename=fname)
    solver.open_file()
    assert not solver.file.closed
    solver.close_file()
    assert solver.file.closed


# ---------------------------------------------------------------------------
# GBBSolver.solve / gbb_solver
# ---------------------------------------------------------------------------

def test_gbb_solve_maxiter_zero_returns_initial_feasible_point():
    np.random.seed(5)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    result = GBBSolver(maxiter=0, verbose=0).solve(problem)
    assert isinstance(result, OptimizeResult)
    assert result.success is False
    assert result.status == 1
    assert result.nbiter == 0
    assert result.solution.shape == (6, 2)
    np.testing.assert_allclose(result.solution.T @ result.solution,
                               np.eye(2), atol=1e-10)


def test_gbb_solve_maxiter_one_reports_failure_and_stats():
    np.random.seed(6)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    result = GBBSolver(maxiter=1, verbose=0).solve(problem)
    assert result.success is False
    assert result.status == 1
    assert result.message == "Maximum number of iterations has been exceeded."
    assert problem.stats["nbiter"] == 1
    assert problem.stats["fev"] >= 2
    assert result.nfev == problem.stats["fev"]
    assert result.cpu >= 0
    assert np.isfinite(result.error)  # Xsol is known
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf
    assert result.solution.shape == (6, 2)


def test_gbb_solve_without_known_solution_gives_inf_error():
    A, B, C = _user_matrices(seed=7)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))
    np.random.seed(7)
    result = GBBSolver(maxiter=1, verbose=0).solve(problem)
    assert result.error == np.inf
    assert result.solution.shape == (4, 2)


def test_gbb_solve_keeps_iterates_feasible():
    np.random.seed(8)
    problem = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)
    result = GBBSolver(maxiter=50, verbose=0).solve(problem)
    X = result.solution
    np.testing.assert_allclose(X.T @ X, np.eye(2), atol=1e-6)
    assert problem.stats["feasi"] < 1e-6
    assert np.isfinite(result.fun)


def test_gbb_solve_writes_to_file(tmp_path):
    np.random.seed(9)
    fname = str(tmp_path / "gbb_log.txt")
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    GBBSolver(maxiter=1, verbose=1, filename=fname).solve(problem)
    with open(fname) as fh:
        content = fh.read()
    assert "Options" in content


def test_gbb_solver_full_results_bookkeeping():
    np.random.seed(10)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    solver = GBBSolver(full_results=True, maxiter=1, verbose=0)
    X0, _ = np.linalg.qr(np.random.rand(6, 2))
    exitcode, f, X, normG, nbiter, msg = gbb_solver(
        problem, 6, 6, X0, problem.A, problem.B, solver.options, False,
        sys.stdout)
    assert exitcode == 1
    assert nbiter == 1
    assert len(problem.stats["total_fun"]) == problem.stats["fev"]
    assert len(problem.stats["total_grad"]) == problem.stats["grad"]
    assert X.shape == (6, 2)
    assert normG >= 0
