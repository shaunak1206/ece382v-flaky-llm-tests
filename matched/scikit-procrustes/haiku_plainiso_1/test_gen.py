"""Unit tests for ProcrustesProblem and GBBSolver in skprocrustes.skprocrustes."""

import sys

import numpy as np
import pytest

from skprocrustes.skprocrustes import (
    GBBSolver,
    OptimizeResult,
    ProcrustesProblem,
    ProcrustesSolver,
    _status_message,
)


def _seed(value):
    np.random.seed(value)


def _orthonormal(X):
    return np.allclose(X.T.dot(X), np.eye(X.shape[1]), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# GBBSolver: option handling
# ---------------------------------------------------------------------------

def test_gbb_default_options():
    opts = GBBSolver().options
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


def test_gbb_solvername_and_base_class():
    solver = GBBSolver(verbose=0)
    assert isinstance(solver, ProcrustesSolver)
    assert solver.solvername == "gbb"
    assert solver.solver is None


def test_gbb_keeps_user_supplied_options():
    solver = GBBSolver(tau=2e-3, maxiter=7, verbose=0,
                       projected_gradient=2, xtol=1e-5)
    assert solver.options["tau"] == 2e-3
    assert solver.options["maxiter"] == 7
    assert solver.options["verbose"] == 0
    assert solver.options["projected_gradient"] == 2
    assert solver.options["xtol"] == 1e-5
    # Unspecified options still receive their defaults.
    assert solver.options["gtol"] == 1e-6


def test_gbb_rejects_non_bool_flags():
    for bad in ({"full_results": 1}, {"etavar": "no"}, {"timer": 0}):
        with pytest.raises(Exception, match="must be"):
            GBBSolver(**bad)


def test_gbb_rejects_non_float_tolerances():
    for name in ("xtol", "gtol", "ftol", "zeta", "kappa", "eta", "tau"):
        with pytest.raises(Exception, match="must be a float"):
            GBBSolver(**{name: 1})


def test_gbb_rejects_bad_option_values():
    with pytest.raises(Exception, match="projected_gradient"):
        GBBSolver(projected_gradient=3)
    with pytest.raises(Exception, match="maxiter must be an integer"):
        GBBSolver(maxiter=10.0)
    with pytest.raises(Exception, match="verbose must be"):
        GBBSolver(verbose=2)
    with pytest.raises(Exception, match="filename must be string"):
        GBBSolver(filename=5)


# ---------------------------------------------------------------------------
# GBBSolver: output handling
# ---------------------------------------------------------------------------

def test_gbb_open_and_close_file_default_stdout():
    solver = GBBSolver(verbose=0)
    solver.open_file()
    assert solver.file is sys.stdout
    solver.close_file()


def test_gbb_open_file_bad_path_exits(tmp_path):
    bad = tmp_path / "no_such_dir" / "log.txt"
    solver = GBBSolver(verbose=0, filename=str(bad))
    with pytest.raises(SystemExit):
        solver.open_file()


def test_gbb_solve_writes_log_to_filename(tmp_path):
    log_path = tmp_path / "gbb.log"
    _seed(18)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    solver = GBBSolver(verbose=1, maxiter=20, filename=str(log_path))
    solver.solve(problem)
    assert log_path.exists()
    assert "Options:" in log_path.read_text()


# ---------------------------------------------------------------------------
# GBBSolver: solve behaviour
# ---------------------------------------------------------------------------

def test_gbb_maxiter_zero_returns_initial_point():
    _seed(11)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    result = GBBSolver(maxiter=0, verbose=0).solve(problem)
    X = result.solution

    assert result.status == 1
    assert result.success is False
    assert result.message == _status_message["maxiter"]
    assert result.nbiter == 0
    assert result.nfev == 1
    assert _orthonormal(X)
    # With zero iterations the reported objective is the residual at X0.
    expected = np.linalg.norm(problem.A.dot(X) - problem.B, "fro") ** 2
    assert np.isclose(result.fun, expected, rtol=1e-10)


def test_gbb_maxiter_one_runs_single_iteration():
    _seed(12)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    result = GBBSolver(maxiter=1, verbose=0).solve(problem)

    assert result.nbiter == 1
    assert result.status == 1
    assert result.message == _status_message["maxiter"]
    # One gradient evaluation at the start plus one per iteration.
    assert problem.stats["grad"] == 2
    assert result.nfev >= 2
    assert _orthonormal(result.solution)


def test_gbb_result_fields_and_status_consistency():
    _seed(13)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    result = GBBSolver(maxiter=300, verbose=0).solve(problem)

    assert isinstance(result, OptimizeResult)
    for key in ("success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"):
        assert key in result
    assert result.solution.shape == (6, 2)
    assert result.status in (0, 1)
    assert result.success == (result.status == 0)
    assert result.message in (_status_message["success"],
                              _status_message["maxiter"])
    assert (result.status == 0) == (result.message == _status_message["success"])
    assert result.nbiter == problem.stats["nbiter"]
    assert 0 <= result.nbiter <= 300
    assert result.cpu >= 0.0
    assert result.fun >= 0.0


def _solve_and_check_feasible(n, p, seed):
    _seed(seed)
    problem = ProcrustesProblem((n, n, p, p), problemnumber=1)
    result = GBBSolver(maxiter=300, verbose=0).solve(problem)
    X = result.solution
    assert X.shape == (n, p)
    assert np.all(np.isfinite(X))
    assert _orthonormal(X)
    assert problem.stats["feasi"] < 1e-6


def test_gbb_solution_feasible_when_p_at_least_half_n():
    # p >= n/2 takes the dense (invH) branch of gbb_solver.
    _solve_and_check_feasible(n=4, p=2, seed=14)


def test_gbb_solution_feasible_when_p_below_half_n():
    # p < n/2 takes the low-rank (Woodbury) branch of gbb_solver.
    _solve_and_check_feasible(n=10, p=2, seed=15)


def test_gbb_error_is_inf_without_known_solution():
    _seed(16)
    A = np.random.rand(6, 6)
    B = np.random.rand(6, 2)
    C = np.eye(2)
    problem = ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C])
    result = GBBSolver(maxiter=20, verbose=0).solve(problem)

    assert result.error == np.inf
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


def test_gbb_error_matches_known_solution_distance():
    _seed(17)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    result = GBBSolver(maxiter=300, verbose=0).solve(problem)

    expected = np.linalg.norm(result.solution - problem.Xsol, np.inf)
    assert np.isfinite(result.error)
    assert result.error == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_gbb_is_deterministic_under_seed():
    def run():
        _seed(19)
        problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
        return GBBSolver(maxiter=200, verbose=0).solve(problem)

    first = run()
    second = run()
    np.testing.assert_allclose(first.solution, second.solution,
                               rtol=0, atol=1e-12)
    assert first.fun == pytest.approx(second.fun, rel=1e-12)


# ---------------------------------------------------------------------------
# ProcrustesProblem: construction
# ---------------------------------------------------------------------------

def test_problem_default_xsol_and_b_consistency():
    _seed(21)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)

    assert problem.problemnumber == 1
    assert problem.A.shape == (6, 6)
    assert problem.B.shape == (6, 2)
    assert problem.C.shape == (2, 2)
    assert np.allclose(problem.C, np.eye(2))
    assert problem.Xsol.shape == (6, 2)
    assert np.allclose(problem.Xsol.T.dot(problem.Xsol), np.eye(2))
    assert np.allclose(problem.B, problem.A.dot(problem.Xsol).dot(problem.C))


def test_problem1_singular_values_lie_in_ten_to_twelve():
    _seed(22)
    problem = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert sv.min() >= 10.0 - 1e-8
    assert sv.max() <= 12.0 + 1e-8


def test_problem3_singular_value_clusters():
    _seed(23)
    problem = ProcrustesProblem((50, 50, 2, 2), problemnumber=3)
    sv = np.linalg.svd(problem.A, compute_uv=False)  # descending order
    tol = 1e-8
    assert np.all((sv[0:15] >= 10 - tol) & (sv[0:15] <= 11 + tol))
    assert np.all((sv[15:30] >= 5 - tol) & (sv[15:30] <= 6 + tol))
    assert np.all((sv[30:42] >= 2 - tol) & (sv[30:42] <= 3 + tol))
    assert np.all((sv[42:] >= -tol) & (sv[42:] <= 1e-3 + tol))


def test_problem_rejects_invalid_sizes():
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((6, 8, 2, 2), problemnumber=1)
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((6, 6, 2, 3), problemnumber=1)
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((6, 6, 4, 4), problemnumber=1)


def test_problem_rejects_unknown_problemnumber():
    with pytest.raises(Exception, match="Problem matrices are empty"):
        ProcrustesProblem((6, 6, 2, 2))
    with pytest.raises(Exception, match="Problem matrices are empty"):
        ProcrustesProblem((6, 6, 2, 2), problemnumber=4)
    with pytest.raises(Exception, match="requires n"):
        ProcrustesProblem((20, 20, 2, 2), problemnumber=3)


def test_problem_user_matrices_are_stored():
    _seed(24)
    A = np.random.rand(6, 6)
    B = np.random.rand(6, 2)
    C = np.random.rand(2, 2)
    Xsol = np.random.rand(6, 2)

    problem = ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C])
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert "Xsol" not in problem.__dict__
    assert problem.stats == {}

    with_known = ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C, Xsol))
    assert with_known.Xsol is Xsol


def test_problem_user_matrices_reject_wrong_shapes():
    good_A = np.zeros((6, 6))
    good_B = np.zeros((6, 2))
    good_C = np.zeros((2, 2))

    with pytest.raises(Exception, match="A must be"):
        ProcrustesProblem((6, 6, 2, 2),
                          matrices=[np.zeros((6, 5)), good_B, good_C])
    with pytest.raises(Exception, match="B must be"):
        ProcrustesProblem((6, 6, 2, 2),
                          matrices=[good_A, np.zeros((6, 3)), good_C])
    with pytest.raises(Exception, match="C must be"):
        ProcrustesProblem((6, 6, 2, 2),
                          matrices=[good_A, good_B, np.zeros((3, 2))])
    with pytest.raises(Exception, match="Xsol must be"):
        ProcrustesProblem((6, 6, 2, 2),
                          matrices=[good_A, good_B, good_C, np.zeros((6, 3))])
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        ProcrustesProblem((6, 6, 2, 2), matrices=[good_A, good_B])
