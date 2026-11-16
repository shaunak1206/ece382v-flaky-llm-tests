import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
from scipy import linalg as sp

import skprocrustes as skp
from skprocrustes.skprocrustes import compute_residual


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_problem_rejects_non_square_sizes():
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 8, 2, 2), problemnumber=1)
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_problem_rejects_m_not_divisible_by_p():
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_problem_requires_problemnumber_or_matrices():
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 10, 2, 2))


@pytest.mark.parametrize("number", [1, 2])
def test_default_problems_shapes_and_consistency(number):
    np.random.seed(0)
    problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=number)
    assert problem.sizes == (10, 10, 2, 2)
    assert problem.problemnumber == number
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    # B is built from the known solution
    expected_B = problem.A @ problem.Xsol @ problem.C
    assert np.allclose(problem.B, expected_B)
    # Xsol has orthonormal columns
    assert np.allclose(problem.Xsol.T @ problem.Xsol, np.eye(2))
    # C is the identity
    assert np.allclose(problem.C, np.eye(2))
    assert problem.stats == {}


def test_problem_one_singular_values_in_range():
    np.random.seed(1)
    problem = skp.ProcrustesProblem((12, 12, 3, 3), problemnumber=1)
    s = sp.svdvals(problem.A)
    assert np.all(s >= 10.0 - 1e-8)
    assert np.all(s <= 12.0 + 1e-8)


def test_problem_three_valid_and_invalid_sizes():
    np.random.seed(2)
    problem = skp.ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
    assert problem.A.shape == (50, 50)
    assert problem.B.shape == (50, 5)
    assert np.allclose(problem.B, problem.A @ problem.Xsol @ problem.C)
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_problem_user_matrices_without_solution():
    A = np.random.RandomState(0).randn(6, 6)
    B = np.random.RandomState(1).randn(6, 2)
    C = np.eye(2)
    problem = skp.ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C))
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert not hasattr(problem, "Xsol")


def test_problem_user_matrices_with_solution_as_list():
    rs = np.random.RandomState(3)
    A = rs.randn(6, 6)
    X = np.linalg.qr(rs.randn(6, 2))[0]
    C = np.eye(2)
    B = A @ X @ C
    problem = skp.ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C, X])
    assert problem.Xsol is X
    assert np.allclose(problem.A @ problem.Xsol @ problem.C, problem.B)


def test_problem_user_matrices_wrong_shapes_raise():
    A = np.ones((6, 6))
    B = np.ones((6, 2))
    C = np.eye(2)
    X = np.ones((6, 2))
    with pytest.raises(Exception):
        skp.ProcrustesProblem((6, 6, 2, 2), matrices=(np.ones((5, 6)), B, C))
    with pytest.raises(Exception):
        skp.ProcrustesProblem((6, 6, 2, 2), matrices=(A, np.ones((6, 3)), C))
    with pytest.raises(Exception):
        skp.ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, np.eye(3)))
    with pytest.raises(Exception):
        skp.ProcrustesProblem((6, 6, 2, 2),
                              matrices=(A, B, C, np.ones((2, 6))))
    # silence unused variable warning
    assert X.shape == (6, 2)


def test_problem_too_few_matrices_raises():
    A = np.ones((6, 6))
    B = np.ones((6, 2))
    with pytest.raises(Exception):
        skp.ProcrustesProblem((6, 6, 2, 2), matrices=(A, B))


# ---------------------------------------------------------------------------
# GBBSolver options
# ---------------------------------------------------------------------------

def test_gbb_default_options():
    solver = skp.GBBSolver()
    opts = solver.options
    assert solver.solvername == "gbb"
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


def test_gbb_custom_options_are_kept():
    solver = skp.GBBSolver(xtol=1e-8, gtol=1e-7, maxiter=50, verbose=0,
                           projected_gradient=2, etavar=True)
    assert solver.options["xtol"] == 1e-8
    assert solver.options["gtol"] == 1e-7
    assert solver.options["maxiter"] == 50
    assert solver.options["verbose"] == 0
    assert solver.options["projected_gradient"] == 2
    assert solver.options["etavar"] is True


@pytest.mark.parametrize("kwargs", [
    {"full_results": 1},
    {"xtol": 1},
    {"gtol": "small"},
    {"ftol": 1},
    {"zeta": 1},
    {"kappa": 1},
    {"eta": 1},
    {"etavar": 1},
    {"tau": 1},
    {"projected_gradient": 3},
    {"maxiter": 10.0},
    {"verbose": 2},
    {"filename": 5},
    {"timer": "yes"},
])
def test_gbb_invalid_options_raise(kwargs):
    with pytest.raises(Exception):
        skp.GBBSolver(**kwargs)


# ---------------------------------------------------------------------------
# GBBSolver.solve
# ---------------------------------------------------------------------------

def test_gbb_solve_returns_feasible_solution_that_improves_residual():
    np.random.seed(10)
    problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    solver = skp.GBBSolver(verbose=0)

    np.random.seed(11)
    result = solver.solve(problem)

    # replicate the starting point drawn inside solve()
    np.random.seed(11)
    X0, _ = sp.qr(np.random.rand(10, 2), mode='economic')
    _, initial_res = compute_residual(problem.A, problem.B, problem.C,
                                      X0, None)

    X = result.solution
    assert X.shape == (10, 2)
    assert np.allclose(X.T @ X, np.eye(2), atol=1e-6)
    _, final_res = compute_residual(problem.A, problem.B, problem.C, X, None)
    assert final_res < initial_res
    assert result.cpu >= 0.0
    assert np.isfinite(result.error)
    assert result.nbiter >= 1
    assert result.nfev >= 1
    assert result.status in (0, 1)
    assert result.success == (result.status == 0)


def test_gbb_solve_fills_problem_stats_and_result_consistently():
    np.random.seed(20)
    problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    solver = skp.GBBSolver(verbose=0)
    result = solver.solve(problem)
    assert result.nbiter == problem.stats["nbiter"]
    assert result.nfev == problem.stats["fev"]
    assert problem.stats["fev"] >= 1
    assert problem.stats["grad"] >= 1
    assert problem.stats["feasi"] < 1e-6
    # full_results is off, so histories are reported as inf
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


def test_gbb_solve_maxiter_one_reports_failure():
    np.random.seed(30)
    problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    solver = skp.GBBSolver(verbose=0, maxiter=1)
    result = solver.solve(problem)
    assert result.success is False
    assert result.status == 1
    assert "Maximum number of iterations" in result.message
    assert result.nbiter == 1


def test_gbb_solve_with_p_half_of_n_and_etavar():
    # p >= smalldim/2 selects the inverse-based Crank-Nicolson update
    np.random.seed(40)
    problem = skp.ProcrustesProblem((6, 6, 3, 3), problemnumber=1)
    solver = skp.GBBSolver(verbose=0, etavar=True, maxiter=200)
    result = solver.solve(problem)
    X = result.solution
    assert X.shape == (6, 3)
    assert np.allclose(X.T @ X, np.eye(3), atol=1e-6)


def test_gbb_solve_projected_gradient_two():
    np.random.seed(50)
    problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    solver = skp.GBBSolver(verbose=0, projected_gradient=2, maxiter=300)
    result = solver.solve(problem)
    X = result.solution
    assert X.shape == (10, 2)
    assert np.allclose(X.T @ X, np.eye(2), atol=1e-6)


def test_gbb_solve_user_problem_without_known_solution_has_inf_error():
    rs = np.random.RandomState(7)
    A = rs.randn(8, 8)
    Q = np.linalg.qr(rs.randn(8, 2))[0]
    C = np.eye(2)
    B = A @ Q @ C
    np.random.seed(60)
    problem = skp.ProcrustesProblem((8, 8, 2, 2), matrices=(A, B, C))
    solver = skp.GBBSolver(verbose=0, maxiter=300)
    result = solver.solve(problem)
    assert result.error == np.inf
    assert result.solution.shape == (8, 2)
    assert np.allclose(result.solution.T @ result.solution, np.eye(2),
                       atol=1e-6)


def test_gbb_solve_writes_to_file(tmp_path):
    np.random.seed(70)
    problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    fname = str(tmp_path / "gbb_output.txt")
    solver = skp.GBBSolver(verbose=1, filename=fname, maxiter=20)
    result = solver.solve(problem)
    assert os.path.exists(fname)
    with open(fname) as fh:
        content = fh.read()
    assert "GBB Solver" in content or "Options" in content
    assert result.solution.shape == (10, 2)
