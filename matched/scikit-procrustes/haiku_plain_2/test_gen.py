"""Unit tests for ProcrustesProblem and GBBSolver in skprocrustes."""

import numpy as np
import pytest

from skprocrustes import GBBSolver, OptimizeResult, ProcrustesProblem
from skprocrustes import ProcrustesSolver
from skprocrustes.skprocrustes import _status_message


def _orthonormal(rows, cols, seed):
    """Random matrix with orthonormal columns, from a private RNG."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((rows, cols)))
    return Q


def _identity_problem(n, p, seed):
    """Planted problem with A = I, C = I, B = Xs, so the optimum is Xs."""
    Xs = _orthonormal(n, p, seed)
    problem = ProcrustesProblem((n, n, p, p),
                                matrices=[np.eye(n), Xs, np.eye(p), Xs])
    return problem, Xs


def test_gbb_default_options():
    solver = GBBSolver()
    assert isinstance(solver, ProcrustesSolver)
    assert solver.solvername == "gbb"
    expected = {
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
    for key, value in expected.items():
        assert solver.options[key] == value, key


def test_gbb_explicit_options_are_kept():
    solver = GBBSolver(maxiter=7, tau=2e-2, projected_gradient=2, verbose=0)
    assert solver.options["maxiter"] == 7
    assert solver.options["tau"] == 2e-2
    assert solver.options["projected_gradient"] == 2
    assert solver.options["verbose"] == 0


def test_gbb_rejects_invalid_options():
    with pytest.raises(Exception, match="tau must be a float"):
        GBBSolver(tau=1)
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        GBBSolver(projected_gradient=3)
    with pytest.raises(Exception, match="maxiter must be an integer"):
        GBBSolver(maxiter=10.0)
    with pytest.raises(Exception, match="verbose must be"):
        GBBSolver(verbose=2)
    with pytest.raises(Exception, match="full_results must be a boolean"):
        GBBSolver(full_results=1)
    with pytest.raises(Exception, match="filename must be string"):
        GBBSolver(filename=123)


def test_problem_stores_sizes_and_empty_stats():
    np.random.seed(0)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    assert problem.sizes == (6, 6, 2, 2)
    assert problem.problemnumber == 1
    assert problem.A.shape == (6, 6)
    assert problem.B.shape == (6, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (6, 2)
    assert problem.stats == {}


def test_problem1_singular_values_in_range():
    np.random.seed(1)
    problem = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)
    svals = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(svals >= 10.0 - 1e-8)
    assert np.all(svals <= 12.0 + 1e-8)


def test_planted_problems_satisfy_B_equals_AXC():
    for number in (1, 2):
        np.random.seed(number)
        problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=number)
        assert np.allclose(problem.C, np.eye(2))
        assert np.allclose(problem.Xsol.T @ problem.Xsol, np.eye(2))
        assert np.allclose(problem.B, problem.A @ problem.Xsol @ problem.C)


def test_problem3_singular_value_clusters():
    np.random.seed(2)
    problem = ProcrustesProblem((50, 50, 2, 2), problemnumber=3)
    svals = np.sort(np.linalg.svd(problem.A, compute_uv=False))[::-1]
    tol = 1e-8
    assert np.all(svals[:15] >= 10.0 - tol)
    assert np.all(svals[:15] < 11.0 + tol)
    assert np.all(svals[15:30] >= 5.0 - tol)
    assert np.all(svals[15:30] < 6.0 + tol)
    assert np.all(svals[30:42] >= 2.0 - tol)
    assert np.all(svals[30:42] < 3.0 + tol)
    assert np.all(svals[42:] < 1e-3 + tol)


def test_problem_rejects_bad_sizes():
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((6, 8, 2, 2), problemnumber=1)
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((6, 6, 2, 3), problemnumber=1)
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)
    with pytest.raises(Exception, match="requires"):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_problem_without_source_raises():
    with pytest.raises(Exception, match="Problem matrices are empty"):
        ProcrustesProblem((4, 4, 2, 2))
    with pytest.raises(Exception, match="Problem matrices are empty"):
        ProcrustesProblem((4, 4, 2, 2), problemnumber=4)


def test_user_matrices_stored_by_reference():
    A = np.eye(4)
    B = np.ones((4, 2))
    C = np.eye(2)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C])
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert "Xsol" not in problem.__dict__


def test_user_matrices_with_known_solution_sets_Xsol():
    Xs = _orthonormal(4, 2, seed=3)
    problem = ProcrustesProblem((4, 4, 2, 2),
                                matrices=(np.eye(4), Xs, np.eye(2), Xs))
    assert problem.Xsol is Xs


def test_user_matrices_bad_shapes_raise():
    with pytest.raises(Exception, match="A must be"):
        ProcrustesProblem((4, 4, 2, 2),
                          matrices=[np.eye(3), np.ones((4, 2)), np.eye(2)])
    with pytest.raises(Exception, match="B must be"):
        ProcrustesProblem((4, 4, 2, 2),
                          matrices=[np.eye(4), np.ones((4, 3)), np.eye(2)])
    with pytest.raises(Exception, match="C must be"):
        ProcrustesProblem((4, 4, 2, 2),
                          matrices=[np.eye(4), np.ones((4, 2)), np.eye(3)])
    with pytest.raises(Exception, match="Xsol must be"):
        ProcrustesProblem((4, 4, 2, 2),
                          matrices=[np.eye(4), np.ones((4, 2)), np.eye(2),
                                    np.ones((3, 2))])
    with pytest.raises(Exception, match="matrices must be"):
        ProcrustesProblem((4, 4, 2, 2), matrices=[np.eye(4), np.ones((4, 2))])


def test_gbb_maxiter_one_reports_maxiter():
    problem, _ = _identity_problem(4, 2, seed=4)
    result = GBBSolver(maxiter=1, verbose=0).solve(problem)
    assert result.success is False
    assert result.status == 1
    assert result.message == _status_message["maxiter"]
    assert result.nbiter == 1


def test_gbb_maxiter_one_stats():
    problem, _ = _identity_problem(4, 2, seed=5)
    result = GBBSolver(maxiter=1, verbose=0).solve(problem)
    # one gradient at the start plus one in the single outer iteration
    assert problem.stats["grad"] == 2
    # one evaluation at the start plus between 1 and 5 backtracking trials
    assert 2 <= problem.stats["fev"] <= 6
    assert result.nfev == problem.stats["fev"]
    assert problem.stats["feasi"] < 1e-8


def test_gbb_error_inf_without_known_solution():
    problem = ProcrustesProblem((4, 4, 2, 2),
                                matrices=[np.eye(4), np.ones((4, 2)), np.eye(2)])
    np.random.seed(6)
    result = GBBSolver(maxiter=3, verbose=0).solve(problem)
    assert result.error == np.inf
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


def test_gbb_error_matches_inf_norm_of_gap():
    np.random.seed(7)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    result = GBBSolver(maxiter=5, verbose=0).solve(problem)
    expected = np.linalg.norm(result.solution - problem.Xsol, np.inf)
    assert np.isclose(result.error, expected)


def test_gbb_solution_feasible_dense_branch():
    # p = 2 is not < n/2 = 2, so the explicit H-matrix update is used.
    np.random.seed(8)
    problem, _ = _identity_problem(4, 2, seed=8)
    result = GBBSolver(maxiter=50, verbose=0).solve(problem)
    X = result.solution
    assert X.shape == (4, 2)
    assert np.allclose(X.T @ X, np.eye(2), atol=1e-6)


def test_gbb_solution_feasible_thin_branch():
    # p = 2 < n/2 = 5, so the low-rank (Sherman-Morrison) update is used.
    for pg in (1, 2):
        np.random.seed(9 + pg)
        problem, _ = _identity_problem(10, 2, seed=9)
        result = GBBSolver(maxiter=50, verbose=0,
                           projected_gradient=pg).solve(problem)
        X = result.solution
        assert X.shape == (10, 2)
        assert np.allclose(X.T @ X, np.eye(2), atol=1e-6)


def test_gbb_converges_planted_identity_small():
    np.random.seed(10)
    problem, _ = _identity_problem(4, 2, seed=10)
    result = GBBSolver(verbose=0).solve(problem)
    assert result.success
    assert result.error < 1e-3


def test_gbb_converges_planted_identity_thin():
    np.random.seed(11)
    problem, _ = _identity_problem(10, 2, seed=11)
    result = GBBSolver(verbose=0).solve(problem)
    assert result.success
    assert result.error < 1e-2


def test_gbb_seeded_runs_reproducible():
    problem, _ = _identity_problem(6, 2, seed=12)
    np.random.seed(123)
    first = GBBSolver(maxiter=30, verbose=0).solve(problem)
    np.random.seed(123)
    second = GBBSolver(maxiter=30, verbose=0).solve(problem)
    assert np.allclose(first.solution, second.solution, atol=1e-12)
    assert first.nbiter == second.nbiter


def test_gbb_writes_log_file(tmp_path):
    problem, _ = _identity_problem(4, 2, seed=13)
    log = tmp_path / "gbb.log"
    GBBSolver(maxiter=2, verbose=1, filename=str(log)).solve(problem)
    text = log.read_text()
    assert "Options:" in text
    assert "nbiter" in text
    assert "Execution date" in text


def test_gbb_result_fields_present():
    problem, _ = _identity_problem(4, 2, seed=14)
    result = GBBSolver(maxiter=3, verbose=0).solve(problem)
    assert isinstance(result, OptimizeResult)
    for key in ("success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"):
        assert key in result, key
    assert result.solution.shape == (4, 2)
    assert result.cpu >= 0.0
