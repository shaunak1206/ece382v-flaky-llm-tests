import numpy as np
import pytest

import skprocrustes as skp
from skprocrustes import skprocrustes as mod


def _small_problem(number=1, sizes=(10, 10, 2, 2)):
    return skp.ProcrustesProblem(sizes, problemnumber=number)


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_problem_rejects_non_square_sizes():
    np.random.seed(0)
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 8, 2, 2), problemnumber=1)
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_problem_rejects_indivisible_dimensions():
    np.random.seed(0)
    with pytest.raises(Exception, match="divisible"):
        skp.ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_problem_one_structure_and_singular_values():
    np.random.seed(0)
    problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert problem.sizes == (10, 10, 2, 2)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    np.testing.assert_allclose(problem.C, np.eye(2))
    # Xsol has orthonormal columns
    np.testing.assert_allclose(problem.Xsol.T @ problem.Xsol, np.eye(2),
                               atol=1e-12)
    # B = A Xsol C
    np.testing.assert_allclose(problem.B,
                               problem.A @ problem.Xsol @ problem.C,
                               atol=1e-10)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(sv >= 10.0 - 1e-8)
    assert np.all(sv <= 12.0 + 1e-8)


def test_problem_two_structure():
    np.random.seed(1)
    problem = skp.ProcrustesProblem((12, 12, 3, 3), problemnumber=2)
    assert problem.A.shape == (12, 12)
    assert problem.B.shape == (12, 3)
    np.testing.assert_allclose(problem.Xsol.T @ problem.Xsol, np.eye(3),
                               atol=1e-12)
    np.testing.assert_allclose(problem.B,
                               problem.A @ problem.Xsol @ problem.C,
                               atol=1e-8)


def test_problem_three_singular_value_groups():
    np.random.seed(2)
    problem = skp.ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
    sv = np.sort(np.linalg.svd(problem.A, compute_uv=False))[::-1]
    eps = 1e-8
    assert np.all((sv[:15] >= 10 - eps) & (sv[:15] <= 11 + eps))
    assert np.all((sv[15:30] >= 5 - eps) & (sv[15:30] <= 6 + eps))
    assert np.all((sv[30:42] >= 2 - eps) & (sv[30:42] <= 3 + eps))
    assert np.all(sv[42:] <= 1e-3 + eps)


def test_problem_three_requires_specific_n():
    np.random.seed(0)
    with pytest.raises(Exception):
        skp.ProcrustesProblem((20, 20, 2, 2), problemnumber=3)


def test_problem_without_matrices_or_valid_number_raises():
    np.random.seed(0)
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 10, 2, 2))
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=4)


def test_problem_user_matrices_with_solution():
    np.random.seed(0)
    A = np.diag([4.0, 3.0, 2.0, 1.0])
    C = np.eye(2)
    X = np.eye(4, 2)
    B = A @ X @ C
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C, X])
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert problem.Xsol is X


def test_problem_user_matrices_without_solution():
    np.random.seed(0)
    A = np.eye(4)
    B = np.eye(4, 2)
    C = np.eye(2)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))
    assert not hasattr(problem, "Xsol")
    assert 'Xsol' not in problem.__dict__


@pytest.mark.parametrize("shapes", [
    ((3, 4), (4, 2), (2, 2), (4, 2)),   # A wrong
    ((4, 4), (4, 3), (2, 2), (4, 2)),   # B wrong
    ((4, 4), (4, 2), (3, 2), (4, 2)),   # C wrong
    ((4, 4), (4, 2), (2, 2), (4, 3)),   # Xsol wrong
])
def test_problem_user_matrices_wrong_shapes(shapes):
    np.random.seed(0)
    mats = [np.zeros(s) for s in shapes]
    with pytest.raises(Exception):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=mats)


def test_problem_too_few_matrices():
    np.random.seed(0)
    with pytest.raises(Exception):
        skp.ProcrustesProblem((4, 4, 2, 2),
                              matrices=[np.eye(4), np.eye(4, 2)])


def test_problem_reproducible_with_seed_and_stats_empty():
    np.random.seed(123)
    p1 = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    np.random.seed(123)
    p2 = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    np.testing.assert_allclose(p1.A, p2.A)
    np.testing.assert_allclose(p1.B, p2.B)
    np.testing.assert_allclose(p1.Xsol, p2.Xsol)
    assert p1.stats == {}
    assert p1.problemnumber == 1


# ---------------------------------------------------------------------------
# GBBSolver
# ---------------------------------------------------------------------------

def test_gbb_default_options():
    np.random.seed(0)
    solver = skp.GBBSolver()
    assert solver.solvername == "gbb"
    assert solver.options == {
        "full_results": False, "xtol": 1e-6, "gtol": 1e-6, "ftol": 1e-12,
        "zeta": 1e-4, "kappa": 0.1, "eta": 0.85, "etavar": False,
        "tau": 1e-3, "projected_gradient": 1, "maxiter": 5000,
        "verbose": 1, "filename": None, "timer": False}


def test_gbb_custom_options_are_kept():
    np.random.seed(0)
    solver = skp.GBBSolver(maxiter=7, verbose=0, xtol=1e-3,
                           projected_gradient=2, etavar=True)
    assert solver.options["maxiter"] == 7
    assert solver.options["verbose"] == 0
    assert solver.options["xtol"] == 1e-3
    assert solver.options["projected_gradient"] == 2
    assert solver.options["etavar"] is True
    # untouched options keep defaults
    assert solver.options["gtol"] == 1e-6


@pytest.mark.parametrize("bad", [
    {"xtol": 1}, {"gtol": 1}, {"ftol": 1}, {"zeta": 1}, {"kappa": 1},
    {"eta": 1}, {"etavar": 1}, {"tau": 1}, {"projected_gradient": 3},
    {"maxiter": 1.5}, {"verbose": 2}, {"filename": 3}, {"timer": 1},
    {"full_results": 1},
])
def test_gbb_invalid_options_raise(bad):
    np.random.seed(0)
    with pytest.raises(Exception):
        skp.GBBSolver(**bad)


def test_gbb_maxiter_reached_reports_failure():
    np.random.seed(0)
    problem = _small_problem()
    solver = skp.GBBSolver(verbose=0, maxiter=1)
    result = solver.solve(problem)
    assert result.success is False
    assert result.status == 1
    assert result.message == mod._status_message["maxiter"]
    assert result.nbiter == 1
    assert problem.stats["nbiter"] == 1
    # one initial evaluation plus 1..5 line search evaluations
    assert 2 <= result.nfev <= 6
    assert result.nfev == problem.stats["fev"]


def test_gbb_result_fields_and_feasibility():
    np.random.seed(0)
    problem = _small_problem()
    solver = skp.GBBSolver(verbose=0, maxiter=10)
    result = solver.solve(problem)
    for key in ("success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"):
        assert key in result
    assert result.solution.shape == (10, 2)
    np.testing.assert_allclose(result.solution.T @ result.solution,
                               np.eye(2), atol=1e-6)
    assert problem.stats["feasi"] < 1e-6
    assert result.cpu >= 0.0
    assert np.isfinite(result.error)
    # full_results not requested: placeholders are infinity
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf
    # attribute access matches dict access
    assert result.solution is result["solution"]


def test_gbb_is_reproducible_with_same_seed():
    np.random.seed(5)
    problem1 = _small_problem()
    np.random.seed(7)
    r1 = skp.GBBSolver(verbose=0, maxiter=8).solve(problem1)

    np.random.seed(5)
    problem2 = _small_problem()
    np.random.seed(7)
    r2 = skp.GBBSolver(verbose=0, maxiter=8).solve(problem2)

    np.testing.assert_allclose(r1.solution, r2.solution)
    assert r1.nbiter == r2.nbiter
    assert r1.status == r2.status


def test_gbb_without_known_solution_gives_infinite_error():
    np.random.seed(0)
    A = np.diag([4.0, 3.0, 2.0, 1.0])
    C = np.eye(2)
    B = A @ np.eye(4, 2) @ C
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C])
    result = skp.GBBSolver(verbose=0, maxiter=2).solve(problem)
    assert result.error == np.inf
    assert result.solution.shape == (4, 2)
    np.testing.assert_allclose(result.solution.T @ result.solution,
                               np.eye(2), atol=1e-6)


def test_gbb_full_results_requires_total_crit_key():
    # gbb_solver stores "total_grad" while solve() expects "total_crit",
    # so asking for full results currently raises an informative error.
    np.random.seed(0)
    problem = _small_problem()
    solver = skp.GBBSolver(verbose=0, maxiter=2, full_results=True)
    with pytest.raises(Exception, match="full results"):
        solver.solve(problem)
    assert len(problem.stats["total_fun"]) >= 3


def test_gbb_projected_gradient_two_runs_and_stays_feasible():
    np.random.seed(0)
    problem = _small_problem()
    solver = skp.GBBSolver(verbose=0, maxiter=5, projected_gradient=2)
    result = solver.solve(problem)
    assert result.nbiter == 5
    np.testing.assert_allclose(result.solution.T @ result.solution,
                               np.eye(2), atol=1e-6)


def test_gbb_writes_to_file_and_closes_it(tmp_path):
    np.random.seed(0)
    problem = _small_problem()
    out = tmp_path / "gbb_output.txt"
    solver = skp.GBBSolver(verbose=1, maxiter=2, filename=str(out))
    result = solver.solve(problem)
    assert solver.file.closed
    assert out.exists()
    content = out.read_text()
    assert "Options:" in content
    assert result.nbiter == 2
