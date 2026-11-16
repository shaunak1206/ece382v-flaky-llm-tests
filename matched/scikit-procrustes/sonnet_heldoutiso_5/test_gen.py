import numpy as np
import pytest

from skprocrustes.skprocrustes import (ProcrustesProblem, GBBSolver,
                                       OptimizeResult)


def _identity_problem(seed=0, n=8, p=2):
    """A = I, B = Xsol: the unique global minimiser is Xsol."""
    rng = np.random.RandomState(seed)
    Xsol, _ = np.linalg.qr(rng.standard_normal((n, p)))
    A = np.eye(n)
    B = Xsol.copy()
    C = np.eye(p)
    return ProcrustesProblem((n, n, p, p), matrices=(A, B, C, Xsol)), Xsol


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_problem_rejects_non_square_a():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 8, 2, 2), problemnumber=1)


def test_problem_rejects_p_not_equal_q():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_problem_rejects_m_not_divisible_by_p():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_problem_without_matrices_or_number_raises():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2))


def test_problem_invalid_problemnumber_raises():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=7)


@pytest.mark.parametrize("number", [1, 2])
def test_default_problems_shapes_and_consistency(number):
    np.random.seed(1)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=number)
    assert prob.sizes == (10, 10, 2, 2)
    assert prob.problemnumber == number
    assert prob.A.shape == (10, 10)
    assert prob.B.shape == (10, 2)
    assert prob.C.shape == (2, 2)
    assert prob.Xsol.shape == (10, 2)
    # Known solution has orthonormal columns and reproduces B exactly.
    np.testing.assert_allclose(prob.Xsol.T @ prob.Xsol, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(prob.A @ prob.Xsol @ prob.C, prob.B,
                               atol=1e-10)
    np.testing.assert_allclose(prob.C, np.eye(2))


def test_problem1_singular_values_in_expected_range():
    np.random.seed(2)
    prob = ProcrustesProblem((12, 12, 3, 3), problemnumber=1)
    sv = np.linalg.svd(prob.A, compute_uv=False)
    assert sv.min() >= 10.0 - 1e-8
    assert sv.max() <= 12.0 + 1e-8


def test_problem3_requires_supported_size_and_builds_for_n50():
    np.random.seed(3)
    prob = ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
    assert prob.A.shape == (50, 50)
    assert prob.Xsol.shape == (50, 5)
    np.testing.assert_allclose(prob.A @ prob.Xsol @ prob.C, prob.B,
                               atol=1e-8)
    with pytest.raises(Exception):
        ProcrustesProblem((20, 20, 2, 2), problemnumber=3)


def test_user_matrices_with_known_solution():
    A = np.arange(16.0).reshape(4, 4)
    B = np.ones((4, 2))
    C = np.eye(2)
    X = np.eye(4, 2)
    prob = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C, X])
    assert prob.A is A
    assert prob.B is B
    assert prob.C is C
    assert prob.Xsol is X
    assert prob.stats == {}


def test_user_matrices_without_solution_has_no_xsol():
    A = np.eye(4)
    B = np.ones((4, 2))
    C = np.eye(2)
    prob = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))
    assert not hasattr(prob, "Xsol")


def test_user_matrices_wrong_shapes_raise():
    A = np.eye(4)
    B = np.ones((4, 2))
    C = np.eye(2)
    X = np.eye(4, 2)
    sizes = (4, 4, 2, 2)
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(np.eye(3), B, C))
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(A, np.ones((3, 2)), C))
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(A, B, np.eye(3)))
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(A, B, C, np.eye(4, 3)))
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(A, B))


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


def test_gbb_custom_options_are_kept():
    solver = GBBSolver(xtol=1e-8, maxiter=10, verbose=0,
                       projected_gradient=2, etavar=True)
    assert solver.options["xtol"] == 1e-8
    assert solver.options["maxiter"] == 10
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
        GBBSolver(**kwargs)


# ---------------------------------------------------------------------------
# GBBSolver.solve
# ---------------------------------------------------------------------------

def test_gbb_solve_recovers_known_solution_for_identity_a():
    np.random.seed(10)
    prob, Xsol = _identity_problem(seed=10)
    result = GBBSolver(verbose=0).solve(prob)
    assert isinstance(result, OptimizeResult)
    assert result.solution.shape == (8, 2)
    assert np.linalg.norm(result.solution - Xsol, "fro") < 1e-2
    assert result.success == (result.status == 0)


def test_gbb_solution_is_feasible_and_result_fields_present():
    np.random.seed(11)
    prob, _ = _identity_problem(seed=11)
    result = GBBSolver(verbose=0).solve(prob)
    X = result.solution
    np.testing.assert_allclose(X.T @ X, np.eye(2), atol=1e-6)
    assert prob.stats["feasi"] < 1e-6
    for key in ("success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"):
        assert key in result
    assert result.cpu >= 0
    assert result.nbiter == prob.stats["nbiter"]
    assert result.nfev == prob.stats["fev"]
    assert np.isfinite(result.error)
    # without full_results the totals are placeholders
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


def test_gbb_solve_maxiter_reports_failure():
    np.random.seed(12)
    prob, _ = _identity_problem(seed=12)
    result = GBBSolver(verbose=0, maxiter=1).solve(prob)
    assert result.success is False
    assert result.status == 1
    assert "Maximum number of iterations" in result.message
    assert result.nbiter == 1


def test_gbb_solve_without_known_solution_gives_inf_error():
    np.random.seed(13)
    n, p = 8, 2
    Xtrue, _ = np.linalg.qr(np.random.randn(n, p))
    A = np.eye(n)
    prob = ProcrustesProblem((n, n, p, p),
                             matrices=(A, Xtrue.copy(), np.eye(p)))
    result = GBBSolver(verbose=0).solve(prob)
    assert result.error == np.inf
    assert result.solution.shape == (n, p)


def test_gbb_solve_default_problem_runs_and_decreases_cost():
    np.random.seed(14)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=200).solve(prob)
    X = result.solution
    assert X.shape == (10, 2)
    np.testing.assert_allclose(X.T @ X, np.eye(2), atol=1e-6)
    assert result.nbiter >= 1
    assert result.nfev >= 2
    assert np.isfinite(result.fun)
    assert result.fun >= 0


def test_gbb_solve_with_invh_branch_and_second_gradient_option():
    # p >= smalldim/2 takes the direct-inverse branch (invH = True)
    np.random.seed(15)
    prob, Xsol = _identity_problem(seed=15, n=6, p=3)
    result = GBBSolver(verbose=0, maxiter=300).solve(prob)
    X = result.solution
    assert X.shape == (6, 3)
    np.testing.assert_allclose(X.T @ X, np.eye(3), atol=1e-6)

    # projected_gradient=2 is used in the low-rank-update branch
    np.random.seed(16)
    prob2, _ = _identity_problem(seed=16, n=8, p=2)
    result2 = GBBSolver(verbose=0, projected_gradient=2,
                        maxiter=300).solve(prob2)
    X2 = result2.solution
    np.testing.assert_allclose(X2.T @ X2, np.eye(2), atol=1e-6)


def test_gbb_solve_writes_to_file_and_closes_it(tmp_path):
    np.random.seed(17)
    prob, _ = _identity_problem(seed=17)
    path = tmp_path / "gbb_output.txt"
    solver = GBBSolver(verbose=1, filename=str(path), maxiter=20)
    solver.solve(prob)
    assert path.exists()
    assert solver.file.closed
    assert "Options" in path.read_text()
