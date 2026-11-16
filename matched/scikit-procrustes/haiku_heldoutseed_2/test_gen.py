import numpy as np
import pytest

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver


@pytest.fixture(autouse=True)
def fixed_seed():
    """Make the randomly generated problems and solver starts reproducible."""
    np.random.seed(0)
    yield


def _identity_problem(n=6, p=2):
    """A = I, C = I, B = first p columns of a random orthogonal matrix.

    The global minimum of ||X - B||_F^2 over X^T X = I is 0 (X = B).
    """
    rng = np.random.RandomState(1)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = np.eye(n)
    B = Q[:, :p].copy()
    C = np.eye(p)
    return ProcrustesProblem((n, n, p, p), matrices=(A, B, C))


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_default_problem_has_expected_shapes():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    assert problem.sizes == (10, 10, 2, 2)


def test_problem1_singular_values_lie_in_interval():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    s = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(s >= 10.0 - 1e-8)
    assert np.all(s <= 12.0 + 1e-8)


def test_problem_b_equals_a_xsol_c():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert np.allclose(problem.B, problem.A.dot(problem.Xsol).dot(problem.C))


def test_predefined_xsol_has_orthonormal_columns():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert np.allclose(problem.Xsol.T.dot(problem.Xsol), np.eye(2))


def test_same_seed_gives_same_problem():
    np.random.seed(123)
    p1 = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    np.random.seed(123)
    p2 = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert np.array_equal(p1.A, p2.A)
    assert np.array_equal(p1.B, p2.B)


def test_stats_dict_starts_empty():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert problem.stats == {}


def test_non_square_problem_rejected():
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((10, 8, 2, 2), problemnumber=1)


def test_p_different_from_q_rejected():
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_m_not_divisible_by_p_rejected():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_no_matrices_and_no_problemnumber_rejected():
    with pytest.raises(Exception, match="problem number"):
        ProcrustesProblem((10, 10, 2, 2))


def test_unknown_problemnumber_rejected():
    with pytest.raises(Exception, match="problem number"):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=7)


def test_problemnumber3_rejects_unsupported_n():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_user_matrices_are_stored_and_optional_xsol_handled():
    A = np.eye(4)
    B = np.ones((4, 2))
    C = np.eye(2)
    Xsol = np.eye(4, 2)

    problem = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert not hasattr(problem, "Xsol")

    problem_with_sol = ProcrustesProblem((4, 4, 2, 2),
                                         matrices=[A, B, C, Xsol])
    assert problem_with_sol.Xsol is Xsol


def test_wrong_shape_matrix_rejected():
    A = np.eye(4)
    B = np.ones((3, 2))  # should be (4, 2)
    C = np.eye(2)
    with pytest.raises(Exception, match="B must be"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))


def test_wrong_shape_xsol_rejected():
    A = np.eye(4)
    B = np.ones((4, 2))
    C = np.eye(2)
    Xsol = np.eye(2, 2)  # should be (4, 2)
    with pytest.raises(Exception, match="Xsol must be"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, Xsol))


def test_matrices_as_ndarray_rejected():
    arr = np.zeros((3, 4, 4))
    with pytest.raises(Exception, match="list or tuple"):
        ProcrustesProblem((4, 4, 2, 2), matrices=arr)


# ---------------------------------------------------------------------------
# GBBSolver
# ---------------------------------------------------------------------------

def test_gbb_default_options_and_name():
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


def test_gbb_invalid_projected_gradient_rejected():
    with pytest.raises(Exception, match="projected_gradient"):
        GBBSolver(projected_gradient=3)


def test_gbb_invalid_verbose_rejected():
    with pytest.raises(Exception, match="verbose"):
        GBBSolver(verbose=2)


def test_gbb_non_float_xtol_rejected():
    with pytest.raises(Exception, match="xtol"):
        GBBSolver(xtol=1)


def test_gbb_solve_returns_feasible_solution_of_correct_shape():
    problem = _identity_problem(n=6, p=2)
    solver = GBBSolver(maxiter=200)
    result = solver.solve(problem)

    X = result.solution
    assert X.shape == (6, 2)
    assert np.allclose(X.T.dot(X), np.eye(2), atol=1e-6)
    assert isinstance(result.success, (bool, np.bool_))
    assert result.error == np.inf  # user problem has no known solution


def test_gbb_solve_reaches_near_zero_objective_on_easy_problem():
    problem = _identity_problem(n=6, p=2)
    result = GBBSolver(maxiter=5000).solve(problem)
    assert np.isfinite(result.fun)
    assert result.fun < 1e-3


def test_gbb_maxiter_one_reports_failure():
    problem = _identity_problem(n=6, p=2)
    result = GBBSolver(maxiter=1).solve(problem)
    assert result.success is False
    assert result.status == 1
    assert result.nbiter == 1
    assert "Maximum number of iterations" in result.message


def test_gbb_error_matches_inf_norm_with_known_solution():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(maxiter=200).solve(problem)
    expected = np.linalg.norm(result.solution - problem.Xsol, np.inf)
    assert np.isfinite(result.error)
    assert result.error >= 0.0
    assert np.isclose(result.error, expected)


def test_gbb_solve_writes_log_to_file(tmp_path):
    problem = _identity_problem(n=6, p=2)
    logfile = tmp_path / "gbb_log.txt"
    solver = GBBSolver(filename=str(logfile), maxiter=5)
    solver.solve(problem)

    assert logfile.exists()
    content = logfile.read_text()
    assert "Options:" in content
    assert solver.file.closed


@pytest.mark.xfail(
    strict=True,
    reason="gbb_solver stores 'total_grad' but GBBSolver.solve checks for "
           "'total_crit' when full_results=True",
)
def test_gbb_full_results_reports_history():
    problem = _identity_problem(n=6, p=2)
    result = GBBSolver(full_results=True, maxiter=5).solve(problem)
    assert len(result.total_fun) > 0
    assert len(result.total_crit) > 0
