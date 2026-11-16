import numpy as np
import pytest
from scipy import linalg as sp

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver


@pytest.fixture(autouse=True)
def _restore_numpy_random_state():
    # Seeding inside each test changes numpy's global generator state.
    # Save it before the test and restore it afterwards.
    state = np.random.get_state()
    yield
    np.random.set_state(state)


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_problem1_stores_sizes_and_shapes():
    np.random.seed(1)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)

    assert problem.sizes == (6, 6, 2, 2)
    assert problem.problemnumber == 1
    assert problem.A.shape == (6, 6)
    assert problem.B.shape == (6, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (6, 2)
    assert np.array_equal(problem.C, np.eye(2))
    assert problem.stats == {}


def test_nonsquare_m_n_raises():
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((6, 5, 2, 2), problemnumber=1)


def test_nonsquare_p_q_raises():
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((6, 6, 2, 3), problemnumber=1)


def test_m_not_divisible_by_p_raises():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((5, 5, 2, 2), problemnumber=1)


def test_no_matrices_and_no_problemnumber_raises():
    with pytest.raises(Exception, match="no problem number"):
        ProcrustesProblem((6, 6, 2, 2), problemnumber=None)


def test_problem3_unsupported_n_raises():
    np.random.seed(2)
    with pytest.raises(Exception, match="requires"):
        ProcrustesProblem((8, 8, 2, 2), problemnumber=3)


def test_problem1_singular_values_lie_in_interval():
    np.random.seed(3)
    problem = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)

    # U and V are orthogonal, so the singular values of A are the diagonal
    # entries of Sigma, which lie in [10, 12) for every random outcome.
    sv = sp.svd(problem.A, compute_uv=False)
    assert np.all(sv >= 10.0 - 1e-8)
    assert np.all(sv <= 12.0 + 1e-8)


def test_problem1_xsol_has_orthonormal_columns():
    np.random.seed(4)
    problem = ProcrustesProblem((6, 6, 3, 3), problemnumber=1)

    # Xsol is a row permutation of eye(n, p), so Xsol^T Xsol = I.
    assert np.allclose(problem.Xsol.T @ problem.Xsol, np.eye(3), atol=1e-12)


def test_problem1_B_equals_A_Xsol_C():
    np.random.seed(5)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)

    expected = np.dot(problem.A, np.dot(problem.Xsol, problem.C))
    assert np.allclose(problem.B, expected, atol=1e-10)


def test_problem3_singular_value_structure():
    np.random.seed(6)
    problem = ProcrustesProblem((50, 50, 2, 2), problemnumber=3)

    # For n = 50, the diagonal blocks are (15, 15, 12, 8) entries in the
    # ranges [10,11), [5,6), [2,3) and [0, 1e-3). Singular values are
    # returned in descending order, so the groups appear in that order.
    sv = sp.svd(problem.A, compute_uv=False)
    tol = 1e-8
    assert np.all((sv[0:15] >= 10.0 - tol) & (sv[0:15] <= 11.0 + tol))
    assert np.all((sv[15:30] >= 5.0 - tol) & (sv[15:30] <= 6.0 + tol))
    assert np.all((sv[30:42] >= 2.0 - tol) & (sv[30:42] <= 3.0 + tol))
    assert np.all(sv[42:] <= 1e-3 + tol)


def test_problem2_xsol_and_B_consistent():
    np.random.seed(7)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=2)

    assert problem.A.shape == (6, 6)
    assert problem.Xsol.shape == (6, 2)
    expected = np.dot(problem.A, np.dot(problem.Xsol, problem.C))
    assert np.allclose(problem.B, expected, atol=1e-10)


def test_user_supplied_matrices_stored_without_xsol():
    A = np.eye(4)
    B = np.ones((4, 2))
    C = np.eye(2)

    problem_list = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C])
    assert problem_list.A is A
    assert problem_list.B is B
    assert problem_list.C is C
    assert problem_list.problemnumber is None
    assert not hasattr(problem_list, "Xsol")

    problem_tuple = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))
    assert problem_tuple.A is A
    assert not hasattr(problem_tuple, "Xsol")


def test_user_supplied_matrices_with_xsol_stored():
    A = np.eye(4)
    B = np.ones((4, 2))
    C = np.eye(2)
    X = np.ones((4, 2))

    problem = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, X))
    assert problem.Xsol is X


def test_user_supplied_wrong_shapes_raise():
    A = np.eye(4)
    B = np.ones((4, 2))
    C = np.eye(2)
    cases = [
        ([np.ones((3, 4)), B, C], "A must be"),
        ([A, np.ones((4, 3)), C], "B must be"),
        ([A, B, np.ones((2, 3))], "C must be"),
        ([A, B, C, np.ones((4, 3))], "Xsol must be"),
    ]
    for matrices, message in cases:
        with pytest.raises(Exception, match=message):
            ProcrustesProblem((4, 4, 2, 2), matrices=matrices)


def test_user_supplied_too_few_matrices_raises():
    A = np.eye(4)
    B = np.ones((4, 2))
    with pytest.raises(Exception, match="matrices must be"):
        ProcrustesProblem((4, 4, 2, 2), matrices=[A, B])


# ---------------------------------------------------------------------------
# GBBSolver
# ---------------------------------------------------------------------------

def test_gbbsolver_default_options_and_user_overrides():
    solver = GBBSolver()
    assert solver.solvername == "gbb"
    expected_defaults = {
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
    assert dict(solver.options) == expected_defaults

    solver = GBBSolver(verbose=0, maxiter=7, gtol=1e-4)
    assert solver.options["verbose"] == 0
    assert solver.options["maxiter"] == 7
    assert solver.options["gtol"] == 1e-4
    assert solver.options["xtol"] == 1e-6


def test_gbbsolver_invalid_option_types_raise():
    bad_options = [
        {"xtol": 1},
        {"gtol": 1},
        {"ftol": "1e-12"},
        {"zeta": 1},
        {"kappa": 1},
        {"eta": 1},
        {"etavar": 1},
        {"tau": 1},
        {"projected_gradient": 3},
        {"maxiter": 1.0},
        {"verbose": 2},
        {"full_results": 1},
        {"filename": 3},
        {"timer": 1},
    ]
    for kwargs in bad_options:
        with pytest.raises(Exception):
            GBBSolver(**kwargs)


def test_gbbsolver_solution_is_feasible():
    np.random.seed(8)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    solver = GBBSolver(verbose=0)

    result = solver.solve(problem)
    X = result.solution

    assert X.shape == (6, 2)
    assert np.allclose(X.T @ X, np.eye(2), atol=1e-6)


def test_gbbsolver_result_contains_expected_keys():
    np.random.seed(9)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0).solve(problem)

    for key in ("success", "status", "message", "solution", "fun",
                "normcrit", "error", "cpu", "nbiter", "nfev",
                "total_fun", "total_crit"):
        assert key in result.keys()
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


def test_gbbsolver_error_is_infinite_without_known_solution():
    np.random.seed(10)
    A = np.diag([3.0, 2.0, 1.0, 0.5])
    B = np.eye(4)[:, 0:2]
    C = np.eye(2)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C])

    result = GBBSolver(verbose=0, maxiter=200).solve(problem)

    assert result.error == np.inf
    assert result.solution.shape == (4, 2)


def test_gbbsolver_error_matches_distance_to_xsol():
    np.random.seed(11)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)

    result = GBBSolver(verbose=0).solve(problem)

    expected = np.linalg.norm(result.solution - problem.Xsol, np.inf)
    assert np.isclose(result.error, expected, rtol=1e-10, atol=1e-12)


def test_gbbsolver_status_consistent_with_maxiter():
    np.random.seed(12)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)

    result = GBBSolver(verbose=0, maxiter=1).solve(problem)

    assert result.nbiter <= 1
    assert result.success == (result.status == 0)
    # The solver reports status 1 exactly when nbiter reached maxiter.
    assert (result.status == 1) == (result.nbiter >= 1)
    if result.status == 1:
        assert result.message == \
            "Maximum number of iterations has been exceeded."


def test_gbbsolver_does_not_modify_problem_inputs():
    np.random.seed(13)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    A0 = problem.A.copy()
    B0 = problem.B.copy()
    C0 = problem.C.copy()
    Xsol0 = problem.Xsol.copy()

    GBBSolver(verbose=0).solve(problem)

    assert np.array_equal(problem.A, A0)
    assert np.array_equal(problem.B, B0)
    assert np.array_equal(problem.C, C0)
    assert np.array_equal(problem.Xsol, Xsol0)


def test_gbbsolver_fills_problem_stats():
    np.random.seed(14)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)

    result = GBBSolver(verbose=0).solve(problem)

    assert problem.stats["nbiter"] == result.nbiter
    assert result.nfev == problem.stats["fev"]
    assert problem.stats["fev"] >= 1
    for key in ("grad", "linear_solver", "feasi"):
        assert key in problem.stats


def test_gbbsolver_same_seed_gives_same_solution():
    A = np.diag(np.arange(1.0, 7.0))
    B = np.arange(12.0).reshape(6, 2) / 10.0
    C = np.eye(2)
    problem = ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C])

    np.random.seed(15)
    first = GBBSolver(verbose=0).solve(problem)

    np.random.seed(15)
    second = GBBSolver(verbose=0).solve(problem)

    assert np.allclose(first.solution, second.solution, rtol=0, atol=1e-12)
    assert first.status == second.status


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="gbb_solver calls print(..., fileobj) which passes the file "
           "object as the 'sep' argument when verbose > 0",
)
def test_gbbsolver_verbose1_solve_runs():
    np.random.seed(16)
    A = np.eye(4)
    B = np.ones((4, 2))
    C = np.eye(2)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C])

    GBBSolver(verbose=1).solve(problem)
