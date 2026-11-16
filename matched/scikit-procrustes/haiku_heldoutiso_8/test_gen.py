import numpy as np
import pytest

from skprocrustes.skprocrustes import (
    GBBSolver,
    OptimizeResult,
    ProcrustesProblem,
    _status_message,
)


@pytest.fixture(autouse=True)
def _seed_rng():
    # Make the random problem generation and solver initialisation repeatable.
    np.random.seed(20240601)
    yield


def _default_problem(n=6, p=2, problemnumber=1):
    return ProcrustesProblem((n, n, p, p), problemnumber=problemnumber)


def _singular_values(M):
    return np.linalg.svd(M, compute_uv=False)


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_init_stores_sizes_and_problemnumber():
    problem = _default_problem(n=6, p=2, problemnumber=1)
    assert problem.sizes == (6, 6, 2, 2)
    assert problem.problemnumber == 1


@pytest.mark.parametrize("sizes", [(10, 12, 2, 2), (10, 10, 2, 3)])
def test_init_rejects_nonsquare_problems(sizes):
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem(sizes, problemnumber=1)


def test_init_rejects_m_not_divisible_by_p():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_default_problem_matrix_shapes():
    m, n, p, q = 8, 8, 2, 2
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)
    assert problem.A.shape == (m, n)
    assert problem.B.shape == (m, q)
    assert problem.C.shape == (p, q)
    assert problem.Xsol.shape == (n, p)


def test_default_problem_C_is_identity():
    problem = _default_problem(n=6, p=2)
    assert np.array_equal(problem.C, np.eye(2))


def test_default_problem_B_equals_A_Xsol_C():
    problem = _default_problem(n=6, p=2)
    expected = np.dot(problem.A, np.dot(problem.Xsol, problem.C))
    assert np.allclose(problem.B, expected, atol=1e-10)


def test_default_problem_Xsol_has_orthonormal_columns():
    problem = _default_problem(n=6, p=2)
    assert np.allclose(problem.Xsol.T @ problem.Xsol, np.eye(2), atol=1e-12)


def test_problem1_singular_values_lie_in_expected_interval():
    # Problem 1: singular values uniformly distributed in [10, 12].
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    s = _singular_values(problem.A)
    assert s.min() >= 10.0 - 1e-8
    assert s.max() <= 12.0 + 1e-8


def test_problem3_has_expected_number_of_tiny_and_large_singular_values():
    # n = 50 uses blocks (15, 15, 12, 8): 42 values >= 2 and 8 values < 0.001.
    problem = ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
    s = _singular_values(problem.A)
    assert s.shape == (50,)
    assert np.sum(s < 1e-2) == 8
    assert np.sum(s >= 2.0 - 1e-8) == 42


def test_problem3_rejects_unsupported_n():
    with pytest.raises(Exception, match="requires n"):
        ProcrustesProblem((7, 7, 1, 1), problemnumber=3)


@pytest.mark.parametrize("problemnumber", [None, 7])
def test_default_generation_without_valid_problemnumber_raises(problemnumber):
    with pytest.raises(Exception, match="empty"):
        ProcrustesProblem((4, 4, 2, 2), problemnumber=problemnumber)


def test_user_matrices_are_stored_by_reference():
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 2)
    C = np.eye(2)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C])
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert not hasattr(problem, "Xsol")


def test_user_known_solution_is_stored_when_given():
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 2)
    C = np.eye(2)
    X = np.random.rand(4, 2)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, X))
    assert problem.Xsol is X


@pytest.mark.parametrize("index, shape, match", [
    (0, (4, 3), "A must be"),
    (1, (4, 3), "B must be"),
    (2, (3, 2), "C must be"),
    (3, (4, 3), "Xsol must be"),
])
def test_user_matrices_with_wrong_shape_raise(index, shape, match):
    mats = [np.ones((4, 4)), np.ones((4, 2)), np.eye(2), np.ones((4, 2))]
    mats[index] = np.ones(shape)
    with pytest.raises(Exception, match=match):
        ProcrustesProblem((4, 4, 2, 2), matrices=mats)


def test_user_matrices_with_too_few_entries_raise():
    with pytest.raises(Exception, match="matrices must be"):
        ProcrustesProblem((4, 4, 2, 2),
                          matrices=[np.ones((4, 4)), np.ones((4, 2))])


def test_stats_dictionary_is_empty_after_init():
    problem = _default_problem(n=6, p=2)
    assert problem.stats == {}


# ---------------------------------------------------------------------------
# GBBSolver
# ---------------------------------------------------------------------------

def test_gbb_default_options_and_overrides():
    solver = GBBSolver()
    assert solver.solvername == "gbb"
    assert solver.options["full_results"] is False
    assert solver.options["xtol"] == 1e-6
    assert solver.options["gtol"] == 1e-6
    assert solver.options["ftol"] == 1e-12
    assert solver.options["zeta"] == 1e-4
    assert solver.options["kappa"] == 0.1
    assert solver.options["eta"] == 0.85
    assert solver.options["etavar"] is False
    assert solver.options["tau"] == 1e-3
    assert solver.options["projected_gradient"] == 1
    assert solver.options["maxiter"] == 5000
    assert solver.options["verbose"] == 1
    assert solver.options["filename"] is None
    assert solver.options["timer"] is False

    custom = GBBSolver(tau=0.01, maxiter=10)
    assert custom.options["tau"] == 0.01
    assert custom.options["maxiter"] == 10


@pytest.mark.parametrize("kwargs, match", [
    ({"xtol": 1}, "xtol must be a float"),
    ({"projected_gradient": 3}, "projected_gradient must be 1 or 2"),
    ({"verbose": 2}, "verbose must be 0"),
    ({"maxiter": 10.0}, "maxiter must be an integer"),
    ({"full_results": 1}, "full_results must be a boolean"),
    ({"timer": "yes"}, "timer must be boolean"),
    ({"filename": 3}, "filename must be string"),
])
def test_gbb_rejects_invalid_options(kwargs, match):
    with pytest.raises(Exception, match=match):
        GBBSolver(**kwargs)


def test_gbb_solve_returns_well_formed_result():
    # full_results is left False: GBBSolver.solve only reads
    # problem.stats["total_fun"]/["total_crit"] when full_results is True,
    # and gbb_solver stores "total_grad" instead of "total_crit".
    problem = _default_problem(n=6, p=2)
    solver = GBBSolver(verbose=0, maxiter=500)
    result = solver.solve(problem)

    assert isinstance(result, OptimizeResult)
    for key in ("success", "status", "message", "solution", "fun",
                "normcrit", "error", "cpu", "nbiter", "nfev",
                "total_fun", "total_crit"):
        assert key in result
    assert result.solution.shape == (6, 2)
    assert result.status in (0, 1)
    assert result.success == (result.status == 0)
    assert result.message in _status_message.values()
    assert 0 <= result.nbiter <= 500
    assert result.cpu >= 0
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


@pytest.mark.parametrize("n", [4, 8])
def test_gbb_solution_has_orthonormal_columns(n):
    # n=4 exercises the invH branch, n=8 the low-rank (invH=False) branch.
    problem = _default_problem(n=n, p=2)
    result = GBBSolver(verbose=0, maxiter=500).solve(problem)
    X = result.solution
    assert np.allclose(X.T @ X, np.eye(2), atol=1e-6)


def test_gbb_error_is_inf_norm_distance_to_known_solution():
    problem = _default_problem(n=6, p=2)
    result = GBBSolver(verbose=0, maxiter=500).solve(problem)
    expected = np.linalg.norm(result.solution - problem.Xsol, np.inf)
    assert np.isfinite(result.error)
    assert result.error == pytest.approx(expected)


def test_gbb_error_is_inf_without_known_solution():
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 2)
    C = np.eye(2)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C])
    result = GBBSolver(verbose=0, maxiter=200).solve(problem)
    assert result.error == np.inf


def test_gbb_solve_does_not_modify_problem_matrices():
    problem = _default_problem(n=6, p=2)
    A0, B0, C0 = problem.A.copy(), problem.B.copy(), problem.C.copy()
    GBBSolver(verbose=0, maxiter=200).solve(problem)
    assert np.array_equal(problem.A, A0)
    assert np.array_equal(problem.B, B0)
    assert np.array_equal(problem.C, C0)
    assert "nbiter" in problem.stats
