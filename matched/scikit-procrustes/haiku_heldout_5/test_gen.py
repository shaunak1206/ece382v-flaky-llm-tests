import numpy as np
import pytest

from skprocrustes.skprocrustes import (
    GBBSolver,
    ProcrustesProblem,
    _status_message,
)


@pytest.fixture(autouse=True)
def _seed_rng():
    """Make every test that uses random numbers deterministic."""
    np.random.seed(0)


@pytest.fixture
def gbb_problem():
    """Small, well-conditioned default problem with a known solution."""
    return ProcrustesProblem((8, 8, 2, 2), problemnumber=1)


def _user_matrices(m=6, n=6, p=2, q=2):
    """Build a user-supplied (A, B, C, Xsol) set with compatible sizes."""
    rng = np.random.RandomState(1)
    A = rng.randn(m, n) + m * np.eye(m, n)
    B = rng.randn(m, q)
    C = np.eye(p, q)
    Xsol = np.eye(n, p)
    return A, B, C, Xsol


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_problem1_shapes_and_attributes():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    assert np.array_equal(problem.C, np.eye(2))
    assert problem.problemnumber == 1
    assert problem.sizes == (10, 10, 2, 2)


def test_problem1_singular_values_in_expected_interval():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    # Problem 1 draws singular values uniformly from [10, 12).
    assert np.all(sv >= 10.0 - 1e-8)
    assert np.all(sv <= 12.0 + 1e-8)


def test_problem_xsol_is_feasible_and_consistent_with_b():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    Xsol = problem.Xsol
    assert np.allclose(Xsol.T @ Xsol, np.eye(2))
    assert np.allclose(problem.B, problem.A @ Xsol @ problem.C)


def test_problem2_builds_with_expected_shapes():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=2)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.Xsol.shape == (10, 2)
    assert np.allclose(problem.B, problem.A @ problem.Xsol @ problem.C)


def test_problem3_singular_value_structure():
    problem = ProcrustesProblem((50, 50, 2, 2), problemnumber=3)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    # For n=50 the blocks are (15, 15, 12, 8): 42 values >= 2 and 8 tiny ones.
    assert np.sum(sv >= 2.0) == 42
    assert np.sum(sv < 0.01) == 8


def test_problem3_rejects_unsupported_n():
    with pytest.raises(Exception, match="requires n"):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


@pytest.mark.parametrize("sizes", [(10, 8, 2, 2), (10, 10, 2, 4)])
def test_non_square_sizes_rejected(sizes):
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem(sizes, problemnumber=1)


def test_m_not_divisible_by_p_rejected():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


@pytest.mark.parametrize("problemnumber", [None, 7])
def test_missing_or_unknown_problemnumber_rejected(problemnumber):
    with pytest.raises(Exception, match="problem number has been chosen"):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=problemnumber)


def test_same_seed_gives_reproducible_problem():
    np.random.seed(7)
    p1 = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)
    np.random.seed(7)
    p2 = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)
    assert np.allclose(p1.A, p2.A)
    assert np.allclose(p1.B, p2.B)
    assert np.allclose(p1.Xsol, p2.Xsol)


def test_user_supplied_matrices_are_stored_and_stats_empty():
    A, B, C, Xsol = _user_matrices()
    problem = ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C, Xsol))
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert np.array_equal(problem.Xsol, Xsol)
    assert problem.problemnumber is None
    assert problem.stats == {}


@pytest.mark.parametrize(
    "pos, badshape, msg",
    [
        (0, (6, 5), "A must be"),
        (1, (6, 3), "B must be"),
        (2, (3, 2), "C must be"),
        (3, (5, 2), "Xsol must be"),
    ],
)
def test_user_supplied_wrong_shapes_rejected(pos, badshape, msg):
    mats = list(_user_matrices())
    mats[pos] = np.zeros(badshape)
    with pytest.raises(Exception, match=msg):
        ProcrustesProblem((6, 6, 2, 2), matrices=mats)


def test_user_supplied_too_few_matrices_rejected():
    A, B, _, _ = _user_matrices()
    with pytest.raises(Exception, match="matrices must be"):
        ProcrustesProblem((6, 6, 2, 2), matrices=(A, B))


def test_no_xsol_attribute_when_only_three_matrices_given():
    A, B, C, _ = _user_matrices()
    problem = ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C))
    assert "Xsol" not in problem.__dict__


# ---------------------------------------------------------------------------
# GBBSolver
# ---------------------------------------------------------------------------

def test_gbb_default_options():
    solver = GBBSolver()
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
    assert solver.solvername == "gbb"


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"gtol": 1}, "gtol must be a float"),
        ({"projected_gradient": 3}, "projected_gradient must be 1 or 2"),
        ({"verbose": 2}, "verbose must be 0"),
        ({"timer": 1}, "timer must be boolean"),
        ({"filename": 5}, "filename must be string"),
        ({"maxiter": 10.0}, "maxiter must be an integer"),
    ],
)
def test_gbb_invalid_options_rejected(kwargs, msg):
    with pytest.raises(Exception, match=msg):
        GBBSolver(**kwargs)


def test_gbb_solve_finds_feasible_solution_near_known_solution(gbb_problem):
    result = GBBSolver(verbose=0).solve(gbb_problem)
    X = result.solution
    assert X.shape == (8, 2)
    assert np.allclose(X.T @ X, np.eye(2), atol=1e-6)
    assert result.success is True
    assert result.status == 0
    assert result.message == _status_message["success"]
    # The problem has a unique minimizer equal to Xsol (A is invertible).
    # Tolerance is loose on purpose; the solver is iterative.
    assert result.error < 1e-2


def test_gbb_result_contains_expected_fields(gbb_problem):
    result = GBBSolver(verbose=0, maxiter=5).solve(gbb_problem)
    for key in ("success", "status", "message", "solution", "fun",
                "error", "cpu", "nbiter", "nfev", "total_fun", "total_crit"):
        assert key in result.keys(), key
    assert result.cpu >= 0.0
    assert result.nfev >= 1


def test_gbb_maxiter_exceeded_reports_failure(gbb_problem):
    result = GBBSolver(maxiter=2, verbose=0).solve(gbb_problem)
    assert result.success is False
    assert result.status == 1
    assert result.message == _status_message["maxiter"]
    assert result.nbiter == 2


def test_gbb_error_is_inf_without_known_solution():
    A, B, C, _ = _user_matrices()
    problem = ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C))
    result = GBBSolver(verbose=0, maxiter=20).solve(problem)
    assert np.isinf(result.error)
    assert result.solution.shape == (6, 2)


def test_gbb_criticality_history_disabled_by_default(gbb_problem):
    result = GBBSolver(verbose=0, maxiter=5).solve(gbb_problem)
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


@pytest.mark.xfail(
    strict=True,
    reason="gbb_solver stores 'total_grad' but GBBSolver.solve checks "
           "for 'total_crit', so full_results=True always raises",
)
def test_gbb_full_results_history(gbb_problem):
    result = GBBSolver(full_results=True, verbose=0, maxiter=5).solve(
        gbb_problem)
    assert len(result.total_fun) > 0
    assert len(result.total_crit) > 0


def test_gbb_writes_log_to_file(gbb_problem, tmp_path):
    log = tmp_path / "gbb.log"
    GBBSolver(filename=str(log), verbose=1, maxiter=3).solve(gbb_problem)
    text = log.read_text()
    assert "Options" in text
    assert "Execution date" in text


def test_gbb_projected_gradient_2_runs(gbb_problem):
    result = GBBSolver(projected_gradient=2, verbose=0, maxiter=10).solve(
        gbb_problem)
    assert result.solution.shape == (8, 2)
    assert result.nbiter <= 10
