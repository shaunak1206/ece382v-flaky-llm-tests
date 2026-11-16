"""Unit tests for ProcrustesProblem and GBBSolver in skprocrustes/skprocrustes.py.

Determinism notes:
- Every test that builds a problem or runs a solver seeds numpy's global
  generator at the top of its body. The `restore_numpy_random_state` fixture
  only saves and restores the global numpy RNG state, so tests do not leak
  state into each other.
- Floating point comparisons use explicit tolerances.
- Tests that need to replace module attributes use pytest's monkeypatch.
- No wall clock, sleeping, sets, or dict ordering is relied upon.
"""
import numpy as np
import pytest

import skprocrustes.skprocrustes as skmod
from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver

SEED = 20240601
TOL = 1e-9


@pytest.fixture
def restore_numpy_random_state():
    """Save the global numpy RNG state and restore it after the test."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _is_feasible(X, atol=1e-6):
    return np.allclose(X.T @ X, np.eye(X.shape[1]), atol=atol)


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_non_square_problem_is_rejected(restore_numpy_random_state):
    np.random.seed(SEED)
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((4, 5, 2, 2), problemnumber=1)
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((4, 4, 2, 3), problemnumber=1)


def test_m_not_divisible_by_p_is_rejected(restore_numpy_random_state):
    np.random.seed(SEED)
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((5, 5, 2, 2), problemnumber=1)


def test_missing_or_unknown_problemnumber_is_rejected(restore_numpy_random_state):
    np.random.seed(SEED)
    with pytest.raises(Exception, match="no problem number"):
        ProcrustesProblem((4, 4, 2, 2))
    with pytest.raises(Exception, match="no problem number"):
        ProcrustesProblem((4, 4, 2, 2), problemnumber=7)


def test_problem3_rejects_unsupported_n(restore_numpy_random_state):
    np.random.seed(SEED)
    with pytest.raises(Exception, match="requires n"):
        ProcrustesProblem((7, 7, 1, 1), problemnumber=3)


def test_problem1_builds_consistent_matrices(restore_numpy_random_state):
    np.random.seed(SEED)
    m, n, p, q = 6, 6, 2, 2
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)
    assert problem.A.shape == (m, n)
    assert problem.B.shape == (m, q)
    assert problem.C.shape == (p, q)
    assert problem.Xsol.shape == (n, p)
    assert np.array_equal(problem.C, np.eye(p, q))
    assert _is_feasible(problem.Xsol, atol=TOL)
    assert np.allclose(problem.B, problem.A @ problem.Xsol @ problem.C,
                       atol=1e-9)


def test_problem1_singular_values_lie_in_interval(restore_numpy_random_state):
    # A = U * Sigma * V^T with orthogonal Householder U, V, so the singular
    # values of A are exactly the diagonal of Sigma, which lie in [10, 12]
    # for every random outcome.
    np.random.seed(SEED)
    problem = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(sv >= 10.0 - TOL)
    assert np.all(sv <= 12.0 + TOL)


def test_problem3_singular_value_clusters(restore_numpy_random_state):
    # For n = 50 the clusters have sizes 15, 15, 12, 8 with values in
    # [10,11), [5,6), [2,3) and [0,0.001). These hold for every random draw.
    np.random.seed(SEED)
    problem = ProcrustesProblem((50, 50, 2, 2), problemnumber=3)
    sv = np.linalg.svd(problem.A, compute_uv=False)  # sorted descending
    assert np.all(sv[0:15] >= 10.0 - TOL) and np.all(sv[0:15] < 11.0 + TOL)
    assert np.all(sv[15:30] >= 5.0 - TOL) and np.all(sv[15:30] < 6.0 + TOL)
    assert np.all(sv[30:42] >= 2.0 - TOL) and np.all(sv[30:42] < 3.0 + TOL)
    assert np.all(sv[42:50] >= -TOL) and np.all(sv[42:50] < 0.001 + TOL)
    assert problem.Xsol.shape == (50, 2)
    assert _is_feasible(problem.Xsol, atol=TOL)


def test_problem2_builds_consistent_matrices(restore_numpy_random_state):
    np.random.seed(SEED)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=2)
    assert problem.A.shape == (6, 6)
    assert problem.B.shape == (6, 2)
    assert problem.Xsol.shape == (6, 2)
    assert np.all(np.isfinite(problem.A))
    assert np.allclose(problem.B, problem.A @ problem.Xsol @ problem.C,
                       atol=1e-9)


def test_same_seed_gives_same_problem(restore_numpy_random_state):
    np.random.seed(SEED)
    first = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    np.random.seed(SEED)
    second = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    assert np.allclose(first.A, second.A, rtol=0, atol=1e-12)
    assert np.allclose(first.B, second.B, rtol=0, atol=1e-12)
    assert np.array_equal(first.Xsol, second.Xsol)


def test_user_matrices_are_stored_and_stats_start_empty(
        restore_numpy_random_state):
    np.random.seed(SEED)
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C])
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert "Xsol" not in problem.__dict__
    assert problem.stats == {}


def test_wrong_matrix_shapes_are_rejected(restore_numpy_random_state):
    np.random.seed(SEED)
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    Xsol = np.zeros((4, 2))
    bad_cases = [
        [np.eye(3), B, C],
        [A, np.zeros((4, 3)), C],
        [A, B, np.eye(3)],
        [A, B, C, np.zeros((3, 2))],  # wrong Xsol shape
    ]
    for case in bad_cases:
        with pytest.raises(Exception):
            ProcrustesProblem((4, 4, 2, 2), matrices=case)
    # The valid four-matrix case must still be accepted.
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C, Xsol])
    assert problem.Xsol is Xsol


def test_matrices_must_be_list_or_tuple(restore_numpy_random_state):
    np.random.seed(SEED)
    with pytest.raises(Exception):
        ProcrustesProblem((4, 4, 2, 2), matrices=np.zeros((3, 4, 4)))


# ---------------------------------------------------------------------------
# GBBSolver
# ---------------------------------------------------------------------------

def test_default_options_are_applied(restore_numpy_random_state):
    np.random.seed(SEED)
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


def test_invalid_option_types_are_rejected(restore_numpy_random_state):
    np.random.seed(SEED)
    bad_options = [
        {"gtol": 1},               # int instead of float
        {"xtol": 1},
        {"ftol": 1},
        {"tau": 1},
        {"verbose": 2},            # GBB only allows 0 or 1
        {"projected_gradient": 3},
        {"maxiter": 5.0},
        {"filename": 3},
        {"timer": "yes"},
        {"etavar": 1},
    ]
    for kwargs in bad_options:
        with pytest.raises(Exception):
            GBBSolver(**kwargs)


def test_user_options_override_defaults(restore_numpy_random_state):
    np.random.seed(SEED)
    solver = GBBSolver(xtol=1e-4, maxiter=7, verbose=0)
    assert solver.options["xtol"] == 1e-4
    assert solver.options["maxiter"] == 7
    assert solver.options["verbose"] == 0
    # Options not given keep their defaults.
    assert solver.options["gtol"] == 1e-6


def test_solve_returns_feasible_solution(restore_numpy_random_state):
    np.random.seed(SEED)
    # (4,4,2,2) uses the Sherman-Morrison branch (invH), while (10,10,2,2)
    # uses the low-rank branch. Both projected-gradient choices are covered.
    configs = [((4, 4, 2, 2), 1), ((10, 10, 2, 2), 1), ((10, 10, 2, 2), 2)]
    for sizes, pg in configs:
        m, n, p, q = sizes
        problem = ProcrustesProblem(sizes, problemnumber=1)
        solver = GBBSolver(verbose=0, maxiter=20, projected_gradient=pg)
        result = solver.solve(problem)

        assert result.solution.shape == (n, p)
        # The Cayley-type update keeps X on the Stiefel manifold; the
        # tolerance covers only floating point error.
        assert _is_feasible(result.solution, atol=1e-6)
        assert result.status in (0, 1)
        assert result.success == (result.status == 0)
        assert isinstance(result.message, str)
        assert isinstance(result.nbiter, int)
        assert result.nfev >= 1


def test_error_matches_distance_to_known_solution(restore_numpy_random_state):
    np.random.seed(SEED)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    solver = GBBSolver(verbose=0, maxiter=20)
    result = solver.solve(problem)
    expected = np.linalg.norm(result.solution - problem.Xsol, np.inf)
    assert result.error == pytest.approx(expected, abs=1e-12)
    assert result.error >= 0.0


def test_error_is_inf_without_known_solution(restore_numpy_random_state):
    np.random.seed(SEED)
    base = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    problem = ProcrustesProblem((4, 4, 2, 2),
                                matrices=(base.A, base.B, base.C))
    result = GBBSolver(verbose=0, maxiter=20).solve(problem)
    assert result.error == np.inf


def test_total_fun_is_inf_when_full_results_disabled(
        restore_numpy_random_state):
    np.random.seed(SEED)
    problem = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=20).solve(problem)
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


@pytest.mark.xfail(
    strict=True,
    raises=Exception,
    reason="GBBSolver.solve checks for 'total_crit' in problem.stats, but "
           "gbb_solver stores 'total_grad', so full_results=True always raises.",
)
def test_full_results_request_currently_raises(restore_numpy_random_state):
    np.random.seed(SEED)
    problem = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    GBBSolver(verbose=0, maxiter=20, full_results=True).solve(problem)


def test_solve_is_reproducible_with_same_seed(restore_numpy_random_state):
    np.random.seed(SEED)
    base = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    matrices = (base.A, base.B, base.C)

    np.random.seed(SEED)
    first = GBBSolver(verbose=0, maxiter=20).solve(
        ProcrustesProblem((6, 6, 2, 2), matrices=matrices))
    np.random.seed(SEED)
    second = GBBSolver(verbose=0, maxiter=20).solve(
        ProcrustesProblem((6, 6, 2, 2), matrices=matrices))

    assert np.allclose(first.solution, second.solution, rtol=0, atol=1e-12)
    assert first.nbiter == second.nbiter
    assert first.status == second.status


def test_solve_delegates_to_gbb_solver(monkeypatch, restore_numpy_random_state):
    np.random.seed(SEED)
    base = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    problem = ProcrustesProblem((4, 4, 2, 2),
                                matrices=(base.A, base.B, base.C))
    calls = []
    fake_X = np.eye(4, 2)

    def fake_gbb_solver(*args):
        calls.append(args)
        args[0].stats.update(nbiter=3, fev=7, grad=2)
        return 0, 1.5, fake_X, 0.25, 3, "fake message"

    monkeypatch.setattr(skmod, "gbb_solver", fake_gbb_solver)

    solver = GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert len(calls) == 1
    args = calls[0]
    assert args[0] is problem
    assert args[1] == 4 and args[2] == 4
    assert args[4] is problem.A
    assert args[5] is problem.B
    assert args[6] is solver.options
    assert args[7] is False

    assert result.success is True
    assert result.status == 0
    assert result.fun == 1.5
    assert result.message == "fake message"
    assert result.nbiter == 3
    assert result.nfev == 7
    assert result.error == np.inf
    assert np.array_equal(result.solution, fake_X)


def test_solve_maps_failure_exitcode_to_unsuccessful_result(
        monkeypatch, restore_numpy_random_state):
    np.random.seed(SEED)
    base = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    problem = ProcrustesProblem((4, 4, 2, 2),
                                matrices=(base.A, base.B, base.C))

    def fake_gbb_solver(*args):
        args[0].stats.update(nbiter=5, fev=9, grad=4)
        return 1, 2.0, np.eye(4, 2), 0.5, 5, "maxiter reached"

    monkeypatch.setattr(skmod, "gbb_solver", fake_gbb_solver)

    result = GBBSolver(verbose=0).solve(problem)
    assert result.success is False
    assert result.status == 1
    assert result.message == "maxiter reached"
