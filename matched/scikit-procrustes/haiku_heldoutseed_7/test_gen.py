import os

import numpy as np
import pytest

import skprocrustes.skprocrustes as skp


def _good_matrices(m=4, n=4, p=2, q=2, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.rand(m, n)
    B = rng.rand(m, q)
    C = rng.rand(p, q)
    Xsol = rng.rand(n, p)
    return [A, B, C, Xsol]


def _default_problem(sizes=(6, 6, 2, 2), problemnumber=1, seed=1):
    np.random.seed(seed)
    return skp.ProcrustesProblem(sizes, problemnumber=problemnumber)


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [(10, 12, 2, 2), (10, 10, 2, 4)])
def test_non_square_problem_is_rejected(sizes):
    with pytest.raises(Exception, match="square"):
        skp.ProcrustesProblem(sizes, problemnumber=1)


def test_m_not_divisible_by_p_is_rejected():
    # m = 10 is not divisible by p = 3
    with pytest.raises(Exception, match="divisible"):
        skp.ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_default_problem_matrix_shapes():
    problem = _default_problem(sizes=(10, 10, 2, 2), problemnumber=1)
    assert problem.sizes == (10, 10, 2, 2)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)


def test_default_problem_satisfies_B_equals_AXC():
    problem = _default_problem(sizes=(8, 8, 2, 2), problemnumber=1)
    expected_B = problem.A @ problem.Xsol @ problem.C
    assert np.allclose(problem.B, expected_B)


def test_problem1_singular_values_lie_in_interval():
    # A = U * Sigma * V^T with orthogonal U, V, so singular values of A
    # equal the diagonal of Sigma, uniformly drawn in [10, 12].
    problem = _default_problem(sizes=(12, 12, 3, 3), problemnumber=1)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(sv >= 10.0 - 1e-8)
    assert np.all(sv <= 12.0 + 1e-8)


def test_problem3_singular_value_structure():
    # n = 50 groups: 15 values in [10,11), 15 in [5,6), 12 in [2,3),
    # and 8 values in [0, 1e-3).
    problem = _default_problem(sizes=(50, 50, 5, 5), problemnumber=3)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(sv[0:15] >= 10.0 - 1e-8) and np.all(sv[0:15] < 11.0)
    assert np.all(sv[15:30] >= 5.0 - 1e-8) and np.all(sv[15:30] < 6.0)
    assert np.all(sv[30:42] >= 2.0 - 1e-8) and np.all(sv[30:42] < 3.0)
    assert np.all(sv[42:] < 1e-3)


def test_problem3_rejects_unsupported_n():
    with pytest.raises(Exception, match="requires n"):
        skp.ProcrustesProblem((7, 7, 1, 1), problemnumber=3)


def test_problem2_builds_finite_matrices():
    problem = _default_problem(sizes=(10, 10, 2, 2), problemnumber=2)
    assert problem.A.shape == (10, 10)
    assert np.all(np.isfinite(problem.A))
    assert np.all(np.isfinite(problem.B))


@pytest.mark.parametrize("problemnumber", [None, 4])
def test_no_matrices_and_no_valid_problemnumber_is_rejected(problemnumber):
    with pytest.raises(Exception, match="Problem matrices are empty"):
        skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=problemnumber)


def test_default_xsol_has_orthonormal_columns():
    problem = _default_problem(sizes=(10, 10, 3, 3), problemnumber=1)
    assert np.allclose(problem.Xsol.T @ problem.Xsol, np.eye(3))


def test_seeded_construction_is_reproducible():
    np.random.seed(123)
    first = skp.ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    np.random.seed(123)
    second = skp.ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    assert np.allclose(first.A, second.A)
    assert np.allclose(first.B, second.B)
    assert np.allclose(first.Xsol, second.Xsol)


def test_user_supplied_matrices_are_stored_without_xsol():
    A, B, C, _ = _good_matrices()
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C])
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert not hasattr(problem, "Xsol")


def test_user_supplied_tuple_with_xsol_stores_all_four():
    A, B, C, Xsol = _good_matrices()
    problem = skp.ProcrustesProblem((4, 4, 2, 2),
                                    matrices=(A, B, C, Xsol))
    assert problem.Xsol is Xsol


def test_too_few_matrices_is_rejected():
    A, B, _, _ = _good_matrices()
    with pytest.raises(Exception, match="matrices must be a list"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=[A, B])


@pytest.mark.parametrize("name, index, bad, message", [
    ("A", 0, np.zeros((3, 4)), "A must be"),
    ("B", 1, np.zeros((4, 3)), "B must be"),
    ("C", 2, np.zeros((3, 2)), "C must be"),
    ("Xsol", 3, np.zeros((4, 3)), "Xsol must be"),
])
def test_wrong_matrix_shape_is_rejected(name, index, bad, message):
    matrices = _good_matrices()
    matrices[index] = bad
    with pytest.raises(Exception, match=message):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=matrices)


# ---------------------------------------------------------------------------
# GBBSolver
# ---------------------------------------------------------------------------

def test_gbb_default_options_and_solvername():
    solver = skp.GBBSolver()
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
    assert solver.options == expected
    assert solver.solvername == "gbb"


@pytest.mark.parametrize("key, value", [
    ("full_results", 1),
    ("xtol", 1),
    ("gtol", 1),
    ("ftol", 1),
    ("zeta", 1),
    ("kappa", 1),
    ("eta", 1),
    ("etavar", 1),
    ("tau", 1),
    ("projected_gradient", 3),
    ("maxiter", 1.0),
    ("verbose", 2),
    ("filename", 5),
    ("timer", "yes"),
])
def test_gbb_invalid_option_values_are_rejected(key, value):
    with pytest.raises(Exception):
        skp.GBBSolver(**{key: value})


def test_gbb_valid_non_default_options_are_kept():
    solver = skp.GBBSolver(xtol=1e-5, gtol=1e-4, ftol=1e-10, tau=1e-2,
                           projected_gradient=2, maxiter=10, verbose=0)
    assert solver.options["xtol"] == 1e-5
    assert solver.options["gtol"] == 1e-4
    assert solver.options["ftol"] == 1e-10
    assert solver.options["tau"] == 1e-2
    assert solver.options["projected_gradient"] == 2
    assert solver.options["maxiter"] == 10
    assert solver.options["verbose"] == 0


def test_gbb_solve_returns_feasible_solution_and_consistent_error():
    problem = _default_problem(sizes=(6, 6, 2, 2), problemnumber=1, seed=1)
    solver = skp.GBBSolver(maxiter=1000, verbose=0)
    result = solver.solve(problem)

    X = result.solution
    assert X.shape == (6, 2)
    assert np.allclose(X.T @ X, np.eye(2), atol=1e-6)
    assert problem.stats["feasi"] < 1e-6
    assert np.isfinite(result.fun)
    assert result.nbiter <= 1000
    # error is the infinity-norm distance to the known solution
    expected_error = np.linalg.norm(X - problem.Xsol, np.inf)
    assert result.error == pytest.approx(expected_error)


def test_gbb_solve_without_known_solution_reports_infinite_error():
    rng = np.random.RandomState(7)
    A = rng.randn(6, 6)
    B = rng.randn(6, 2)
    C = np.eye(2)
    problem = skp.ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C])
    solver = skp.GBBSolver(maxiter=20, verbose=0)
    result = solver.solve(problem)
    assert result.error == np.inf
    assert result.solution.shape == (6, 2)


def test_gbb_reports_failure_when_maxiter_is_reached():
    problem = _default_problem(sizes=(4, 4, 2, 2), problemnumber=1, seed=2)
    # Tolerances this tiny can never be met, and the first three
    # iterations never stop early, so maxiter is always reached.
    solver = skp.GBBSolver(maxiter=3, gtol=1e-300, xtol=1e-300,
                           ftol=1e-300, verbose=0)
    result = solver.solve(problem)
    assert result.status == 1
    assert result.success is False
    assert result.nbiter == 3
    assert result.message == skp._status_message["maxiter"]


def test_gbb_writes_log_to_file_and_closes_it(tmp_path):
    problem = _default_problem(sizes=(4, 4, 2, 2), problemnumber=1, seed=3)
    log = tmp_path / "gbb.log"
    solver = skp.GBBSolver(filename=str(log), maxiter=3, verbose=1)
    solver.solve(problem)
    assert log.exists()
    content = log.read_text()
    assert "Options" in content
    assert solver.file.closed


def test_gbb_verbose_output_goes_to_stdout(capsys):
    problem = _default_problem(sizes=(4, 4, 2, 2), problemnumber=1, seed=4)
    solver = skp.GBBSolver(maxiter=2, verbose=1)
    solver.solve(problem)
    captured = capsys.readouterr()
    assert "GBB Solver" in captured.out


def test_gbb_full_results_raises_with_current_stats_keys():
    # gbb_solver records "total_grad" in problem.stats, but
    # GBBSolver.solve checks for "total_crit", so full_results=True
    # currently raises. This test documents that behavior.
    problem = _default_problem(sizes=(4, 4, 2, 2), problemnumber=1, seed=5)
    solver = skp.GBBSolver(full_results=True, maxiter=2, verbose=0)
    with pytest.raises(Exception, match="For full results"):
        solver.solve(problem)
