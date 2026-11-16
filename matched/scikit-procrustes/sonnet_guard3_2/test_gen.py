import numpy as np
import pytest

import skprocrustes.skprocrustes as skp_mod
from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver


def _orth_err(X):
    p = X.shape[1]
    return np.linalg.norm(X.T.dot(X) - np.eye(p), "fro")


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_problem_rejects_non_square_a():
    np.random.seed(0)
    with pytest.raises(Exception):
        ProcrustesProblem((6, 4, 2, 2), problemnumber=1)


def test_problem_rejects_p_not_equal_q():
    np.random.seed(0)
    with pytest.raises(Exception):
        ProcrustesProblem((6, 6, 2, 3), problemnumber=1)


def test_problem_rejects_m_not_divisible_by_p():
    np.random.seed(0)
    with pytest.raises(Exception):
        ProcrustesProblem((7, 7, 2, 2), problemnumber=1)


def test_problem1_structure_and_singular_values():
    np.random.seed(1)
    problem = ProcrustesProblem((6, 6, 3, 3), problemnumber=1)
    assert problem.sizes == (6, 6, 3, 3)
    assert problem.problemnumber == 1
    assert problem.A.shape == (6, 6)
    assert problem.B.shape == (6, 3)
    assert problem.C.shape == (3, 3)
    assert problem.Xsol.shape == (6, 3)
    # Xsol has orthonormal columns, C is the identity
    assert _orth_err(problem.Xsol) < 1e-12
    assert np.allclose(problem.C, np.eye(3))
    # B = A Xsol C holds exactly
    assert np.allclose(problem.B, problem.A.dot(problem.Xsol).dot(problem.C))
    # Singular values of A are in [10, 12] for every random outcome
    s = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(s >= 10.0 - 1e-8)
    assert np.all(s <= 12.0 + 1e-8)


def test_problem2_shapes_and_consistency():
    np.random.seed(2)
    problem = ProcrustesProblem((8, 8, 2, 2), problemnumber=2)
    assert problem.A.shape == (8, 8)
    assert problem.B.shape == (8, 2)
    assert problem.Xsol.shape == (8, 2)
    assert np.allclose(problem.B, problem.A.dot(problem.Xsol).dot(problem.C))
    assert _orth_err(problem.Xsol) < 1e-12


def test_problem3_requires_specific_n():
    np.random.seed(3)
    with pytest.raises(Exception):
        ProcrustesProblem((20, 20, 2, 2), problemnumber=3)
    problem = ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
    assert problem.A.shape == (50, 50)
    assert problem.B.shape == (50, 5)
    assert np.allclose(problem.B, problem.A.dot(problem.Xsol).dot(problem.C))


def test_problem_without_matrices_or_number_raises():
    np.random.seed(4)
    with pytest.raises(Exception):
        ProcrustesProblem((4, 4, 2, 2))
    with pytest.raises(Exception):
        ProcrustesProblem((4, 4, 2, 2), problemnumber=7)


def test_problem_user_matrices_stored():
    np.random.seed(5)
    A = np.eye(4)
    B = np.ones((4, 2))
    C = np.eye(2)
    X = np.zeros((4, 2))
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, X))
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert problem.Xsol is X
    assert problem.stats == {}


def test_problem_user_matrices_without_xsol():
    np.random.seed(6)
    problem = ProcrustesProblem((4, 4, 2, 2),
                                matrices=[np.eye(4), np.ones((4, 2)),
                                          np.eye(2)])
    assert not hasattr(problem, "Xsol")
    assert problem.A.shape == (4, 4)


def test_problem_user_matrices_wrong_shapes_raise():
    np.random.seed(7)
    A, B, C, X = np.eye(4), np.ones((4, 2)), np.eye(2), np.zeros((4, 2))
    sizes = (4, 4, 2, 2)
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(np.eye(3), B, C))
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(A, np.ones((4, 3)), C))
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(A, B, np.eye(3)))
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(A, B, C, np.zeros((2, 4))))
    # too few matrices
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(A, B))


# ---------------------------------------------------------------------------
# GBBSolver: options
# ---------------------------------------------------------------------------

def test_gbb_default_options():
    np.random.seed(10)
    solver = GBBSolver()
    o = solver.options
    assert solver.solvername == "gbb"
    assert o["full_results"] is False
    assert o["xtol"] == pytest.approx(1e-6)
    assert o["gtol"] == pytest.approx(1e-6)
    assert o["ftol"] == pytest.approx(1e-12)
    assert o["zeta"] == pytest.approx(1e-4)
    assert o["kappa"] == pytest.approx(0.1)
    assert o["eta"] == pytest.approx(0.85)
    assert o["etavar"] is False
    assert o["tau"] == pytest.approx(1e-3)
    assert o["projected_gradient"] == 1
    assert o["maxiter"] == 5000
    assert o["verbose"] == 1
    assert o["filename"] is None
    assert o["timer"] is False


def test_gbb_custom_options_are_kept():
    np.random.seed(11)
    solver = GBBSolver(xtol=1e-4, gtol=1e-5, maxiter=7, verbose=0,
                       projected_gradient=2, etavar=True, kappa=0.5)
    o = solver.options
    assert o["xtol"] == pytest.approx(1e-4)
    assert o["gtol"] == pytest.approx(1e-5)
    assert o["maxiter"] == 7
    assert o["verbose"] == 0
    assert o["projected_gradient"] == 2
    assert o["etavar"] is True
    assert o["kappa"] == pytest.approx(0.5)
    # untouched options still get defaults
    assert o["tau"] == pytest.approx(1e-3)


@pytest.mark.parametrize("key,value", [
    ("xtol", 1), ("gtol", 1), ("ftol", 1), ("zeta", 1), ("kappa", 1),
    ("eta", 1), ("tau", 1), ("etavar", 1), ("full_results", 1),
    ("projected_gradient", 3), ("maxiter", 1.5), ("verbose", 2),
    ("filename", 3), ("timer", "yes"),
])
def test_gbb_invalid_options_raise(key, value):
    np.random.seed(12)
    with pytest.raises(Exception):
        GBBSolver(**{key: value})


# ---------------------------------------------------------------------------
# GBBSolver: solving
# ---------------------------------------------------------------------------

def test_gbb_maxiter_one_reports_failure():
    np.random.seed(20)
    problem = ProcrustesProblem((6, 6, 3, 3), problemnumber=1)
    result = GBBSolver(maxiter=1, verbose=0).solve(problem)
    assert result.success is False
    assert result.status == 1
    assert result.message == skp_mod._status_message["maxiter"]
    assert result.nbiter == 1
    assert result.solution.shape == (6, 3)
    assert result["cpu"] >= 0.0
    # no full results requested
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


def test_gbb_converges_on_easy_problem():
    np.random.seed(21)
    Xsol = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    A = np.eye(4)
    C = np.eye(2)
    B = A.dot(Xsol).dot(C)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, Xsol))
    result = GBBSolver(verbose=0).solve(problem)
    X = result.solution
    assert X.shape == (4, 2)
    assert _orth_err(X) < 1e-8
    assert result.success is True
    assert result.status == 0
    assert result.nbiter == problem.stats["nbiter"]
    assert result.nfev == problem.stats["fev"]
    residual = np.linalg.norm(A.dot(X).dot(C) - B, "fro") ** 2
    assert residual < 1e-3
    assert np.isfinite(result.error)
    assert result.error < 1e-2


@pytest.mark.parametrize("projected_gradient", [1, 2])
@pytest.mark.parametrize("sizes", [(6, 6, 3, 3), (8, 8, 2, 2)])
def test_gbb_solution_stays_orthonormal(sizes, projected_gradient):
    # (8,8,2,2) exercises the low-rank update branch (p < n/2),
    # (6,6,3,3) exercises the direct inverse branch.
    np.random.seed(22)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    solver = GBBSolver(maxiter=4, verbose=0,
                       projected_gradient=projected_gradient)
    result = solver.solve(problem)
    assert result.solution.shape == (sizes[1], sizes[2])
    assert _orth_err(result.solution) < 1e-8
    assert problem.stats["feasi"] < 1e-8
    assert result.nbiter == problem.stats["nbiter"]
    assert result.nbiter <= 4


def test_gbb_same_seed_gives_same_solution():
    np.random.seed(23)
    problem = ProcrustesProblem((6, 6, 3, 3), problemnumber=1)
    solver = GBBSolver(maxiter=10, verbose=0)

    np.random.seed(99)
    r1 = solver.solve(problem)
    sol1 = r1.solution.copy()
    np.random.seed(99)
    r2 = solver.solve(problem)

    assert np.allclose(sol1, r2.solution)
    assert r1.nbiter == r2.nbiter


def test_gbb_full_results_records_function_values():
    np.random.seed(24)
    problem = ProcrustesProblem((6, 6, 3, 3), problemnumber=1)
    # solve() looks for "total_crit" in stats, which gbb_solver does not set
    problem.stats["total_crit"] = []
    result = GBBSolver(full_results=True, maxiter=5, verbose=0).solve(problem)
    assert isinstance(result.total_fun, list)
    assert len(result.total_fun) == result.nfev
    assert len(result.total_fun) >= 2
    assert all(np.isfinite(v) and v >= 0.0 for v in result.total_fun)
    assert result.total_crit == []


def test_gbb_error_is_inf_without_known_solution():
    np.random.seed(25)
    A = np.diag([3.0, 2.0, 1.5, 1.0])
    B = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    C = np.eye(2)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))
    result = GBBSolver(maxiter=5, verbose=0).solve(problem)
    assert result.error == np.inf
    assert _orth_err(result.solution) < 1e-8


def test_gbb_error_matches_inf_norm_with_known_solution():
    np.random.seed(26)
    problem = ProcrustesProblem((6, 6, 3, 3), problemnumber=1)
    result = GBBSolver(maxiter=3, verbose=0).solve(problem)
    expected = np.linalg.norm(result.solution - problem.Xsol, np.inf)
    assert result.error == pytest.approx(expected)


def test_gbb_writes_log_to_file(tmp_path):
    np.random.seed(27)
    problem = ProcrustesProblem((6, 6, 3, 3), problemnumber=1)
    logfile = tmp_path / "gbb_log.txt"
    solver = GBBSolver(filename=str(logfile), maxiter=2, verbose=1)
    result = solver.solve(problem)
    assert logfile.exists()
    content = logfile.read_text()
    assert "Options" in content
    assert "nbiter" in content
    assert result.solution.shape == (6, 3)


def test_gbb_invalid_filename_exits(tmp_path):
    np.random.seed(28)
    problem = ProcrustesProblem((6, 6, 3, 3), problemnumber=1)
    bad = tmp_path / "missing_dir" / "log.txt"
    solver = GBBSolver(filename=str(bad), verbose=0)
    with pytest.raises(SystemExit):
        solver.solve(problem)
