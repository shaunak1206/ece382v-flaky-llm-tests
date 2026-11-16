import numpy as np
import pytest

import skprocrustes as skp
from skprocrustes.skprocrustes import GBBSolver, ProcrustesProblem


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(12345)


def _user_problem(m=6, p=2):
    A = np.random.randn(m, m)
    Xsol = np.linalg.qr(np.random.randn(m, p))[0]
    C = np.eye(p)
    B = A @ Xsol @ C
    return A, B, C, Xsol


# ---------------------------------------------------------------- Problem

def test_problem_rejects_nonsquare_a():
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((10, 8, 2, 2), problemnumber=1)


def test_problem_rejects_p_not_equal_q():
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_problem_rejects_m_not_divisible_by_p():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_problem_requires_problemnumber_or_matrices():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2))


def test_problem1_structure():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert problem.sizes == (10, 10, 2, 2)
    assert problem.problemnumber == 1
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    assert problem.stats == {}
    np.testing.assert_allclose(problem.C, np.eye(2))
    # B is built from the known solution
    np.testing.assert_allclose(problem.B,
                               problem.A @ problem.Xsol @ problem.C,
                               atol=1e-12)
    # Xsol has orthonormal columns
    np.testing.assert_allclose(problem.Xsol.T @ problem.Xsol, np.eye(2),
                               atol=1e-12)


def test_problem1_singular_values_in_range():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    s = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(s >= 10.0 - 1e-8)
    assert np.all(s <= 12.0 + 1e-8)


def test_problem2_shapes():
    problem = ProcrustesProblem((12, 12, 3, 3), problemnumber=2)
    assert problem.A.shape == (12, 12)
    assert problem.B.shape == (12, 3)
    assert problem.Xsol.shape == (12, 3)


def test_problem3_requires_special_n():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_problem3_n50():
    problem = ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
    assert problem.A.shape == (50, 50)
    s = np.linalg.svd(problem.A, compute_uv=False)
    # the largest group is in [10, 11], the smallest below 1e-3
    assert s.max() <= 11.0 + 1e-8
    assert s.min() <= 1e-3 + 1e-8


def test_problem_user_matrices_with_and_without_xsol():
    A, B, C, Xsol = _user_problem()
    p1 = ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C))
    assert p1.A is A and p1.B is B and p1.C is C
    assert not hasattr(p1, "Xsol")
    p2 = ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C, Xsol])
    assert p2.Xsol is Xsol


def test_problem_user_matrices_wrong_shapes():
    A, B, C, Xsol = _user_problem()
    with pytest.raises(Exception, match="A must"):
        ProcrustesProblem((6, 6, 2, 2), matrices=(A[:, :5], B, C))
    with pytest.raises(Exception, match="B must"):
        ProcrustesProblem((6, 6, 2, 2), matrices=(A, B[:, :1], C))
    with pytest.raises(Exception, match="C must"):
        ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, np.eye(3)))
    with pytest.raises(Exception, match="Xsol must"):
        ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C, Xsol[:, :1]))


def test_problem_user_matrices_too_few():
    A, B, C, _ = _user_problem()
    with pytest.raises(Exception):
        ProcrustesProblem((6, 6, 2, 2), matrices=(A, B))


# ----------------------------------------------------------------- Solver

def test_gbb_default_options():
    solver = GBBSolver()
    assert solver.solvername == "gbb"
    o = solver.options
    assert o["full_results"] is False
    assert o["xtol"] == 1e-6
    assert o["gtol"] == 1e-6
    assert o["ftol"] == 1e-12
    assert o["zeta"] == 1e-4
    assert o["kappa"] == 0.1
    assert o["eta"] == 0.85
    assert o["etavar"] is False
    assert o["tau"] == 1e-3
    assert o["projected_gradient"] == 1
    assert o["maxiter"] == 5000
    assert o["verbose"] == 1
    assert o["filename"] is None
    assert o["timer"] is False


def test_gbb_custom_options_are_kept():
    solver = GBBSolver(xtol=1e-4, maxiter=10, verbose=0,
                       projected_gradient=2, etavar=True)
    assert solver.options["xtol"] == 1e-4
    assert solver.options["maxiter"] == 10
    assert solver.options["verbose"] == 0
    assert solver.options["projected_gradient"] == 2
    assert solver.options["etavar"] is True
    # untouched options still get defaults
    assert solver.options["gtol"] == 1e-6


@pytest.mark.parametrize("kwargs", [
    {"full_results": 1},
    {"xtol": 1},
    {"gtol": "a"},
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


def test_gbb_solve_returns_feasible_solution():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    solver = GBBSolver(verbose=0)
    result = solver.solve(problem)
    for key in ("success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"):
        assert key in result
    X = result.solution
    assert X.shape == (10, 2)
    np.testing.assert_allclose(X.T @ X, np.eye(2), atol=1e-8)
    assert np.isfinite(result.fun)
    assert result.cpu >= 0
    assert result.nbiter == problem.stats["nbiter"]
    assert problem.stats["feasi"] < 1e-8
    assert problem.stats["fev"] >= 1
    assert result.success == (result.status == 0)
    # Xsol is known, so error is finite
    assert np.isfinite(result.error)
    # full_results is off
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


def test_gbb_solve_full_square_orthogonal_branch():
    # p == n: uses the direct (invH) Cayley update branch
    A, B, C, Xsol = _user_problem(m=4, p=4)
    problem = ProcrustesProblem((4, 4, 4, 4), matrices=(A, B, C))
    result = GBBSolver(verbose=0).solve(problem)
    X = result.solution
    np.testing.assert_allclose(X.T @ X, np.eye(4), atol=1e-8)
    # no known solution -> error is infinite
    assert result.error == np.inf


def test_gbb_projected_gradient_2():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, projected_gradient=2).solve(problem)
    X = result.solution
    np.testing.assert_allclose(X.T @ X, np.eye(2), atol=1e-8)


def test_gbb_maxiter_reached_reports_failure():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=1).solve(problem)
    assert result.status == 1
    assert result.success is False
    assert "Maximum number of iterations" in result.message
    assert problem.stats["nbiter"] == 1


def test_gbb_writes_to_file(tmp_path):
    fname = str(tmp_path / "gbb_out.txt")
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    solver = GBBSolver(verbose=1, filename=fname, maxiter=3)
    solver.solve(problem)
    with open(fname) as fh:
        content = fh.read()
    assert "Options" in content
    assert len(content) > 0
