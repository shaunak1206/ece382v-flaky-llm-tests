import os

import numpy as np
import pytest

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(12345)


def _identity_problem(n=4, p=2):
    """A = I, C = I, so the known solution is unique and easy to find."""
    A = np.eye(n)
    C = np.eye(p)
    Xsol = np.zeros((n, p))
    for j in range(p):
        Xsol[j + 1, j] = 1.0
    B = A.dot(Xsol).dot(C)
    return ProcrustesProblem((n, n, p, p), matrices=(A, B, C, Xsol)), Xsol


# --------------------------------------------------------------------------
# ProcrustesProblem
# --------------------------------------------------------------------------

def test_problem_default_1_shapes_and_relations():
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert prob.sizes == (10, 10, 2, 2)
    assert prob.A.shape == (10, 10)
    assert prob.B.shape == (10, 2)
    assert prob.C.shape == (2, 2)
    assert prob.Xsol.shape == (10, 2)
    np.testing.assert_allclose(prob.C, np.eye(2))
    np.testing.assert_allclose(prob.Xsol.T.dot(prob.Xsol), np.eye(2),
                               atol=1e-12)
    np.testing.assert_allclose(prob.B,
                               prob.A.dot(prob.Xsol).dot(prob.C),
                               atol=1e-10)


def test_problem_1_singular_values_in_range():
    prob = ProcrustesProblem((12, 12, 3, 3), problemnumber=1)
    s = np.linalg.svd(prob.A, compute_uv=False)
    assert np.all(s >= 10.0 - 1e-8)
    assert np.all(s <= 12.0 + 1e-8)


def test_problem_default_2_builds():
    prob = ProcrustesProblem((6, 6, 2, 2), problemnumber=2)
    assert prob.A.shape == (6, 6)
    assert prob.B.shape == (6, 2)
    np.testing.assert_allclose(prob.B, prob.A.dot(prob.Xsol).dot(prob.C),
                               atol=1e-8)


def test_problem_stats_initially_empty_dict():
    prob = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    assert prob.stats == {}
    assert prob.problemnumber == 1


def test_problem_non_square_raises():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 8, 2, 2), problemnumber=1)
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_problem_m_not_divisible_by_p_raises():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_problem_without_matrices_or_number_raises():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2))


def test_problem_number_3_requires_specific_n():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_problem_number_3_n50():
    prob = ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
    assert prob.A.shape == (50, 50)
    assert prob.Xsol.shape == (50, 5)
    np.testing.assert_allclose(prob.B, prob.A.dot(prob.Xsol).dot(prob.C),
                               atol=1e-8)


def test_problem_user_matrices_with_and_without_xsol():
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 2)
    C = np.eye(2)
    X = np.random.rand(4, 2)
    prob = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C, X])
    assert prob.A is A and prob.B is B and prob.C is C and prob.Xsol is X

    prob3 = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))
    assert not hasattr(prob3, "Xsol")


def test_problem_user_matrices_wrong_shapes_raise():
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 2)
    C = np.eye(2)
    X = np.random.rand(4, 2)
    with pytest.raises(Exception, match="A must"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(np.zeros((3, 4)), B, C))
    with pytest.raises(Exception, match="B must"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, np.zeros((4, 3)), C))
    with pytest.raises(Exception, match="C must"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, np.zeros((3, 2))))
    with pytest.raises(Exception, match="Xsol must"):
        ProcrustesProblem((4, 4, 2, 2),
                          matrices=(A, B, C, np.zeros((2, 4))))


def test_problem_too_few_matrices_raises():
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 2)
    with pytest.raises(Exception):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B))


# --------------------------------------------------------------------------
# GBBSolver options
# --------------------------------------------------------------------------

def test_gbb_default_options():
    s = GBBSolver()
    assert s.solvername == "gbb"
    o = s.options
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


def test_gbb_custom_options_kept():
    s = GBBSolver(xtol=1e-4, maxiter=10, verbose=0, projected_gradient=2,
                  etavar=True)
    assert s.options["xtol"] == 1e-4
    assert s.options["maxiter"] == 10
    assert s.options["verbose"] == 0
    assert s.options["projected_gradient"] == 2
    assert s.options["etavar"] is True


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


# --------------------------------------------------------------------------
# GBBSolver.solve
# --------------------------------------------------------------------------

def test_gbb_solve_identity_problem_invH_branch():
    prob, Xsol = _identity_problem(n=4, p=2)
    result = GBBSolver(verbose=0).solve(prob)
    X = result.solution
    assert X.shape == (4, 2)
    np.testing.assert_allclose(X.T.dot(X), np.eye(2), atol=1e-6)
    assert result.error < 1e-2
    np.testing.assert_allclose(X, Xsol, atol=1e-2)


def test_gbb_solve_identity_problem_lowrank_branch():
    prob, Xsol = _identity_problem(n=8, p=2)
    result = GBBSolver(verbose=0).solve(prob)
    X = result.solution
    np.testing.assert_allclose(X.T.dot(X), np.eye(2), atol=1e-6)
    np.testing.assert_allclose(X, Xsol, atol=1e-2)


def test_gbb_solve_projected_gradient_2_feasible():
    prob, Xsol = _identity_problem(n=8, p=2)
    result = GBBSolver(verbose=0, projected_gradient=2).solve(prob)
    X = result.solution
    np.testing.assert_allclose(X.T.dot(X), np.eye(2), atol=1e-6)


def test_gbb_solve_result_fields_and_stats():
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0).solve(prob)
    for key in ("success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"):
        assert key in result
    assert result.solution.shape == (10, 2)
    assert result.success == (result.status == 0)
    assert result.cpu >= 0
    assert result.nbiter == prob.stats["nbiter"]
    assert result.nfev == prob.stats["fev"]
    assert prob.stats["fev"] >= 1
    assert np.isfinite(result.error)
    assert np.isfinite(result.fun)
    # without full_results these are placeholders
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf
    # feasibility statistic recorded by the solver
    assert prob.stats["feasi"] < 1e-6


def test_gbb_solve_without_known_solution_error_is_inf():
    prob_full, _ = _identity_problem(n=4, p=2)
    prob = ProcrustesProblem((4, 4, 2, 2),
                             matrices=(prob_full.A, prob_full.B, prob_full.C))
    result = GBBSolver(verbose=0).solve(prob)
    assert result.error == np.inf


def test_gbb_solve_maxiter_reached():
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=1).solve(prob)
    assert result.success is False
    assert result.status == 1
    assert "Maximum number of iterations" in result.message
    assert result.nbiter == 1


def test_gbb_solve_writes_to_file(tmp_path):
    fname = str(tmp_path / "gbb_out.txt")
    prob, _ = _identity_problem(n=4, p=2)
    solver = GBBSolver(verbose=1, filename=fname)
    result = solver.solve(prob)
    assert os.path.exists(fname)
    with open(fname) as fh:
        content = fh.read()
    assert "Options" in content
    assert result.solution.shape == (4, 2)
    assert solver.file.closed
