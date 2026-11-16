import numpy as np
import pytest
from scipy import linalg as sp

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_problem_rejects_non_square_sizes():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 8, 2, 2), problemnumber=1)
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_problem_rejects_m_not_divisible_by_p():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_problem_requires_problemnumber_or_matrices():
    with pytest.raises(Exception):
        ProcrustesProblem((6, 6, 2, 2))


def test_problem_invalid_problemnumber_raises():
    with pytest.raises(Exception):
        ProcrustesProblem((6, 6, 2, 2), problemnumber=7)


def test_problem_number_1_shapes_and_consistency():
    np.random.seed(0)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert prob.sizes == (10, 10, 2, 2)
    assert prob.A.shape == (10, 10)
    assert prob.B.shape == (10, 2)
    assert prob.C.shape == (2, 2)
    assert prob.Xsol.shape == (10, 2)
    # B is built from the known solution
    np.testing.assert_allclose(prob.B, prob.A @ prob.Xsol @ prob.C,
                               atol=1e-10)
    assert prob.stats == {}


def test_problem_number_1_xsol_orthonormal_and_C_identity():
    np.random.seed(1)
    prob = ProcrustesProblem((12, 12, 3, 3), problemnumber=1)
    np.testing.assert_allclose(prob.Xsol.T @ prob.Xsol, np.eye(3),
                               atol=1e-12)
    np.testing.assert_allclose(prob.C, np.eye(3))


def test_problem_number_1_singular_values_in_range():
    np.random.seed(2)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    s = sp.svdvals(prob.A)
    assert np.all(s >= 10.0 - 1e-8)
    assert np.all(s <= 12.0 + 1e-8)


def test_problem_number_2_shapes():
    np.random.seed(3)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=2)
    assert prob.A.shape == (10, 10)
    assert prob.B.shape == (10, 2)
    np.testing.assert_allclose(prob.B, prob.A @ prob.Xsol @ prob.C,
                               atol=1e-8)


def test_problem_number_3_requires_specific_n():
    with pytest.raises(Exception):
        ProcrustesProblem((20, 20, 2, 2), problemnumber=3)
    np.random.seed(4)
    prob = ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
    assert prob.A.shape == (50, 50)
    assert prob.Xsol.shape == (50, 5)


def test_problem_user_matrices_without_xsol():
    A = np.random.rand(6, 6)
    B = np.random.rand(6, 2)
    C = np.eye(2)
    prob = ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C))
    assert prob.A is A
    assert prob.B is B
    assert prob.C is C
    assert not hasattr(prob, "Xsol")


def test_problem_user_matrices_with_xsol_as_list():
    A = np.random.rand(6, 6)
    X = np.eye(6, 2)
    C = np.eye(2)
    B = A @ X @ C
    prob = ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C, X])
    assert prob.Xsol is X


def test_problem_user_matrices_wrong_shapes_raise():
    A = np.random.rand(6, 6)
    B = np.random.rand(6, 2)
    C = np.eye(2)
    X = np.eye(6, 2)
    with pytest.raises(Exception):
        ProcrustesProblem((6, 6, 2, 2), matrices=(np.random.rand(5, 6), B, C))
    with pytest.raises(Exception):
        ProcrustesProblem((6, 6, 2, 2), matrices=(A, np.random.rand(6, 3), C))
    with pytest.raises(Exception):
        ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, np.eye(3)))
    with pytest.raises(Exception):
        ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C, np.eye(6, 3)))
    # sanity: the correct one works
    ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C, X))


def test_problem_too_few_matrices_raises():
    A = np.random.rand(6, 6)
    B = np.random.rand(6, 2)
    with pytest.raises(Exception):
        ProcrustesProblem((6, 6, 2, 2), matrices=(A, B))


# ---------------------------------------------------------------------------
# GBBSolver: options
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
    solver = GBBSolver(xtol=1e-4, gtol=1e-5, maxiter=10, verbose=0,
                       projected_gradient=2, etavar=True)
    assert solver.options["xtol"] == 1e-4
    assert solver.options["gtol"] == 1e-5
    assert solver.options["maxiter"] == 10
    assert solver.options["verbose"] == 0
    assert solver.options["projected_gradient"] == 2
    assert solver.options["etavar"] is True


@pytest.mark.parametrize("kwargs", [
    {"xtol": 1},
    {"gtol": "small"},
    {"ftol": 1},
    {"zeta": 1},
    {"kappa": 1},
    {"eta": 1},
    {"tau": 1},
    {"etavar": 1},
    {"full_results": "yes"},
    {"projected_gradient": 3},
    {"maxiter": 10.0},
    {"verbose": 2},
    {"filename": 5},
    {"timer": 1},
])
def test_gbb_invalid_options_raise(kwargs):
    with pytest.raises(Exception):
        GBBSolver(**kwargs)


# ---------------------------------------------------------------------------
# GBBSolver: solving
# ---------------------------------------------------------------------------

def test_gbb_solve_returns_valid_result():
    np.random.seed(10)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    solver = GBBSolver(verbose=0)
    result = solver.solve(prob)
    for key in ("success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"):
        assert key in result
    assert result.solution.shape == (10, 2)
    np.testing.assert_allclose(result.solution.T @ result.solution,
                               np.eye(2), atol=1e-6)
    assert np.isfinite(result.fun)
    assert result.fun >= 0
    assert result.cpu >= 0
    assert result.nbiter == prob.stats["nbiter"]
    assert result.nfev == prob.stats["fev"]
    assert prob.stats["feasi"] < 1e-6
    assert np.isfinite(result.error)
    # without full_results, histories are placeholders
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf
    assert result.status in (0, 1)
    assert result.success == (result.status == 0)


def test_gbb_maxiter_one_reports_failure():
    np.random.seed(11)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=1).solve(prob)
    assert result.status == 1
    assert result.success is False
    assert result.nbiter == 1
    assert "Maximum number of iterations" in result.message


def test_gbb_small_problem_uses_inverse_branch():
    # p >= smalldim/2 so the Cayley update is done with a direct solve
    np.random.seed(12)
    prob = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=20).solve(prob)
    assert result.solution.shape == (4, 2)
    np.testing.assert_allclose(result.solution.T @ result.solution,
                               np.eye(2), atol=1e-6)
    assert np.isfinite(result.fun)


def test_gbb_projected_gradient_option_2():
    np.random.seed(13)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, projected_gradient=2,
                       maxiter=30).solve(prob)
    np.testing.assert_allclose(result.solution.T @ result.solution,
                               np.eye(2), atol=1e-6)
    assert np.isfinite(result.fun)


def test_gbb_without_known_solution_error_is_inf():
    np.random.seed(14)
    A = np.random.rand(6, 6) + 6 * np.eye(6)
    Q, _ = np.linalg.qr(np.random.rand(6, 2))
    C = np.eye(2)
    B = A @ Q @ C
    prob = ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C))
    result = GBBSolver(verbose=0, maxiter=5).solve(prob)
    assert result.error == np.inf
    assert result.solution.shape == (6, 2)


def test_gbb_writes_to_filename_when_given(tmp_path):
    np.random.seed(15)
    fname = str(tmp_path / "gbb_out.txt")
    prob = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    solver = GBBSolver(verbose=0, maxiter=3, filename=fname)
    solver.solve(prob)
    assert (tmp_path / "gbb_out.txt").exists()
    assert solver.file.closed
