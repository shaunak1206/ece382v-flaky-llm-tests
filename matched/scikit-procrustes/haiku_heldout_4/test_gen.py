import numpy as np
import pytest

from skprocrustes.skprocrustes import (
    ProcrustesProblem,
    GBBSolver,
    OptimizeResult,
    _status_message,
)


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_problem_stores_sizes_problemnumber_and_empty_stats():
    np.random.seed(0)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert problem.sizes == (10, 10, 2, 2)
    assert problem.problemnumber == 1
    assert problem.stats == {}


def test_nonsquare_sizes_are_rejected():
    bad_sizes = [(10, 12, 2, 2), (10, 10, 2, 3), (10, 10, 3, 2)]
    for sizes in bad_sizes:
        with pytest.raises(Exception, match="only square"):
            ProcrustesProblem(sizes, problemnumber=1)


def test_m_not_divisible_by_p_is_rejected():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((9, 9, 2, 2), problemnumber=1)


def test_default_problem_1_shapes_and_C_identity():
    np.random.seed(1)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    assert np.allclose(problem.C, np.eye(2))


def test_default_problem_1_singular_values_in_expected_range():
    np.random.seed(2)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(sv >= 10.0 - 1e-8)
    assert np.all(sv <= 12.0 + 1e-8)


def test_default_problem_xsol_orthonormal_and_B_consistent():
    np.random.seed(3)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert np.allclose(problem.Xsol.T @ problem.Xsol, np.eye(2))
    assert np.allclose(problem.B, problem.A @ problem.Xsol @ problem.C)


def test_default_problem_is_reproducible_with_same_seed():
    np.random.seed(42)
    p1 = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    np.random.seed(42)
    p2 = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert np.allclose(p1.A, p2.A)
    assert np.allclose(p1.B, p2.B)
    assert np.allclose(p1.Xsol, p2.Xsol)


def test_problem_3_n50_singular_value_structure():
    np.random.seed(4)
    problem = ProcrustesProblem((50, 50, 2, 2), problemnumber=3)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    # Group 1 (15 values in [10, 11]), group 2 (15 values in [5, 6]),
    # group 3 (12 values in [2, 3]), group 4 (8 values below 1e-3).
    assert np.sum(sv >= 9.99) == 15
    assert np.sum((sv > 4.9) & (sv < 6.1)) == 15
    assert np.sum((sv > 1.9) & (sv < 3.1)) == 12
    assert np.sum(sv < 0.01) == 8


def test_problem_3_unsupported_n_raises():
    with pytest.raises(Exception, match="requires n"):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_no_problemnumber_and_no_matrices_raises():
    with pytest.raises(Exception, match="problem number"):
        ProcrustesProblem((10, 10, 2, 2))


def test_user_matrices_are_stored_and_no_xsol_when_three_given():
    A = np.random.rand(6, 6)
    B = np.random.rand(6, 2)
    C = np.eye(2)
    problem = ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C])
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert "Xsol" not in problem.__dict__


def test_user_matrices_with_xsol_are_stored():
    A = np.random.rand(6, 6)
    B = np.random.rand(6, 2)
    C = np.eye(2)
    Xsol = np.eye(6, 2)
    problem = ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C, Xsol))
    assert problem.Xsol is Xsol


def test_wrong_matrix_shapes_raise():
    good_A = np.zeros((6, 6))
    good_B = np.zeros((6, 2))
    good_C = np.eye(2)
    good_X = np.zeros((6, 2))
    cases = [
        ([np.zeros((5, 6)), good_B, good_C], "A must be"),
        ([good_A, np.zeros((6, 3)), good_C], "B must be"),
        ([good_A, good_B, np.zeros((3, 2))], "C must be"),
        ([good_A, good_B, good_C, np.zeros((4, 2))], "Xsol must be"),
    ]
    for matrices, message in cases:
        with pytest.raises(Exception, match=message):
            ProcrustesProblem((6, 6, 2, 2), matrices=matrices)


def test_too_few_or_wrong_container_matrices_raise():
    A = np.zeros((6, 6))
    B = np.zeros((6, 2))
    with pytest.raises(Exception, match="must be a list or tuple"):
        ProcrustesProblem((6, 6, 2, 2), matrices=[A, B])
    with pytest.raises(Exception, match="must be a list or tuple"):
        ProcrustesProblem((6, 6, 2, 2), matrices={"A": A, "B": B, "C": np.eye(2)})


# ---------------------------------------------------------------------------
# GBBSolver: options
# ---------------------------------------------------------------------------

def test_gbb_default_options_and_solvername():
    solver = GBBSolver()
    assert solver.solvername == "gbb"
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


def test_gbb_option_overrides_are_stored():
    solver = GBBSolver(tau=0.01, maxiter=42, verbose=0, projected_gradient=2)
    assert solver.options["tau"] == 0.01
    assert solver.options["maxiter"] == 42
    assert solver.options["verbose"] == 0
    assert solver.options["projected_gradient"] == 2


def test_gbb_invalid_option_values_raise():
    invalid = [
        ({"full_results": "yes"}, "full_results must be a boolean"),
        ({"xtol": 1}, "xtol must be a float"),
        ({"gtol": 1}, "gtol must be a float"),
        ({"ftol": 1}, "ftol must be a float"),
        ({"zeta": 1}, "zeta must be a float"),
        ({"kappa": 1}, "kappa must be a float"),
        ({"eta": 1}, "eta must be a float"),
        ({"etavar": 1}, "etavar must be a boolean"),
        ({"tau": 1}, "tau must be a float"),
        ({"maxiter": 1.5}, "maxiter must be an integer"),
        ({"verbose": 2}, "verbose must be 0"),
        ({"projected_gradient": 3}, "projected_gradient must be 1 or 2"),
        ({"filename": 12}, "filename must be string"),
        ({"timer": "no"}, "timer must be boolean"),
    ]
    for kwargs, message in invalid:
        with pytest.raises(Exception, match=message):
            GBBSolver(**kwargs)


# ---------------------------------------------------------------------------
# GBBSolver: solve
# ---------------------------------------------------------------------------

def test_solve_returns_optimize_result_with_expected_fields():
    np.random.seed(5)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    solver = GBBSolver(verbose=0, maxiter=200)
    result = solver.solve(problem)
    assert isinstance(result, OptimizeResult)
    for key in ["success", "status", "message", "solution", "fun",
                "normcrit", "error", "cpu", "nbiter", "nfev",
                "total_fun", "total_crit"]:
        assert key in result, key
    assert result.solution.shape == (10, 2)
    assert result.status in (0, 1)
    assert result.success == (result.status == 0)
    assert result.fun >= 0
    assert result.nbiter <= 200
    assert problem.stats["nbiter"] == result.nbiter


def test_solve_solution_is_orthonormal():
    np.random.seed(6)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=200).solve(problem)
    X = result.solution
    assert np.allclose(X.T @ X, np.eye(2), atol=1e-6)


def test_solve_error_is_norm_to_known_solution():
    np.random.seed(7)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=200).solve(problem)
    expected = np.linalg.norm(result.solution - problem.Xsol, np.inf)
    assert np.isfinite(result.error)
    assert np.allclose(result.error, expected)


def test_solve_with_maxiter_one_reports_failure():
    # With maxiter=1 the loop runs once, nbiter becomes 1 and the
    # maxiter status is reported deterministically.
    np.random.seed(8)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=1).solve(problem)
    assert result.status == 1
    assert result.success is False
    assert result.message == _status_message["maxiter"]
    assert result.nbiter == 1


def test_solve_user_defined_problem_without_xsol_has_infinite_error():
    np.random.seed(9)
    A = np.random.rand(6, 6)
    B = np.random.rand(6, 2)
    C = np.eye(2)
    problem = ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C])
    result = GBBSolver(verbose=0, maxiter=200).solve(problem)
    assert result.error == np.inf
    assert result.solution.shape == (6, 2)
    assert np.allclose(result.solution.T @ result.solution, np.eye(2),
                       atol=1e-6)


def test_solve_writes_log_to_filename(tmp_path):
    np.random.seed(10)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    path = tmp_path / "gbb_log.txt"
    solver = GBBSolver(verbose=1, maxiter=5, filename=str(path))
    solver.solve(problem)
    assert path.exists()
    assert "Options" in path.read_text()


@pytest.mark.xfail(
    strict=False,
    reason="gbb_solver stores 'total_grad' but GBBSolver.solve checks "
           "for 'total_crit' when full_results=True",
)
def test_solve_full_results_returns_lists():
    np.random.seed(11)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(full_results=True, verbose=0, maxiter=5).solve(problem)
    assert isinstance(result.total_fun, list)
    assert isinstance(result.total_crit, list)
