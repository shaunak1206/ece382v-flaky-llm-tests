import numpy as np
import pytest
import scipy.linalg as sla

import skprocrustes as skp


def _good_matrices(with_xsol=True):
    A = np.eye(4)
    B = np.eye(4, 2)
    C = np.eye(2)
    if with_xsol:
        return [A, B, C, np.eye(4, 2)]
    return [A, B, C]


def _residual(problem, X):
    return np.linalg.norm(problem.A.dot(X).dot(problem.C) - problem.B) ** 2


def _initial_point(seed, n, p):
    np.random.seed(seed)
    X0 = np.random.rand(n, p)
    Q, _ = sla.qr(X0, mode='economic')
    return Q


def _identity_problem():
    A = np.eye(4)
    Xsol = np.eye(4, 2)
    C = np.eye(2)
    B = A.dot(Xsol).dot(C)
    return skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, Xsol))


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [(4, 5, 2, 2), (4, 4, 2, 3)])
def test_problem_rejects_non_square_sizes(sizes):
    with pytest.raises(Exception, match="square"):
        skp.ProcrustesProblem(sizes, problemnumber=1)


def test_problem_rejects_m_not_divisible_by_p():
    with pytest.raises(Exception, match="divisible"):
        skp.ProcrustesProblem((5, 5, 2, 2), problemnumber=1)


def test_problem_requires_matrices_or_problemnumber():
    with pytest.raises(Exception, match="no problem number"):
        skp.ProcrustesProblem((4, 4, 2, 2))


def test_problem_user_matrices_without_xsol():
    mats = _good_matrices(with_xsol=False)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=tuple(mats))
    assert problem.sizes == (4, 4, 2, 2)
    assert problem.A is mats[0]
    assert problem.B is mats[1]
    assert problem.C is mats[2]
    assert not hasattr(problem, "Xsol")
    assert problem.problemnumber is None
    assert problem.stats == {}


def test_problem_user_matrices_with_xsol():
    mats = _good_matrices(with_xsol=True)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=mats)
    assert problem.Xsol is mats[3]
    assert problem.Xsol.shape == (4, 2)


@pytest.mark.parametrize("bad_index, bad_shape, message", [
    (0, (3, 4), "A must be"),
    (1, (4, 3), "B must be"),
    (2, (3, 2), "C must be"),
    (3, (3, 2), "Xsol must be"),
])
def test_problem_rejects_wrong_matrix_shapes(bad_index, bad_shape, message):
    mats = _good_matrices(with_xsol=True)
    mats[bad_index] = np.zeros(bad_shape)
    with pytest.raises(Exception, match=message):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=mats)


def test_problem_rejects_too_few_matrices():
    mats = _good_matrices(with_xsol=False)
    with pytest.raises(Exception, match="matrices must be"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=mats[:2])


def test_problem_number_1_structure():
    np.random.seed(0)
    problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    assert problem.problemnumber == 1
    assert np.allclose(problem.C, np.eye(2))
    # known solution has orthonormal columns and reproduces B exactly
    assert np.allclose(problem.Xsol.T.dot(problem.Xsol), np.eye(2))
    assert np.allclose(problem.A.dot(problem.Xsol).dot(problem.C), problem.B)
    # singular values of A lie in [10, 12]
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert sv.min() >= 10.0 - 1e-8
    assert sv.max() <= 12.0 + 1e-8


def test_problem_number_2_structure():
    np.random.seed(1)
    problem = skp.ProcrustesProblem((6, 6, 3, 3), problemnumber=2)
    assert problem.A.shape == (6, 6)
    assert problem.B.shape == (6, 3)
    assert problem.C.shape == (3, 3)
    assert np.all(np.isfinite(problem.A))
    assert np.allclose(problem.Xsol.T.dot(problem.Xsol), np.eye(3))
    assert np.allclose(problem.A.dot(problem.Xsol).dot(problem.C), problem.B)


def test_problem_number_3_singular_value_blocks():
    np.random.seed(2)
    problem = skp.ProcrustesProblem((50, 50, 2, 2), problemnumber=3)
    assert problem.A.shape == (50, 50)
    sv = np.linalg.svd(problem.A, compute_uv=False)  # descending
    eps = 1e-8
    assert np.all(sv[:15] >= 10.0 - eps) and np.all(sv[:15] <= 11.0 + eps)
    assert np.all(sv[15:30] >= 5.0 - eps) and np.all(sv[15:30] <= 6.0 + eps)
    assert np.all(sv[30:42] >= 2.0 - eps) and np.all(sv[30:42] <= 3.0 + eps)
    assert np.all(sv[42:] <= 1e-3 + eps)


def test_problem_number_3_requires_supported_dimension():
    with pytest.raises(Exception, match="requires n"):
        skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


# ---------------------------------------------------------------------------
# GBBSolver: options
# ---------------------------------------------------------------------------

def test_gbb_default_options():
    solver = skp.GBBSolver()
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
    solver = skp.GBBSolver(xtol=1e-4, gtol=1e-5, maxiter=10, verbose=0,
                           etavar=True, projected_gradient=2, timer=True,
                           filename="somefile.txt")
    assert solver.options["xtol"] == 1e-4
    assert solver.options["gtol"] == 1e-5
    assert solver.options["maxiter"] == 10
    assert solver.options["verbose"] == 0
    assert solver.options["etavar"] is True
    assert solver.options["projected_gradient"] == 2
    assert solver.options["timer"] is True
    assert solver.options["filename"] == "somefile.txt"
    # unspecified options still get defaults
    assert solver.options["ftol"] == 1e-12


@pytest.mark.parametrize("kwargs", [
    {"full_results": 1},
    {"xtol": 1},
    {"gtol": "small"},
    {"ftol": 1},
    {"zeta": 1},
    {"kappa": 1},
    {"eta": 1},
    {"etavar": 1},
    {"tau": 1},
    {"projected_gradient": 3},
    {"maxiter": 1.5},
    {"verbose": 2},
    {"filename": 5},
    {"timer": 1},
])
def test_gbb_rejects_invalid_options(kwargs):
    with pytest.raises(Exception, match="must be"):
        skp.GBBSolver(**kwargs)


# ---------------------------------------------------------------------------
# GBBSolver: solve
# ---------------------------------------------------------------------------

def test_gbb_solve_identity_problem_reduces_residual():
    problem = _identity_problem()
    X0 = _initial_point(0, 4, 2)
    initial = _residual(problem, X0)

    np.random.seed(0)
    result = skp.GBBSolver(verbose=0).solve(problem)

    assert result.solution.shape == (4, 2)
    assert np.allclose(result.solution.T.dot(result.solution), np.eye(2),
                       atol=1e-6)
    final = _residual(problem, result.solution)
    assert np.isfinite(final)
    assert final < initial
    assert result.success == (result.status == 0)
    assert result.cpu >= 0.0
    assert np.isfinite(result.error)


@pytest.mark.parametrize("projected_gradient", [1, 2])
def test_gbb_solve_generated_problem_is_feasible(projected_gradient):
    np.random.seed(123)
    problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    initial = _residual(problem, _initial_point(5, 10, 2))

    np.random.seed(5)
    solver = skp.GBBSolver(verbose=0, projected_gradient=projected_gradient)
    result = solver.solve(problem)

    X = result.solution
    assert X.shape == (10, 2)
    assert np.linalg.norm(X.T.dot(X) - np.eye(2)) < 1e-6
    assert problem.stats["feasi"] < 1e-6
    assert _residual(problem, X) < initial
    assert result.nbiter == problem.stats["nbiter"]
    assert result.nbiter >= 3
    assert result.nfev == problem.stats["fev"]
    assert result.nfev >= result.nbiter + 1
    assert np.isfinite(result.error)


def test_gbb_solve_is_deterministic_for_fixed_seed():
    problem = _identity_problem()
    np.random.seed(11)
    first = skp.GBBSolver(verbose=0, maxiter=5).solve(problem)
    np.random.seed(11)
    second = skp.GBBSolver(verbose=0, maxiter=5).solve(problem)
    assert np.allclose(first.solution, second.solution)
    assert first.nbiter == second.nbiter
    assert first.nfev == second.nfev


def test_gbb_maxiter_zero_returns_initial_orthonormal_point():
    problem = _identity_problem()
    expected = _initial_point(3, 4, 2)

    np.random.seed(3)
    result = skp.GBBSolver(verbose=0, maxiter=0).solve(problem)

    assert result.success is False
    assert result.status == 1
    assert result.nbiter == 0
    assert result.nfev == 1
    assert np.allclose(result.solution, expected)
    assert np.isclose(result.fun, _residual(problem, expected))


def test_gbb_maxiter_exceeded_reports_failure(capsys):
    problem = _identity_problem()
    np.random.seed(4)
    result = skp.GBBSolver(verbose=0, maxiter=2).solve(problem)

    assert result.success is False
    assert result.status == 1
    assert result.nbiter == 2
    assert result.message == "Maximum number of iterations has been exceeded."
    out = capsys.readouterr().out
    assert "Maximum number of iterations" in out


def test_gbb_without_known_solution_reports_infinite_error():
    mats = _good_matrices(with_xsol=False)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=mats)
    np.random.seed(6)
    result = skp.GBBSolver(verbose=0, maxiter=0).solve(problem)
    assert result.error == np.inf
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


def test_gbb_full_results_requires_total_crit_in_stats():
    problem = _identity_problem()
    np.random.seed(7)
    solver = skp.GBBSolver(verbose=0, maxiter=2, full_results=True)
    with pytest.raises(Exception, match="full results"):
        solver.solve(problem)


def test_gbb_full_results_returns_recorded_function_values():
    problem = _identity_problem()
    problem.stats["total_crit"] = []
    np.random.seed(8)
    result = skp.GBBSolver(verbose=0, maxiter=2, full_results=True).solve(problem)

    assert isinstance(result.total_fun, list)
    assert len(result.total_fun) == result.nfev
    assert result.total_crit == []
    assert all(np.isfinite(v) for v in result.total_fun)


def test_gbb_writes_log_to_file(tmp_path):
    problem = _identity_problem()
    logfile = tmp_path / "gbb_log.txt"
    np.random.seed(9)
    solver = skp.GBBSolver(verbose=1, maxiter=2, filename=str(logfile))
    solver.solve(problem)

    assert logfile.exists()
    content = logfile.read_text()
    assert "Options" in content
    assert "Execution date" in content


def test_gbb_invalid_filename_exits(tmp_path):
    problem = _identity_problem()
    bad = tmp_path / "missing_dir" / "log.txt"
    solver = skp.GBBSolver(verbose=0, filename=str(bad))
    with pytest.raises(SystemExit):
        solver.solve(problem)
