import numpy as np
import pytest

import skprocrustes as skp


def _make_problem(sizes=(10, 10, 2, 2), number=1, seed=0):
    np.random.seed(seed)
    return skp.ProcrustesProblem(sizes, problemnumber=number)


def _user_problem(n=8, p=2, seed=1, with_xsol=True):
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    Xsol, _ = np.linalg.qr(rng.randn(n, p))
    C = np.eye(p)
    B = A.dot(Xsol).dot(C)
    if with_xsol:
        return skp.ProcrustesProblem((n, n, p, p), matrices=(A, B, C, Xsol))
    return skp.ProcrustesProblem((n, n, p, p), matrices=(A, B, C))


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_problem_rejects_non_square_sizes():
    with pytest.raises(Exception, match="square"):
        skp.ProcrustesProblem((10, 8, 2, 2), problemnumber=1)
    with pytest.raises(Exception, match="square"):
        skp.ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_problem_rejects_m_not_divisible_by_p():
    with pytest.raises(Exception, match="divisible"):
        skp.ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_problem_requires_matrices_or_problem_number():
    with pytest.raises(Exception, match="no problem number"):
        skp.ProcrustesProblem((10, 10, 2, 2))


def test_problem_three_requires_special_n():
    with pytest.raises(Exception, match="requires n"):
        skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_problem_one_shapes_and_attributes():
    problem = _make_problem((10, 10, 2, 2), number=1)
    assert problem.sizes == (10, 10, 2, 2)
    assert problem.problemnumber == 1
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    assert problem.stats == {}


def test_problem_one_known_solution_is_exact_and_orthonormal():
    problem = _make_problem((10, 10, 2, 2), number=1, seed=3)
    np.testing.assert_allclose(problem.C, np.eye(2))
    np.testing.assert_allclose(problem.Xsol.T.dot(problem.Xsol), np.eye(2),
                               atol=1e-12)
    np.testing.assert_allclose(
        problem.A.dot(problem.Xsol).dot(problem.C), problem.B, atol=1e-10)


def test_problem_one_singular_values_in_expected_range():
    problem = _make_problem((12, 12, 3, 3), number=1, seed=5)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert np.all(sv >= 10.0 - 1e-8)
    assert np.all(sv <= 12.0 + 1e-8)


def test_problem_three_has_small_singular_values():
    problem = _make_problem((50, 50, 5, 5), number=3, seed=4)
    assert problem.A.shape == (50, 50)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert np.sum(sv < 0.01) == 8
    assert np.sum(sv >= 2.0 - 1e-8) == 42


def test_problem_user_matrices_without_solution():
    problem = _user_problem(with_xsol=False)
    assert problem.A.shape == (8, 8)
    assert problem.B.shape == (8, 2)
    assert problem.C.shape == (2, 2)
    assert not hasattr(problem, "Xsol")
    assert problem.stats == {}


def test_problem_user_matrices_with_solution_and_list_input():
    n, p = 6, 3
    rng = np.random.RandomState(7)
    A = rng.randn(n, n)
    Xsol, _ = np.linalg.qr(rng.randn(n, p))
    C = np.eye(p)
    B = A.dot(Xsol)
    problem = skp.ProcrustesProblem((n, n, p, p), matrices=[A, B, C, Xsol])
    assert problem.Xsol is Xsol
    assert problem.A is A


def test_problem_rejects_bad_user_matrices():
    n, p = 6, 2
    A = np.zeros((n, n))
    B = np.zeros((n, p))
    C = np.zeros((p, p))
    X = np.zeros((n, p))
    sizes = (n, n, p, p)
    with pytest.raises(Exception, match="A must be"):
        skp.ProcrustesProblem(sizes, matrices=(np.zeros((n, n + 1)), B, C))
    with pytest.raises(Exception, match="B must be"):
        skp.ProcrustesProblem(sizes, matrices=(A, np.zeros((n, p + 1)), C))
    with pytest.raises(Exception, match="C must be"):
        skp.ProcrustesProblem(sizes, matrices=(A, B, np.zeros((p + 1, p))))
    with pytest.raises(Exception, match="Xsol must be"):
        skp.ProcrustesProblem(sizes, matrices=(A, B, C, np.zeros((n, p + 1))))
    with pytest.raises(Exception, match="matrices must be"):
        skp.ProcrustesProblem(sizes, matrices=(A, B))
    # a valid call for contrast
    skp.ProcrustesProblem(sizes, matrices=(A, B, C, X))


# ---------------------------------------------------------------------------
# GBBSolver options
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
    solver = skp.GBBSolver(xtol=1e-4, gtol=1e-5, maxiter=17, verbose=0,
                           projected_gradient=2, etavar=True)
    assert solver.options["xtol"] == 1e-4
    assert solver.options["gtol"] == 1e-5
    assert solver.options["maxiter"] == 17
    assert solver.options["verbose"] == 0
    assert solver.options["projected_gradient"] == 2
    assert solver.options["etavar"] is True


@pytest.mark.parametrize("kwargs, message", [
    ({"full_results": "yes"}, "full_results"),
    ({"xtol": 1}, "xtol"),
    ({"gtol": 1}, "gtol"),
    ({"ftol": "small"}, "ftol"),
    ({"zeta": 1}, "zeta"),
    ({"kappa": 1}, "kappa"),
    ({"eta": 1}, "eta"),
    ({"etavar": 1}, "etavar"),
    ({"tau": 1}, "tau"),
    ({"projected_gradient": 3}, "projected_gradient"),
    ({"maxiter": 1.5}, "maxiter"),
    ({"verbose": 2}, "verbose"),
    ({"filename": 3}, "filename"),
    ({"timer": 1}, "timer"),
])
def test_gbb_rejects_invalid_options(kwargs, message):
    with pytest.raises(Exception, match=message):
        skp.GBBSolver(**kwargs)


# ---------------------------------------------------------------------------
# GBBSolver.solve
# ---------------------------------------------------------------------------

def test_gbb_solve_returns_expected_result_fields():
    problem = _make_problem((10, 10, 2, 2), number=1, seed=11)
    np.random.seed(12)
    result = skp.GBBSolver(verbose=0, maxiter=50).solve(problem)

    assert isinstance(result, skp.OptimizeResult)
    for key in ("success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"):
        assert key in result
    assert result.solution.shape == (10, 2)
    assert isinstance(result.success, bool)
    assert result.status in (0, 1)
    assert result.success == (result.status == 0)
    assert result.cpu >= 0.0
    assert np.isfinite(result.error)
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf
    assert result.nbiter == problem.stats["nbiter"]
    assert result.nfev == problem.stats["fev"]
    assert result.nfev >= result.nbiter + 1


@pytest.mark.parametrize("sizes", [(10, 10, 2, 2), (6, 6, 3, 3), (8, 8, 4, 4)])
def test_gbb_solution_stays_on_stiefel_manifold(sizes):
    problem = _make_problem(sizes, number=1, seed=21)
    np.random.seed(22)
    result = skp.GBBSolver(verbose=0, maxiter=30).solve(problem)
    p = sizes[2]
    X = result.solution
    np.testing.assert_allclose(X.T.dot(X), np.eye(p), atol=1e-8)
    assert problem.stats["feasi"] < 1e-8


def test_gbb_reduces_the_residual_on_well_conditioned_problem():
    problem = _make_problem((10, 10, 2, 2), number=1, seed=31)
    np.random.seed(32)
    result = skp.GBBSolver(verbose=0, maxiter=500).solve(problem)

    A, B, C = problem.A, problem.B, problem.C
    final_res = np.linalg.norm(A.dot(result.solution).dot(C) - B, 'fro') ** 2

    rng = np.random.RandomState(33)
    X0, _ = np.linalg.qr(rng.rand(10, 2))
    initial_res = np.linalg.norm(A.dot(X0).dot(C) - B, 'fro') ** 2
    assert final_res < initial_res


def test_gbb_projected_gradient_two_and_etavar_run_cleanly():
    problem = _make_problem((10, 10, 2, 2), number=1, seed=41)
    np.random.seed(42)
    solver = skp.GBBSolver(verbose=0, maxiter=20, projected_gradient=2,
                           etavar=True)
    result = solver.solve(problem)
    X = result.solution
    np.testing.assert_allclose(X.T.dot(X), np.eye(2), atol=1e-8)
    assert result.nbiter <= 20


def test_gbb_maxiter_one_reports_maxiter_failure(capsys):
    problem = _make_problem((10, 10, 2, 2), number=1, seed=51)
    np.random.seed(52)
    result = skp.GBBSolver(verbose=0, maxiter=1).solve(problem)
    assert result.nbiter == 1
    assert result.status == 1
    assert result.success is False
    assert "Maximum number of iterations" in result.message
    out = capsys.readouterr().out
    assert "Warning" in out


def test_gbb_error_is_inf_without_known_solution():
    problem = _user_problem(n=8, p=2, with_xsol=False)
    np.random.seed(61)
    result = skp.GBBSolver(verbose=0, maxiter=10).solve(problem)
    assert result.error == np.inf
    assert result.solution.shape == (8, 2)


def test_gbb_full_results_requires_total_crit_in_stats():
    # gbb_solver stores "total_grad", but solve() looks for "total_crit".
    problem = _make_problem((10, 10, 2, 2), number=1, seed=71)
    np.random.seed(72)
    solver = skp.GBBSolver(verbose=0, maxiter=5, full_results=True)
    with pytest.raises(Exception, match="full results"):
        solver.solve(problem)
    assert len(problem.stats["total_fun"]) >= 1
    assert len(problem.stats["total_grad"]) >= 1


def test_gbb_verbose_prints_header_to_stdout(capsys):
    problem = _make_problem((10, 10, 2, 2), number=1, seed=81)
    np.random.seed(82)
    skp.GBBSolver(verbose=1, maxiter=3).solve(problem)
    out = capsys.readouterr().out
    assert "Options:" in out
    assert "Execution date" in out
    assert "nbiter" in out


def test_gbb_filename_writes_log_and_closes_file(tmp_path):
    logfile = tmp_path / "gbb_log.txt"
    problem = _make_problem((10, 10, 2, 2), number=1, seed=91)
    np.random.seed(92)
    solver = skp.GBBSolver(verbose=1, maxiter=3, filename=str(logfile))
    solver.solve(problem)
    assert logfile.exists()
    assert solver.file.closed
    content = logfile.read_text()
    assert "Options:" in content
    assert "Execution date" in content


def test_gbb_invalid_filename_exits(tmp_path):
    bad = tmp_path / "missing_dir" / "log.txt"
    problem = _make_problem((10, 10, 2, 2), number=1, seed=95)
    solver = skp.GBBSolver(verbose=0, maxiter=3, filename=str(bad))
    with pytest.raises(SystemExit):
        solver.solve(problem)


def test_gbb_solver_function_returns_six_tuple_and_updates_stats():
    problem = _make_problem((10, 10, 2, 2), number=1, seed=101)
    solver = skp.GBBSolver(verbose=0, maxiter=10)
    rng = np.random.RandomState(102)
    X0, _ = np.linalg.qr(rng.rand(10, 2))
    out = skp.gbb_solver(problem, 10, 10, X0, problem.A, problem.B,
                         solver.options, False, None)
    exitcode, fval, X, normgrad, nbiter, msg = out
    assert exitcode in (0, 1)
    assert X.shape == (10, 2)
    assert normgrad >= 0.0
    assert nbiter == problem.stats["nbiter"]
    assert 1 <= nbiter <= 10
    assert isinstance(msg, str)
    assert problem.stats["grad"] >= 1
    assert problem.stats["linear_solver"] == 0
