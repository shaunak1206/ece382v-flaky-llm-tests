import numpy as np
import pytest
import os
import tempfile
import builtins

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, OptimizeResult


def test_procrustesproblem_default_problem_1_shapes():
    sizes = (10, 10, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    m, n, p, q = sizes
    assert problem.A.shape == (m, n)
    assert problem.B.shape == (m, q)
    assert problem.C.shape == (p, q)
    assert problem.Xsol.shape == (n, p)


def test_procrustesproblem_default_problem_2_shapes():
    sizes = (10, 10, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=2)
    m, n, p, q = sizes
    assert problem.A.shape == (m, n)
    assert problem.B.shape == (m, q)
    assert problem.C.shape == (p, q)
    assert problem.Xsol.shape == (n, p)


def test_procrustesproblem_default_problem_3_valid_n():
    # n must be one of (50,95,500,1000) for problemnumber=3
    for n_allowed in (50, 95, 500, 1000):
        sizes = (n_allowed, n_allowed, 10, 10)
        problem = ProcrustesProblem(sizes, problemnumber=3)
        assert problem.A.shape == (n_allowed, n_allowed)
        assert problem.Xsol.shape == (n_allowed, 10)


def test_procrustesproblem_default_problem_3_invalid_n():
    sizes = (51, 51, 10, 10)
    with pytest.raises(Exception) as e:
        ProcrustesProblem(sizes, problemnumber=3)
    assert "Problem 6 requires" in str(e.value)


def test_procrustes_invalid_sizes_non_square():
    with pytest.raises(Exception) as e:
        ProcrustesProblem((5, 6, 2, 2))
    assert "square problems are supported" in str(e.value)

    with pytest.raises(Exception) as e:
        ProcrustesProblem((6, 6, 2, 3))
    assert "square problems are supported" in str(e.value)


def test_procrustes_invalid_sizes_not_divisible():
    with pytest.raises(Exception) as e:
        ProcrustesProblem((10, 10, 3, 3))
    assert "m must be divisible by p" in str(e.value)


def test_procrustes_user_defined_matrices_correct_sizes():
    m, n, p, q = 6, 6, 2, 2
    A = np.random.randn(m, n)
    B = np.random.randn(m, q)
    C = np.eye(p, q)
    Xsol = np.random.randn(n, p)
    problem = ProcrustesProblem((m, n, p, q), matrices=[A, B, C, Xsol])
    np.testing.assert_array_equal(problem.A, A)
    np.testing.assert_array_equal(problem.B, B)
    np.testing.assert_array_equal(problem.C, C)
    np.testing.assert_array_equal(problem.Xsol, Xsol)


def test_procrustes_user_defined_matrices_incorrect_shapes():
    m, n, p, q = 6, 6, 2, 2
    A = np.random.randn(m, n)
    B = np.random.randn(m, q)
    C = np.eye(p, q)
    Xsol = np.random.randn(n, p)

    # wrong shape for A
    A_bad = np.random.randn(m + 1, n)
    with pytest.raises(Exception) as e:
        ProcrustesProblem((m, n, p, q), matrices=[A_bad, B, C])
    assert "A must be (m,n)" in str(e.value)

    # wrong shape for B
    B_bad = np.random.randn(m, q + 1)
    with pytest.raises(Exception) as e:
        ProcrustesProblem((m, n, p, q), matrices=[A, B_bad, C])
    assert "B must be (m,q)" in str(e.value)

    # wrong shape for C
    C_bad = np.eye(p + 1, q)
    with pytest.raises(Exception) as e:
        ProcrustesProblem((m, n, p, q), matrices=[A, B, C_bad])
    assert "C must be (p,q)" in str(e.value)

    # wrong shape for Xsol
    Xsol_bad = np.random.randn(n + 1, p)
    with pytest.raises(Exception) as e:
        ProcrustesProblem((m, n, p, q), matrices=[A, B, C, Xsol_bad])
    assert "Xsol must be (n,p)" in str(e.value)

    # invalid container type for matrices
    with pytest.raises(Exception) as e:
        ProcrustesProblem((m, n, p, q), matrices="notalist")
    assert "matrices must be a list or tuple" in str(e.value)

    # less than three matrices given
    with pytest.raises(Exception) as e:
        ProcrustesProblem((m, n, p, q), matrices=[A])
    assert "matrices must be a list or tuple containing matrices" in str(e.value)


def test_gbbsolver_default_options_and_validation():
    # Creating GBBSolver with defaults
    solver = GBBSolver()
    opts = solver.options
    # Check default keys and types
    assert isinstance(opts, dict)
    # defaults keys present
    for key in ("full_results", "xtol", "gtol", "ftol", "zeta", "kappa",
                "eta", "etavar", "tau", "projected_gradient", "maxiter",
                "verbose", "filename", "timer"):
        assert key in opts

    # full_results must be bool
    with pytest.raises(Exception):
        GBBSolver(full_results="yes")

    # xtol must be float
    with pytest.raises(Exception):
        GBBSolver(xtol="low")

    # gtol must be float
    with pytest.raises(Exception):
        GBBSolver(gtol="low")

    # ftol must be float
    with pytest.raises(Exception):
        GBBSolver(ftol="low")

    # zeta must be float
    with pytest.raises(Exception):
        GBBSolver(zeta="low")

    # kappa must be float
    with pytest.raises(Exception):
        GBBSolver(kappa="low")

    # eta must be float
    with pytest.raises(Exception):
        GBBSolver(eta="low")

    # etavar must be bool
    with pytest.raises(Exception):
        GBBSolver(etavar="no")

    # tau must be float
    with pytest.raises(Exception):
        GBBSolver(tau="low")

    # projected_gradient must be 1 or 2
    with pytest.raises(Exception):
        GBBSolver(projected_gradient=3)

    # maxiter must be int
    with pytest.raises(Exception):
        GBBSolver(maxiter=1.5)

    # verbose must be 0 or 1
    with pytest.raises(Exception):
        GBBSolver(verbose=5)

    # filename must be string if provided
    with pytest.raises(Exception):
        GBBSolver(filename=123)

    # timer must be bool
    with pytest.raises(Exception):
        GBBSolver(timer="yes")


def test_gbbsolver_solve_returns_optimize_result(monkeypatch):
    
    m, n, p, q = 6, 6, 2, 2
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)

    # Patch gbb_solver to return fixed values (exitcode, fval, X, normG, nbiter, msg)
    def fake_gbb_solver(problem, largedim, smalldim, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        Xsol = np.eye(n, p)
        return 0, 1.23, Xsol, 0.001, 10, "success"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_gbb_solver)

    solver = GBBSolver(full_results=False)
    result = solver.solve(problem)

    assert isinstance(result, OptimizeResult)
    assert result.success is True
    assert result.status == 0
    assert result.message == "success"
    assert result.fun == 1.23
    assert np.allclose(result.solution.T @ result.solution, np.eye(p), atol=1e-6)
    assert result.error >= 0
    assert result.nbiter == problem.stats["nbiter"]
    assert result.nfev == problem.stats["fev"]
    # total_fun and total_crit should be inf because full_results is False
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


def test_gbbsolver_solve_with_full_results_checks_keys(monkeypatch):
    m, n, p, q = 6, 6, 2, 2
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)
    # Set problem.stats keys for full_results
    problem.stats["total_fun"] = [0.1, 0.05]
    problem.stats["total_crit"] = [0.01, 0.005]

    def fake_gbb_solver(problem, largedim, smalldim, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        Xsol = np.eye(n, p)
        return 0, 0.02, Xsol, 0.0001, 5, "success"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_gbb_solver)

    solver = GBBSolver(full_results=True)
    result = solver.solve(problem)

    assert isinstance(result.total_fun, list)
    assert isinstance(result.total_crit, list)
    assert len(result.total_fun) == len(problem.stats["total_fun"])
    assert len(result.total_crit) == len(problem.stats["total_crit"])


def test_gbbsolver_open_close_file_sys_stdout(monkeypatch):
    solver = GBBSolver(filename=None)
    solver.open_file()
    # file should be sys.stdout
    import sys
    assert solver.file == sys.stdout
    solver.close_file()  # should not raise


def test_gbbsolver_open_file_invalid_filename(monkeypatch):
    solver = GBBSolver(filename="/invalid/path/to/file.txt")

    # Patch sys.exit to raise SystemExit so pytest can catch it
    monkeypatch.setattr("sys.exit", lambda code=-1: (_ for _ in ()).throw(SystemExit(code)))

    # Patch builtins.print to suppress output
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit):
        solver.open_file()


def test_gbbsolver_open_close_file_tmpfile():
    # Write to a temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        fname = tmpfile.name
    try:
        solver = GBBSolver(filename=fname)
        solver.open_file()
        assert not solver.file.closed
        solver.file.write("Test\n")
        solver.close_file()
        assert solver.file.closed
    finally:
        os.remove(fname)


def test_optimize_result_dict_attr_access():
    res = OptimizeResult({"success": True, "fun": 2.5, "solution": np.array([[1, 2], [3, 4]])})
    assert res.success is True
    assert res.fun == 2.5
    np.testing.assert_array_equal(res.solution, np.array([[1, 2], [3, 4]]))

    # Attribute error on missing key
    with pytest.raises(AttributeError):
        _ = res.nonexistent


def test_optimize_result_repr_and_dir():
    res = OptimizeResult({"key1": 123, "key2": "abc"})
    rep = repr(res)
    assert "key1" in rep
    assert "key2" in rep

    dirlist = dir(res)
    assert "key1" in dirlist
    assert "key2" in dirlist


def test_optimize_result_show(capsys):
    res = OptimizeResult({"success": True, "fun": 1.0, "solution": np.array([1])})
    res.show()
    captured = capsys.readouterr()
    assert "success" in captured.out
    assert "fun" in captured.out
    assert "solution" not in captured.out


def test_gbbsolver_solve_error_when_full_results_missing_stats(monkeypatch):
    m, n, p, q = 6, 6, 2, 2
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)
    # Remove total_fun/total_crit keys to simulate missing data
    if "total_fun" in problem.stats:
        del problem.stats["total_fun"]
    if "total_crit" in problem.stats:
        del problem.stats["total_crit"]

    def fake_gbb_solver(problem, largedim, smalldim, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        Xsol = np.eye(n, p)
        return 0, 1.0, Xsol, 0.001, 10, "success"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_gbb_solver)
    solver = GBBSolver(full_results=True)
    with pytest.raises(Exception) as e:
        solver.solve(problem)
    assert "For full results, set problem.stats" in str(e.value)
