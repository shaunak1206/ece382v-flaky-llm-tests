import numpy as np
import pytest
import sys
import os
from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, OptimizeResult


def make_square_problem(m=10, n=10, p=2, q=2):
    # Create random matrices A, B, C with compatible sizes and Xsol
    A = np.random.randn(m, n)
    Q, _ = np.linalg.qr(np.random.randn(n, p))  # orthonormal n x p
    C = np.eye(p, q)
    Xsol = Q[:, :p]
    B = A @ Xsol @ C
    return A, B, C, Xsol


def test_procrustesproblem_init_with_predefined_problemnumber():
    # Checks that predefined problems build matrices with correct dimensions
    for pn in [1, 2, 3]:
        # 3 requires specific n
        n = 50 if pn == 3 else 10
        sizes = (n, n, 2, 2)
        problem = ProcrustesProblem(sizes, problemnumber=pn)
        assert problem.A.shape == (n, n)
        assert problem.B.shape == (n, 2)
        assert problem.C.shape == (2, 2)
        assert problem.Xsol.shape == (n, 2)
        # Xsol should be orthonormal: Xsol.T @ Xsol ~ I
        np.testing.assert_allclose(problem.Xsol.T @ problem.Xsol, np.eye(2), atol=1e-12)


def test_procrustesproblem_init_with_user_matrices_accepts_valid():
    m, n, p, q = 10, 10, 3, 3
    A, B, C, Xsol = make_square_problem(m, n, p, q)
    problem = ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))
    assert np.allclose(problem.A, A)
    assert np.allclose(problem.B, B)
    assert np.allclose(problem.C, C)
    assert np.allclose(problem.Xsol, Xsol)


def test_procrustesproblem_init_raises_on_invalid_sizes():
    # m != n or p != q raises Exception
    with pytest.raises(Exception, match="square problems"):
        ProcrustesProblem((5, 6, 2, 2))  # m != n
    with pytest.raises(Exception, match="square problems"):
        ProcrustesProblem((10, 10, 2, 3))  # p != q
    # m must be divisible by p
    with pytest.raises(Exception, match="divisible by p"):
        ProcrustesProblem((10, 10, 3, 3))


def test_procrustesproblem_init_raises_on_invalid_matrices():
    m, n, p, q = 10, 10, 3, 3
    A, B, C, Xsol = make_square_problem(m, n, p, q)
    # matrices must be tuple or list
    with pytest.raises(Exception, match="list or tuple"):
        ProcrustesProblem((m, n, p, q), matrices="not a list")
    # less than 3 matrices
    with pytest.raises(Exception, match="containing matrices A, B and C"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B))
    # wrong shape of A
    A_bad = np.random.randn(m+1, n)
    with pytest.raises(Exception, match="A must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A_bad, B, C))
    # wrong shape of B
    B_bad = np.random.randn(m, q+1)
    with pytest.raises(Exception, match="B must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B_bad, C))
    # wrong shape of C
    C_bad = np.random.randn(p, q+1)
    with pytest.raises(Exception, match="C must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C_bad))
    # wrong shape of Xsol
    Xsol_bad = np.random.randn(n+1, p)
    with pytest.raises(Exception, match="Xsol must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol_bad))


def test_gbbsovler_setoptions_defaults():
    solver = GBBSolver()
    opts = solver.options
    # Check default keys and types
    assert isinstance(opts["full_results"], bool)
    assert isinstance(opts["xtol"], float)
    assert isinstance(opts["gtol"], float)
    assert isinstance(opts["ftol"], float)
    assert isinstance(opts["zeta"], float)
    assert isinstance(opts["kappa"], float)
    assert isinstance(opts["eta"], float)
    assert isinstance(opts["etavar"], bool)
    assert isinstance(opts["tau"], float)
    assert opts["projected_gradient"] in [1, 2]
    assert isinstance(opts["maxiter"], int)
    assert opts["verbose"] in [0, 1]
    assert opts["filename"] is None
    assert isinstance(opts["timer"], bool)


@pytest.mark.parametrize("invalid_opt,value,errmsg", [
    ("full_results", "no", "full_results must be a boolean"),
    ("xtol", "1e-6", "xtol must be a float"),
    ("gtol", 1, "gtol must be a float"),
    ("ftol", None, "ftol must be a float"),
    ("zeta", "0.1", "zeta must be a float"),
    ("kappa", [], "kappa must be a float"),
    ("eta", None, "eta must be a float"),
    ("etavar", 1, "etavar must be a boolean"),
    ("tau", "1e-3", "tau must be a float"),
    ("projected_gradient", 3, "projected_gradient must be 1 or 2"),
    ("maxiter", 0.1, "maxiter must be an integer"),
    ("verbose", 5, "verbose must be 0, or 1"),
    ("filename", 123, "filename must be string"),
    ("timer", "False", "timer must be boolean"),
])
def test_gbbsovler_setoptions_invalid(invalid_opt, value, errmsg):
    # Attempt to create GBBSolver with invalid option triggers exception
    opts = {invalid_opt: value}
    if invalid_opt == "verbose":
        # verbose is only 0 or 1 allowed for GBBSolver
        opts["xtol"] = 1e-6  # to avoid other errors
        opts["gtol"] = 1e-6
        opts["ftol"] = 1e-12
        opts["kappa"] = 0.1
        opts["zeta"] = 1e-4
        opts["tau"] = 1e-3
        opts["projected_gradient"] = 1
        opts["maxiter"] = 10
        opts["full_results"] = False
    with pytest.raises(Exception, match=errmsg):
        GBBSolver(**opts)


def test_gbbsovler_open_close_file(tmp_path):
    # Test opening and closing file when filename is given and None
    filename = tmp_path / "tempfile.txt"
    solver = GBBSolver(filename=str(filename))
    # file should be open file handle
    solver.open_file()
    assert not solver.file.closed
    assert solver.file.name == str(filename)
    solver.close_file()
    assert solver.file.closed
    # With filename None, fileobj is sys.stdout
    solver2 = GBBSolver()
    solver2.open_file()
    assert solver2.file == sys.stdout
    solver2.close_file()  # Nothing to close for sys.stdout


def test_gbbsovler_open_file_invalid_name():
    solver = GBBSolver(filename="<>invalid_filename<>")
    # Opening file with invalid name prints error and exits,
    # so we will catch SystemExit here
    with pytest.raises(SystemExit):
        solver.open_file()


def test_gbbsovler_solve_returns_optimizeresult(monkeypatch):
    m, n, p, q = 10, 10, 3, 3
    A, B, C, Xsol = make_square_problem(m, n, p, q)
    problem = ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))
    # patch gbb_solver called inside solve to return controlled output
    def fake_gbb_solver(problem, largedim, smalldim, X, Aarg, Barg, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        Xout = np.eye(smalldim, p)
        exitcode = 0
        fval = 1.23
        normG = 0.001
        nbiter = 5
        msg = "success"
        return exitcode, fval, Xout, normG, nbiter, msg
    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_gbb_solver)

    solver = GBBSolver(full_results=True)
    # Provide fake stats expected by solve()
    problem.stats["nbiter"] = 5
    problem.stats["fev"] = 10
    problem.stats["total_fun"] = [1.5, 1.3]
    problem.stats["total_crit"] = [0.002, 0.001]

    result = solver.solve(problem)
    assert isinstance(result, OptimizeResult)
    assert result.success is True
    assert result.status == 0
    assert hasattr(result, "solution")
    assert result.fun == 1.23
    assert np.allclose(result.solution.T @ result.solution, np.eye(p), atol=1e-12)
    assert result.error >= 0
    assert "nbiter" in result
    assert "nfev" in result
    assert isinstance(result.cpu, float)
    assert isinstance(result.total_fun, list)
    assert isinstance(result.total_crit, list)


def test_gbbsovler_solve_no_xsol(monkeypatch):
    m, n, p, q = 10, 10, 2, 2
    A, B, C, Xsol = make_square_problem(m, n, p, q)
    # Remove Xsol attribute deliberately
    problem = ProcrustesProblem((m, n, p, q), matrices=(A, B, C))
    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", lambda *a, **k: (0, 1.0, np.eye(n, p), 0.1, 1, "msg"))
    problem.stats = {"nbiter": 1, "fev": 1, "total_fun": [], "total_crit": []}
    solver = GBBSolver()
    result = solver.solve(problem)
    assert result.error == float('inf')
    assert result.success


def test_procrustesproblem_problemnumber_3_invalid_n():
    # problemnumber=3 raises if n not in allowed set
    n_invalid = 30
    with pytest.raises(Exception, match="requires n = 50,"):
        ProcrustesProblem((n_invalid, n_invalid, 2, 2), problemnumber=3)


def test_gbbsovler_setoptions_boundary_values():
    # Boundary checks that options at edge values don't raise
    options = {
        "full_results": True,
        "xtol": 1e-10,
        "gtol": 1e-10,
        "ftol": 1e-15,
        "zeta": 0.0,
        "kappa": 1.0,
        "eta": 0.0,
        "etavar": False,
        "tau": 0.0,
        "projected_gradient": 2,
        "maxiter": 1,
        "verbose": 1,
        "filename": None,
        "timer": False
    }
    solver = GBBSolver(**options)
    for k, v in options.items():
        assert solver.options[k] == v


def test_optimize_result_repr_and_attr():
    d = {"success": True, "fun": 1.23, "solution": np.ones((3, 2))}
    res = OptimizeResult(d)
    r = repr(res)
    assert "success" in r and "fun" in r
    # attribute access
    assert res.success is True
    assert np.all(res.solution == 1)
    with pytest.raises(AttributeError):
        _ = res.notexistingattr
    res.newattr = 100
    assert res["newattr"] == 100


def test_optimize_result_show(capsys):
    res = OptimizeResult({"success": True, "fun": 2.5, "solution": np.eye(2), "error": float('inf')})
    res.show()
    out = capsys.readouterr().out
    assert "success" in out
    assert "fun" in out
    assert "error" not in out


def test_gbbsovler_solve_verbose_output(monkeypatch, capsys):
    m, n, p, q = 10, 10, 2, 2
    A, B, C, Xsol = make_square_problem(m, n, p, q)
    problem = ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))
    problem.stats.update(nbiter=1, fev=1, total_fun=[], total_crit=[])
    def fake_solver(*args, **kwargs):
        return 0, 1.0, np.eye(n, p), 0.0, 1, "ok"
    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_solver)
    solver = GBBSolver(verbose=1)
    solver.solve(problem)
    out = capsys.readouterr().out
    assert "GBB Solver" in out
    assert "Options:" in out
    assert "nbiter" in out


def test_procrustesproblem_xsol_is_permutation():
    # According to code, Xsol is generated via np.random.permutation on identity matrix.
    n = 6
    sizes = (n, n, 3, 3)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    # Check that each row of Xsol has exactly one 1, rest 0; (is permutation matrix)
    row_sums = np.sum(np.abs(problem.Xsol), axis=1)
    col_sums = np.sum(np.abs(problem.Xsol), axis=0)
    assert np.allclose(row_sums, np.ones_like(row_sums))
    assert np.allclose(col_sums, np.ones_like(col_sums))


def test_gbbsovler_invalid_projected_gradient_option():
    # projected_gradient can only be 1 or 2
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        GBBSolver(projected_gradient=3)


def test_gbbsovler_verbose_option_out_of_range():
    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        GBBSolver(verbose=3)


def test_gbbsovler_timer_option_type():
    with pytest.raises(Exception, match="timer must be boolean"):
        GBBSolver(timer="yes")


def test_gbbsovler_filename_option_type():
    with pytest.raises(Exception, match="filename must be string"):
        GBBSolver(filename=123)


def test_procrustesproblem_raises_on_no_problemnumber_nor_matrices():
    with pytest.raises(Exception, match="no problem number has been chosen"):
        ProcrustesProblem((10, 10, 2, 2))


def test_gbbsovler_solve_errors_propagate(monkeypatch):
    # patch gbb_solver to raise, ensure solve propagates
    def fake_solver(*args, **kwargs):
        raise RuntimeError("solver failure")
    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_solver)
    m = n = 10
    p = q = 2
    A, B, C, Xsol = make_square_problem(m, n, p, q)
    problem = ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))
    solver = GBBSolver()
    with pytest.raises(RuntimeError, match="solver failure"):
        solver.solve(problem)
