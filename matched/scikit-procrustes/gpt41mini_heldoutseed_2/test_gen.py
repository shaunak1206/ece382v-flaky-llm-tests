import numpy as np
import pytest
import os
import sys
from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, OptimizeResult


def make_default_problem():
    # Use problemnumber=1 with sizes divisible accordingly (m must be divisible by p)
    m, n, p, q = 8, 8, 2, 2
    return ProcrustesProblem((m, n, p, q), problemnumber=1)


def make_user_problem():
    # Create consistent matrices for user defined problem
    m, n, p, q = 6, 6, 3, 3
    A = np.random.randn(m, n)
    B = np.random.randn(m, q)
    C = np.eye(p, q)
    Xsol = np.eye(n, p)
    return ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))


def test_procrustes_problem_default_instantiation():
    problem = make_default_problem()
    assert problem.A.shape == (8, 8)
    assert problem.B.shape == (8, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (8, 2)


def test_procrustes_problem_user_defined_correct_shapes():
    problem = make_user_problem()
    assert problem.A.shape == (6, 6)
    assert problem.B.shape == (6, 3)
    assert problem.C.shape == (3, 3)
    assert problem.Xsol.shape == (6, 3)


def test_procrustes_problem_user_defined_invalid_shapes():
    m, n, p, q = 4, 4, 2, 2
    A = np.random.randn(m, n + 1)  # wrong shape
    B = np.random.randn(m, q)
    C = np.eye(p, q)

    with pytest.raises(Exception, match="A must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C))

    # Correct A, wrong B
    A = np.random.randn(m, n)
    B = np.random.randn(m + 1, q)

    with pytest.raises(Exception, match="B must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C))

    # Correct B, wrong C
    B = np.random.randn(m, q)
    C = np.eye(p + 1, q)
    with pytest.raises(Exception, match="C must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C))

    # Xsol wrong shape
    C = np.eye(p, q)
    Xsol = np.eye(n + 1, p)
    with pytest.raises(Exception, match="Xsol must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))


def test_procrustes_problem_invalid_sizes_non_square():
    with pytest.raises(Exception, match="Currently only square problems are supported"):
        ProcrustesProblem((5, 4, 2, 2))

    with pytest.raises(Exception, match="Currently only square problems are supported"):
        ProcrustesProblem((6, 6, 3, 2))


def test_procrustes_problem_invalid_sizes_not_divisible():
    with pytest.raises(Exception, match="Dimensions are incompatible: m must be divisible by p"):
        ProcrustesProblem((7, 7, 3, 3))


def test_gbbsolver_default_options_and_solve(monkeypatch):
    problem = make_default_problem()
    solver = GBBSolver()

    # The solve calls gbb_solver internally, patch it to a dummy function
    def fake_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0):
        # Return expected tuple based on code: exitcode, f, X, normG, nbiter, msg
        Xout = np.eye(n, problem.sizes[2])
        return 0, 0.0, Xout, 0.0, 1, "success"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_gbb_solver)

    result = solver.solve(problem)
    assert isinstance(result, OptimizeResult)
    assert result.success
    assert result.status == 0
    assert np.allclose(result.solution.T @ result.solution, np.eye(result.solution.shape[1]), atol=1e-7)
    assert result.nbiter == 1
    assert isinstance(result.fun, float)


def test_gbbsolver_options_validation():
    # Only test a few invalid option cases for coverage
    # full_results must be bool
    with pytest.raises(Exception, match="full_results must be a boolean"):
        GBBSolver(full_results="yes")

    with pytest.raises(Exception, match="xtol must be a float"):
        GBBSolver(xtol="small")

    with pytest.raises(Exception, match="gtol must be a float"):
        GBBSolver(gtol="tiny")

    with pytest.raises(Exception, match="ftol must be a float"):
        GBBSolver(ftol="tiny")

    with pytest.raises(Exception, match="zeta must be a float"):
        GBBSolver(zeta="small")

    with pytest.raises(Exception, match="kappa must be a float"):
        GBBSolver(kappa="value")

    with pytest.raises(Exception, match="eta must be a float"):
        GBBSolver(eta="mid")

    with pytest.raises(Exception, match="etavar must be a boolean"):
        GBBSolver(etavar="False")

    with pytest.raises(Exception, match="tau must be a float"):
        GBBSolver(tau="val")

    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        GBBSolver(projected_gradient=3)

    with pytest.raises(Exception, match="maxiter must be an integer"):
        GBBSolver(maxiter=9.5)

    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        GBBSolver(verbose=4)

    with pytest.raises(Exception, match="filename must be string"):
        GBBSolver(filename=123)

    with pytest.raises(Exception, match="timer must be boolean"):
        GBBSolver(timer="yes")


def test_gbbsolver_with_custom_options_and_monkeypatch(monkeypatch):
    problem = make_default_problem()

    # Patch gbb_solver to return a fixed result for us
    def fake_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0):
        Xout = np.eye(n, problem.sizes[2]) * 2
        return 0, 123.456, Xout, 0.001, 5, "custom success"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_gbb_solver)

    solver = GBBSolver(full_results=True, xtol=1e-5, maxiter=10, verbose=0)
    problem.stats["nbiter"] = 0
    problem.stats["fev"] = 0

    result = solver.solve(problem)
    assert result.success
    assert result.fun == 123.456
    assert result.message == "custom success"
    assert result.nbiter == 0  # as we patched, nbiter remains same; could be inaccurate
    assert "total_fun" not in result  # no total_fun set for GBBSolver by default


def test_procrustes_problem_problemnumber_3_allowed_n(monkeypatch):
    # This is to test allowed n values for problemnumber=3
    for n in (50, 95, 500, 1000):
        m = n
        p = 10
        q = 10
        # Patch np.random.rand to fixed values so that test is reliable
        monkeypatch.setattr("numpy.random.rand", lambda *args, **kwargs: np.zeros(args))

        problem = ProcrustesProblem((m, n, p, q), problemnumber=3)
        assert problem.A.shape == (m, n)
        assert problem.B.shape == (m, q)
        assert problem.C.shape == (p, q)
        # Check that Sigma diagonal is non-negative
        sigma_diag = np.linalg.svd(problem.A, compute_uv=False)
        assert np.all(sigma_diag >= 0)


def test_procrustes_problem_problemnumber_3_invalid_n():
    with pytest.raises(Exception, match="Problem 6 requires n = 50,"):
        ProcrustesProblem((7, 7, 2, 2), problemnumber=3)


def test_optimize_result_repr_and_dir():
    res = OptimizeResult()
    assert "OptimizeResult()" in repr(res)
    res['success'] = True
    res['status'] = 0
    assert 'success' in dir(res)
    assert repr(res).count('success')
    # attribute access works
    assert res.success is True
    with pytest.raises(AttributeError):
        _ = res.nonexistent_attr


def test_optimize_result_show(capsys):
    res = OptimizeResult(success=True, status=0, message="ok")
    res.show()
    captured = capsys.readouterr()
    assert "Summary:" in captured.out
    assert "success" in captured.out


def test_gbbsolver_file_handling(tmp_path):
    # Create a file path that does not exist to test open_file and close_file
    filename = tmp_path / "testfile.txt"
    solver = GBBSolver(filename=str(filename))
    problem = make_default_problem()

    # Patch gbb_solver to bypass actual call
    def fake_gbb_solver(*args, **kwargs):
        Xout = np.eye(problem.sizes[1], problem.sizes[2])
        return 0, 0.0, Xout, 0.0, 0, "fake"

    import skprocrustes.skprocrustes as skp
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    result = solver.solve(problem)
    assert os.path.exists(filename)
    assert solver.file.closed
    monkeypatch.undo()


def test_gbbsolver_file_invalid_filename(monkeypatch):
    solver = GBBSolver(filename="<>invalid_filename")

    # Override sys.exit to test exit call without stopping test
    called = {}

    def fake_exit(code):
        called['exit'] = code
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        solver.open_file()
    assert called['exit'] == -1


def test_gbbsolver_projected_gradient_values():
    # Values 1 and 2 allowed
    solver1 = GBBSolver(projected_gradient=1)
    assert solver1.options["projected_gradient"] == 1
    solver2 = GBBSolver(projected_gradient=2)
    assert solver2.options["projected_gradient"] == 2
    # Default is 1
    solver3 = GBBSolver()
    assert solver3.options["projected_gradient"] == 1


def test_gbbsolver_maxiter_and_verbose_defaults():
    solver = GBBSolver()
    assert solver.options["maxiter"] == 5000
    assert solver.options["verbose"] == 1


def test_gbbsolver_options_accept_valid_values():
    # No exceptions expected for valid inputs
    solver = GBBSolver(
        full_results=True,
        xtol=1e-7,
        gtol=1e-5,
        ftol=1e-14,
        zeta=1e-5,
        kappa=0.2,
        eta=0.9,
        etavar=True,
        tau=5e-4,
        projected_gradient=2,
        maxiter=100,
        verbose=0,
        filename=None,
        timer=True
    )
    opts = solver.options
    assert opts["full_results"] is True
    assert isinstance(opts["xtol"], float)
    assert opts["projected_gradient"] == 2
    assert opts["verbose"] == 0
    assert opts["timer"] is True


def test_gbbsolver_solution_orthogonality(monkeypatch):
    # Test that solution returned is approximately orthogonal

    problem = make_default_problem()

    # Patch the gbb_solver to return an orthogonal matrix
    def fake_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0):
        # Create an orthonormal matrix of size n by p
        n, p = X.shape
        Q, _ = np.linalg.qr(np.random.randn(n, p))
        return 0, 0.0, Q, 0.0, 10, "success"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_gbb_solver)

    solver = GBBSolver()
    result = solver.solve(problem)
    sol = result.solution
    # Check orthogonality: sol^T sol ≈ I
    assert sol.shape[0] == problem.sizes[1]
    assert sol.shape[1] == problem.sizes[2]
    assert np.allclose(np.dot(sol.T, sol), np.eye(sol.shape[1]), atol=1e-7)


def test_gbbsolver_with_full_results_and_stats(monkeypatch):
    problem = make_default_problem()
    problem.stats["total_fun"] = []
    problem.stats["total_crit"] = []

    def fake_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0):
        return 0, 10.0, np.eye(n, problem.sizes[2]), 0.0, 2, "success"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_gbb_solver)
    solver = GBBSolver(full_results=True)
    result = solver.solve(problem)

    assert isinstance(result, OptimizeResult)
    assert hasattr(result, "total_fun")
    assert hasattr(result, "total_crit")
    assert result.fun == 10.0
    assert result.message == "success"


def test_optimize_result_keys_and_attribute_behavior():
    res = OptimizeResult()
    res["test_key"] = 123
    assert res.test_key == 123
    res.test_key = 456
    assert res["test_key"] == 456
    del res.test_key
    with pytest.raises(AttributeError):
        _ = res.test_key


# Additional tests for coverage of open_file and close_file with None filename (stdout)
def test_gbbsolver_open_close_file_stdout():
    solver = GBBSolver(filename=None)
    solver.open_file()
    assert solver.file is sys.stdout
    solver.close_file()
    # No exception raised


def test_gbbsolver_open_close_file_with_filename(tmp_path):
    filename = tmp_path / "output.log"
    solver = GBBSolver(filename=str(filename))
    solver.open_file()
    assert not solver.file.closed
    solver.close_file()
    assert solver.file.closed
    assert os.path.exists(filename)


# Check string representation of solvername
def test_solvername_attribute():
    solver = GBBSolver()
    assert solver.solvername == "gbb"

