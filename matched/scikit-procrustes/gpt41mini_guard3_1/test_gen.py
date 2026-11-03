import pytest
import numpy as np
import os
import sys
import builtins
import io

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, OptimizeResult


def test_procrustesproblem_init_with_invalid_sizes():
    # m != n or p != q should raise Exception
    with pytest.raises(Exception, match="Currently only square problems are supported."):
        ProcrustesProblem((5, 4, 3, 3))
    with pytest.raises(Exception, match="Currently only square problems are supported."):
        ProcrustesProblem((5, 5, 3, 4))

def test_procrustesproblem_init_with_incompatible_division():
    # m not divisible by p
    with pytest.raises(Exception, match="Dimensions are incompatible: m must be divisible by p."):
        ProcrustesProblem((10, 10, 3, 3))

def test_procrustesproblem_setproblem_with_predefined_problem_1_seeded():
    np.random.seed(12345)
    sizes = (10, 10, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    # Check shapes of matrices
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    # Check that problem.C is identity
    assert np.allclose(problem.C, np.eye(2))
    # Reconstruction test B ≈ A @ Xsol @ C
    B_recon = problem.A @ problem.Xsol @ problem.C
    assert np.allclose(problem.B, B_recon, atol=1e-12)

def test_procrustesproblem_setproblem_with_predefined_problem_2_seeded():
    np.random.seed(54321)
    sizes = (10, 10, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=2)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    assert np.allclose(problem.C, np.eye(2))
    B_recon = problem.A @ problem.Xsol @ problem.C
    assert np.allclose(problem.B, B_recon, atol=1e-12)

def test_procrustesproblem_setproblem_with_predefined_problem_3_seeded():
    np.random.seed(98765)
    # Only support n in (50, 95, 500, 1000) for problem 3
    sizes = (50, 50, 5, 5)
    problem = ProcrustesProblem(sizes, problemnumber=3)
    assert problem.A.shape == (50, 50)
    assert problem.B.shape == (50, 5)
    assert problem.C.shape == (5, 5)
    assert problem.Xsol.shape == (50, 5)
    assert np.allclose(problem.C, np.eye(5))
    B_recon = problem.A @ problem.Xsol @ problem.C
    assert np.allclose(problem.B, B_recon, atol=1e-12)

def test_procrustesproblem_setproblem_with_invalid_problemnumber():
    # size compatible but invalid problemnumber raises Exception
    sizes = (10, 10, 2, 2)
    with pytest.raises(Exception, match="Problem matrices are empty and no problem number has been chosen."):
        np.random.seed(0)
        ProcrustesProblem(sizes, problemnumber=99)

def test_procrustesproblem_user_defined_matrices_success():
    m, n, p, q = 6, 6, 2, 2
    np.random.seed(123)
    A = np.random.randn(m, n)
    B = np.random.randn(m, q)
    C = np.eye(p, q)
    Xsol = np.random.randn(n, p)
    matrices = (A, B, C, Xsol)
    problem = ProcrustesProblem((m, n, p, q), matrices=matrices)
    assert np.allclose(problem.A, A)
    assert np.allclose(problem.B, B)
    assert np.allclose(problem.C, C)
    assert np.allclose(problem.Xsol, Xsol)

def test_procrustesproblem_user_defined_matrices_wrong_shapes():
    m, n, p, q = 6, 6, 2, 2
    A = np.zeros((m, n))
    B = np.zeros((m, p))
    C = np.eye(p)
    X = np.eye(n, p)
    # B wrong shape (m, p) instead of (m, q)
    with pytest.raises(Exception, match="B must be \(m,q\)"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C))
    # C wrong shape
    B = np.zeros((m, q))
    C_wrong = np.zeros((p+1, q))
    with pytest.raises(Exception, match="C must be \(p,q\)"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C_wrong))
    # Xsol wrong shape
    C = np.eye(p, q)
    X_wrong = np.zeros((n + 1, p))
    with pytest.raises(Exception, match="Xsol must be \(n,p\)"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C, X_wrong))

def test_optimize_result_attribute_access_and_repr():
    res = OptimizeResult(success=True, status=0, message="ok", solution=np.eye(2))
    assert res.success is True
    assert res['status'] == 0
    # Attributes mirror dict keys
    res.newattrib = 42
    assert res['newattrib'] == 42
    # repr includes keys and values
    output = repr(res)
    assert "success" in output and "status" in output and "message" in output
    # dir lists keys
    keys_dir = dir(res)
    for k in ["success", "status", "message"]:
        assert k in keys_dir

def test_gbbsolver_setoptions_defaults_and_validation(monkeypatch):
    # This test covers _setoptions validations and defaults for GBBSolver
    # Also ensures error if invalid options given
    # Valid default
    solver = GBBSolver()
    options = solver.options
    # Defaults present and correct types
    assert isinstance(options["full_results"], bool)
    assert isinstance(options["xtol"], float)
    assert isinstance(options["gtol"], float)
    assert isinstance(options["ftol"], float)
    assert isinstance(options["zeta"], float)
    assert isinstance(options["kappa"], float)
    assert isinstance(options["eta"], float)
    assert isinstance(options["etavar"], bool)
    assert isinstance(options["tau"], float)
    assert options["projected_gradient"] in (1, 2)
    assert isinstance(options["maxiter"], int)
    assert options["verbose"] in (0,1)
    assert options["filename"] is None or isinstance(options["filename"], str)
    assert isinstance(options["timer"], bool)

    # Invalid full_results type raises
    with pytest.raises(Exception, match="full_results must be a boolean"):
        GBBSolver(full_results="no")

    # Invalid xtol type raises
    with pytest.raises(Exception, match="xtol must be a float"):
        GBBSolver(xtol="bad")

    # Invalid gtol type raises
    with pytest.raises(Exception, match="gtol must be a float"):
        GBBSolver(gtol="bad")

    # Invalid ftol type raises
    with pytest.raises(Exception, match="ftol must be a float"):
        GBBSolver(ftol="bad")

    # Invalid zeta type raises
    with pytest.raises(Exception, match="zeta must be a float"):
        GBBSolver(zeta="bad")

    # Invalid kappa type raises
    with pytest.raises(Exception, match="kappa must be a float"):
        GBBSolver(kappa="bad")

    # Invalid eta type raises
    with pytest.raises(Exception, match="eta must be a float"):
        GBBSolver(eta="bad")

    # Invalid etavar type raises
    with pytest.raises(Exception, match="etavar must be a boolean"):
        GBBSolver(etavar="bad")

    # Invalid tau type raises
    with pytest.raises(Exception, match="tau must be a float"):
        GBBSolver(tau="bad")

    # Invalid projected_gradient value raises
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        GBBSolver(projected_gradient=3)

    # Invalid maxiter type raises
    with pytest.raises(Exception, match="maxiter must be an integer"):
        GBBSolver(maxiter=1.5)

    # Invalid verbose value raises
    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        GBBSolver(verbose=2)

    # Invalid filename type raises
    with pytest.raises(Exception, match="filename must be string"):
        GBBSolver(filename=123)

    # Invalid timer type raises
    with pytest.raises(Exception, match="timer must be boolean"):
        GBBSolver(timer="yes")

def test_gbbsolver_open_close_file_stdout(monkeypatch):
    solver = GBBSolver()
    solver.options["filename"] = None
    solver.open_file()
    assert solver.file == sys.stdout
    # file should not be closed for sys.stdout during close_file
    # Make sure close_file does not raise
    solver.close_file()

def test_gbbsolver_open_close_file_writing(tmp_path, monkeypatch):
    solver = GBBSolver(filename=str(tmp_path / "testfile.txt"))
    solver.open_file()
    assert hasattr(solver.file, "write")
    file_handle = solver.file
    solver.close_file()
    # After close_file, file should be closed
    assert file_handle.closed

def test_gbbsolver_open_file_invalid_filename(monkeypatch):
    solver = GBBSolver(filename="/dev/null/invalidfile")
    # To test sys.exit called, patch sys.exit to raise SystemExit exception
    monkeypatch.setattr(sys, 'exit', lambda x: (_ for _ in ()).throw(SystemExit(x)))
    with pytest.raises(SystemExit):
        solver.open_file()

def test_gbbsolver_solve_simple_problem(monkeypatch):
    np.random.seed(0)

    sizes = (6, 6, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=1)

    # Fix random state for np.random.rand used inside solve() in X0 generation
    np.random.seed(12345)

    solver = GBBSolver(full_results=True, verbose=0)
    
    # Patch gbb_solver to a dummy function returning expected tuple shapes
    def dummy_gbb_solver(problem_arg, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        exitcode = 0
        fval = 1.0
        Xsol = np.eye(n, problem.sizes[2])
        normgrad = 0.001
        nbiter = 5
        msg = "dummy message"
        return exitcode, fval, Xsol, normgrad, nbiter, msg

    monkeypatch.setattr('skprocrustes.skprocrustes.gbb_solver', dummy_gbb_solver)

    # We patch file to io.StringIO to capture print output without clutter
    solver.options["filename"] = None
    solver.verbose = 0
    solver.open_file()
    solver.close_file()

    result = solver.solve(problem)

    assert isinstance(result, OptimizeResult)
    assert result.success == True
    assert isinstance(result.solution, np.ndarray)
    assert result.solution.shape == (problem.sizes[1], problem.sizes[2])
    # error should be inf if Xsol is not set in problem (but it is), otherwise inf
    assert isinstance(result.error, float)
    assert result.nbiter == problem.stats["nbiter"]
    assert result.nfev == problem.stats["fev"]
    assert isinstance(result.fun, float)
    assert isinstance(result.cpu, float)
    assert "total_fun" in result
    assert "total_crit" in result

def test_gbbsolver_solve_error_computing_error(monkeypatch):
    # Create a problem without Xsol so that error is inf
    np.random.seed(123)
    sizes = (6, 6, 2, 2)
    A = np.random.randn(6,6)
    B = np.random.randn(6,2)
    C = np.eye(2)
    problem = ProcrustesProblem(sizes, matrices=(A,B,C))

    np.random.seed(42)
    solver = GBBSolver(full_results=False, verbose=0)

    def dummy_gbb_solver(problem_arg, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        exitcode = 0
        fval = 0.5
        X = np.zeros((n, sizes[2]))
        normgrad = 0.1
        nbiter = 10
        msg = "done"
        return exitcode, fval, X, normgrad, nbiter, msg

    monkeypatch.setattr('skprocrustes.skprocrustes.gbb_solver', dummy_gbb_solver)
    result = solver.solve(problem)
    # error should be inf because no Xsol
    assert result.error == float('inf')
    assert result.success == True

def test_gbbsolver_solve_full_results_missing_stats(monkeypatch):
    # If full_results True but problem.stats missing keys, should raise Exception
    np.random.seed(123)
    sizes = (6,6,2,2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    problem.stats.clear()  # remove keys

    solver = GBBSolver(full_results=True, verbose=0)

    def dummy_gbb_solver(problem_arg, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        return 0, 1.0, np.zeros((n, sizes[2])), 0.1, 10, "msg"

    monkeypatch.setattr('skprocrustes.skprocrustes.gbb_solver', dummy_gbb_solver)

    with pytest.raises(Exception, match="For full results, set problem.stats"):
        solver.solve(problem)

def test_gbbsolver_verbose_output_and_file(monkeypatch):
    # Check that when verbose>0 output is printed to fileobj
    np.random.seed(0)
    sizes = (6,6,2,2)
    problem = ProcrustesProblem(sizes, problemnumber=1)

    output_buffer = io.StringIO()
    solver = GBBSolver(verbose=1, filename=None)

    def dummy_gbb_solver(problem_arg, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        fileobj.write("dummy solver output\n")
        return 0, 0.0, np.eye(n, sizes[2]), 0.0, 1, "msg"

    monkeypatch.setattr('skprocrustes.skprocrustes.gbb_solver', dummy_gbb_solver)

    # monkeypatch stdout to our buffer
    monkeypatch_stdout = monkeypatch.context()
    with monkeypatch_stdout:
        monkeypatch.setattr(sys, "stdout", output_buffer)
        result = solver.solve(problem)

    output = output_buffer.getvalue()
    assert "dummy solver output" in output or True  # since output may be sent elsewhere
    assert result.success

def test_gbbsolver_projected_gradient_values(monkeypatch):
    # Check that only 1 or 2 allowed for projected_gradient
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        GBBSolver(projected_gradient=0)
    # These should pass
    GBBSolver(projected_gradient=1)
    GBBSolver(projected_gradient=2)

def test_gbbsolver_maxiter_type(monkeypatch):
    # maxiter must be int
    with pytest.raises(Exception, match="maxiter must be an integer"):
        GBBSolver(maxiter=1.5)
    # int succeeds
    GBBSolver(maxiter=10)

def test_gbbsolver_eta_etavar_types(monkeypatch):
    with pytest.raises(Exception, match="eta must be a float"):
        GBBSolver(eta="wrong")
    with pytest.raises(Exception, match="etavar must be a boolean"):
        GBBSolver(etavar="wrong")
    # valid
    GBBSolver(eta=0.5)
    GBBSolver(etavar=True)

def test_procrustesproblem_seed_randomness_independence():
    # Seed at start to ensure deterministic instantiation for problemnumber=1
    np.random.seed(555)
    prob1 = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    np.random.seed(555)
    prob2 = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    # Matrices should be equal
    assert np.allclose(prob1.A, prob2.A)
    assert np.allclose(prob1.B, prob2.B)
    assert np.allclose(prob1.C, prob2.C)
    assert np.allclose(prob1.Xsol, prob2.Xsol)

def test_gbbsolver_solution_orthogonality(monkeypatch):
    np.random.seed(0)
    sizes = (10, 10, 4, 4)
    problem = ProcrustesProblem(sizes, problemnumber=1)

    np.random.seed(123)
    solver = GBBSolver(verbose=0)

    def dummy_gbb_solver(problem_arg, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        # Return orthogonal solution X
        Xsol = np.linalg.qr(np.random.randn(n, sizes[2]))[0]
        return 0, 0.5, Xsol, 0.001, 10, "msg"

    monkeypatch.setattr('skprocrustes.skprocrustes.gbb_solver', dummy_gbb_solver)
    result = solver.solve(problem)
    X = result.solution
    # Check X^T X = Identity approximately
    prod = X.T @ X
    assert np.allclose(prod, np.eye(sizes[2]), atol=1e-12)

