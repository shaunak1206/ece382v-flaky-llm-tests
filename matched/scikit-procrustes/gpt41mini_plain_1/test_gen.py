import pytest
import numpy as np
from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, OptimizeResult
from numpy.testing import assert_array_equal, assert_allclose
import builtins
import sys


def test_procrustesproblem_default_problem_1():
    # Use default problem number 1, sizes compatible: m=n=10, p=q=2, m divisible by p
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)
    # Xsol should be orthogonal columns (Xsol.T Xsol = eye)
    XTX = np.dot(problem.Xsol.T, problem.Xsol)
    assert_allclose(XTX, np.eye(2), atol=1e-12)

def test_procrustesproblem_default_problem_2():
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=2)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)

def test_procrustesproblem_default_problem_3_valid_n50():
    problem = ProcrustesProblem((50, 50, 10, 10), problemnumber=3)
    assert problem.A.shape == (50, 50)
    assert problem.B.shape == (50, 10)
    assert problem.C.shape == (10, 10)
    assert problem.Xsol.shape == (50, 10)    

def test_procrustesproblem_default_problem_3_invalid_n_raises():
    with pytest.raises(Exception) as excinfo:
        ProcrustesProblem((51, 51, 10, 10), problemnumber=3)
    assert "requires n = 50" in str(excinfo.value)

def test_procrustesproblem_matrices_valid_shapes():
    m, n, p, q = 6, 6, 2, 2
    A = np.random.rand(m, n)
    B = np.random.rand(m, q)
    C = np.eye(p, q)
    X = np.eye(n, p)
    problem = ProcrustesProblem((m, n, p, q), matrices=[A, B, C, X])
    assert np.all(problem.A == A)
    assert np.all(problem.B == B)
    assert np.all(problem.C == C)
    assert np.all(problem.Xsol == X)

def test_procrustesproblem_matrices_invalid_less_than_3():
    m, n, p, q = 6, 6, 2, 2
    A = np.random.rand(m, n)
    B = np.random.rand(m, q)
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        ProcrustesProblem((m, n, p, q), matrices=[A, B])  # Only two matrices

def test_procrustesproblem_matrices_invalid_shapes():
    m, n, p, q = 6, 6, 2, 2
    A = np.random.rand(m, n)
    B = np.random.rand(m, q)
    C = np.eye(p, q)
    # A wrong shape for X
    X = np.zeros((n+1, p))
    with pytest.raises(Exception, match="Xsol must be"):
        ProcrustesProblem((m, n, p, q), matrices=[A, B, C, X])

@pytest.mark.parametrize("sizes", [ (10, 9, 2, 2), (10, 10, 2, 3), (9, 9, 2, 2) ])
def test_procrustesproblem_invalid_squares_raises(sizes):
    # sizes where m != n or p != q or m%p != 0
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, problemnumber=1)

def make_minimal_problem():
    m, n, p, q = 6, 6, 2, 2
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)
    # Provide some stats to allow solvers to run without full error
    problem.stats = {
        "nbiter": 1,
        "fev": 1,
        "total_fun": [1.0],
        "total_grad": [0.5],
        "total_crit": [0.5],
        "nfev": 1,
        "blocksteps": 1,
    }
    return problem

def test_gbbsolver_setoptions_default_and_custom():
    solver = GBBSolver()
    # Default options present & correct types
    assert isinstance(solver.options, dict)
    assert solver.options["full_results"] is False
    assert solver.options["xtol"] == 1e-6
    assert solver.options["projected_gradient"] == 1
    # Provide some invalid option types
    with pytest.raises(Exception, match="full_results must be a boolean"):
        GBBSolver(full_results=1)
    with pytest.raises(Exception, match="xtol must be a float"):
        GBBSolver(xtol="wrong")
    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        GBBSolver(verbose=5)
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        GBBSolver(projected_gradient=3)

def test_gbbsolver_open_close_file_stdout(monkeypatch):
    solver = GBBSolver()
    solver.options["filename"] = None
    solver.open_file()
    # file should be sys.stdout
    assert solver.file == sys.stdout
    solver.close_file()  # should do nothing without error

def test_gbbsolver_open_close_file_with_file(tmp_path):
    solver = GBBSolver(filename=str(tmp_path / "testfile.txt"))
    solver.open_file()
    assert solver.file.name.endswith("testfile.txt")
    solver.close_file()
    # File should exist after close
    assert (tmp_path / "testfile.txt").exists()

def test_gbbsolver_open_file_invalid_filename(monkeypatch):
    # Simulate OSError on file open and check sys.exit called
    solver = GBBSolver(filename="/this/path/does/not/exist/file.txt")
    # Patch sys.exit to raise SystemExit for pytest capturing
    called = {}
    def fake_exit(code):
        called['code'] = code
        raise SystemExit(code)
    monkeypatch.setattr(sys, "exit", fake_exit)
    # Patch print to suppress output during test
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        solver.open_file()
    assert called['code'] == -1

def test_gbbsolver_solve_returns_optimize_result(monkeypatch):
    problem = make_minimal_problem()
    solver = GBBSolver()
    # Patch gbb_solver function used in solve with a dummy function
    def dummy_gbb_solver(*args, **kwargs):
        return 0, 0.5, np.eye(problem.sizes[1], problem.sizes[2]), 0.1, 1, "success"
    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", dummy_gbb_solver)
    # Patch sp.qr to return orthonormal X0
    def dummy_qr(X0, mode='economic'):
        q, r = np.linalg.qr(X0)
        # Return q, r, but gbb_solver expects only X, temp = qr()
        return q, r
    monkeypatch.setattr("skprocrustes.skprocrustes.sp.qr", dummy_qr)
    res = solver.solve(problem)
    assert isinstance(res, OptimizeResult)
    # Should include keys and attributes
    assert "success" in res
    assert res.success is True
    assert hasattr(res, "solution")
    assert res.solution.shape == (problem.sizes[1], problem.sizes[2])
    # error should be finite and <= inf
    assert res.error <= np.inf

def test_optimize_result_set_and_get_attributes():
    optres = OptimizeResult()
    optres.success = True
    optres.message = "msg"
    assert optres.success is True
    assert optres.message == "msg"
    # __repr__ contains the keys and values
    reprstr = repr(optres)
    assert "success" in reprstr and "msg" in reprstr

def test_optimize_result_attribute_error_on_missing():
    optres = OptimizeResult()
    with pytest.raises(AttributeError):
        _ = optres.nonexistent

def test_compute_residual_and_optimality_basic_shapes():
    from skprocrustes.skprocrustes import compute_residual, optimality
    m, n, p, q = 6, 6, 2, 2
    A = np.eye(m, n)
    B = np.eye(m, q)
    C = np.eye(p, q)
    X = np.eye(n, p)
    R, residual = compute_residual(A, B, C, X, None)
    assert R.shape == (m, q)
    assert isinstance(residual, float)
    normproj, normgrad, grad = optimality(A, C, R, X, None)
    assert isinstance(normproj, float)
    assert isinstance(normgrad, float)
    assert grad.shape == (n, p)

def test_gbb_solver_invH_branch(monkeypatch):
    # Test gbb_solver branch where p < smalldim/2, forcing invH = False code path
    from skprocrustes.skprocrustes import gbb_solver
    m, n, p = 6, 6, 1  # p=1 < 6/2 == 3
    problem = ProcrustesProblem((m, n, p, p), problemnumber=1)
    problem.stats = {"nbiter":0, "fev":0, "grad":0}
    X = np.eye(n, p)
    options = {
        "tau": 1e-3,
        "xtol": 1e-6,
        "ftol": 1e-12,
        "gtol": 1e-6,
        "memory": 5,
        "full_results": False,
        "eta": 0.85,
        "etavar": False,
        "kappa": 0.1,
        "zeta": 1e-4,
        "verbose":0,
        "projected_gradient":1,
        "maxiter": 2,
        "filename": None,
        "timer": False,
    }
    # Run the gbb_solver, should finish due to maxiter=2 quickly
    exitcode, f, Xout, normG, nbiter, msg = gbb_solver(problem, m, n, X, problem.A, problem.B,
                                                       options, False, sys.stdout)
    assert exitcode in (0,1)
    assert isinstance(Xout, np.ndarray)

def test_blockbidiag_shapes_and_output():
    from skprocrustes.skprocrustes import blockbidiag
    m, n, p, q = 6, 6, 2, 2
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)
    U = np.zeros((m, m))
    V = np.zeros((n, n))
    T = np.zeros((m, n+q))
    steps = 1
    partial = 0
    halfreorth = False
    Uout, Vout, Tout, B1, reorth = blockbidiag(problem, U, V, T, steps, partial, halfreorth)
    assert Uout.shape == (m, m)
    assert Vout.shape == (n, n)
    assert Tout.shape == (m, n+q)
    assert B1.shape == (p, p)
    assert isinstance(reorth, int)

def test_bidiaggs_output_shapes_and_basic_properties():
    from skprocrustes.skprocrustes import bidiaggs
    m, n = 6, 6
    s = 2
    inds = 0
    # Create random prod of shape (m,s)
    prod = np.random.randn(m, s)
    # mat is random orthonormal-like (m,m)
    mat = np.eye(m)
    gstol = 1e-10
    reorth = 0
    mat_out, R, reorth = bidiaggs(inds, prod, mat, gstol, reorth)
    assert mat_out.shape == (m, m)
    assert R.shape == (s, s)
    assert isinstance(reorth, int)

def test_polar_newton_schultz_returns_correct_shape():
    from skprocrustes.skprocrustes import polar_newton_schultz
    m, n = 6, 4
    A = np.random.randn(m, n)
    tol_cgce = 1e-8
    U = polar_newton_schultz(A, tol_cgce)
    assert U.shape == (m, n)

def test_procrustesproblem_invalid_divisible():
    with pytest.raises(Exception, match="must be divisible by p"):
        ProcrustesProblem((9, 9, 4, 4), problemnumber=1)

def test_gbbsolver_setoptions_projected_gradient_values():
    # Projected gradient can only be 1 or 2
    with pytest.raises(Exception):
        GBBSolver(projected_gradient=0)
    with pytest.raises(Exception):
        GBBSolver(projected_gradient=3)
    # Acceptable values do not raise
    solver1 = GBBSolver(projected_gradient=1)
    solver2 = GBBSolver(projected_gradient=2)
    assert solver1.options["projected_gradient"] == 1
    assert solver2.options["projected_gradient"] == 2

def test_procrustesproblem_xsol_optional():
    # If no Xsol provided, the attribute may not exist
    m, n, p, q = 4, 4, 2, 2
    A = np.eye(m, n)
    B = np.eye(m, q)
    C = np.eye(p, q)
    problem = ProcrustesProblem((m, n, p, q), matrices=[A, B, C])
    # Xsol should not be present
    assert not hasattr(problem, "Xsol")

