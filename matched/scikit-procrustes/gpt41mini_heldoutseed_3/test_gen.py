import pytest
import numpy as np
from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, OptimizeResult, _status_message
from numpy.linalg import norm


def test_procrustes_problem_valid_default_problem():
    sizes = (6, 6, 2, 2)  # m=n=6, p=q=2, 6 divisible by 2
    problem = ProcrustesProblem(sizes, problemnumber=1)
    assert problem.sizes == sizes
    assert problem.A.shape == (6, 6)
    assert problem.B.shape == (6, 2)
    assert problem.C.shape == (2, 2)
    # Xsol should have shape (6, 2)
    assert problem.Xsol.shape == (6, 2)
    # Known solution Xsol is orthogonal: Xsol.T @ Xsol == I
    np.testing.assert_allclose(problem.Xsol.T @ problem.Xsol, np.eye(2), atol=1e-12)


def test_procrustes_problem_invalid_sizes_square_check():
    with pytest.raises(Exception, match="Currently only square problems are supported"):
        ProcrustesProblem((6, 5, 2, 2), problemnumber=1)
    with pytest.raises(Exception, match="Currently only square problems are supported"):
        ProcrustesProblem((6, 6, 2, 3), problemnumber=1)


def test_procrustes_problem_invalid_dimension_divisible():
    with pytest.raises(Exception, match="Dimensions are incompatible: m must be divisible by p"):
        ProcrustesProblem((7, 7, 3, 3), problemnumber=1)


def test_procrustes_problem_user_matrices_valid():
    m, n, p, q = 4, 4, 2, 2
    A = np.eye(m, n)
    B = np.ones((m, q))
    C = np.eye(p, q)
    Xsol = np.eye(n, p)
    matrices = (A, B, C, Xsol)
    problem = ProcrustesProblem((m, n, p, q), matrices=matrices)
    np.testing.assert_array_equal(problem.A, A)
    np.testing.assert_array_equal(problem.B, B)
    np.testing.assert_array_equal(problem.C, C)
    np.testing.assert_array_equal(problem.Xsol, Xsol)


def test_procrustes_problem_user_matrices_invalid_shapes():
    m, n, p, q = 4, 4, 2, 2
    A = np.eye(m, n)
    B = np.ones((m, q))
    C = np.eye(p, q)
    # Invalid Xsol shape:
    Xsol = np.eye(n, p + 1)
    with pytest.raises(Exception, match="Xsol must be \\(n,p\\)"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))

    # Not enough matrices
    with pytest.raises(Exception, match="matrices must be a list or tuple containing matrices A, B and C"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B))


def test_gbbsolver_default_options_and_setoptions():
    solver = GBBSolver()
    opts = solver.options
    # Check default types and values for some options:
    assert opts["full_results"] is False
    assert isinstance(opts["xtol"], float)
    assert isinstance(opts["gtol"], float)
    assert isinstance(opts["ftol"], float)
    assert isinstance(opts["kappa"], float)
    assert opts["projected_gradient"] in (1, 2)
    assert opts["verbose"] in (0, 1)
    assert opts["filename"] is None
    assert opts["timer"] is False


@pytest.mark.parametrize("opt_key, invalid_value, error_msg", [
    ("full_results", "yes", "full_results must be a boolean"),
    ("xtol", "small", "xtol must be a float"),
    ("gtol", "small", "gtol must be a float"),
    ("ftol", "tiny", "ftol must be a float"),
    ("zeta", "value", "zeta must be a float"),
    ("kappa", "factor", "kappa must be a float"),
    ("eta", "high", "eta must be a float"),
    ("etavar", "maybe", "etavar must be a boolean"),
    ("tau", "fast", "tau must be a float"),
    ("projected_gradient", 3, "projected_gradient must be 1 or 2"),
    ("maxiter", 3.5, "maxiter must be an integer"),
    ("verbose", 4, "verbose must be 0, or 1"),
    ("filename", 123, "filename must be string"),
    ("timer", "yes", "timer must be boolean"),
])
def test_gbbsolver_invalid_option_types(opt_key, invalid_value, error_msg):
    with pytest.raises(Exception, match=error_msg):
        GBBSolver(**{opt_key: invalid_value})


def test_gbbsolver_solve_returns_optimize_result(monkeypatch):
    # Prepare a simple problem
    sizes = (4, 4, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    
    # Patch gbb_solver to avoid complex internal computation and just return dummy output
    def dummy_gbb_solver(problem_, largedim, smalldim, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        exitcode = 0
        fval = 1.234
        # Make X orthogonal of correct size (smalldim, p)
        Xshape = X.shape
        # Generate orthogonal X approx
        Xnew = np.eye(Xshape[0], Xshape[1])
        normG = 0.01
        nbiter = 2
        msg = _status_message['success']
        return exitcode, fval, Xnew, normG, nbiter, msg

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", dummy_gbb_solver)

    solver = GBBSolver(full_results=True)
    # Add dummy stats needed when full_results True
    problem.stats["nbiter"] = 2
    problem.stats["fev"] = 4
    problem.stats["total_fun"] = [1.5, 1.3]
    problem.stats["total_crit"] = [0.1, 0.05]
    result = solver.solve(problem)

    # Check OptimizeResult attributes
    assert isinstance(result, OptimizeResult)
    assert hasattr(result, "success")
    assert hasattr(result, "status")
    assert hasattr(result, "message")
    assert hasattr(result, "solution")
    assert hasattr(result, "fun")
    assert hasattr(result, "error")
    assert hasattr(result, "cpu")
    assert hasattr(result, "nbiter")
    assert hasattr(result, "nfev")
    assert hasattr(result, "total_fun")
    assert hasattr(result, "total_crit")

    assert result.success is True
    assert result.status == 0
    assert isinstance(result.solution, np.ndarray)
    assert result.solution.shape == (problem.sizes[1], problem.sizes[2])
    assert np.allclose(result.fun, 1.234)
    assert np.allclose(result.total_fun, problem.stats["total_fun"])
    assert np.allclose(result.total_crit, problem.stats["total_crit"])
    assert result.nbiter == 2
    assert result.nfev == 4
    assert "successfully" in result.message.lower()

def test_gbbsolver_solve_error_if_full_results_without_stats(monkeypatch):
    sizes = (4, 4, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    # Patch gbb_solver as before
    def dummy_gbb_solver(*args, **kwargs):
        return 0, 0.0, np.eye(2,2), 0, 0, _status_message['success']

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", dummy_gbb_solver)
    solver = GBBSolver(full_results=True)
    # No total_fun or total_crit in problem.stats
    problem.stats.clear()
    problem.stats["nbiter"] = 0
    problem.stats["fev"] = 0

    with pytest.raises(Exception, match="For full results, set problem.stats\\[\"total_fun\"\\] and problem.stats\\[\"total_crit\"\\]"):
        solver.solve(problem)


def test_gbbsolver_open_close_file_stdout(monkeypatch):
    solver = GBBSolver()
    solver.options["filename"] = None
    solver.open_file()
    import sys
    # file should be sys.stdout
    assert solver.file == sys.stdout
    solver.close_file()  # Should not error


def test_gbbsolver_open_file_with_invalid_filename(monkeypatch):
    # This test triggers the OSError condition and sys.exit(-1)
    solver = GBBSolver()
    solver.options["filename"] = "/invalid/path/filename.txt"

    # Patch sys.exit and file open to simulate error
    exited = {}
    def fake_exit(code):
        exited["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", fake_exit)
    with pytest.raises(SystemExit):
        solver.open_file()

    assert exited.get("code") == -1


def test_optimize_result_repr_and_attr_access():
    d = {"success": True, "fun": 10}
    res = OptimizeResult(d)
    # repr includes keys and values
    rep = repr(res)
    assert "success" in rep and "fun" in rep
    # attribute access
    assert res.success is True
    assert res.fun == 10
    with pytest.raises(AttributeError):
        _ = res.unknown_attribute


def test_compare_solvers_basic(monkeypatch):
    # Mock solvers with solve method returning OptimizeResult with total_fun
    class DummySolver:
        def __init__(self, solvername):
            self.solvername = solvername

        def solve(self, problem):
            problem.stats["nbiter"] = 1
            problem.stats["fev"] = 1
            return OptimizeResult({
                "total_fun": [1.0, 0.5],
                "solution": np.eye(2),
                "success": True,
                "status": 0,
                "message": "OK"
            })

    from skprocrustes.skprocrustes import compare_solvers
    problem = ProcrustesProblem((4,4,2,2), problemnumber=1)

    solver_spg = DummySolver("spg")
    solver_eb = DummySolver("eb")
    solver_gpi = DummySolver("gpi")
    solver_gkb = DummySolver("gkb")

    results = compare_solvers(problem, solver_spg, solver_eb, solver_gpi, solver_gkb, plot=False)
    assert len(results) == 4
    for r in results:
        assert isinstance(r, OptimizeResult)
        assert hasattr(r, "total_fun")


def test_compute_residual_and_optimality_consistency():
    from skprocrustes.skprocrustes import compute_residual, optimality
    m, n, p, q = 4, 4, 2, 2
    A = np.eye(m, n)
    B = 2*np.ones((m, q))
    C = np.eye(p, q)
    X = np.ones((n, p))
    R, residual = compute_residual(A, B, C, X, None)
    assert R.shape == (m, q)
    assert np.isclose(residual, norm(R, 'fro')**2)

    precond = 0.5
    R2, residual2 = compute_residual(A, B, C, X, precond)
    assert np.isclose(residual2, (precond**2)*norm(R2, 'fro')**2)

    normgradproj, normgrad, grad = optimality(A, C, R, X, None)
    assert isinstance(normgrad, float)
    assert isinstance(normgradproj, float)
    assert grad.shape == A.T.shape[0], or grad.shape == A.T.shape[0] or grad.shape == (n, p)  # grad shape matches X shape
    assert grad.shape == X.shape

    normgradproj_p, normgrad_p, grad_p = optimality(A, C, R, X, precond)
    assert normgrad_p >= 0
    assert grad_p.shape == grad.shape


def test_blockbidiag_output_shapes_and_reorth():
    from skprocrustes.skprocrustes import blockbidiag
    m, n, p, q = 6, 6, 2, 2
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)
    U = np.zeros((m, m))
    V = np.zeros((n, n))
    T = np.zeros((m, n+q))
    steps = 2
    partial = 0
    halfreorth = False

    U_out, V_out, T_out, B1, reorth = blockbidiag(problem, U.copy(), V.copy(), T.copy(), steps, partial, halfreorth)
    assert U_out.shape == (m, m)
    assert V_out.shape == (n, n)
    assert T_out.shape == (m, n+q)
    assert B1.shape == (p, p)
    assert isinstance(reorth, int)


def test_bidiaggs_reorthogonalization_effects():
    from skprocrustes.skprocrustes import bidiaggs
    m, n, p, q = 6, 6, 2, 2
    problem = ProcrustesProblem((m,n,p,q), problemnumber=1)
    mat = np.zeros((m,m))
    prod = np.random.randn(m, p)
    gstol = 1e-10
    reorth = 0
    mat[:, :p] = np.linalg.qr(np.random.randn(m,p), mode='economic')[0]

    mat_new, R, reorth_new = bidiaggs(p, prod.copy(), mat, gstol, reorth)
    assert mat_new.shape == (m,m)
    assert R.shape == (p,p)
    assert isinstance(reorth_new, int)
    assert reorth_new >= reorth


def test_polardecomp_returns_ns(monkeypatch):
    from skprocrustes.skprocrustes import polardecomp

    W = np.random.randn(4,2)

    # Should run without error
    options = {"polar":"ns"}
    U = polardecomp(W, options)
    assert U.shape == W.shape


def test_polar_newton_schultz_small_matrix():
    from skprocrustes.skprocrustes import polar_newton_schultz
    A = np.random.randn(6,3)
    U = polar_newton_schultz(A, 1e-5)
    assert U.shape == A.shape
    # If m < n, error string is raised
    A2 = np.random.randn(3,6)
    with pytest.raises(Exception):
        polar_newton_schultz(A2, 1e-5)
