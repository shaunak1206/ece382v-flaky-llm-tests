import numpy as np
import pytest
import numpy.linalg as la

import skprocrustes.skprocrustes as skp


def test_procrustesproblem_init_predefined_problem1():
    sizes = (10, 10, 2, 2)
    problem = skp.ProcrustesProblem(sizes, problemnumber=1)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)


def test_procrustesproblem_init_predefined_problem2():
    sizes = (10, 10, 5, 5)
    problem = skp.ProcrustesProblem(sizes, problemnumber=2)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 5)
    assert problem.C.shape == (5, 5)
    assert problem.Xsol.shape == (10, 5)


def test_procrustesproblem_init_predefined_problem3_n50():
    sizes = (50, 50, 10, 10)
    problem = skp.ProcrustesProblem(sizes, problemnumber=3)
    assert problem.A.shape == (50, 50)
    assert problem.B.shape == (50, 10)
    assert problem.C.shape == (10, 10)
    assert problem.Xsol.shape == (50, 10)


def test_procrustesproblem_init_raises_invalid_sizes():
    # sizes[0] != sizes[1]
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 9, 2, 2))
    # sizes[2] != sizes[3]
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 10, 2, 3))
    # m not divisible by p
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 10, 3, 3))


def test_procrustesproblem_init_with_user_matrices():
    m, n, p, q = (6, 6, 3, 3)
    A = np.eye(m, n)
    B = np.ones((m, q))
    C = np.eye(p, q)
    X = np.eye(n, p)
    problem = skp.ProcrustesProblem((m, n, p, q), matrices=[A, B, C, X])
    np.testing.assert_array_equal(problem.A, A)
    np.testing.assert_array_equal(problem.B, B)
    np.testing.assert_array_equal(problem.C, C)
    np.testing.assert_array_equal(problem.Xsol, X)


def test_procrustesproblem_init_user_matrices_invalid_shapes():
    m, n, p, q = (6, 6, 3, 3)
    A = np.eye(m, n)
    B = np.ones((m, q))
    C = np.eye(p, q)
    X = np.eye(n, p)

    # Wrong A shape
    wrongA = np.eye(m+1, n)
    with pytest.raises(Exception):
        skp.ProcrustesProblem((m, n, p, q), matrices=[wrongA, B, C])

    # Wrong B shape
    wrongB = np.ones((m, q+1))
    with pytest.raises(Exception):
        skp.ProcrustesProblem((m, n, p, q), matrices=[A, wrongB, C])

    # Wrong C shape
    wrongC = np.eye(p+1, q)
    with pytest.raises(Exception):
        skp.ProcrustesProblem((m, n, p, q), matrices=[A, B, wrongC])

    # Wrong X shape
    wrongX = np.eye(n+1, p)
    with pytest.raises(Exception):
        skp.ProcrustesProblem((m, n, p, q), matrices=[A, B, C, wrongX])


def make_simple_problem():
    m, n, p, q = 6, 6, 3, 3
    problem = skp.ProcrustesProblem((m, n, p, q), problemnumber=1)
    # Overwrite problem stats for test usage.
    problem.stats = {
        "nbiter": 1,
        "fev": 2,
        "total_fun": [0.1, 0.01],
        "total_grad": [0.5, 0.3]
    }
    return problem


def test_gbbsolver_options_defaults_and_valid():
    solver = skp.GBBSolver()
    opts = solver.options
    # Check default types and keys
    assert isinstance(opts, dict)
    for key in ["full_results", "xtol", "gtol", "ftol", "zeta", "kappa", "eta",
                "etavar", "tau", "projected_gradient", "maxiter", "verbose",
                "filename", "timer"]:
        assert key in opts
    # Check some defaults
    assert isinstance(opts["full_results"], bool)
    assert isinstance(opts["xtol"], float)
    assert opts["projected_gradient"] in (1, 2)
    assert opts["verbose"] in (0, 1)

    # Invalid full_results type
    with pytest.raises(Exception):
        skp.GBBSolver(full_results="notbool")

    # Invalid projected_gradient value
    with pytest.raises(Exception):
        skp.GBBSolver(projected_gradient=3)


def test_gbbsolver_invalid_verbose():
    with pytest.raises(Exception):
        skp.GBBSolver(verbose=5)


def test_gbbsolver_invalid_filename_type():
    with pytest.raises(Exception):
        skp.GBBSolver(filename=123)  # non-str filename


def test_gbbsolver_solve_runs(monkeypatch):
    problem = make_simple_problem()

    # Patch np.random.rand to fixed numbers for reproducibility
    np.random.seed(123)

    solver = skp.GBBSolver(full_results=True, verbose=0)

    # Patch problem.stats to have required keys for full_results
    problem.stats = {
        "nbiter": 5,
        "fev": 10,
        "total_fun": [1.0, 0.9, 0.8],
        "total_crit": [0.1, 0.05, 0.01]
    }

    result = solver.solve(problem)

    # Result is an OptimizeResult instance (subclass of dict)
    assert isinstance(result, skp.OptimizeResult)

    # Result keys
    for key in ["success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"]:
        assert key in result

    assert result.success in [True, False]
    assert isinstance(result.fun, float)
    assert result.solution.shape == (problem.sizes[1], problem.sizes[2])


def test_gbbsolver_open_close_file(monkeypatch):
    solver = skp.GBBSolver(filename=None)
    # Open file should set to sys.stdout (test basic)
    solver.open_file()
    assert solver.file is not None
    solver.close_file()

    # With invalid filename (simulate OSError)
    class DummyOpen:
        def __init__(self):
            self.files_opened = 0

        def open(self, *args, **kwargs):
            raise OSError("fake error")

    dummy = DummyOpen()
    old_open = open

    def _fake_open(*args, **kwargs):
        raise OSError("fake error")

    import builtins
    builtins.open = _fake_open
    solver = skp.GBBSolver(filename="invalidfile")

    with pytest.raises(SystemExit):
        solver.open_file()

    builtins.open = old_open


def test_procrustesproblem_problemnumber3_invalid_n():
    # n outside allowed values for problem 3
    with pytest.raises(Exception):
        skp.ProcrustesProblem((60, 60, 2, 2), problemnumber=3)


def test_procrustesproblem_init_no_problemnumber_or_matrices():
    with pytest.raises(Exception):
        skp.ProcrustesProblem((10, 10, 2, 2))


def test_optimize_result_attribute_access():
    d = {"success": True, "fun": 10.0, "solution": np.eye(3)}
    res = skp.OptimizeResult(d)
    assert res.success is True
    assert np.array_equal(res.solution, np.eye(3))
    with pytest.raises(AttributeError):
        _ = res.non_existent_attr


def test_optimize_result_repr_and_dir():
    d = {"success": True, "fun": 0.0}
    res = skp.OptimizeResult(d)
    r = repr(res)
    assert "success" in r and "fun" in r
    keys = dir(res)
    assert "success" in keys and "fun" in keys


def test_compute_residual_and_optimality_basic():
    m, n, p, q = 6, 6, 3, 3
    problem = skp.ProcrustesProblem((m, n, p, q), problemnumber=1)
    X = np.eye(n, p)
    R, residual = skp.compute_residual(problem.A, problem.B, problem.C, X, None)
    assert R.shape == (m, q)
    assert residual >= 0

    normgradproj, normgrad, grad = skp.optimality(problem.A, problem.C, R, X, None)
    assert isinstance(normgradproj, float)
    assert isinstance(normgrad, float)
    assert grad.shape == (n, p)


def test_polardecomp_and_polar_newton_schultz_runs():
    W = np.eye(4)
    options = {"polar": "ns"}
    U = skp.polardecomp(W, options)
    assert np.allclose(U.T @ U, np.eye(4), atol=1e-8)

    # Test for matrix with m > n
    A = np.random.rand(5, 3)
    U = skp.polar_newton_schultz(A, 1e-8)
    assert U.shape == A.shape
    assert la.norm(U.T @ U - np.eye(3)) < 1e-6


def test_bidiaggs_reorthogonalization_behavior():
    m = 5
    n = 5
    s = 2
    prod = np.random.randn(m, s)
    mat = np.eye(m, n)
    gstol = 1e-10
    reorth = 0

    mat_out, R, reorth_out = skp.bidiaggs(0, prod, mat.copy(), gstol, reorth)
    assert mat_out.shape == mat.shape
    assert R.shape == (s, s)
    assert isinstance(reorth_out, int)
    # If rank deficient columns produce reorth > 0 (might depend on random)
    assert reorth_out >= 0


def test_blockbidiag_simple_call():
    m, n, p, q = 6, 6, 3, 3
    problem = skp.ProcrustesProblem((m, n, p, q), problemnumber=1)
    U = np.zeros((m, m))
    V = np.zeros((n, n))
    T = np.zeros((m, n+q))
    steps = m // q
    partial = 0
    halfreorth = False

    U2, V2, T2, B1, reorth = skp.blockbidiag(problem, U, V, T, steps, partial, halfreorth)

    assert U2.shape == U.shape
    assert V2.shape == V.shape
    assert T2.shape == T.shape
    assert B1.shape == (p, p)
    assert isinstance(reorth, int)


def test_spg_solver_runs_and_returns_expected():
    problem = make_simple_problem()
    options = {"full_results": False, "verbose": 0, "changevar": False,
               "precond": None, "gtol": 1e-3, "maxiter": 10,
               "strategy": "newfw", "eta": 0.85, "etavar": False,
               "filename": None, "polar": None, "timer": False}
    fileobj = None  # handlers defaults to sys.stdout internally

    X, f, opt, exitcode, msg = skp.spg_solver(problem, "spg", options, fileobj)
    assert isinstance(X, np.ndarray)
    assert isinstance(f, float)
    assert isinstance(opt, float)
    assert exitcode in (0, 1)
    assert isinstance(msg, str)


def test_gkb_setup_runs():
    problem = make_simple_problem()
    options = {"full_results": False, "verbose": 0, "inner_solver": "spg",
               "gtol": 1e-3, "maxiter": 10, "strategy": "newfw", "eta": 0.85,
               "etavar": False, "filename": None, "polar": None,
               "bloboptest": False, "timer": False, "precond": None,
               "horfreorth": False}
    fileobj = None
    X, fval, opt, exitcode, msg = skp.gkb_setup(problem, "gkb", options, fileobj)
    assert isinstance(X, np.ndarray)
    assert isinstance(fval, float)
    assert isinstance(opt, float)
    assert exitcode in (0, 1)
    assert isinstance(msg, str)


def test_spatial_solver_invalid_constraint_violation(monkeypatch):
    problem = make_simple_problem()
    options = {"full_results": False, "verbose": 0, "changevar": False,
               "precond": None, "gtol": 1e-3, "maxiter": 10,
               "strategy": "monot", "eta": 0.85, "etavar": False,
               "filename": None, "polar": None, "timer": False}

    def fake_svd(W, full_matrices=False):
        # Return matrices that yield Xtrial violating constraint (non orthogonal)
        U, _, Vt = la.svd(W)
        U[:, 0] = U[:, 0] * 10  # Intentionally break orthogonality
        return U, np.diag(np.ones(U.shape[1])), Vt
    monkeypatch.setattr(skp.sp, "svd", fake_svd)

    with pytest.raises(Exception, match="constraint violation"):
        skp.spectral_solver(problem, 6, 3, np.zeros((6, 3)), problem.A, problem.B,
                           "spg", options, False, None)


def test_spg_solver_with_changevar_precond(monkeypatch):
    problem = make_simple_problem()
    options = {"full_results": False, "verbose": 0, "changevar": True,
               "precond": "stupid", "gtol": 1e-3, "maxiter": 2,
               "strategy": "newfw", "eta": 0.85, "etavar": False,
               "filename": None, "polar": None, "timer": False}

    # patch spectral_solver to short-circuit real calls
    def dummy_spectral_solver(*args, **kwargs):
        return 0, 0.0, np.zeros((problem.sizes[1], problem.sizes[2])), 0.0, 0.0, 1, "success"
    monkeypatch.setattr(skp, "spectral_solver", dummy_spectral_solver)
    Xk, f, opt, exitcode, msg = skp.spg_solver(problem, "spg", options, None)
    assert isinstance(Xk, np.ndarray)
    assert exitcode == 0


def test_gbb_solver_invH_off_condition(monkeypatch):
    problem = make_simple_problem()
    options = {"full_results": False, "verbose": 0, "xtol": 1e-6, "gtol": 1e-6,
               "ftol": 1e-12, "zeta": 1e-4, "kappa": 0.1, "eta": 0.85,
               "etavar": False, "tau": 1e-3, "projected_gradient": 1,
               "maxiter": 2, "filename": None, "timer": False}
    # Use p < smalldim/2 condition True
    largedim = 6
    smalldim = 6
    X = np.eye(smalldim, 2)  # p=2 < 3 = smalldim/2
    A = problem.A
    B = problem.B
    # patch compute_residual to a trivial residual with zero normgradproj
    def dummy_optimality(A, C, R, X, precond):
        return 0.0, 0.0, np.zeros_like(X)
    monkeypatch.setattr(skp, "optimality", dummy_optimality)
    monkeypatch.setattr(skp, "compute_residual", lambda *a, **k: (np.zeros_like(B), 0.0))
    exitcode, f, Xout, normG, nbiter, msg = skp.gbb_solver(problem, largedim, smalldim, X, A, B,
                                                          options, False, None)
    assert exitcode in (0, 1)
    assert isinstance(Xout, np.ndarray)


def test_gbb_solver_runs_with_projected_gradient_2(monkeypatch):
    problem = make_simple_problem()
    options = {"full_results": False, "verbose": 0, "xtol": 1e-6, "gtol": 1e-6,
               "ftol": 1e-12, "zeta": 1e-4, "kappa": 0.1, "eta": 0.85,
               "etavar": False, "tau": 1e-3, "projected_gradient": 2,
               "maxiter": 1, "filename": None, "timer": False}
    largedim = 6
    smalldim = 6
    X = np.eye(smalldim, 3)
    A = problem.A
    B = problem.B
    monkeypatch.setattr(skp, "optimality", lambda *a, **k: (0.0, 0.0, np.zeros_like(X)))
    monkeypatch.setattr(skp, "compute_residual", lambda *a, **k: (np.zeros_like(B), 0.0))
    exitcode, f, Xout, normG, nbiter, msg = skp.gbb_solver(problem, largedim, smalldim, X, A, B,
                                                          options, False, None)
    assert exitcode in (0, 1)
    assert isinstance(Xout, np.ndarray)


# -- END OF TESTS --
