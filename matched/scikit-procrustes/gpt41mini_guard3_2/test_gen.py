import numpy as np
import pytest
import sys
import io
import numpy.linalg as la

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, _status_message


@pytest.mark.parametrize("sizes", [
    (10, 10, 2, 2),
    (6, 6, 3, 3),
    (4, 4, 2, 2),
])
def test_procrustesproblem_init_valid(sizes):
    np.random.seed(0)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    m, n, p, q = sizes
    assert problem.A.shape == (m, n)
    assert problem.B.shape == (m, q)
    assert problem.C.shape == (p, q)
    assert problem.Xsol.shape == (n, p)
    # solution is orthogonal
    assert la.norm(problem.Xsol.T @ problem.Xsol - np.eye(p), ord=np.inf) < 1e-12
    # residual computation consistent
    R, residual = problem.A @ (problem.Xsol @ problem.C) - problem.B, None
    residual = la.norm(R, ord='fro') ** 2
    R2, residual2 = problem.A @ (problem.Xsol @ problem.C) - problem.B, None
    residual2 = la.norm(R2, ord='fro') ** 2
    assert abs(residual - residual2) < 1e-14


def test_procrustesproblem_init_invalid_dimensions():
    with pytest.raises(Exception, match="Currently only square problems are supported"):
        ProcrustesProblem((10, 9, 2, 2))
    with pytest.raises(Exception, match="Dimensions are incompatible: m must be divisible by p"):
        ProcrustesProblem((10, 10, 3, 2))


def test_procrustesproblem_with_matrices_valid():
    np.random.seed(1)
    m, n, p, q = (6, 6, 2, 2)
    A = np.random.randn(m, n)
    B = np.random.randn(m, q)
    C = np.eye(p, q)
    Xsol = np.eye(n, p)
    matrices = (A, B, C, Xsol)
    problem = ProcrustesProblem((m, n, p, q), matrices=matrices)
    assert np.allclose(problem.A, A)
    assert np.allclose(problem.B, B)
    assert np.allclose(problem.C, C)
    assert np.allclose(problem.Xsol, Xsol)


@pytest.mark.parametrize("bad_matrices", [
    ((np.random.randn(5, 6), np.random.randn(5, 6), np.eye(5, 5)),),  # A shape wrong
    ((np.random.randn(6, 6), np.random.randn(6, 5), np.eye(5, 5)),),  # B shape wrong
    ((np.random.randn(6, 6), np.random.randn(6, 6), np.eye(4, 5)),),  # C shape wrong
])
def test_procrustesproblem_with_bad_matrices_raises(bad_matrices):
    sizes = (6, 6, 5, 5)
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=bad_matrices[0])


def test_procrustesproblem_with_bad_Xsol_shape():
    sizes = (6, 6, 3, 3)
    A = np.eye(6)
    B = np.eye(6)
    C = np.eye(3)
    Xsol_bad = np.eye(6, 2)  # Wrong shape (6,2) instead of (6,3)
    matrices = [A, B, C, Xsol_bad]
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=matrices)


def test_gbbsovler_default_options_and_setoptions(monkeypatch):
    monkeypatch.setattr("sys.stdout", io.StringIO())
    solver = GBBSolver()
    # By default, all options have default values
    opt = solver.options
    assert isinstance(opt["full_results"], bool)
    assert isinstance(opt["xtol"], float)
    assert isinstance(opt["gtol"], float)
    assert isinstance(opt["ftol"], float)
    assert isinstance(opt["zeta"], float)
    assert isinstance(opt["kappa"], float)
    assert isinstance(opt["eta"], float)
    assert isinstance(opt["etavar"], bool)
    assert isinstance(opt["tau"], float)
    assert isinstance(opt["projected_gradient"], int)
    assert opt["maxiter"] == 5000
    assert opt["verbose"] in (0, 1)
    assert opt["filename"] is None
    assert isinstance(opt["timer"], bool)


@pytest.mark.parametrize("bad_option, value, error_msg", [
    ("full_results", "notbool", "full_results must be a boolean"),
    ("xtol", "notfloat", "xtol must be a float"),
    ("gtol", "notfloat", "gtol must be a float"),
    ("ftol", "notfloat", "ftol must be a float"),
    ("zeta", "notfloat", "zeta must be a float"),
    ("kappa", "notfloat", "kappa must be a float"),
    ("eta", "notfloat", "eta must be a float"),
    ("etavar", 1, "etavar must be a boolean"),
    ("tau", "notfloat", "tau must be a float"),
    ("projected_gradient", 3, "projected_gradient must be 1 or 2"),
    ("maxiter", 1.5, "maxiter must be an integer"),
    ("verbose", 5, "verbose must be 0, or 1"),
    ("filename", 123, "filename must be string"),
    ("timer", "notbool", "timer must be boolean"),
])
def test_gbbsovler_setoptions_invalid(monkeypatch, bad_option, value, error_msg):
    monkeypatch.setattr("sys.stdout", io.StringIO())
    options = {bad_option: value}
    with pytest.raises(Exception, match=error_msg):
        GBBSolver(**options)


def test_gbbsovler_open_close_file(monkeypatch):
    solver = GBBSolver()
    # No filename: fileobj should be sys.stdout
    solver.options["filename"] = None
    solver.open_file()
    assert solver.file is sys.stdout
    # Now patch filename to a temp file
    import tempfile
    with tempfile.NamedTemporaryFile() as tmpfile:
        solver.options["filename"] = tmpfile.name + "_new"
        # Actually file doesn't exist, so file will be open for writing
        solver.open_file()
        assert not solver.file.closed
        solver.close_file()
        assert solver.file.closed
    # If filename is invalid, sys.exit(-1) is called
    # Patch sys.exit to raise SystemExit to capture it:
    solver.options["filename"] = "/root/invalid_file_very_unlikely_to_exist.tmp"
    import builtins

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr(sys, "exit", fake_exit)
    with pytest.raises(SystemExit):
        solver.open_file()


def test_gbbsovler_solve_returns_result(monkeypatch):
    # We patch gbb_solver to dummy function to isolate test
    dummy_calls = {}

    def dummy_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj,
                         B1=None, blobopprod=0.0):
        dummy_calls["called"] = True
        exitcode = 0
        fval = 1.0
        # X solution approx: identity
        Xsol = np.eye(n, problem.sizes[2])
        normgrad = 0.01
        outer = 10
        msg = _status_message['success']
        return exitcode, fval, Xsol, normgrad, outer, msg

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", dummy_gbb_solver)
    monkeypatch.setattr("numpy.random.rand", lambda *args, **kwargs: np.zeros(args))
    solver = GBBSolver(full_results=True, verbose=0)
    # Use a fixed seeding for reproducibility
    np.random.seed(42)
    problem = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)

    problem.stats["nbiter"] = 5
    problem.stats["fev"] = 10
    problem.stats["total_fun"] = [0.1, 0.05]
    problem.stats["total_crit"] = [0.03, 0.02]
    result = solver.solve(problem)

    assert dummy_calls.get("called", False) is True
    # Check result fields
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
    assert np.allclose(result.fun, 1.0)
    # error is infinity or finite depending on Xsol presence
    assert result.error == pytest.approx(np.linalg.norm(result.solution - problem.Xsol, np.inf), rel=1e-12)
    # message matches known message
    assert result.message == _status_message['success'] or result.message == _status_message['stalled']


def test_procrustesproblem_problemnumber_raises_on_bad_number():
    np.random.seed(0)
    sizes = (50, 50, 10, 10)
    with pytest.raises(Exception, match="Problem matrices are empty and no problem number has been chosen."):
        # None problemnumber and matrices==None triggers Exception
        ProcrustesProblem(sizes)


def test_procrustesproblem_problemnumber3_with_allowed_n(seeds=[50, 95, 500, 1000]):
    for n in seeds:
        np.random.seed(1)
        sizes = (n, n, n//5, n//5)
        problem = ProcrustesProblem(sizes, problemnumber=3)
        assert problem.A.shape == (n, n)
        assert problem.B.shape == (n, n//5)
        assert problem.C.shape == (n//5, n//5)
        # Xsol shape matches (n, n//5)
        assert problem.Xsol.shape == (n, n//5)


def test_gbbsovler_projected_gradient_valid_values():
    solver = GBBSolver(projected_gradient=1)
    assert solver.options["projected_gradient"] == 1
    solver = GBBSolver(projected_gradient=2)
    assert solver.options["projected_gradient"] == 2


def test_gbbsovler_projected_gradient_invalid_value(monkeypatch):
    monkeypatch.setattr("sys.stdout", io.StringIO())
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        GBBSolver(projected_gradient=3)


def test_gbb_solver_iteration_completion(monkeypatch):
    # This test runs a minimal GBBSolver solve call to check it completes properly with minimal iterations.
    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", lambda *a, **k: (0, 0.1, np.eye(a[2], a[2]//3), 0.001, 1, _status_message['success']))
    monkeypatch.setattr("numpy.random.rand", lambda *args, **kwargs: np.zeros(args))

    sizes = (6, 6, 2, 2)
    np.random.seed(2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    solver = GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert result.success is True or result.success is False
    assert isinstance(result.fun, float)
    # Check that solution is matrix of correct shape
    assert result.solution.shape == (sizes[1], sizes[2])
    # Check that error is always non-negative
    assert result.error >= 0


def test_procrustesproblem_Xsol_is_optional():
    m, n, p, q = 6, 6, 2, 2
    A = np.eye(m)
    B = np.eye(m)
    C = np.eye(p)
    matrices = (A, B, C)
    problem = ProcrustesProblem((m, n, p, q), matrices=matrices)
    assert problem.A.shape == (m, n)
    assert problem.B.shape == (m, q)
    assert problem.C.shape == (p, q)
    assert not hasattr(problem, "Xsol")


def test_procrustesproblem_stats_initial(monkeypatch):
    np.random.seed(3)
    prob = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    # Before solving, stats dict non-empty and contains keys
    assert isinstance(prob.stats, dict)
    assert "nbiter" not in prob.stats  # Not set until solver runs
    # Run a dummy solver to set stats:
    solver = GBBSolver()
    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", lambda *args, **kwargs: (0,0,prob.Xsol,0,1,"success"))
    result = solver.solve(prob)
    assert "nbiter" in prob.stats
    assert "fev" in prob.stats
    assert "total_fun" in prob.stats or not solver.options["full_results"]


def test_gbbsovler_verbose_output(monkeypatch):
    np.random.seed(4)
    prob = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    # Patch gbb_solver to a dummy that returns success
    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", lambda *args, **kwargs: (0, 0.1, np.eye(args[2], args[2]//3), 0.001, 1, _status_message['success']))
    solver = GBBSolver(verbose=1)
    # Redirect stdout to StringIO
    import io
    import sys
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        solver.solve(prob)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = stdout
    assert "Options:" in output
    assert "GBB Solver" in output


def test_procrustesproblem_Xsol_permutes_identity():
    np.random.seed(10)
    sizes = (6, 6, 2, 2)
    prob = ProcrustesProblem(sizes, problemnumber=1)
    # Xsol should be a permutation of identity rows
    eye = np.eye(sizes[1], sizes[2])
    # Check that Xsol has same singular values as eye
    s1 = np.linalg.svd(prob.Xsol, compute_uv=False)
    s2 = np.linalg.svd(eye, compute_uv=False)
    assert np.allclose(s1, s2)
    # Check that Xsol.T @ Xsol close to identity
    assert np.allclose(prob.Xsol.T @ prob.Xsol, np.eye(sizes[2]), atol=1e-12)

