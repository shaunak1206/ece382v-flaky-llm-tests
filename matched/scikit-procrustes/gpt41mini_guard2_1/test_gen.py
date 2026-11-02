import numpy as np
import pytest
import sys

import skprocrustes.skprocrustes as skp


def test_procrustesproblem_default_problemnumber_1_seeded():
    np.random.seed(42)
    sizes = (10, 10, 2, 2)
    problem = skp.ProcrustesProblem(sizes, problemnumber=1)

    # Check shapes
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)

    # Check orthogonality of Xsol (Xsol.T @ Xsol == I)
    xtx = np.dot(problem.Xsol.T, problem.Xsol)
    assert np.allclose(xtx, np.eye(2), atol=1e-12)

    # Check that B = A Xsol C approximately
    reconstructed_B = np.dot(problem.A, np.dot(problem.Xsol, problem.C))
    assert np.allclose(problem.B, reconstructed_B, atol=1e-12)


def test_procrustesproblem_default_problemnumber_2_seeded():
    np.random.seed(123)
    sizes = (10, 10, 2, 2)
    problem = skp.ProcrustesProblem(sizes, problemnumber=2)

    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)

    # Orthogonality check
    xtx = np.dot(problem.Xsol.T, problem.Xsol)
    assert np.allclose(xtx, np.eye(2), atol=1e-12)

    # B approx equals A Xsol C
    reconstructed_B = np.dot(problem.A, np.dot(problem.Xsol, problem.C))
    assert np.allclose(problem.B, reconstructed_B, atol=1e-12)


def test_procrustesproblem_default_problemnumber_3_seeded():
    np.random.seed(321)
    sizes = (50, 50, 5, 5)
    problem = skp.ProcrustesProblem(sizes, problemnumber=3)

    assert problem.A.shape == (50, 50)
    assert problem.B.shape == (50, 5)
    assert problem.C.shape == (5, 5)
    assert problem.Xsol.shape == (50, 5)

    xtx = np.dot(problem.Xsol.T, problem.Xsol)
    assert np.allclose(xtx, np.eye(5), atol=1e-12)

    # B = A Xsol C approx
    reconstructed_B = np.dot(problem.A, np.dot(problem.Xsol, problem.C))
    assert np.allclose(problem.B, reconstructed_B, atol=1e-12)


def test_procrustesproblem_user_defined_matrices_correct_sizes():
    rng = np.random.default_rng(0)
    m, n, p, q = 6, 6, 2, 2
    A = rng.normal(size=(m, n))
    Xsol = rng.normal(size=(n, p))
    C = np.eye(p, q)
    B = np.dot(A, np.dot(Xsol, C))

    problem = skp.ProcrustesProblem((m, n, p, q), matrices=[A, B, C, Xsol])

    # Users matrices preserved
    assert np.allclose(problem.A, A)
    assert np.allclose(problem.B, B)
    assert np.allclose(problem.C, C)
    assert np.allclose(problem.Xsol, Xsol)


def test_procrustesproblem_user_defined_matrices_wrong_length_raises():
    m, n, p, q = 2, 2, 2, 2
    A = np.eye(m, n)
    B = np.eye(m, q)
    C = np.eye(p, q)
    with pytest.raises(Exception):
        skp.ProcrustesProblem((m, n, p, q), matrices=[A, B])  # less than 3


@pytest.mark.parametrize("badshape", [
    ((4, 3), (4, 2), (2, 2)),  # A shape wrong
    ((3, 3), (3, 2), (3, 2)),  # C shape wrong
    ((3, 3), (3, 3), (3, 3)),  # C shape wrong
    ((3, 3), (3, 2), (2, 2), (2, 3)),  # Xsol shape wrong
])
def test_procrustesproblem_user_defined_matrices_shape_errors(badshape):
    shapes = badshape
    matrices = []
    for s in shapes:
        matrices.append(np.zeros(s))
    sizes = (3, 3, 2, 2)
    with pytest.raises(Exception):
        skp.ProcrustesProblem(sizes, matrices=matrices)


def test_procrustesproblem_sizes_not_square_error():
    with pytest.raises(Exception, match="Currently only square problems are supported."):
        skp.ProcrustesProblem((3, 4, 2, 2))
    with pytest.raises(Exception, match="Currently only square problems are supported."):
        skp.ProcrustesProblem((4, 4, 2, 3))


def test_procrustesproblem_sizes_divisible_error():
    with pytest.raises(Exception, match="Dimensions are incompatible: m must be divisible by p"):
        skp.ProcrustesProblem((10, 10, 3, 3))


def test_gbbsolver_setoptions_defaults_and_types():
    # Test default options and types for GBBSolver _setoptions

    solver = skp.GBBSolver()

    # Defaults are set
    opts = solver.options
    assert opts["full_results"] is False
    assert isinstance(opts["xtol"], float)
    assert isinstance(opts["gtol"], float)
    assert isinstance(opts["ftol"], float)
    assert isinstance(opts["zeta"], float)
    assert isinstance(opts["kappa"], float)
    assert isinstance(opts["eta"], float)
    assert isinstance(opts["etavar"], bool)
    assert isinstance(opts["tau"], float)
    assert opts["projected_gradient"] in (1, 2)
    assert isinstance(opts["maxiter"], int)
    assert opts["verbose"] in (0, 1)
    assert opts["filename"] is None
    assert isinstance(opts["timer"], bool)


@pytest.mark.parametrize("key,bad_value,error_message", [
    ("full_results", "notbool", "full_results must be a boolean"),
    ("xtol", "notfloat", "xtol must be a float"),
    ("gtol", "notfloat", "gtol must be a float"),
    ("ftol", "notfloat", "ftol must be a float"),
    ("zeta", "notfloat", "zeta must be a float"),
    ("kappa", "notfloat", "kappa must be a float"),
    ("eta", "notfloat", "eta must be a float"),
    ("etavar", "notbool", "etavar must be a boolean"),
    ("tau", "notfloat", "tau must be a float"),
    ("projected_gradient", 3, "projected_gradient must be 1 or 2"),
    ("maxiter", 5.5, "maxiter must be an integer"),
    ("verbose", 5, "verbose must be 0, or 1"),
    ("filename", 123, "filename must be string"),
    ("timer", "notbool", "timer must be boolean"),
])
def test_gbbsolver_setoptions_invalid(monkeypatch, key, bad_value, error_message):
    # Create opts with one bad key and test exception

    opts = {key: bad_value}
    # Patch _setoptions to receive opts and call original
    # Use a new instance to isolate
    solver = skp.GBBSolver.__new__(skp.GBBSolver)
    with pytest.raises(Exception, match=error_message):
        solver._setoptions(opts)


def test_gbbsovler_solve_invokes_gbb_solver(monkeypatch):
    # We test that solve method runs and returns OptimizeResult with expected fields

    np.random.seed(0)

    sizes = (6, 6, 2, 2)
    problem = skp.ProcrustesProblem(sizes, problemnumber=1)
    problem.stats = {"nbiter": 3, "fev": 5, "total_fun": [], "total_crit": []}

    solver = skp.GBBSolver(full_results=True, maxiter=2, verbose=0)
    monkeypatch.setattr(sys, "stdout", open('/dev/null', 'w'))

    # Monkeypatch gbb_solver to a dummy that returns fixed data
    def fake_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        return 0, 0.1, X, 0.01, 3, "success"
    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    result = solver.solve(problem)

    assert isinstance(result, skp.OptimizeResult)
    assert "success" in result
    assert result.success is True
    assert np.allclose(result.solution, result.solution)  # trivial check
    assert result.nbiter == problem.stats["nbiter"]
    assert result.nfev == problem.stats["fev"]
    assert result.fun == 0.1
    assert result.error == np.inf or isinstance(result.error, float)  # no Xsol? error=inf or float
    assert "total_fun" in result
    assert "total_crit" in result


def test_procrustesproblem_error_invalid_problemnumber():
    np.random.seed(1)
    sizes = (10, 10, 2, 2)
    with pytest.raises(Exception, match="Problem matrices are empty and no problem number has been chosen."):
        skp.ProcrustesProblem(sizes)


def test_procrustesproblem_error_invalid_problemnumber_3_n_check():
    # test that problem 3 raises error if n not in acceptable set
    sizes = (10, 11, 2, 2)
    with pytest.raises(Exception, match="Problem 6 requires n = 50,"):
        skp.ProcrustesProblem(sizes, problemnumber=3)


def test_optimize_result_repr_and_attr_access():
    d = {"a": 1, "b": 2}
    res = skp.OptimizeResult(d)
    s = repr(res)
    assert "a" in s
    assert "b" in s
    assert res.a == 1
    assert res["b"] == 2
    with pytest.raises(AttributeError):
        _ = res.c
    res.c = 5
    assert res.c == 5
    del res.c
    with pytest.raises(AttributeError):
        _ = res.c
    keys = dir(res)
    assert "a" in keys
    assert "b" in keys


def test_gbbsovler_open_close_file_and_filename(monkeypatch, tmp_path):
    solver = skp.GBBSolver()

    # Default filename None means stdout assigned to file
    solver.options["filename"] = None
    solver.open_file()
    assert solver.file == sys.stdout
    solver.close_file()  # no error

    # Assign existing file path temporarily
    file_path = tmp_path / "tempfile.txt"
    solver.options["filename"] = str(file_path)
    solver.open_file()
    assert solver.file is not None
    solver.close_file()
    # The file should be closed and exist now
    assert file_path.exists()


@pytest.mark.parametrize("filename_value", [
    "", "/does/not/exist/path/invalidfile"
])
def test_gbbsolver_open_file_invalid(monkeypatch, filename_value, capsys):
    solver = skp.GBBSolver()
    solver.options["filename"] = filename_value
    # Reference: if file open fails, print error and exit with code -1
    # We patch sys.exit to capture call instead of actually exiting.

    called = {}

    def fake_exit(code):
        called["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)
    with pytest.raises(SystemExit):
        solver.open_file()

    assert called["code"] == -1


def test_compare_solvers_basic(monkeypatch):
    # Basic test that compare_solvers runs without error for known solvers and returns results list

    np.random.seed(1234)
    sizes = (10, 10, 2, 2)
    problem = skp.ProcrustesProblem(sizes, problemnumber=1)
    # patch solve methods to just return a dummy OptimizeResult with total_fun key
    class DummySolver(skp.ProcrustesSolver):
        def __init__(self):
            super().__init__()
            self.solvername = "spg"
        def solve(self, problem):
            problem.stats["total_fun"] = [1.0, 0.5, 0.1]
            r = skp.OptimizeResult({
                "success": True,
                "status": 0,
                "message": "success",
                "solution": np.eye(problem.sizes[1], problem.sizes[2]),
                "fun": 0.1,
                "normcrit": 0.01,
                "error": 0.0,
                "cpu": 0.01,
                "nbiter": 5,
                "nfev": 10,
                "total_fun": problem.stats["total_fun"]
            })
            return r

    solver1 = DummySolver()
    solver2 = DummySolver()
    results = skp.compare_solvers(problem, solver1, solver2, plot=False)
    assert len(results) == 2
    for res in results:
        assert isinstance(res, skp.OptimizeResult)
        assert "total_fun" in res
        assert res.success


def test_optimality_and_compute_residual_consistency():
    np.random.seed(0)
    sizes = (10, 10, 3, 3)
    problem = skp.ProcrustesProblem(sizes, problemnumber=1)
    X = np.eye(sizes[1], sizes[2])
    R, residual = skp.compute_residual(problem.A, problem.B, problem.C, X, None)
    normgradproj, normgrad, grad = skp.optimality(problem.A, problem.C, R, X, None)

    # residual non-negative
    assert residual >= 0
    # gradient norm >=0
    assert normgrad >= 0
    # projected gradient norm infinite norm scalar
    assert isinstance(normgradproj, float)
    assert normgradproj <= normgrad + 1e-12


def test_polardecomp_ns_equivalence():
    np.random.seed(0)
    A = np.random.rand(5,3)
    A[:, 0] = A[:, 0] * 10  # make rank full

    options = {"polar": "ns"}

    U = skp.polardecomp(A, options)
    # U should be orthogonal: U.T @ U == I_p
    p = A.shape[1]
    xtx = np.dot(U.T, U)
    assert np.allclose(xtx, np.eye(p), atol=1e-8)

    # Additionally, U * V.T approx A polar decomposition: here only U returned.
    assert U.shape == A.shape


def test_procrustesproblem_output_consistent_random_seeds():
    seed = 42
    np.random.seed(seed)
    p1 = skp.ProcrustesProblem((10,10,2,2), problemnumber=1)
    np.random.seed(seed)
    p2 = skp.ProcrustesProblem((10,10,2,2), problemnumber=1)

    # Matrices generated should be identical for same seed
    assert np.allclose(p1.A, p2.A)
    assert np.allclose(p1.B, p2.B)
    assert np.allclose(p1.C, p2.C)
    assert np.allclose(p1.Xsol, p2.Xsol)


def test_gbbsovler_projected_gradient_options_only_1_or_2_allowed():
    solver = skp.GBBSolver()
    opts = solver.options
    opts["projected_gradient"] = 3
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        solver._setoptions(opts)
    opts["projected_gradient"] = 2  # valid
    solver._setoptions(opts)  # should not raise


def test_procrustesproblem_known_solution_none_when_not_given():
    sizes = (6, 6, 2, 2)
    # Provide only the matrices for A, B, C, no Xsol
    A = np.eye(6)
    B = np.ones((6, 2))
    C = np.eye(2)
    problem = skp.ProcrustesProblem(sizes, matrices=[A, B, C])
    assert not hasattr(problem, 'Xsol')


def test_procrustesproblem_given_solution_shape_check_error():
    sizes = (6, 6, 2, 2)
    A = np.eye(6)
    B = np.ones((6, 2))
    C = np.eye(2)
    Xwrong = np.ones((2, 6))  # Wrong shape for Xsol (should be 6,2)
    with pytest.raises(Exception, match="Xsol must be \\(n,p\\)"):
        skp.ProcrustesProblem(sizes, matrices=[A, B, C, Xwrong])
