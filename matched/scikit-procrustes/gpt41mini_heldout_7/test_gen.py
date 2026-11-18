import numpy as np
import pytest
import builtins
import sys
import io
from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, OptimizeResult


def test_procrustesproblem_init_with_problemnumber_1():
    p = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert p.A.shape == (10, 10)
    assert p.B.shape == (10, 2)
    assert p.C.shape == (2, 2)
    assert p.Xsol.shape == (10, 2)
    assert p.problemnumber == 1


def test_procrustesproblem_init_with_problemnumber_2():
    p = ProcrustesProblem((10, 10, 2, 2), problemnumber=2)
    assert p.A.shape == (10, 10)
    assert p.B.shape == (10, 2)
    assert p.C.shape == (2, 2)
    assert p.Xsol.shape == (10, 2)


def test_procrustesproblem_init_with_problemnumber_3_valid_n50():
    p = ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
    assert p.A.shape == (50, 50)
    assert p.B.shape == (50, 5)
    assert p.C.shape == (5, 5)
    assert p.Xsol.shape == (50, 5)


def test_procrustesproblem_init_with_problemnumber_3_invalid_n():
    with pytest.raises(Exception, match="requires n = 50,"):
        ProcrustesProblem((51, 51, 5, 5), problemnumber=3)


def test_procrustesproblem_init_with_matrices_valid():
    m, n, p, q = 4, 4, 2, 2
    A = np.ones((m, n))
    B = np.ones((m, q))
    C = np.eye(p, q)
    Xsol = np.eye(n, p)
    p1 = ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))
    assert np.array_equal(p1.A, A)
    assert np.array_equal(p1.B, B)
    assert np.array_equal(p1.C, C)
    assert np.array_equal(p1.Xsol, Xsol)


def test_procrustesproblem_init_with_matrices_invalid_length():
    m, n, p, q = 4, 4, 2, 2
    matrices = (np.ones((m, n)), np.ones((m, q)))  # only 2 matrices
    with pytest.raises(Exception, match="matrices must be a list or tuple containing matrices"):
        ProcrustesProblem((m, n, p, q), matrices=matrices)


def test_procrustesproblem_init_with_incorrect_shapes():
    m, n, p, q = 4, 4, 2, 2
    A = np.ones((m+1, n))  # Wrong shape
    B = np.ones((m, q))
    C = np.eye(p, q)
    with pytest.raises(Exception, match="A must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C))
    A = np.ones((m, n))
    B_bad = np.ones((m+1, q))
    with pytest.raises(Exception, match="B must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B_bad, C))
    B = np.ones((m, q))
    C_bad = np.eye(p+1, q)
    with pytest.raises(Exception, match="C must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C_bad))


def test_procrustesproblem_init_with_xsol_incorrect_shape():
    m, n, p, q = 4, 4, 2, 2
    A = np.ones((m, n))
    B = np.ones((m, q))
    C = np.eye(p, q)
    Xsol_bad = np.eye(n+1, p)
    with pytest.raises(Exception, match="Xsol must be"):
        ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol_bad))


def test_procrustesproblem_init_fails_on_non_square_m_n():
    sizes = (3, 4, 2, 2)
    with pytest.raises(Exception, match="only square problems"):
        ProcrustesProblem(sizes)


def test_procrustesproblem_init_fails_on_non_square_p_q():
    sizes = (4, 4, 2, 3)
    with pytest.raises(Exception, match="only square problems"):
        ProcrustesProblem(sizes)


def test_procrustesproblem_init_fails_if_m_not_divisible_by_p():
    sizes = (4, 4, 3, 3)
    with pytest.raises(Exception, match="Dimensions are incompatible"):
        ProcrustesProblem(sizes)


@pytest.mark.parametrize("invalid_opt, val, msg", [
    ("full_results", "notbool", "full_results must be a boolean"),
    ("filename", 123, "filename must be a string"),
    ("strategy", "unknown", "strategy not implemented"),
    ("gtol", "notfloat", "gtol must be a float"),
    ("eta", "notfloat", "eta must be a float"),
    ("etavar", "notbool", "etavar must be a boolean"),
    ("maxiter", "notint", "maxiter must be an integer"),
    ("verbose", 5, "verbose must be 0, 1, 2 or 3"),
    ("changevar", "notbool", "changevar must be True or False"),
    ("polar", "badvalue", "polar must be ns or None"),
    ("timer", "notbool", "timer must be boolean"),
    ("precond", "badvalue", "precond must be stupid or None"),
])
def test_gbbsovler_setoptions_invalid_values(invalid_opt, val, msg):
    opts = {invalid_opt: val}
    if invalid_opt == "changevar":
        opts["full_results"] = True  # to allow setting changevar in _setoptions
    solver = GBBSolver()
    # Patch options to empty first to avoid interference
    solver.options = {}
    # Inject the invalid option
    with pytest.raises(Exception, match=msg):
        solver._setoptions(opts)


def test_gbbsovler_setoptions_defaults():
    solver = GBBSolver()
    # Should not raise
    solver._setoptions({})
    opts = solver.options
    # Check defaults keys
    assert isinstance(opts["full_results"], bool)
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
    assert opts["filename"] is None or isinstance(opts["filename"], str)
    assert isinstance(opts["timer"], bool)


def test_gbbsovler_open_file_stdout_and_close(monkeypatch):
    solver = GBBSolver(filename=None)
    solver._setoptions({})
    solver.open_file()
    assert solver.file == sys.stdout
    # monkeypatch close to catch close call
    called = []
    def fakeclose():
        called.append(True)
    solver.file.close = fakeclose
    solver.close_file()
    assert called


def test_gbbsovler_open_file_file_and_close(tmp_path):
    file_path = tmp_path / "tmpfile.txt"
    solver = GBBSolver(filename=str(file_path))
    solver._setoptions({})
    solver.open_file()
    assert not solver.file.closed
    solver.close_file()
    assert solver.file.closed


def test_gbbsovler_open_file_invalid_filename(monkeypatch):
    solver = GBBSolver(filename="/invalid_path/this_file_cannot_exist.txt")
    solver._setoptions({"filename": "/invalid_path/this_file_cannot_exist.txt"})
    # Replace sys.exit to raise instead so we can test
    with pytest.raises(SystemExit):
        solver.open_file()


def test_gbbsovler_solve_runs_and_returns(monkeypatch):
    # Patch gbb_solver to fake internal solver call
    import skprocrustes.skprocrustes as skp

    def fake_gbb_solver(problem, largedim, smalldim, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        n, p = smalldim, problem.sizes[2]
        X_sol = np.eye(n, p)
        return 0, 1.0, X_sol, 0.0, 3, "success"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    sizes = (6, 6, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    problem.stats = {"nbiter": 3, "fev": 5, "total_fun": [1.0], "total_crit": [0.1]}

    solver = GBBSolver(full_results=True)
    solver._setoptions({"full_results": True})
    result = solver.solve(problem)
    assert isinstance(result, OptimizeResult)
    assert result.success is True
    assert hasattr(result, "solution")
    assert result.solution.shape == (6, 2)
    assert result.nbiter == 3
    assert result.nfev == 5
    assert isinstance(result.cpu, float)
    assert "total_fun" in result
    assert "total_crit" in result
    assert isinstance(result.total_fun, (list, np.ndarray))
    assert isinstance(result.total_crit, (list, np.ndarray))


def test_gbbsovler_solve_without_xsol(monkeypatch):
    # Patch gbb_solver to fake internal solver call
    import skprocrustes.skprocrustes as skp

    def fake_gbb_solver(problem, largedim, smalldim, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        n, p = smalldim, problem.sizes[2]
        X_sol = np.eye(n, p)
        return 0, 1.0, X_sol, 0.0, 3, "success"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    sizes = (6, 6, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    if hasattr(problem, "Xsol"):
        del problem.Xsol
    problem.stats = {"nbiter": 3, "fev": 5, "total_fun": [1.0], "total_crit": [0.1]}

    solver = GBBSolver(full_results=False)
    solver._setoptions({"full_results": False})
    result = solver.solve(problem)
    assert np.isinf(result.error)


def test_gbbsovler_solve_with_missing_full_results_stats(monkeypatch):
    # Patch gbb_solver to fake internal solver call
    import skprocrustes.skprocrustes as skp

    def fake_gbb_solver(problem, largedim, smalldim, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        n, p = smalldim, problem.sizes[2]
        X_sol = np.eye(n, p)
        return 0, 1.0, X_sol, 0.0, 3, "success"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    sizes = (6, 6, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    problem.stats = {"nbiter": 3, "fev": 5}  # missing total_fun/total_crit

    solver = GBBSolver(full_results=True)
    solver._setoptions({"full_results": True})

    with pytest.raises(Exception, match="For full results, set problem.stats"):
        solver.solve(problem)


def test_optimize_result_access_and_repr():
    res = OptimizeResult({"success": True, "status": 0, "message": "ok"})
    assert res.success is True
    assert res.status == 0
    assert "message" in dir(res)
    # Attribute error if non-existent
    with pytest.raises(AttributeError):
        _ = res.nonexistent_attr
    # Test string representation has keys listed
    s = repr(res)
    assert "success" in s and "status" in s


def test_procrustesproblem_stats_initially_empty():
    p = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert isinstance(p.stats, dict)
    assert p.stats == {}


def test_gbbsovler_setoptions_projected_gradient_valid():
    solver = GBBSolver()
    solver._setoptions({"projected_gradient": 1})
    assert solver.options["projected_gradient"] == 1
    solver._setoptions({"projected_gradient": 2})
    assert solver.options["projected_gradient"] == 2


def test_gbbsovler_setoptions_projected_gradient_invalid():
    solver = GBBSolver()
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        solver._setoptions({"projected_gradient": 3})


def test_gbbsovler_verbose_prints(capsys, monkeypatch):
    # Patch gbb_solver to fake internal solver call
    import skprocrustes.skprocrustes as skp

    def fake_gbb_solver(problem, largedim, smalldim, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        n, p = smalldim, problem.sizes[2]
        X_sol = np.eye(n, p)
        return 0, 1.0, X_sol, 0.0, 3, "success"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    sizes = (6, 6, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    problem.stats = {"nbiter": 3, "fev": 5, "total_fun": [1.0], "total_crit": [0.1]}

    solver = GBBSolver(verbose=1)
    solver._setoptions({"verbose": 1, "full_results": True})
    solver.solve(problem)
    captured = capsys.readouterr()
    assert "Options" in captured.out
    assert "success" in captured.out


def test_procrustesproblem_randomness_seeded_behavior():
    # Check that repeated instance creation produce different A when seed changes
    np.random.seed(42)
    p1 = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    A1 = p1.A.copy()
    np.random.seed(43)
    p2 = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    A2 = p2.A.copy()
    assert not np.array_equal(A1, A2)


def test_gbbsovler_solve_reproducible_solution(monkeypatch):
    import skprocrustes.skprocrustes as skp

    called = {}
    def fake_gbb_solver(problem, largedim, smalldim, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        called["called"] = True
        n, p = smalldim, problem.sizes[2]
        # Return X as ones matrix to check assignment
        return 0, 1.0, np.ones((n, p)), 0.0, 3, "success"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    sizes = (4, 4, 2, 2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    problem.stats = {"nbiter": 2, "fev": 3, "total_fun": [1.0], "total_crit": [0.1]}

    solver = GBBSolver()
    solver._setoptions({})
    result = solver.solve(problem)
    assert called.get("called")
    assert np.all(result.solution == 1.0)


def test_procrustesproblem_invalid_sizes_type():
    with pytest.raises(IndexError):
        ProcrustesProblem((3,))


def test_procrustesproblem_with_zero_dimension():
    with pytest.raises(Exception, match="Dimensions are incompatible"):
        ProcrustesProblem((0, 0, 0, 0))


def test_gbbsovler_setoptions_accepts_minimal_options():
    solver = GBBSolver()
    solver._setoptions({})
    # Should have default maxiter, etc.
    assert solver.options["maxiter"] == 5000
    assert solver.options["verbose"] == 1


def test_gbbsovler_setoptions_rejects_non_int_maxiter():
    solver = GBBSolver()
    with pytest.raises(Exception, match="maxiter must be an integer"):
        solver._setoptions({"maxiter": 1.5})
