import numpy as np
import pytest
from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, OptimizeResult


def generate_square_problem(m=10, n=10, p=2, q=2):
    """Helper to generate a ProcrustesProblem with square sizes and default matrices."""
    # sizes must satisfy: m == n, p == q, and m % p == 0
    sizes = (m, n, p, q)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    return problem


def test_procrustesproblem_init_predefined_1():
    p = generate_square_problem()
    assert p.A.shape == (10, 10)
    assert p.B.shape == (10, 2)
    assert p.C.shape == (2, 2)
    assert p.Xsol.shape == (10, 2)


def test_procrustesproblem_init_predefined_2():
    p = ProcrustesProblem((10, 10, 2, 2), problemnumber=2)
    assert p.A.shape == (10, 10)
    assert p.B.shape == (10, 2)
    assert p.C.shape == (2, 2)
    assert p.Xsol.shape == (10, 2)


def test_procrustesproblem_init_predefined_3_valid_n():
    # n=50 is valid for problemnumber 3
    p = ProcrustesProblem((50, 50, 2, 2), problemnumber=3)
    assert p.A.shape == (50, 50)
    assert p.B.shape == (50, 2)
    assert p.C.shape == (2, 2)
    assert p.Xsol.shape == (50, 2)


def test_procrustesproblem_init_predefined_3_invalid_n():
    with pytest.raises(Exception) as e:
        ProcrustesProblem((51, 51, 2, 2), problemnumber=3)
    assert "Problem 6 requires n = 50" in str(e.value)


def test_procrustesproblem_init_invalid_sizes_non_square():
    with pytest.raises(Exception) as e:
        ProcrustesProblem((10, 9, 2, 2))
    assert "only square problems are supported" in str(e.value)


def test_procrustesproblem_init_invalid_sizes_not_divisible():
    with pytest.raises(Exception) as e:
        ProcrustesProblem((10, 10, 3, 3))
    assert "Dimensions are incompatible: m must be divisible by p" in str(e.value)


def test_procrustesproblem_user_supplied_matrices_correct_shapes():
    m, n, p, q = 6, 6, 2, 2
    A = np.random.rand(m, n)
    B = np.random.rand(m, q)
    C = np.eye(p, q)
    Xsol = np.random.rand(n, p)
    p = ProcrustesProblem((m, n, p, q), matrices=[A, B, C, Xsol])
    assert np.array_equal(p.A, A)
    assert np.array_equal(p.B, B)
    assert np.array_equal(p.C, C)
    assert np.array_equal(p.Xsol, Xsol)


def test_procrustesproblem_user_supplied_matrices_wrong_shapes():
    m, n, p, q = 6, 6, 2, 2
    A = np.random.rand(m, n)
    B = np.random.rand(m, q)
    C = np.eye(p, q)
    Xsol = np.random.rand(n, p + 1)  # wrong shape
    with pytest.raises(Exception) as e:
        ProcrustesProblem((m, n, p, q), matrices=[A, B, C, Xsol])
    assert "Xsol must be (n,p)" in str(e.value)


def test_gbbsolver_default_options():
    solver = GBBSolver()
    opts = solver.options
    # Check all default keys and their types or values
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


def test_gbbsolver_options_invalid_projected_gradient():
    with pytest.raises(Exception) as e:
        GBBSolver(projected_gradient=3)
    assert "projected_gradient must be 1 or 2" in str(e.value)


def test_gbbsolver_options_invalid_verbose():
    with pytest.raises(Exception) as e:
        GBBSolver(verbose=4)
    assert "verbose must be 0, or 1" in str(e.value)


def test_gbbsolver_options_invalid_full_results():
    with pytest.raises(Exception) as e:
        GBBSolver(full_results="yes")
    assert "full_results must be a boolean" in str(e.value)


def test_gbbsolver_options_invalid_filename_type():
    with pytest.raises(Exception) as e:
        GBBSolver(filename=123)
    assert "filename must be string" in str(e.value)


def test_gbbsolver_options_invalid_timer_type():
    with pytest.raises(Exception) as e:
        GBBSolver(timer="no")
    assert "timer must be boolean" in str(e.value)


def test_gbbsolver_open_close_file_stdout(monkeypatch):
    solver = GBBSolver()
    solver.options["filename"] = None
    solver.open_file()
    assert solver.file == sys.stdout
    # Use close_file (should be no error)
    solver.close_file()


def test_gbbsolver_open_file_invalid_filename(monkeypatch):
    import sys

    logs = []

    def fake_exit(code):
        logs.append(f"exit_called with code {code}")
        raise SystemExit(code)

    monkeypatch.setattr("builtins.print", lambda *a, **k: logs.append(" ".join(map(str, a))))
    monkeypatch.setattr("sys.exit", fake_exit)

    solver = GBBSolver()
    solver.options["filename"] = "/this/path/does/not/exist/invalid_file.txt"
    with pytest.raises(SystemExit):
        solver.open_file()
    assert any("ERROR" in msg for msg in logs)


def test_gbbsolver_solve_returns_result(monkeypatch):
    # Patch gbb_solver to a dummy that returns fixed values
    called = {}

    def dummy_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        called["called"] = True
        return (0, 1.23, X, 0.456, 10, "Success message")

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", dummy_gbb_solver)

    problem = generate_square_problem(m=6, n=6, p=2, q=2)
    problem.stats = {"nbiter": 10, "fev": 5}  # minimal stats needed
    solver = GBBSolver()
    # Patch random init X0 to a fixed seed for reproducibility
    monkeypatch.setattr("numpy.random.rand", lambda n, p: np.ones((n, p)))
    result = solver.solve(problem)

    assert isinstance(result, OptimizeResult)
    assert result.success is True
    assert result.status == 0
    assert result.message == "Success message"
    assert np.allclose(result.solution.T @ result.solution, np.eye(problem.sizes[2]))
    assert "nbiter" in result
    assert "nfev" in result

    assert called.get("called")


def test_optimize_result_attributes():
    d = {"success": True, "status": 0, "message": "msg", "solution": np.array([[1]])}
    res = OptimizeResult(d)
    assert res.success is True
    assert res.status == 0
    assert res.message == "msg"
    assert np.array_equal(res.solution, np.array([[1]]))
    assert "success" in dir(res)

    # AttributeError on unknown attr
    with pytest.raises(AttributeError):
        _ = res.unknownattr


def test_gbbsolver_solve_error_when_full_results_but_missing_keys(monkeypatch):
    # Patch gbb_solver to a dummy
    def dummy_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        return (0, 0.0, X, 0.0, 0, "ok")

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", dummy_gbb_solver)

    problem = generate_square_problem(6, 6, 2, 2)
    solver = GBBSolver(full_results=True)
    # Remove keys intentionally to test Exception
    problem.stats = {"nbiter": 1, "fev": 1}

    with pytest.raises(Exception) as e:
        solver.solve(problem)
    assert "problem.stats[\"total_fun\"]" in str(e.value)


def test_gbbsolver_solve_with_full_results(monkeypatch):
    def dummy_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        return (0, 0.0, X, 0.0, 0, "ok")

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", dummy_gbb_solver)

    problem = generate_square_problem(6, 6, 2, 2)
    problem.stats = {
        "nbiter": 1,
        "fev": 1,
        "total_fun": [1.0, 0.5],
        "total_crit": [0.1, 0.05],
    }
    solver = GBBSolver(full_results=True)
    result = solver.solve(problem)
    assert isinstance(result.total_fun, list)
    assert isinstance(result.total_crit, list)


def test_gbbsolver_verbose_print(capsys):
    def dummy_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        print("Called dummy solver")
        return (0, 0.0, X, 0.0, 0, "ok")

    import sys
    import skprocrustes.skprocrustes as skpmod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(skpmod, "gbb_solver", dummy_gbb_solver)

    problem = generate_square_problem(6, 6, 2, 2)
    # Add required stats keys
    problem.stats = {"nbiter": 0, "fev": 0}

    solver = GBBSolver(verbose=1)
    solver.solve(problem)

    out = capsys.readouterr().out
    assert "GBB Solver" in out
    monkeypatch.undo()


def test_procrustesproblem_stats_initially_empty():
    p = generate_square_problem()
    assert isinstance(p.stats, dict)
    assert len(p.stats) == 0


def test_gbbsolver_options_projected_gradient_valid_values():
    # Should accept 1 and 2
    solver1 = GBBSolver(projected_gradient=1)
    solver2 = GBBSolver(projected_gradient=2)
    assert solver1.options["projected_gradient"] == 1
    assert solver2.options["projected_gradient"] == 2


def test_gbbsolver_options_set_via_kwargs():
    opts = {
        "full_results": True,
        "xtol": 1e-5,
        "gtol": 1e-4,
        "ftol": 1e-10,
        "zeta": 1e-3,
        "kappa": 0.2,
        "eta": 0.9,
        "etavar": True,
        "tau": 1e-2,
        "projected_gradient": 2,
        "maxiter": 1000,
        "verbose": 0,
        "filename": None,
        "timer": True,
    }
    solver = GBBSolver(**opts)
    for k, v in opts.items():
        assert solver.options[k] == v
