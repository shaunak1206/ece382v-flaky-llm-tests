import pytest
import numpy as np
from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, OptimizeResult, _status_message, spg_solver


@pytest.mark.parametrize("sizes", [
    (10, 10, 2, 2),
    (6, 6, 3, 3),
])
def test_procrustesproblem_default_problems(sizes):
    # Test constructing ProcrustesProblem with default problem 1, 2, 3 when possible
    m, n, p, q = sizes

    # Problem 1 always supported
    problem1 = ProcrustesProblem(sizes, problemnumber=1)
    assert problem1.A.shape == (m, n)
    assert problem1.B.shape == (m, q)
    assert problem1.C.shape == (p, q)
    assert problem1.Xsol.shape == (n, p)

    # Problem 2 always supported
    problem2 = ProcrustesProblem(sizes, problemnumber=2)
    assert problem2.A.shape == (m, n)
    assert problem2.B.shape == (m, q)
    assert problem2.C.shape == (p, q)
    assert problem2.Xsol.shape == (n, p)

    # Problem 3 supports only specific sizes for n
    if n in [50, 95, 500, 1000]:
        problem3 = ProcrustesProblem(sizes, problemnumber=3)
        assert problem3.A.shape == (m, n)
        assert problem3.B.shape == (m, q)
        assert problem3.C.shape == (p, q)
        assert problem3.Xsol.shape == (n, p)
    else:
        with pytest.raises(Exception):
            ProcrustesProblem(sizes, problemnumber=3)


def test_procrustesproblem_invalid_sizes():
    # m != n or p != q should raise Exception
    with pytest.raises(Exception):
        ProcrustesProblem((10, 9, 2, 2))
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 3))

    # m must be divisible by p
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 3, 3))


def test_procrustesproblem_user_matrices_valid():
    m, n, p, q = 6, 6, 3, 3
    A = np.random.randn(m, n)
    B = np.random.randn(m, q)
    C = np.eye(p, q)
    Xsol = np.random.randn(n, p)
    problem = ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))
    assert np.allclose(problem.A, A)
    assert np.allclose(problem.B, B)
    assert np.allclose(problem.C, C)
    assert np.allclose(problem.Xsol, Xsol)

    # Without Xsol
    problem2 = ProcrustesProblem((m, n, p, q), matrices=(A, B, C))
    assert np.allclose(problem2.A, A)
    assert np.allclose(problem2.B, B)
    assert np.allclose(problem2.C, C)
    assert not hasattr(problem2, 'Xsol')


@pytest.mark.parametrize("matrices_invalid", [
    ([np.random.randn(5, 5), np.random.randn(5, 4), np.eye(3, 3)]),  # B shape invalid
    ([np.random.randn(5, 5), np.random.randn(5, 5), np.eye(2, 3)]),  # C shape invalid
    ([np.random.randn(5, 5), np.random.randn(5, 5)]),  # less than 3 matrices
    ([np.random.randn(5, 5), np.random.randn(5, 5), np.eye(3, 3), np.ones((4, 4))]),  # Xsol shape invalid
])
def test_procrustesproblem_user_matrices_invalid(matrices_invalid):
    sizes = (5, 5, 3, 3)
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=matrices_invalid)


def test_optimize_result_attribute_access_and_repr():
    res = OptimizeResult()
    # setting values
    res.success = True
    res.status = 0
    res.message = "ok"
    res.solution = np.eye(3)
    # attribute access
    assert res.success is True
    assert res['status'] == 0
    with pytest.raises(AttributeError):
        _ = res.nonexistent_attribute
    repr_str = repr(res)
    assert "success" in repr_str and "status" in repr_str


def test_gbbs_solver_default_options():
    # Instantiate GBBSolver without options; defaults should be set
    solver = GBBSolver()
    options = solver.options
    # Check default option keys and types
    assert isinstance(options["full_results"], bool)
    assert isinstance(options["xtol"], float)
    assert isinstance(options["gtol"], float)
    assert options["projected_gradient"] in (1, 2)
    assert options["verbose"] in (0, 1)


def test_gbbs_solver_setoptions_invalid_values():
    # Test invalid option: full_results must be bool
    with pytest.raises(Exception):
        GBBSolver(full_results="yes")

    # xtol must be float
    with pytest.raises(Exception):
        GBBSolver(xtol="1e-6")

    # gtol must be float
    with pytest.raises(Exception):
        GBBSolver(gtol="1e-6")

    # ftol must be float
    with pytest.raises(Exception):
        GBBSolver(ftol="1e-12")

    # zeta must be float
    with pytest.raises(Exception):
        GBBSolver(zeta="1e-4")

    # kappa must be float
    with pytest.raises(Exception):
        GBBSolver(kappa="0.1")

    # eta must be float
    with pytest.raises(Exception):
        GBBSolver(eta="0.85")

    # etavar must be bool
    with pytest.raises(Exception):
        GBBSolver(etavar=1)

    # tau must be float
    with pytest.raises(Exception):
        GBBSolver(tau="1e-3")

    # projected_gradient must be 1 or 2
    with pytest.raises(Exception):
        GBBSolver(projected_gradient=3)

    # maxiter must be int
    with pytest.raises(Exception):
        GBBSolver(maxiter=5.5)

    # verbose must be 0 or 1
    with pytest.raises(Exception):
        GBBSolver(verbose=5)

    # filename must be string
    with pytest.raises(Exception):
        GBBSolver(filename=123)

    # timer must be bool
    with pytest.raises(Exception):
        GBBSolver(timer="no")


def test_gbbs_solver_open_close_file(tmp_path):
    # Test file open and close for valid filename
    solver = GBBSolver(filename=str(tmp_path / "logfile.txt"))
    solver.open_file()
    assert not solver.file.closed
    solver.close_file()
    assert solver.file.closed

    # Test that open_file defaults to sys.stdout
    solver2 = GBBSolver()
    solver2.open_file()
    import sys
    assert solver2.file == sys.stdout
    solver2.close_file()


def _setup_simple_problem_and_solver():
    m = n = 6
    p = q = 3
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)
    solver = GBBSolver(full_results=False, verbose=0)
    return problem, solver


def test_gbbs_solver_solve_basic_runs(monkeypatch):
    problem, solver = _setup_simple_problem_and_solver()

    # We patch gbb_solver to a dummy function returning expected tuple to isolate test
    def fake_gbb_solver(problem, largedim, smalldim, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        n, p = smalldim, problem.sizes[2]
        Xsol = np.eye(n, p)
        exitcode = 0
        fval = 1.23
        normG = 0.001
        nbiter = 2
        msg = _status_message['success']
        return exitcode, fval, Xsol, normG, nbiter, msg

    # Patch the function in the module
    import skprocrustes.skprocrustes as skp
    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    result = solver.solve(problem)
    assert isinstance(result, OptimizeResult)
    assert result.success is True
    assert result.status == 0
    assert np.allclose(result.solution.T @ result.solution, np.eye(problem.sizes[2]))
    assert result.nbiter == 2
    assert result.message == _status_message['success']


def test_gbbs_solver_solve_with_known_solution(monkeypatch):
    problem, solver = _setup_simple_problem_and_solver()
    # Provide known solution very close to identity
    Xsol = np.eye(problem.sizes[1], problem.sizes[2])
    problem.Xsol = Xsol

    # Patch gbb_solver to return a solution close to Xsol
    def fake_gbb_solver(problem, largedim, smalldim, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        exitcode = 0
        fval = 0.5
        Xsol_perturbed = X + 0.1
        normG = 0.002
        nbiter = 1
        msg = _status_message['success']
        return exitcode, fval, Xsol_perturbed, normG, nbiter, msg

    import skprocrustes.skprocrustes as skp
    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    result = solver.solve(problem)
    assert np.isfinite(result.error)
    assert np.abs(result.fun - 0.5) < 1e-6
    assert result.nbiter == 1
    assert isinstance(result.cpu, float)


def test_spg_solver_returns_expected_structure():
    # Setup a simple problem for spg_solver
    m = n = 6
    p = q = 3
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)
    options = {
        "full_results": False,
        "changevar": False,
        "precond": None,
        "verbose": 0,
        "filename": None,
        "strategy": "newfw",
        "gtol": 1e-3,
        "eta": 0.85,
        "etavar": False,
        "maxiter": 10,
        "polar": None,
        "timer": False,
        "precond": None
    }
    import sys
    Xk, f, opt, exitcode, msg = spg_solver(problem, "spg", options, sys.stdout)
    assert Xk.shape == (n, p)
    assert isinstance(f, float)
    assert exitcode in (0, 1)
    assert isinstance(msg, str)


def test_optimality_shapes_and_values():
    from skprocrustes.skprocrustes import optimality
    m = n = 6
    p = q = 3
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)
    X = np.eye(n, p)
    R = np.dot(problem.A, np.dot(X, problem.C)) - problem.B

    normgradproj, normgrad, grad = optimality(problem.A, problem.C, R, X, None)
    assert np.isscalar(normgradproj)
    assert np.isscalar(normgrad)
    assert grad.shape == (n, p)


def test_compute_residual_with_and_without_precond():
    from skprocrustes.skprocrustes import compute_residual
    m = n = 6
    p = q = 3
    problem = ProcrustesProblem((m, n, p, q), problemnumber=1)
    X = np.eye(n, p)

    R, res1 = compute_residual(problem.A, problem.B, problem.C, X, precond=None)
    R2, res2 = compute_residual(problem.A, problem.B, problem.C, X, precond=2.0)
    # Residual with precond should be scaled squared
    assert np.allclose(R, R2)
    assert np.isclose(res2, 4 * res1)


def test_open_close_file_with_invalid_filename(monkeypatch):
    solver = GBBSolver(filename="/root/invalid_file")
    # Patch sys.exit to prevent test exit
    exited = {}
    def fake_exit(code):
        exited['called'] = True
        raise SystemExit(code)

    monkeypatch.setattr(solver, "file", None)
    monkeypatch.setattr("sys.exit", fake_exit)
    with pytest.raises(SystemExit):
        solver.open_file()
    assert exited.get('called')


def test_optimize_result_show(capsys):
    res = OptimizeResult()
    res.success = True
    res.status = 0
    res.error = np.inf
    res.message = "done"
    res.solution = np.eye(2)
    res.show()
    out = capsys.readouterr().out
    assert "success" in out
    assert "error" not in out   # because error == np.inf is skipped
    assert "done" in out


def test_procrustesproblem_stats_empty_on_init():
    problem = ProcrustesProblem((6, 6, 3, 3), problemnumber=1)
    assert isinstance(problem.stats, dict)
    assert not problem.stats  # Empty dict on init


def test_gbbs_solver_options_custom_values():
    opts = {
        "full_results": True,
        "xtol": 1e-4,
        "gtol": 1e-4,
        "ftol": 1e-9,
        "zeta": 1e-3,
        "kappa": 0.05,
        "eta": 0.8,
        "etavar": True,
        "tau": 5e-4,
        "projected_gradient": 2,
        "maxiter": 50,
        "verbose": 1,
        "filename": None,
        "timer": True
    }
    solver = GBBSolver(**opts)
    for k, v in opts.items():
        assert solver.options[k] == v


def test_procrustesproblem_invalid_matrices_type():
    sizes = (6, 6, 3, 3)
    # matrices not a list/tuple
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices="invalid")


def test_procrustesproblem_matrices_length_less_than_3():
    sizes = (6, 6, 3, 3)
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=[np.eye(6)])


def test_procrustesproblem_matrices_shapes_mismatch():
    sizes = (6, 6, 3, 3)
    A = np.eye(6)
    B = np.eye(6, 2)  # wrong shape (6,2) instead of (6,3)
    C = np.eye(3)
    # B shape error
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(A, B, C))

    B = np.eye(6, 3)
    C = np.eye(2, 3)  # wrong shape (2,3) instead of (3,3)
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(A, B, C))

    Xsol = np.eye(5, 3)  # wrong shape (5,3) instead of (6,3)
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=(A, B, C, Xsol))
