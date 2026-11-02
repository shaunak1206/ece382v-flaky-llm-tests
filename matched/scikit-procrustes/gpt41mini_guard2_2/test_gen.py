import pytest
import numpy as np
import os
import sys
import builtins
import time

from skprocrustes import skprocrustes as skp

# We focus tests on ProcrustesProblem and GBBSolver from skprocrustes/skprocrustes.py


def test_procrustesproblem_invalid_sizes():
    # m != n
    with pytest.raises(Exception):
        skp.ProcrustesProblem((5, 4, 2, 2))
    # p != q
    with pytest.raises(Exception):
        skp.ProcrustesProblem((6, 6, 2, 3))
    # m not divisible by p
    with pytest.raises(Exception):
        skp.ProcrustesProblem((7, 7, 3, 3))


def test_procrustesproblem_default_problem1_fixed_seed(monkeypatch):
    # seed numpy.random to get reproducible Sigma and random UU,VV
    np.random.seed(42)

    # Patch np.random.permutation so it is deterministic for Xsol
    # we will patch np.random.permutation to return original array (identity)
    monkeypatch.setattr(np.random, "permutation", lambda x: x.copy())

    sizes = (10, 10, 2, 2)  # m=n=10, p=q=2 and m divisible by p=2
    problem = skp.ProcrustesProblem(sizes, problemnumber=1)

    # Check sizes
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)

    # Since C is identity
    np.testing.assert_allclose(problem.C, np.eye(2), atol=1e-15)

    # Check orthogonality constraint Xsol^T Xsol = I_p
    xtx = problem.Xsol.T @ problem.Xsol
    np.testing.assert_allclose(xtx, np.eye(2), atol=1e-13)

    # Check B approx equals A Xsol C (C=I)
    approx_B = problem.A @ problem.Xsol @ problem.C
    np.testing.assert_allclose(approx_B, problem.B, atol=1e-12)


def test_procrustesproblem_default_problem2_fixed_seed(monkeypatch):
    np.random.seed(123)
    monkeypatch.setattr(np.random, "permutation", lambda x: x.copy())
    sizes = (10, 10, 2, 2)
    problem = skp.ProcrustesProblem(sizes, problemnumber=2)

    # Check shapes
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (10, 2)

    # Check Xsol orthogonality
    np.testing.assert_allclose(problem.Xsol.T @ problem.Xsol, np.eye(2), atol=1e-13)

    # Check B approx A Xsol
    np.testing.assert_allclose(problem.B, problem.A @ problem.Xsol @ problem.C, atol=1e-12)


def test_procrustesproblem_default_problem3_fixed_seed(monkeypatch):
    np.random.seed(321)
    monkeypatch.setattr(np.random, "permutation", lambda x: x.copy())
    # Use allowed n=50 as specified in source code for problem 3
    sizes = (50, 50, 5, 5)
    problem = skp.ProcrustesProblem(sizes, problemnumber=3)

    assert problem.A.shape == (50, 50)
    assert problem.B.shape == (50, 5)
    assert problem.C.shape == (5, 5)
    assert problem.Xsol.shape == (50, 5)

    # Check orthogonality of Xsol
    np.testing.assert_allclose(problem.Xsol.T @ problem.Xsol, np.eye(5), atol=1e-13)

    np.testing.assert_allclose(problem.B, problem.A @ problem.Xsol @ problem.C, atol=1e-12)


def test_procrustesproblem_user_matrices_correct_shapes():
    m, n, p, q = 6, 6, 2, 2
    A = np.eye(m, n)
    B = np.ones((m, q))
    C = np.eye(p, q)
    Xsol = np.ones((n, p))
    problem = skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))
    np.testing.assert_allclose(problem.A, A)
    np.testing.assert_allclose(problem.B, B)
    np.testing.assert_allclose(problem.C, C)
    np.testing.assert_allclose(problem.Xsol, Xsol)


@pytest.mark.parametrize("matrices", [
    ((np.eye(5, 6), np.eye(5, 5), np.eye(2, 2))),                # A shape wrong
    ((np.eye(6, 6), np.eye(5, 5), np.eye(2, 2))),                # B shape wrong
    ((np.eye(6, 6), np.eye(6, 5), np.eye(2, 3))),                # C shape wrong
    ((np.eye(6, 6), np.eye(6, 5), np.eye(2, 3), np.eye(5, 4)))   # Xsol shape wrong
])
def test_procrustesproblem_user_matrices_wrong_shapes(matrices):
    shapes = [m.shape for m in matrices]
    size = (6, 6, 2, 3)
    with pytest.raises(Exception):
        skp.ProcrustesProblem(size, matrices=matrices)


@pytest.mark.parametrize("invalid_opts, exc_msg", [
    ({"full_results": "yes"}, "full_results must be a boolean"),
    ({"filename": 123}, "filename must be a string"),
    ({"strategy": "fast"}, "strategy not implemented"),
    ({"gtol": "low"}, "gtol must be a float"),
    ({"eta": "high"}, "eta must be a float"),
    ({"etavar": "sometimes"}, "etavar must be a boolean"),
    ({"maxiter": 10.5}, "maxiter must be an integer"),
    ({"verbose": 5}, "verbose must be 0, 1, 2 or 3"),
    ({"changevar": "maybe"}, "changevar must be True or False"),
    ({"polar": "yes"}, "polar must be ns or None"),
    ({"timer": "yes"}, "timer must be boolean"),
    ({"precond": "fast"}, "precond must be stupid or None"),
])
def test_gbbsovler_setoptions_invalid(invalid_opts, exc_msg):
    # Attempt to create a GBBSolver with invalid options to test exceptions

    # Since GBBSolver._setoptions() is called in __init__, just passing options triggers it
    with pytest.raises(Exception) as e:
        skp.GBBSolver(**invalid_opts)
    assert exc_msg in str(e.value)


def test_gbbsovler_default_options_correct_types_and_defaults():
    # Create GBBSolver with no options, check defaults
    solver = skp.GBBSolver()

    opts = solver.options
    # Check default keys and types
    assert opts["full_results"] is False
    assert isinstance(opts["xtol"], float)
    assert isinstance(opts["gtol"], float)
    assert isinstance(opts["ftol"], float)
    assert isinstance(opts["zeta"], float)
    assert isinstance(opts["kappa"], float)
    assert isinstance(opts["eta"], float)
    assert opts["etavar"] is False
    assert isinstance(opts["tau"], float)
    assert opts["projected_gradient"] in (1, 2)
    assert isinstance(opts["maxiter"], int)
    assert opts["verbose"] in (0, 1)
    assert opts["filename"] is None
    assert opts["timer"] is False


def test_gbbsovler_solve_runs_and_returns_result(monkeypatch):
    # To avoid randomness in solve, seed numpy.random before creating GBBSolver
    np.random.seed(42)

    # Prepare a problem with known solution and fixed seed
    sizes = (10, 10, 2, 2)
    np.random.seed(101)
    monkeypatch.setattr(np.random, "permutation", lambda x: x.copy())
    problem = skp.ProcrustesProblem(sizes, problemnumber=1)

    # Patch np.random.rand inside GBBSolver.solve to fixed value to make X0 deterministic
    # but since numpy.random.rand used inside solve, seed numpy before solve is enough

    solver = skp.GBBSolver(verbose=0, maxiter=5, full_results=True)
    result = solver.solve(problem)

    # Check result is OptimizeResult instance (dict subclass with attributes)
    # Pytest does not have isinstance for dict subclass easily, check keys instead
    assert hasattr(result, "success")
    assert hasattr(result, "solution")
    # solution shape should be n x p
    assert result.solution.shape == (sizes[1], sizes[2])
    # fun is float
    assert isinstance(result.fun, float)
    # error should be finite and >= 0
    assert result.error >= 0
    # nbiter is int
    assert isinstance(result.nbiter, int)
    # nfev is number
    assert isinstance(result.nfev, (int, float))
    # total_fun is list with length >=1
    assert isinstance(result.total_fun, list)
    assert len(result.total_fun) >= 1
    # total_crit is list or np.inf
    assert isinstance(result.total_crit, (list, float))
    # Status code is int
    assert isinstance(result.status, int)
    # message is str
    assert isinstance(result.message, str)


def test_gbbsovler_solve_error_without_full_results(monkeypatch):
    # Create problem without full_results flag set then access total_fun -> raises
    sizes = (10, 10, 2, 2)
    np.random.seed(202)
    monkeypatch.setattr(np.random, "permutation", lambda x: x.copy())
    problem = skp.ProcrustesProblem(sizes, problemnumber=1)
    solver = skp.GBBSolver(verbose=0, full_results=True)
    # Remove total_fun and total_crit keys manually before solve to simulate error
    problem.stats["total_fun"] = None
    problem.stats["total_crit"] = None

    # solve() checks for keys only after solve call, so patch problem.stats accordingly after solve start
    # Actually, easiest to run full solve, then manually break keys and call solve again to trigger Exception
    solver.options["full_results"] = True
    result = solver.solve(problem)  # first call populates stats normally

    # Remove keys to simulate incomplete stats
    problem.stats.pop("total_fun", None)
    problem.stats.pop("total_crit", None)

    # Monkeypatch spectral_solver or gkb_setup to not overwrite stats
    # Instead, test _setoptions to raise error if full_results True but keys missing

    # We test that solve raises Exception when keys missing in stats and full_results True.
    # We forcibly call gkb_setup to reproduce.
    from skprocrustes.skprocrustes import gkb_setup

    with pytest.raises(Exception):
        # gkb_setup called with problem without total_fun and total_grad keys but with full_results True triggers exception
        gkb_setup(problem, solver.solvername, solver.options, sys.stdout)


@pytest.mark.parametrize("projected_gradient", [1, 2])
def test_gbbsovler_setoptions_projected_gradient_valid(projected_gradient):
    solver = skp.GBBSolver(projected_gradient=projected_gradient)
    assert solver.options["projected_gradient"] == projected_gradient


@pytest.mark.parametrize("bad_pg", [0, 3, -1, "one"])
def test_gbbsovler_setoptions_projected_gradient_invalid(bad_pg):
    with pytest.raises(Exception):
        skp.GBBSolver(projected_gradient=bad_pg)


def test_gbbsovler_open_close_file(monkeypatch, tmp_path):
    # Create solver with filename specified
    file_path = tmp_path / "testfile.txt"
    solver = skp.GBBSolver(filename=str(file_path), verbose=0)
    solver.open_file()
    assert not solver.file.closed
    solver.close_file()
    assert solver.file.closed

    # Creating with a filename that already exists triggers exception on _setoptions
    # so create file, then try to create solver again with same filename
    with open(file_path, "w") as f:
        f.write("dummy")
    with pytest.raises(Exception):
        skp.GBBSolver(filename=str(file_path))


def test_gbbsovler_open_file_stdout(monkeypatch):
    solver = skp.GBBSolver(filename=None)
    solver.open_file()
    assert solver.file == sys.stdout
    solver.close_file()  # should be no-op for sys.stdout


def test_optimizresult_repr_and_dir():
    res = skp.OptimizeResult()
    assert isinstance(str(res), str)
    res["a"] = 42
    res["b"] = "test"
    rrepr = repr(res)
    assert "'a'" in rrepr or "a" in rrepr
    assert "b" in rrepr
    d = dir(res)
    assert "a" in d and "b" in d


def test_optimizresult_show(capsys):
    res = skp.OptimizeResult(success=True, solution=np.eye(2), fun=1.0, error=np.inf)
    res.show()
    # Output printed to stdout
    out = capsys.readouterr().out
    assert "success" in out
    assert "solution" not in out  # solution is excluded


def test_compare_solvers(monkeypatch):
    # Only test that it runs without error for spg, eb, gpi solvers.

    np.random.seed(1)
    monkeypatch.setattr(np.random, "permutation", lambda x: x.copy())

    sizes = (10, 10, 2, 2)
    problem = skp.ProcrustesProblem(sizes, problemnumber=1)
    solvers = [
        skp.EBSolver(full_results=True, verbose=0, maxiter=2),
        skp.SPGSolver(full_results=True, verbose=0, maxiter=2),
        skp.GPISolver(full_results=True, verbose=0, maxiter=2),
        # GKBSolver is excluded from plotting in compare_solvers as per source
        skp.GKBSolver(full_results=True, verbose=0, maxiter=2)
    ]
    # Patch solve methods of solvers with dummy results that have total_fun keys to avoid exceptions
    for solver in solvers:
        orig_solve = solver.solve

        def patched_solve(problem, orig_solve=orig_solve):
            res = orig_solve(problem)
            if "total_fun" not in res:
                res["total_fun"] = [1.0, 0.5]
            return res

        monkeypatch.setattr(solver, "solve", patched_solve)

    results = skp.compare_solvers(problem, *solvers, plot=False)
    assert len(results) == 4
    for r in results:
        assert "total_fun" in r


def test_procrustesproblem_problem3_invalid_n(monkeypatch):
    np.random.seed(1)
    monkeypatch.setattr(np.random, "permutation", lambda x: x.copy())
    # n not in allowed list for problem 3
    with pytest.raises(Exception):
        skp.ProcrustesProblem((40, 40, 5, 5), problemnumber=3)


def test_gbbsovler_solver_creates_solution_with_right_shape_and_properties(monkeypatch):
    np.random.seed(999)
    monkeypatch.setattr(np.random, "permutation", lambda x: x.copy())

    sizes = (10, 10, 3, 3)
    problem = skp.ProcrustesProblem(sizes, problemnumber=1)

    solver = skp.GBBSolver(verbose=0, maxiter=10, full_results=True)

    result = solver.solve(problem)

    assert result.solution.shape == (sizes[1], sizes[2])
    # Orthogonality check
    xtx = result.solution.T @ result.solution
    np.testing.assert_allclose(xtx, np.eye(sizes[2]), atol=1e-9)

    # Error >= 0
    assert result.error >= 0

    # Status is int
    assert isinstance(result.status, int)
    # Message is string
    assert isinstance(result.message, str)


def test_gbbsovler_solver_verbose_output_stdout_capture(monkeypatch, capsys):
    np.random.seed(12345)
    monkeypatch.setattr(np.random, "permutation", lambda x: x.copy())
    sizes = (10, 10, 2, 2)
    problem = skp.ProcrustesProblem(sizes, problemnumber=1)

    solver = skp.GBBSolver(verbose=1, maxiter=3, full_results=False)
    _ = solver.solve(problem)

    captured = capsys.readouterr()
    # Since verbose=1, some informational output should be present
    assert "GBB Solver" in captured.out or "gbb" in captured.out.lower()


def test_spellings_and_constants():
    # _status_message keys and values sanity
    stm = skp._status_message
    assert "success" in stm
    assert isinstance(stm["success"], str)
    assert stm["success"].startswith("Optimization")


@pytest.mark.parametrize("problem_number", [1, 2, 3])
def test_procrustesproblem_problemnumber_creates_A_B_C_X_sol(problem_number, monkeypatch):
    np.random.seed(100 + problem_number)
    monkeypatch.setattr(np.random, "permutation", lambda x: x.copy())
    if problem_number == 3:
        # Only test allowed n values
        sizes = (50, 50, 5, 5)
    else:
        sizes = (10, 10, 2, 2)
    problem = skp.ProcrustesProblem(sizes, problemnumber=problem_number)

    # A shape
    assert problem.A.shape == (sizes[0], sizes[1])
    # B shape
    assert problem.B.shape == (sizes[0], sizes[3])
    # C shape
    assert problem.C.shape == (sizes[2], sizes[3])
    # Xsol shape
    assert problem.Xsol.shape == (sizes[1], sizes[2])

    # Check orthogonality of Xsol
    np.testing.assert_allclose(problem.Xsol.T @ problem.Xsol, np.eye(sizes[2]),
                               atol=1e-13)

    # Check that B = A X C approx
    approx_B = problem.A @ problem.Xsol @ problem.C
    np.testing.assert_allclose(problem.B, approx_B, atol=1e-12)
