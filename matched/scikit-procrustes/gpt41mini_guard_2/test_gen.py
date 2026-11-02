import numpy as np
import pytest
import numpy.linalg as nla

import skprocrustes.skprocrustes as skp


def test_procrustesproblem_invalid_sizes_non_square():
    with pytest.raises(Exception, match="square problems are supported"):
        skp.ProcrustesProblem((3, 4, 2, 2))


def test_procrustesproblem_invalid_sizes_not_divisible():
    # m=6, p=4 but 6 not divisible by 4
    with pytest.raises(Exception, match="divisible by p"):
        skp.ProcrustesProblem((6, 6, 4, 4))


def test_procrustesproblem_default_problem_1_fixed_seed(monkeypatch):
    # Fix random seed for reproducibility
    np.random.seed(12345)
    # m=n=10, p=q=2, m divisible by p=2
    problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    A = problem.A
    B = problem.B
    C = problem.C
    Xsol = problem.Xsol

    # Shapes must match
    assert A.shape == (10, 10)
    assert B.shape == (10, 2)
    assert C.shape == (2, 2)
    assert Xsol.shape == (10, 2)

    # C should be identity
    assert np.allclose(C, np.eye(2))

    # B == A * Xsol * C within floating tolerance
    B_reconstructed = A @ Xsol @ C
    assert np.allclose(B, B_reconstructed, atol=1e-12)

    # Xsol orthogonal: Xsol.T @ Xsol = I
    identity = Xsol.T @ Xsol
    assert np.allclose(identity, np.eye(Xsol.shape[1]), atol=1e-12)

    # A should be non-singular (since well conditioned)
    sv = nla.svdvals(A)
    assert np.min(sv) > 0


def test_procrustesproblem_default_problem_2_fixed_seed():
    np.random.seed(54321)
    problem = skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=2)
    A = problem.A
    B = problem.B
    C = problem.C
    Xsol = problem.Xsol

    assert A.shape == (10, 10)
    assert B.shape == (10, 2)
    assert C.shape == (2, 2)
    assert Xsol.shape == (10, 2)
    assert np.allclose(C, np.eye(2))

    B_reconstructed = A @ Xsol @ C
    assert np.allclose(B, B_reconstructed, atol=1e-12)
    identity = Xsol.T @ Xsol
    assert np.allclose(identity, np.eye(Xsol.shape[1]), atol=1e-12)

    sv = nla.svdvals(A)
    assert np.min(sv) > 0


def test_procrustesproblem_default_problem_3_fixed_seed():
    np.random.seed(99999)
    # Using n=50 because problem 3 restricts n
    problem = skp.ProcrustesProblem((50, 50, 10, 10), problemnumber=3)
    A = problem.A
    B = problem.B
    C = problem.C
    Xsol = problem.Xsol

    assert A.shape == (50, 50)
    assert B.shape == (50, 10)
    assert C.shape == (10, 10)
    assert Xsol.shape == (50, 10)
    assert np.allclose(C, np.eye(10))

    B_reconstructed = A @ Xsol @ C
    assert np.allclose(B, B_reconstructed, atol=1e-12)

    identity = Xsol.T @ Xsol
    assert np.allclose(identity, np.eye(Xsol.shape[1]), atol=1e-12)


def test_procrustesproblem_user_defined_matrices_valid():
    m, n, p, q = 6, 6, 3, 3
    np.random.seed(123)
    A = np.random.randn(m, n)
    Xsol = np.eye(n, p)
    B = A @ Xsol @ np.eye(p, q)
    C = np.eye(p, q)

    problem = skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))
    # Matrices retain shape and data
    assert np.allclose(problem.A, A)
    assert np.allclose(problem.B, B)
    assert np.allclose(problem.C, C)
    assert np.allclose(problem.Xsol, Xsol)


def test_procrustesproblem_user_defined_matrices_invalid_shapes():
    m, n, p, q = 4, 4, 2, 2
    A = np.eye(m, n)
    B = np.eye(m, q)
    C = np.eye(p, q)
    X_sol = np.eye(n, p)

    # Modify A shape invalid
    with pytest.raises(Exception, match="A must be"):
        skp.ProcrustesProblem((m,n,p,q), matrices=(np.eye(m+1,n), B, C))

    # Modify B shape invalid
    with pytest.raises(Exception, match="B must be"):
        skp.ProcrustesProblem((m,n,p,q), matrices=(A, np.eye(m, q+1), C))

    # Modify C shape invalid
    with pytest.raises(Exception, match="C must be"):
        skp.ProcrustesProblem((m,n,p,q), matrices=(A, B, np.eye(p+1, q)))

    # Modify Xsol shape invalid
    with pytest.raises(Exception, match="Xsol must be"):
        skp.ProcrustesProblem((m,n,p,q), matrices=(A, B, C, np.eye(n+1, p)))


def test_gbbsolver_default_options_and_setoptions_defaults():
    solver = skp.GBBSolver()
    opts = solver.options

    # Check default options are set and types
    assert "full_results" in opts and opts["full_results"] is False
    assert "xtol" in opts and isinstance(opts["xtol"], float)
    assert "gtol" in opts and isinstance(opts["gtol"], float)
    assert "ftol" in opts and isinstance(opts["ftol"], float)
    assert "zeta" in opts and isinstance(opts["zeta"], float)
    assert "kappa" in opts and isinstance(opts["kappa"], float)
    assert "eta" in opts and isinstance(opts["eta"], float)
    assert "etavar" in opts and isinstance(opts["etavar"], bool)
    assert "tau" in opts and isinstance(opts["tau"], float)
    assert "projected_gradient" in opts and opts["projected_gradient"] in (1, 2)
    assert "maxiter" in opts and isinstance(opts["maxiter"], int)
    assert "verbose" in opts and opts["verbose"] in (0, 1)
    assert "filename" in opts and (opts["filename"] is None or isinstance(opts["filename"], str))
    assert "timer" in opts and isinstance(opts["timer"], bool)


@pytest.mark.parametrize("invalid_opts, error_msg", [
    ({"full_results": "yes"}, "full_results must be a boolean"),
    ({"xtol": "small"}, "xtol must be a float"),
    ({"gtol": "small"}, "gtol must be a float"),
    ({"ftol": "small"}, "ftol must be a float"),
    ({"zeta": "param"}, "zeta must be a float"),
    ({"kappa": "param"}, "kappa must be a float"),
    ({"eta": "param"}, "eta must be a float"),
    ({"etavar": "no"}, "etavar must be a boolean"),
    ({"tau": "small"}, "tau must be a float"),
    ({"projected_gradient": 3}, "projected_gradient must be 1 or 2"),
    ({"maxiter": 10.5}, "maxiter must be an integer"),
    ({"verbose": 5}, "verbose must be 0, or 1"),
    ({"filename": 123}, "filename must be string"),
    ({"timer": "yes"}, "timer must be boolean"),
])
def test_gbbsolver_setoptions_invalid_types(invalid_opts, error_msg):
    # Passing invalid options dictionary should raise Exception with correct message
    with pytest.raises(Exception, match=error_msg):
        skp.GBBSolver(**invalid_opts)


def test_gbbsolver_open_close_file(monkeypatch):
    solver = skp.GBBSolver()
    # Default filename None, file should be sys.stdout
    solver.open_file()
    import sys
    assert solver.file == sys.stdout
    solver.close_file()  # nothing to do but coverage

    # Provide filename, test file open/close
    import tempfile
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        filename = tmp.name
    # File exists, _setoptions raises Exception in constructor; workaround:
    # set filename after __init__
    solver.options["filename"] = "test_gbbsolver_file.txt"

    # The file should not exist to allow open
    import os
    if os.path.exists(solver.options["filename"]):
        os.remove(solver.options["filename"])

    # Test open_file creates file descriptor
    solver.open_file()
    assert hasattr(solver.file, "write")
    solver.file.write("test\n")
    solver.close_file()

    with open(solver.options["filename"], "r") as f:
        content = f.read()
    assert "test" in content
    os.remove(solver.options["filename"])


def test_gbbsolver_solve_runs_and_returns(monkeypatch):
    # This test ensures that the solve() method runs
    # This cannot test correct optimization, but can verify outputs shape and types

    rng = np.random.default_rng(2021)

    # Fix the seed for reproducibility inside the function
    np.random.seed(2021)

    problem = skp.ProcrustesProblem((6, 6, 3, 3), problemnumber=1)

    solver = skp.GBBSolver(full_results=True, maxiter=2, verbose=0)

    result = solver.solve(problem)

    # Check returned type is OptimizeResult (dict subclass)
    assert isinstance(result, skp.OptimizeResult)

    # Check success boolean is present and boolean
    assert "success" in result
    assert isinstance(result.success, bool)

    # solution shape should be (n,p)
    sol = result.solution
    assert isinstance(sol, np.ndarray)
    assert sol.shape == (problem.sizes[1], problem.sizes[2])

    # objective function value should be float
    assert isinstance(result.fun, float)

    # error is inf or float
    assert result.error == np.inf or isinstance(result.error, float)

    # Number of iterations must be int
    assert isinstance(result.nbiter, int)

    # total_fun and total_crit should be lists if full_results else inf
    if solver.options["full_results"]:
        assert isinstance(result.total_fun, list)
        assert isinstance(result.total_crit, list)
    else:
        assert result.total_fun == np.inf
        assert result.total_crit == np.inf


def test_gbbsolver_solve_without_xsol_sets_error_inf(monkeypatch):
    # Create problem without Xsol to verify error is inf
    m, n, p, q = 4, 4, 2, 2
    A = np.eye(m, n)
    B = np.eye(m, q)
    C = np.eye(p, q)
    problem = skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C))

    solver = skp.GBBSolver(maxiter=1, verbose=0)

    result = solver.solve(problem)

    assert "error" in result
    # Without Xsol, error is np.inf
    assert result.error == np.inf


def test_procrustesproblem_problemnumber_none_and_no_matrices():
    # If both matrices and problemnumber are None, should raise Exception

    with pytest.raises(Exception, match="no problem number has been chosen"):
        skp.ProcrustesProblem((4, 4, 2, 2))


def test_gbbsolver_with_full_results_requires_stats(monkeypatch):
    # Manually create a problem and set stats without total_fun keys

    m, n, p, q = 4, 4, 2, 2
    problem = skp.ProcrustesProblem((m, n, p, q), problemnumber=1)

    problem.stats["nbiter"] = 0
    problem.stats["fev"] = 0
    problem.stats["blocksteps"] = 0
    # Not setting total_fun or total_crit keys

    solver = skp.GBBSolver(full_results=True, maxiter=1, verbose=0)

    # Without total_fun and total_crit should raise Exception in solve
    with pytest.raises(Exception, match="For full results, set"):
        solver.solve(problem)


def test_optimizresult_dict_behavior():
    d = skp.OptimizeResult()
    d["a"] = 123
    d["b"] = [1,2,3]

    assert d.a == 123
    assert d.b == [1,2,3]
    with pytest.raises(AttributeError):
        _ = d.c

    d.c = "hello"
    assert d.c == "hello"
    del d.c
    with pytest.raises(AttributeError):
        _ = d.c

    rep = repr(d)
    assert isinstance(rep, str)
    assert "a" in rep and "b" in rep

    keys = dir(d)
    assert "a" in keys
    assert "b" in keys


def test_procrustesproblem_user_matrices_not_list_or_tuple():
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices="notalist")


def test_gbbsolver_projected_gradient_values():
    for val in [1, 2]:
        solver = skp.GBBSolver(projected_gradient=val)
        assert solver.options["projected_gradient"] == val

    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        skp.GBBSolver(projected_gradient=3)


@pytest.mark.parametrize("verbose_level", [0, 1])
def test_gbbsolver_verbose_levels(verbose_level):
    solver = skp.GBBSolver(verbose=verbose_level)
    assert solver.options["verbose"] == verbose_level


@pytest.mark.parametrize("verbose_invalid", [-1, 2])
def test_gbbsolver_invalid_verbose(verbose_invalid):
    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        skp.GBBSolver(verbose=verbose_invalid)
