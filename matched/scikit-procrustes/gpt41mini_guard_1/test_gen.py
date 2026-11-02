import numpy as np
import pytest
import sys
import os
import io

import skprocrustes.skprocrustes as skp

# Seed numpy random to ensure no randomness in tests that depends on it
np.random.seed(12345)


def make_square_problem(m=6, n=6, p=3, q=3, problemnumber=1):
    # Helper to create ProcrustesProblem with m=n and p=q and m divisible by p
    return skp.ProcrustesProblem((m, n, p, q), problemnumber=problemnumber)


def test_procrustesproblem_valid_init_with_predefined_problem():
    problem = make_square_problem(6, 6, 3, 3, problemnumber=1)
    assert problem.A.shape == (6, 6)
    assert problem.B.shape == (6, 3)
    assert problem.C.shape == (3, 3)
    assert problem.Xsol.shape == (6, 3)
    # Check known solution orthogonality Xsol^T Xsol == I
    val = problem.Xsol.T @ problem.Xsol
    np.testing.assert_allclose(val, np.eye(3), atol=1e-12)


def test_procrustesproblem_invalid_size_raises():
    with pytest.raises(Exception, match="Currently only square problems are supported."):
        skp.ProcrustesProblem((4, 5, 2, 2))
    with pytest.raises(Exception, match="Dimensions are incompatible: m must be divisible by p."):
        skp.ProcrustesProblem((7, 7, 3, 3))


def test_procrustesproblem_invalid_matrices_input():
    m, n, p, q = 6, 6, 3, 3
    A = np.zeros((m, n))
    B = np.zeros((m, q))
    C = np.zeros((p, q))
    # not enough matrices
    with pytest.raises(Exception, match="matrices must be a list or tuple containing"):
        skp.ProcrustesProblem((m, n, p, q), matrices=[A, B])
    # wrong shape A
    with pytest.raises(Exception, match="A must be"):
        badshape = np.zeros((m + 1, n))
        skp.ProcrustesProblem((m, n, p, q), matrices=[badshape, B, C])
    # wrong shape B
    with pytest.raises(Exception, match="B must be"):
        badshape = np.zeros((m, q + 1))
        skp.ProcrustesProblem((m, n, p, q), matrices=[A, badshape, C])
    # wrong shape C
    with pytest.raises(Exception, match="C must be"):
        badshape = np.zeros((p + 1, q))
        skp.ProcrustesProblem((m, n, p, q), matrices=[A, B, badshape])
    # wrong shape Xsol
    XX = np.zeros((n, p + 1))
    with pytest.raises(Exception, match="Xsol must be"):
        skp.ProcrustesProblem((m, n, p, q), matrices=[A, B, C, XX])


def test_procrustesproblem_user_supplied_matrices_and_xsol():
    m, n, p, q = 6, 6, 3, 3
    A = np.eye(m, n)
    B = np.eye(m, q)
    C = np.eye(p, q)
    Xsol = np.eye(n, p)
    problem = skp.ProcrustesProblem((m, n, p, q), matrices=[A, B, C, Xsol])
    # Expect matrices equal to input
    np.testing.assert_allclose(problem.A, A)
    np.testing.assert_allclose(problem.B, B)
    np.testing.assert_allclose(problem.C, C)
    np.testing.assert_allclose(problem.Xsol, Xsol)


def test_gbbsovler_default_options_and_setoptions_validation():
    solver = skp.GBBSolver()
    opts = solver.options
    # Check defaults present and valid types
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

    # Test invalid projected_gradient
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        skp.GBBSolver(projected_gradient=3)

    # Test invalid verbose
    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        skp.GBBSolver(verbose=4)

    # Test invalid filename type
    with pytest.raises(Exception, match="filename must be string"):
        skp.GBBSolver(filename=5)

    # Test invalid full_results type
    with pytest.raises(Exception, match="full_results must be a boolean"):
        skp.GBBSolver(full_results="no")


def test_gbbsovler_open_close_file_stdout(monkeypatch):
    solver = skp.GBBSolver()
    solver.options["filename"] = None
    # We replace sys.stdout temporarily to a StringIO to test open_file effects
    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    solver.open_file()
    assert solver.file is fake_stdout
    solver.close_file()  # Should not raise


def test_gbbsovler_open_close_file_with_file(tmp_path):
    solver = skp.GBBSolver()
    fpath = tmp_path / "tmpfile.txt"
    solver.options["filename"] = str(fpath)
    solver.open_file()
    assert not solver.file.closed
    solver.close_file()
    assert solver.file.closed


def test_gbbsovler_open_file_invalid_filename(monkeypatch):
    solver = skp.GBBSolver()
    solver.options["filename"] = "\0invalid_path"
    # Patch sys.exit to raise instead so test can continue
    def fake_exit(code):
        raise RuntimeError("Exited with code {}".format(code))
    monkeypatch.setattr(sys, "exit", fake_exit)
    with pytest.raises(RuntimeError, match="Exited with code -1"):
        solver.open_file()


def test_gbbsovler_solve_returns_expected_keys(monkeypatch):
    problem = make_square_problem(6, 6, 3, 3, problemnumber=1)

    # Patch random.rand to fixed values for reproducible X0 in solve()
    # Actually, seeding numpy globally suffices as done on module load for consistency

    solver = skp.GBBSolver()

    # monkeypatch the gbb_solver function to a simple mock
    def fake_gbb_solver(problem_, m, n, X, A, B, options, inner, fileobj,
                   B1=None, blobopprod=0.0):
        # return simple predefined output: exitcode, fval, Xout, normgrad, nbiter, msg
        n_, p_ = X.shape
        Xout = np.eye(n_, p_)
        return 0, 0.0, Xout, 0.0, 0, "success"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    res = solver.solve(problem)
    assert isinstance(res, skp.OptimizeResult)
    # Expected keys
    expected_keys = [
        "success", "status", "message", "solution", "fun", "error",
        "cpu", "nbiter", "nfev", "total_fun", "total_crit"
    ]
    for key in expected_keys:
        assert key in res

    # solution shape matches problem expected shape (n,p)
    assert res.solution.shape == (problem.sizes[1], problem.sizes[2])

    # success True if exitcode==0
    assert res.success is True


@pytest.mark.parametrize("invalid_opt, exc_match", [
    ({"xtol": "not_float"}, "xtol must be a float"),
    ({"gtol": "not_float"}, "gtol must be a float"),
    ({"ftol": "not_float"}, "ftol must be a float"),
    ({"zeta": "not_float"}, "zeta must be a float"),
    ({"kappa": "not_float"}, "kappa must be a float"),
    ({"eta": "not_float"}, "eta must be a float"),
    ({"etavar": "not_bool"}, "etavar must be a boolean"),
    ({"tau": "not_float"}, "tau must be a float"),
    ({"projected_gradient": 3}, "projected_gradient must be 1 or 2"),
    ({"maxiter": "not_int"}, "maxiter must be an integer"),
    ({"verbose": 5}, "verbose must be 0, or 1"),
    ({"filename": 55}, "filename must be string"),
    ({"timer": "not_bool"}, "timer must be boolean"),
])
def test_gbbsovler_invalid_option_types(invalid_opt, exc_match):
    opt = {**invalid_opt}
    with pytest.raises(Exception, match=exc_match):
        skp.GBBSolver(**opt)


def test_gbbsovler_runs_with_full_results(monkeypatch):
    problem = make_square_problem(6, 6, 3, 3, problemnumber=1)
    problem.stats["total_fun"] = []
    problem.stats["total_crit"] = []

    solver = skp.GBBSolver(full_results=True)

    def fake_gbb_solver(problem_, m, n, X, A, B, options, inner, fileobj,
                   B1=None, blobopprod=0.0):
        # append stats to simulate full_results
        problem_.stats["total_fun"].append(1.0)
        problem_.stats["total_crit"].append(0.1)
        return 0, 0.0, np.eye(n, 3), 0.0, 0, "success"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    res = solver.solve(problem)
    assert isinstance(res.total_fun, (list, np.ndarray))
    assert isinstance(res.total_crit, (list, np.ndarray))


def test_procrustesproblem_problemnumber3_valid_and_invalid():
    # Valid n in {50, 95, 500, 1000}
    for n in (50, 95, 500, 1000):
        m = n
        p = n // 2
        q = p
        problem = skp.ProcrustesProblem((m, n, p, q), problemnumber=3)
        assert problem.A.shape == (m, n)
        assert problem.B.shape == (m, q)
        assert problem.C.shape == (p, q)

    # Invalid n, expect Exception
    with pytest.raises(Exception, match="Problem 6 requires n = 50,"):
        skp.ProcrustesProblem((30, 30, 10, 10), problemnumber=3)


def test_procrustesproblem_known_solution_orthogonality_and_solution_b_relation():
    problem = make_square_problem(6, 6, 3, 3, problemnumber=1)
    # Xsol should be orthogonal
    np.testing.assert_allclose(problem.Xsol.T @ problem.Xsol, np.eye(3), atol=1e-10)
    # Check B = A Xsol C
    B_calc = problem.A @ problem.Xsol @ problem.C
    np.testing.assert_allclose(B_calc, problem.B, atol=1e-12)


def test_gbbsovler_solve_integrity_with_basic_run(monkeypatch):
    problem = make_square_problem(6, 6, 3, 3, problemnumber=1)

    # Setup gb_solver to actually execute (not patched) would be heavy;
    # patch to pass through with dummy output

    def fake_gbb_solver(problem_, m, n, X, A, B, options, inner, fileobj,
                        B1=None, blobopprod=0.0):
        exitcode = 0
        fval = 5.0
        Xres = np.eye(n, 3)
        normgrad = 1e-4
        nbiter = 5
        msg = "success"
        return exitcode, fval, Xres, normgrad, nbiter, msg

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    solver = skp.GBBSolver()
    result = solver.solve(problem)

    # Check result values
    assert result.success
    assert result.fun == 5.0
    assert result.nbiter == 5
    assert isinstance(result.solution, np.ndarray)
    assert result.solution.shape == (6, 3)
    assert result.message == "success"


def test_optimize_result_attr_access_and_repr():
    data = {"a": 1, "value": 3.14}
    res = skp.OptimizeResult(data)
    assert res.a == 1
    assert res["value"] == 3.14
    # Test attribute error for missing key
    with pytest.raises(AttributeError):
        _ = res.missing
    repr_str = repr(res)
    assert "a" in repr_str and "value" in repr_str


def test_gbbsovler_solve_error_when_full_results_but_stats_missing(monkeypatch):
    problem = make_square_problem(6, 6, 3, 3, problemnumber=1)

    # Remove stats keys to trigger exception
    problem.stats = {}

    def fake_gbb_solver_fail(problem_, m, n, X, A, B, options, inner, fileobj,
                             B1=None, blobopprod=0.0):
        return 0, 0.0, np.eye(n, 3), 0.0, 0, "success"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver_fail)

    solver = skp.GBBSolver(full_results=True)

    with pytest.raises(Exception, match="For full results, set problem.stats"):
        solver.solve(problem)
