import io
import os

import numpy as np
import pytest

import skprocrustes.skprocrustes as skp


def make_problem(m=4, n=4, p=2, q=2, with_x=True):
    a = np.eye(m, n)
    b = np.ones((m, q))
    c = np.eye(p, q)
    matrices = (a, b, c, np.eye(n, p)) if with_x else (a, b, c)
    return skp.ProcrustesProblem((m, n, p, q), matrices=matrices)


def test_problem_rejects_non_square_dimensions():
    with pytest.raises(Exception, match="square problems"):
        skp.ProcrustesProblem((3, 4, 2, 2), matrices=(np.eye(3, 4), np.eye(3, 2), np.eye(2, 2)))


def test_problem_rejects_incompatible_divisibility():
    with pytest.raises(Exception, match="divisible by p"):
        skp.ProcrustesProblem((5, 5, 2, 2), matrices=(np.eye(5), np.eye(5, 2), np.eye(2)))


def test_problem_rejects_invalid_matrices_container_type():
    with pytest.raises(Exception, match="must be a list or tuple"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=np.eye(4))


def test_problem_rejects_too_few_matrices():
    with pytest.raises(Exception, match="must be a list or tuple"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=(np.eye(4), np.eye(4, 2)))


def test_problem_rejects_bad_A_shape():
    with pytest.raises(Exception, match="A must be \\(m,n\\)"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=(np.eye(3), np.eye(4, 2), np.eye(2)))


def test_problem_rejects_bad_B_shape():
    with pytest.raises(Exception, match="B must be \\(m,q\\)"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=(np.eye(4), np.eye(3, 2), np.eye(2)))


def test_problem_rejects_bad_C_shape():
    with pytest.raises(Exception, match="C must be \\(p,q\\)"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=(np.eye(4), np.eye(4, 2), np.eye(3)))


def test_problem_rejects_bad_xsol_shape():
    with pytest.raises(Exception, match="Xsol must be \\(n,p\\)"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=(np.eye(4), np.eye(4, 2), np.eye(2), np.eye(3, 2)))


def test_problem_stores_given_matrices_and_solution():
    a = np.arange(16, dtype=float).reshape(4, 4)
    b = np.arange(8, dtype=float).reshape(4, 2)
    c = np.eye(2)
    x = np.eye(4, 2)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(a, b, c, x))
    assert problem.sizes == (4, 4, 2, 2)
    assert np.array_equal(problem.A, a)
    assert np.array_equal(problem.B, b)
    assert np.array_equal(problem.C, c)
    assert np.array_equal(problem.Xsol, x)
    assert isinstance(problem.stats, dict)
    assert problem.stats == {}


def test_problem_default_generation_sets_expected_shapes(monkeypatch):
    rng = np.random.RandomState(0)
    monkeypatch.setattr(np.random, "rand", rng.rand)
    monkeypatch.setattr(np.random, "randn", rng.randn)
    monkeypatch.setattr(np.random, "permutation", lambda x: x)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    assert problem.A.shape == (4, 4)
    assert problem.B.shape == (4, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (4, 2)
    assert np.allclose(problem.C, np.eye(2))
    assert np.allclose(problem.B, problem.A @ problem.Xsol @ problem.C)


def test_gbbsolver_default_options_are_set():
    solver = skp.GBBSolver()
    assert solver.solvername == "gbb"
    assert solver.options["full_results"] is False
    assert solver.options["xtol"] == 1e-6
    assert solver.options["gtol"] == 1e-6
    assert solver.options["ftol"] == 1e-12
    assert solver.options["zeta"] == 1e-4
    assert solver.options["kappa"] == 0.1
    assert solver.options["eta"] == 0.85
    assert solver.options["etavar"] is False
    assert solver.options["tau"] == 1e-3
    assert solver.options["projected_gradient"] == 1
    assert solver.options["maxiter"] == 5000
    assert solver.options["verbose"] == 1
    assert solver.options["filename"] is None
    assert solver.options["timer"] is False


def test_gbbsolver_rejects_invalid_option_types_and_values():
    with pytest.raises(Exception, match="full_results must be a boolean"):
        skp.GBBSolver(full_results="yes")
    with pytest.raises(Exception, match="xtol must be a float"):
        skp.GBBSolver(xtol=1)
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        skp.GBBSolver(projected_gradient=3)
    with pytest.raises(Exception, match="inner_solver not implemented"):
        skp.GBBSolver(inner_solver="bad")
    with pytest.raises(Exception, match="filename must be string"):
        skp.GBBSolver(filename=123)


def test_gbbsolver_inner_solver_gbb_sets_gbb_specific_defaults():
    solver = skp.GBBSolver(inner_solver="gbb")
    assert solver.options["inner_solver"] == "gbb"
    assert solver.options["polar"] is None
    assert solver.options["xtol"] == 1e-6
    assert solver.options["ftol"] == 1e-12
    assert solver.options["zeta"] == 1e-4
    assert solver.options["kappa"] == 0.1
    assert solver.options["tau"] == 1e-3
    assert solver.options["projected_gradient"] == 1


def test_gbbsolver_inner_solver_spg_sets_spg_specific_defaults():
    solver = skp.GBBSolver(inner_solver="spg")
    assert solver.options["inner_solver"] == "spg"
    assert solver.options["changevar"] is False
    assert solver.options["polar"] is None


def test_gbbsolver_open_and_close_file_with_filename(tmp_path):
    filename = tmp_path / "out.txt"
    solver = skp.GBBSolver(filename=str(filename))
    solver.open_file()
    try:
        assert solver.file is not None
        solver.file.write("hello")
    finally:
        solver.close_file()
    assert filename.read_text() == "hello"


def test_gbbsolver_open_file_uses_stdout_when_no_filename():
    solver = skp.GBBSolver()
    solver.open_file()
    try:
        assert solver.file is os.sys.stdout
    finally:
        solver.close_file()


def test_gbbsolver_solve_calls_gbb_solver_and_builds_result(monkeypatch):
    problem = make_problem()
    solver = skp.GBBSolver()

    def fake_gbb_solver(problem_arg, m, n, x0, a, b, options, inner, fileobj, B1=None, blobopprod=0.0):
        assert problem_arg is problem
        assert (m, n) == (4, 4)
        assert x0.shape == (4, 2)
        assert a.shape == (4, 4)
        assert b.shape == (4, 2)
        assert inner is False
        return 0, 12.5, np.eye(4, 2), 0.25, 7, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 10.0)

    result = solver.solve(problem)

    assert isinstance(result, skp.OptimizeResult)
    assert result.success is True
    assert result.status == 0
    assert result.message == "ok"
    assert np.array_equal(result.solution, np.eye(4, 2))
    assert result.fun == 12.5
    assert result.nbiter == 7
    assert result.nfev == problem.stats["fev"]
    assert result.error == 0.0
    assert result.cpu == 0.0


def test_gbbsolver_solve_sets_failure_status(monkeypatch):
    problem = make_problem()
    solver = skp.GBBSolver()

    def fake_gbb_solver(problem_arg, m, n, x0, a, b, options, inner, fileobj, B1=None, blobopprod=0.0):
        return 1, 3.0, np.zeros((4, 2)), 9.0, 2, "failed"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 5.0)

    result = solver.solve(problem)
    assert result.success is False
    assert result.status == 1
    assert result.message == "failed"
    assert result.normcrit == 9.0


def test_gbbsolver_solve_computes_error_inf_when_no_xsol(monkeypatch):
    a = np.eye(4)
    b = np.ones((4, 2))
    c = np.eye(2)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(a, b, c))
    solver = skp.GBBSolver()

    def fake_gbb_solver(problem_arg, m, n, x0, a, b, options, inner, fileobj, B1=None, blobopprod=0.0):
        return 0, 1.0, np.eye(4, 2), 0.0, 1, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 1.0)

    result = solver.solve(problem)
    assert result.error == np.inf


def test_gbbsolver_solve_requires_full_results_stats_when_enabled(monkeypatch):
    problem = make_problem()
    solver = skp.GBBSolver(full_results=True)

    def fake_gbb_solver(problem_arg, m, n, x0, a, b, options, inner, fileobj, B1=None, blobopprod=0.0):
        return 0, 1.0, np.eye(4, 2), 0.0, 1, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 2.0)

    with pytest.raises(Exception, match="For full results"):
        solver.solve(problem)
