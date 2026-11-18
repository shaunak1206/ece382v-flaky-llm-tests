import io
import os

import numpy as np
import pytest

import skprocrustes.skprocrustes as skp


def test_procrustes_problem_rejects_non_square_problem():
    with pytest.raises(Exception, match="square problems are supported"):
        skp.ProcrustesProblem((2, 3, 1, 1), problemnumber=1)


def test_procrustes_problem_rejects_incompatible_dimensions():
    with pytest.raises(Exception, match="divisible by p"):
        skp.ProcrustesProblem((3, 3, 2, 2), problemnumber=1)


def test_procrustes_problem_requires_problemnumber_or_matrices():
    with pytest.raises(Exception, match="Problem matrices are empty"):
        skp.ProcrustesProblem((2, 2, 1, 1))


def test_procrustes_problem_problem1_shapes_and_solution_match():
    np.random.seed(0)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)

    assert problem.A.shape == (4, 4)
    assert problem.B.shape == (4, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (4, 2)
    assert np.allclose(problem.C, np.eye(2))
    assert np.allclose(problem.Xsol.T @ problem.Xsol, np.eye(2), atol=1e-12)
    assert problem.stats == {}


def test_procrustes_problem_problem2_is_deterministic_with_seed():
    np.random.seed(123)
    p1 = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=2)

    np.random.seed(123)
    p2 = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=2)

    assert np.allclose(p1.A, p2.A)
    assert np.allclose(p1.B, p2.B)
    assert np.allclose(p1.C, p2.C)
    assert np.allclose(p1.Xsol, p2.Xsol)


def test_procrustes_problem_problem3_invalid_n_raises():
    with pytest.raises(Exception, match="requires n = 50"):
        skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=3)


def test_procrustes_problem_accepts_user_matrices_and_optional_xsol():
    A = np.arange(16.0).reshape(4, 4)
    B = np.arange(8.0).reshape(4, 2)
    C = np.eye(2)
    X = np.ones((4, 2))

    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, X))

    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert problem.Xsol is X


def test_procrustes_problem_rejects_bad_user_matrix_shapes():
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    C = np.zeros((2, 2))
    X = np.zeros((3, 2))

    with pytest.raises(Exception, match="Xsol must be"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, X))


def test_gbbsolver_defaults_are_set():
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


def test_gbbsolver_rejects_invalid_projected_gradient():
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        skp.GBBSolver(projected_gradient=3)


def test_gbbsolver_rejects_invalid_verbose():
    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        skp.GBBSolver(verbose=2)


def test_gbbsolver_open_file_with_filename_creates_writable_file(tmp_path):
    filename = tmp_path / "out.txt"
    solver = skp.GBBSolver(filename=str(filename))
    solver.open_file()
    try:
        assert solver.file is not None
        solver.file.write("hello")
    finally:
        solver.close_file()

    assert filename.read_text() == "hello"


def test_gbbsolver_open_file_invalid_filename_exits(monkeypatch):
    solver = skp.GBBSolver(filename="dummy.txt")

    def fake_open(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr(skp, "open", fake_open, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        solver.open_file()
    assert excinfo.value.code == -1


def test_gbbsolver_solve_uses_stubbed_solver_and_returns_result(monkeypatch):
    np.random.seed(0)
    problem = skp.ProcrustesProblem((2, 2, 1, 1), problemnumber=1)
    solver = skp.GBBSolver(full_results=False, verbose=0)

    def fake_gbb_solver(problem_arg, largedim, smalldim, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        assert problem_arg is problem
        assert largedim == 2
        assert smalldim == 2
        assert inner is False
        return 0, 1.5, np.array([[1.0], [0.0]]), 0.25, 3, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp, "time", type("T", (), {"time": staticmethod(lambda: 100.0)}))

    result = solver.solve(problem)

    assert result.success is True
    assert result.status == 0
    assert result.message == "ok"
    assert result.fun == 1.5
    assert result.normcrit == 0.25
    assert result.nbiter == 0
    assert result.nfev == 0
    assert result.blocksteps is None if "blocksteps" in result else True


def test_gbbsolver_solve_computes_error_when_xsol_present(monkeypatch):
    np.random.seed(1)
    problem = skp.ProcrustesProblem((2, 2, 1, 1), problemnumber=1)
    solver = skp.GBBSolver(full_results=False, verbose=0)

    Xsol = np.array([[1.0], [0.0]])
    problem.Xsol = Xsol

    def fake_gbb_solver(problem_arg, largedim, smalldim, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        return 0, 0.0, Xsol.copy(), 0.0, 0, "done"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    result = solver.solve(problem)

    assert result.error == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(result.solution, Xsol)


def test_gbbsolver_solve_full_results_requires_stats(monkeypatch):
    np.random.seed(2)
    problem = skp.ProcrustesProblem((2, 2, 1, 1), problemnumber=1)
    solver = skp.GBBSolver(full_results=True, verbose=0)

    def fake_gbb_solver(problem_arg, largedim, smalldim, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        return 0, 1.0, np.array([[1.0], [0.0]]), 0.0, 0, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    with pytest.raises(Exception, match='problem.stats\\["total_fun"\\]'):
        solver.solve(problem)


def test_gbbsolver_solve_full_results_returns_stats_when_present(monkeypatch):
    np.random.seed(3)
    problem = skp.ProcrustesProblem((2, 2, 1, 1), problemnumber=1)
    problem.stats["total_fun"] = [3.0, 2.0]
    problem.stats["total_crit"] = [1.0, 0.5]
    solver = skp.GBBSolver(full_results=True, verbose=0)

    def fake_gbb_solver(problem_arg, largedim, smalldim, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        problem_arg.stats["nbiter"] = 4
        problem_arg.stats["fev"] = 7
        return 0, 2.0, np.array([[0.0], [1.0]]), 0.1, 4, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    result = solver.solve(problem)

    assert result.total_fun == [3.0, 2.0]
    assert result.total_crit == [1.0, 0.5]
    assert result.nbiter == 4
    assert result.nfev == 7
    assert result.success is True
