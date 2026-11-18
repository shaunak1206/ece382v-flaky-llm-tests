import io
import os

import numpy as np
import pytest

import skprocrustes.skprocrustes as skp


def test_problem_rejects_non_square_sizes():
    with pytest.raises(Exception, match="square problems"):
        skp.ProcrustesProblem((3, 4, 2, 2), problemnumber=1)


def test_problem_rejects_incompatible_dimensions():
    with pytest.raises(Exception, match="divisible by p"):
        skp.ProcrustesProblem((5, 5, 2, 2), problemnumber=1)


def test_problem_default_generated_has_expected_shapes():
    np.random.seed(0)
    problem = skp.ProcrustesProblem((6, 6, 2, 2), problemnumber=1)

    assert problem.A.shape == (6, 6)
    assert problem.B.shape == (6, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (6, 2)
    assert problem.stats == {}


def test_problem_default_generated_data_is_consistent():
    np.random.seed(1)
    problem = skp.ProcrustesProblem((6, 6, 2, 2), problemnumber=1)

    # B is generated from A, Xsol, and C, so the residual should be tiny.
    R = problem.A @ problem.Xsol @ problem.C - problem.B
    assert np.linalg.norm(R, ord="fro") < 1e-10


def test_problem_problemnumber3_invalid_n_raises():
    with pytest.raises(Exception, match="Problem 6 requires n = 50"):
        skp.ProcrustesProblem((60, 60, 2, 2), problemnumber=3)


def test_problem_user_matrices_accepts_valid_inputs():
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    X = np.eye(4, 2)

    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, X))

    assert np.all(problem.A == A)
    assert np.all(problem.B == B)
    assert np.all(problem.C == C)
    assert np.all(problem.Xsol == X)


def test_problem_user_matrices_rejects_non_tuple_or_list():
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)

    with pytest.raises(Exception, match="list or tuple"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices="not-a-sequence")


def test_problem_user_matrices_rejects_wrong_A_shape():
    A = np.eye(5)
    B = np.zeros((4, 2))
    C = np.eye(2)

    with pytest.raises(Exception, match="A must be \\(m,n\\)"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))


def test_problem_user_matrices_rejects_wrong_B_shape():
    A = np.eye(4)
    B = np.zeros((5, 2))
    C = np.eye(2)

    with pytest.raises(Exception, match="B must be \\(m,q\\)"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))


def test_problem_user_matrices_rejects_wrong_C_shape():
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(3)

    with pytest.raises(Exception, match="C must be \\(p,q\\)"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))


def test_problem_user_matrices_rejects_wrong_xsol_shape():
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    X = np.eye(4, 3)

    with pytest.raises(Exception, match="Xsol must be \\(n,p\\)"):
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


def test_gbbsolver_rejects_invalid_option_types():
    with pytest.raises(Exception, match="full_results must be a boolean"):
        skp.GBBSolver(full_results="yes")

    with pytest.raises(Exception, match="xtol must be a float"):
        skp.GBBSolver(xtol=1)

    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        skp.GBBSolver(projected_gradient=3)


def test_gbbsolver_rejects_invalid_inner_option_values():
    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        skp.GBBSolver(verbose=2)

    with pytest.raises(Exception, match="filename must be string"):
        skp.GBBSolver(filename=123)


def test_gbbsolver_open_file_uses_filename(tmp_path):
    filename = tmp_path / "out.txt"
    solver = skp.GBBSolver(filename=str(filename))
    solver.open_file()
    try:
        assert hasattr(solver, "file")
        assert solver.file is not None
        solver.file.write("hello")
    finally:
        solver.close_file()

    assert filename.read_text() == "hello"


def test_gbbsolver_open_file_invalid_path_exits(monkeypatch):
    solver = skp.GBBSolver(filename="does_not_exist/forbidden.txt")

    def fake_open(*args, **kwargs):
        raise OSError("bad path")

    monkeypatch.setattr("builtins.open", fake_open)

    with pytest.raises(SystemExit):
        solver.open_file()


def test_gbbsolver_solve_returns_result_and_computes_error(monkeypatch):
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    Xsol = np.eye(4, 2)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, Xsol))
    problem.stats = {"nbiter": 7, "fev": 11, "blocksteps": 3}

    def fake_gbb_solver(problem_arg, m, n, X, A_arg, B_arg, options, inner, fileobj, B1=None, blobopprod=0.0):
        assert problem_arg is problem
        assert m == 4 and n == 4
        assert X.shape == (4, 2)
        return 0, 1.25, Xsol.copy(), 1e-9, 9, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape))

    solver = skp.GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert isinstance(result, skp.OptimizeResult)
    assert result.success is True
    assert result.status == 0
    assert result.message == "ok"
    assert np.allclose(result.solution, Xsol)
    assert result.fun == 1.25
    assert result.error == 0.0
    assert result.nbiter == 7
    assert result.nfev == 11
    assert result.blocksteps == 3


def test_gbbsolver_solve_without_xsol_sets_error_to_inf(monkeypatch):
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))
    problem.stats = {"nbiter": 1, "fev": 2, "blocksteps": 0}

    X = np.eye(4, 2)

    def fake_gbb_solver(problem_arg, m, n, X0, A_arg, B_arg, options, inner, fileobj, B1=None, blobopprod=0.0):
        return 0, 0.0, X.copy(), 0.0, 0, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape))

    solver = skp.GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert result.error == np.inf


def test_gbbsolver_full_results_requires_problem_stats(monkeypatch):
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    Xsol = np.eye(4, 2)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, Xsol))
    problem.stats = {"nbiter": 0, "fev": 0, "blocksteps": 0}

    def fake_gbb_solver(problem_arg, m, n, X, A_arg, B_arg, options, inner, fileobj, B1=None, blobopprod=0.0):
        return 0, 0.0, Xsol.copy(), 0.0, 0, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape))

    solver = skp.GBBSolver(full_results=True, verbose=0)

    with pytest.raises(Exception, match="problem.stats\\[\"total_fun\"\\]"):
        solver.solve(problem)


def test_gbbsolver_solve_populates_full_results(monkeypatch):
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    Xsol = np.eye(4, 2)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, Xsol))
    problem.stats = {
        "nbiter": 4,
        "fev": 6,
        "blocksteps": 2,
        "total_fun": [3.0, 2.0],
        "total_grad": [1.0, 0.5],
    }

    def fake_gbb_solver(problem_arg, m, n, X, A_arg, B_arg, options, inner, fileobj, B1=None, blobopprod=0.0):
        return 0, 0.0, Xsol.copy(), 0.0, 0, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape))

    solver = skp.GBBSolver(full_results=True, verbose=0)
    result = solver.solve(problem)

    assert result.total_fun == [3.0, 2.0]
    assert result.total_crit == [1.0, 0.5]
