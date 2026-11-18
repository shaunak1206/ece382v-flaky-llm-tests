import io
import os
from unittest.mock import Mock

import numpy as np
import pytest

import skprocrustes.skprocrustes as skp


def test_procrustes_problem_rejects_non_square_problem():
    with pytest.raises(Exception, match="Currently only square problems are supported"):
        skp.ProcrustesProblem((4, 5, 2, 2), problemnumber=1)


def test_procrustes_problem_rejects_incompatible_dimensions():
    with pytest.raises(Exception, match="Dimensions are incompatible"):
        skp.ProcrustesProblem((5, 5, 2, 3), problemnumber=1)


def test_procrustes_problem_builds_default_problem_and_stats_are_empty():
    np.random.seed(0)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)

    assert problem.sizes == (4, 4, 2, 2)
    assert problem.A.shape == (4, 4)
    assert problem.B.shape == (4, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (4, 2)
    assert problem.stats == {}
    assert np.allclose(problem.C, np.eye(2))
    assert np.allclose(problem.Xsol.T @ problem.Xsol, np.eye(2))


def test_procrustes_problem_problemnumber_3_invalid_n_raises():
    with pytest.raises(Exception, match="Problem 6 requires n = 50"):
        skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=3)


def test_procrustes_problem_accepts_user_matrices_without_xsol():
    A = np.arange(16.0).reshape(4, 4)
    B = np.arange(8.0).reshape(4, 2)
    C = np.eye(2)

    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))

    assert np.array_equal(problem.A, A)
    assert np.array_equal(problem.B, B)
    assert np.array_equal(problem.C, C)
    assert not hasattr(problem, "Xsol")


def test_procrustes_problem_accepts_user_matrices_with_xsol():
    A = np.arange(16.0).reshape(4, 4)
    B = np.arange(8.0).reshape(4, 2)
    C = np.eye(2)
    Xsol = np.eye(4, 2)

    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C, Xsol])

    assert np.array_equal(problem.Xsol, Xsol)
    assert problem.A.shape == (4, 4)
    assert problem.B.shape == (4, 2)
    assert problem.C.shape == (2, 2)


def test_procrustes_problem_rejects_bad_matrices_container_type():
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    C = np.eye(2)
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices={"A": A, "B": B, "C": C})


def test_procrustes_problem_rejects_bad_matrix_shapes():
    A = np.zeros((3, 4))
    B = np.zeros((4, 2))
    C = np.eye(2)
    with pytest.raises(Exception, match="A must be \\(m,n\\)"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))


def test_gbbsolver_defaults_are_set():
    solver = skp.GBBSolver()

    assert solver.solvername == "gbb"
    assert solver.options["full_results"] is False
    assert solver.options["xtol"] == pytest.approx(1e-6)
    assert solver.options["gtol"] == pytest.approx(1e-6)
    assert solver.options["ftol"] == pytest.approx(1e-12)
    assert solver.options["zeta"] == pytest.approx(1e-4)
    assert solver.options["kappa"] == pytest.approx(0.1)
    assert solver.options["eta"] == pytest.approx(0.85)
    assert solver.options["etavar"] is False
    assert solver.options["tau"] == pytest.approx(1e-3)
    assert solver.options["projected_gradient"] == 1
    assert solver.options["maxiter"] == 5000
    assert solver.options["verbose"] == 1
    assert solver.options["filename"] is None
    assert solver.options["timer"] is False
    assert solver.options["precond"] is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"full_results": 1}, "full_results must be a boolean"),
        ({"xtol": 1}, "xtol must be a float"),
        ({"gtol": 1}, "gtol must be a float"),
        ({"ftol": 1}, "ftol must be a float"),
        ({"zeta": 1}, "zeta must be a float"),
        ({"kappa": 1}, "kappa must be a float"),
        ({"eta": 1}, "eta must be a float"),
        ({"etavar": 1}, "etavar must be a boolean"),
        ({"tau": 1}, "tau must be a float"),
        ({"projected_gradient": 3}, "projected_gradient must be 1 or 2"),
        ({"maxiter": 1.5}, "maxiter must be an integer"),
        ({"verbose": 2}, "verbose must be 0, or 1"),
        ({"filename": 123}, "filename must be string"),
        ({"timer": 1}, "timer must be boolean"),
        ({"precond": "bad"}, "precond must be stupid or None"),
    ],
)
def test_gbbsolver_rejects_invalid_options(kwargs, match):
    with pytest.raises(Exception, match=match):
        skp.GBBSolver(**kwargs)


def test_gbbsolver_inner_solver_spg_accepts_inner_options():
    solver = skp.GBBSolver(inner_solver="spg", changevar=True, polar="ns")
    assert solver.options["inner_solver"] == "spg"
    assert solver.options["changevar"] is True
    assert solver.options["polar"] == "ns"


def test_gbbsolver_inner_solver_gbb_sets_gbb_specific_defaults():
    solver = skp.GBBSolver(inner_solver="gbb")
    assert solver.options["inner_solver"] == "gbb"
    assert solver.options["polar"] is None
    assert solver.options["xtol"] == pytest.approx(1e-6)
    assert solver.options["ftol"] == pytest.approx(1e-12)
    assert solver.options["zeta"] == pytest.approx(1e-4)
    assert solver.options["kappa"] == pytest.approx(0.1)
    assert solver.options["tau"] == pytest.approx(1e-3)
    assert solver.options["projected_gradient"] == 1


def test_gbbsolver_open_file_uses_stdout_when_filename_is_none():
    solver = skp.GBBSolver()
    solver.open_file()
    assert solver.file is os.sys.stdout
    solver.close_file()  # should be a no-op


def test_gbbsolver_open_file_and_close_file_with_real_filename(tmp_path):
    out = tmp_path / "gbb_output.txt"
    solver = skp.GBBSolver(filename=str(out))
    solver.open_file()
    assert solver.file is not None
    assert not solver.file.closed
    solver.close_file()
    assert solver.file.closed


def test_gbbsolver_open_file_invalid_filename_exits(monkeypatch):
    solver = skp.GBBSolver(filename="bad/name/that/cannot/be/opened.txt")

    def fake_open(*args, **kwargs):
        raise OSError("cannot open")

    monkeypatch.setattr("builtins.open", fake_open)
    with pytest.raises(SystemExit):
        solver.open_file()


def test_gbbsolver_solve_calls_underlying_solver_and_returns_optimize_result(monkeypatch):
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    Xsol = np.zeros((4, 2))
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, Xsol))
    problem.stats.clear()

    def fake_gbb_solver(problem_arg, m, n, X, A_arg, B_arg, options, inner, fileobj, B1=None, blobopprod=0.0):
        assert problem_arg is problem
        assert (m, n) == (4, 4)
        assert X.shape == (4, 2)
        assert np.array_equal(A_arg, A)
        assert np.array_equal(B_arg, B)
        assert inner is False
        assert B1 is None
        assert blobopprod == 0.0
        problem_arg.stats["nbiter"] = 3
        problem_arg.stats["fev"] = 7
        return 0, 12.5, np.ones((4, 2)), 0.123, 3, "done"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.sp, "qr", lambda X, mode="economic": (np.eye(X.shape[0], X.shape[1]), X.copy()))
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape))

    solver = skp.GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert isinstance(result, skp.OptimizeResult)
    assert result.success is True
    assert result.status == 0
    assert result.message == "done"
    assert result.fun == pytest.approx(12.5)
    assert result.normcrit == pytest.approx(0.123)
    assert result.nbiter == 3
    assert result.nfev == 7
    assert result.solution.shape == (4, 2)
    assert result.error == pytest.approx(np.linalg.norm(np.ones((4, 2)) - Xsol, np.inf))


def test_gbbsolver_solve_uses_infinite_error_when_xsol_missing(monkeypatch):
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))
    problem.stats.clear()

    def fake_gbb_solver(*args, **kwargs):
        prob = args[0]
        prob.stats["nbiter"] = 1
        prob.stats["fev"] = 2
        return 0, 0.0, np.zeros((4, 2)), 0.0, 1, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.sp, "qr", lambda X, mode="economic": (np.eye(X.shape[0], X.shape[1]), X.copy()))
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape))

    solver = skp.GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert result.error == np.inf


def test_gbbsolver_solve_raises_when_full_results_requested_but_stats_missing(monkeypatch):
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    Xsol = np.zeros((4, 2))
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, Xsol))
    problem.stats.clear()

    def fake_gbb_solver(*args, **kwargs):
        prob = args[0]
        prob.stats["nbiter"] = 1
        prob.stats["fev"] = 2
        return 0, 0.0, np.zeros((4, 2)), 0.0, 1, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.sp, "qr", lambda X, mode="economic": (np.eye(X.shape[0], X.shape[1]), X.copy()))
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape))

    solver = skp.GBBSolver(full_results=True, verbose=0)
    with pytest.raises(Exception, match='problem.stats\\["total_fun"\\] and problem.stats\\["total_grad"\\]'):
        solver.solve(problem)


def test_gbbsolver_solve_closes_file_when_filename_used(monkeypatch, tmp_path):
    A = np.eye(4)
    B = np.zeros((4, 2))
    C = np.eye(2)
    Xsol = np.zeros((4, 2))
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, Xsol))
    problem.stats.clear()

    def fake_gbb_solver(*args, **kwargs):
        prob = args[0]
        prob.stats["nbiter"] = 1
        prob.stats["fev"] = 2
        return 0, 0.0, np.zeros((4, 2)), 0.0, 1, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.sp, "qr", lambda X, mode="economic": (np.eye(X.shape[0], X.shape[1]), X.copy()))
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape))

    out = tmp_path / "solver.log"
    solver = skp.GBBSolver(filename=str(out), verbose=0)
    result = solver.solve(problem)

    assert result.success is True
    assert solver.file.closed
