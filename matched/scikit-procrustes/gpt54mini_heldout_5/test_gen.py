import io
import os
import types

import numpy as np
import pytest

import skprocrustes.skprocrustes as skp


def make_user_problem():
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(1)
    X = np.array([[1.0], [0.0]])
    return skp.ProcrustesProblem((2, 2, 1, 1), matrices=(A, B, C, X))


def test_procrustes_problem_user_matrices_sets_attributes_and_stats_empty():
    problem = make_user_problem()

    assert problem.sizes == (2, 2, 1, 1)
    assert np.allclose(problem.A, np.eye(2))
    assert np.allclose(problem.B, np.eye(2))
    assert np.allclose(problem.C, np.eye(1))
    assert np.allclose(problem.Xsol, np.array([[1.0], [0.0]]))
    assert problem.stats == {}


def test_procrustes_problem_rejects_non_square_sizes():
    with pytest.raises(Exception, match="square problems"):
        skp.ProcrustesProblem((2, 3, 1, 1), matrices=(np.eye(2), np.eye(2), np.eye(1)))


def test_procrustes_problem_rejects_incompatible_dimensions():
    with pytest.raises(Exception, match="divisible by p"):
        skp.ProcrustesProblem((3, 3, 2, 2), matrices=(np.eye(3), np.eye(3), np.eye(2)))


def test_procrustes_problem_rejects_invalid_matrices_container_type():
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        skp.ProcrustesProblem((2, 2, 1, 1), matrices=np.eye(2))


def test_procrustes_problem_rejects_too_few_matrices():
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        skp.ProcrustesProblem((2, 2, 1, 1), matrices=(np.eye(2), np.eye(2)))


def test_procrustes_problem_rejects_wrong_a_shape():
    with pytest.raises(Exception, match="A must be \\(m,n\\)"):
        skp.ProcrustesProblem((2, 2, 1, 1), matrices=(np.ones((1, 1)), np.eye(2), np.eye(1)))


def test_procrustes_problem_rejects_wrong_b_shape():
    with pytest.raises(Exception, match="B must be \\(m,q\\)"):
        skp.ProcrustesProblem((2, 2, 1, 1), matrices=(np.eye(2), np.ones((1, 1)), np.eye(1)))


def test_procrustes_problem_rejects_wrong_c_shape():
    with pytest.raises(Exception, match="C must be \\(p,q\\)"):
        skp.ProcrustesProblem((2, 2, 1, 1), matrices=(np.eye(2), np.eye(2), np.ones((2, 2))))


def test_procrustes_problem_rejects_wrong_xsol_shape():
    with pytest.raises(Exception, match="Xsol must be \\(n,p\\)"):
        skp.ProcrustesProblem(
            (2, 2, 1, 1),
            matrices=(np.eye(2), np.eye(2), np.eye(1), np.ones((1, 1))),
        )


def test_predefined_problem_has_consistent_shapes_and_orthonormal_xsol(monkeypatch):
    rng = np.random.RandomState(0)
    monkeypatch.setattr(skp.np.random, "rand", lambda *args: rng.rand(*args))
    monkeypatch.setattr(skp.np.random, "randn", lambda *args: rng.randn(*args))
    monkeypatch.setattr(skp.np.random, "permutation", lambda x: x[::-1].copy())

    problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)

    assert problem.A.shape == (4, 4)
    assert problem.B.shape == (4, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (4, 2)
    assert np.allclose(problem.Xsol.T @ problem.Xsol, np.eye(2))


def test_problemnumber_3_rejects_unsupported_n():
    with pytest.raises(Exception, match="requires n = 50"):
        skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=3)


def test_optimize_result_attribute_access_and_repr():
    res = skp.OptimizeResult(success=True, status=0, fun=1.5)
    assert res.success is True
    assert res.status == 0
    assert res.fun == 1.5
    assert "success" in dir(res)

    text = repr(res)
    assert "success" in text
    assert "status" in text
    assert "fun" in text


def test_optimize_result_missing_attribute_raises_attribute_error():
    res = skp.OptimizeResult()
    with pytest.raises(AttributeError):
        _ = res.nope


def test_gbbsolver_defaults_and_solvername():
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
    assert solver.options["filename"] is None


def test_gbbsolver_rejects_invalid_projected_gradient():
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        skp.GBBSolver(projected_gradient=3)


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


def test_gbbsolver_open_file_uses_stdout_when_filename_none():
    solver = skp.GBBSolver()
    solver.open_file()
    assert solver.file is os.sys.stdout


def test_gbbsolver_open_file_exits_on_invalid_filename(monkeypatch):
    solver = skp.GBBSolver(filename="bad-name.txt")

    def fake_open(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr("builtins.open", fake_open)

    with pytest.raises(SystemExit) as excinfo:
        solver.open_file()
    assert excinfo.value.code == -1


def test_gbbsolver_solve_returns_optimize_result(monkeypatch):
    problem = make_user_problem()
    problem.stats = {}

    def fake_gbb_solver(problem_arg, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        problem_arg.stats.update({"nbiter": 7, "fev": 11})
        return 0, 3.25, np.array([[1.0], [0.0]]), 0.01, 4, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.np.random, "rand", lambda *args: np.ones(args))
    monkeypatch.setattr(skp.sp, "qr", lambda x, mode="economic": (np.eye(x.shape[0], x.shape[1]), np.eye(x.shape[1])))

    solver = skp.GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert isinstance(result, skp.OptimizeResult)
    assert result.success is True
    assert result.status == 0
    assert result.message == "ok"
    assert result.fun == pytest.approx(3.25)
    assert result.normcrit == pytest.approx(0.01)
    assert result.nbiter == 7
    assert result.nfev == 11
    assert result.solution.shape == (2, 1)


def test_gbbsolver_solve_uses_infinite_error_when_no_xsol(monkeypatch):
    problem = skp.ProcrustesProblem((2, 2, 1, 1), matrices=(np.eye(2), np.eye(2), np.eye(1)))
    problem.stats = {}

    def fake_gbb_solver(problem_arg, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        problem_arg.stats.update({"nbiter": 1, "fev": 2})
        return 0, 1.0, np.array([[1.0], [0.0]]), 0.0, 1, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.np.random, "rand", lambda *args: np.ones(args))
    monkeypatch.setattr(skp.sp, "qr", lambda x, mode="economic": (np.eye(x.shape[0], x.shape[1]), np.eye(x.shape[1])))

    solver = skp.GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert result.error == np.inf


def test_gbbsolver_full_results_requires_problem_stats_lists(monkeypatch):
    problem = make_user_problem()
    problem.stats = {}

    def fake_gbb_solver(problem_arg, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        problem_arg.stats.update({"nbiter": 1, "fev": 2})
        return 0, 1.0, np.array([[1.0], [0.0]]), 0.0, 1, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.np.random, "rand", lambda *args: np.ones(args))
    monkeypatch.setattr(skp.sp, "qr", lambda x, mode="economic": (np.eye(x.shape[0], x.shape[1]), np.eye(x.shape[1])))

    solver = skp.GBBSolver(full_results=True, verbose=0)

    with pytest.raises(Exception, match='problem\\.stats\\["total_fun"\\]'):
        solver.solve(problem)
