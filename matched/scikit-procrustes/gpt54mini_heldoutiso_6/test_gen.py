import io
import os
import numpy as np
import pytest

import skprocrustes.skprocrustes as skp


def make_problem_with_matrices():
    m = n = 4
    p = q = 2
    A = np.arange(m * n, dtype=float).reshape(m, n) + np.eye(m)
    B = np.arange(m * q, dtype=float).reshape(m, q) - 1.0
    C = np.eye(p, q)
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    return skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C, X))


def test_problem_rejects_nonsquare_dimensions():
    with pytest.raises(Exception, match="square problems"):
        skp.ProcrustesProblem((4, 5, 2, 2), matrices=(np.zeros((4, 5)), np.zeros((4, 2)), np.eye(2), np.zeros((5, 2))))


def test_problem_rejects_incompatible_divisibility():
    with pytest.raises(Exception, match="divisible by p"):
        skp.ProcrustesProblem((5, 5, 2, 2), matrices=(np.zeros((5, 5)), np.zeros((5, 2)), np.eye(2), np.zeros((5, 2))))


def test_problem_accepts_user_matrices_and_sets_attributes():
    problem = make_problem_with_matrices()
    assert problem.sizes == (4, 4, 2, 2)
    assert np.array_equal(problem.A, np.arange(16, dtype=float).reshape(4, 4) + np.eye(4))
    assert np.array_equal(problem.B, np.arange(8, dtype=float).reshape(4, 2) - 1.0)
    assert np.array_equal(problem.C, np.eye(2))
    assert np.array_equal(problem.Xsol, np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    assert problem.stats == {}


def test_problem_rejects_invalid_matrices_container_type():
    with pytest.raises(Exception, match="list or tuple"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices="not-a-sequence")


def test_problem_rejects_too_few_matrices():
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        skp.ProcrustesProblem((4, 4, 2, 2), matrices=(np.zeros((4, 4)), np.zeros((4, 2))))


def test_problem_rejects_bad_A_shape():
    with pytest.raises(Exception, match="A must be \\(m,n\\)"):
        skp.ProcrustesProblem(
            (4, 4, 2, 2),
            matrices=(np.zeros((3, 4)), np.zeros((4, 2)), np.eye(2)),
        )


def test_problem_rejects_bad_B_shape():
    with pytest.raises(Exception, match="B must be \\(m,q\\)"):
        skp.ProcrustesProblem(
            (4, 4, 2, 2),
            matrices=(np.zeros((4, 4)), np.zeros((3, 2)), np.eye(2)),
        )


def test_problem_rejects_bad_C_shape():
    with pytest.raises(Exception, match="C must be \\(p,q\\)"):
        skp.ProcrustesProblem(
            (4, 4, 2, 2),
            matrices=(np.zeros((4, 4)), np.zeros((4, 2)), np.zeros((1, 2))),
        )


def test_problem_rejects_bad_Xsol_shape():
    with pytest.raises(Exception, match="Xsol must be \\(n,p\\)"):
        skp.ProcrustesProblem(
            (4, 4, 2, 2),
            matrices=(np.zeros((4, 4)), np.zeros((4, 2)), np.eye(2), np.zeros((3, 2))),
        )


def test_gbbsolver_defaults_set_expected_options():
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
    assert solver.options["inner_solver"] == "spg"


def test_gbbsolver_rejects_invalid_verbose_and_projected_gradient():
    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        skp.GBBSolver(verbose=2)
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        skp.GBBSolver(projected_gradient=3)


def test_gbbsolver_rejects_invalid_inner_solver_and_dependent_options():
    with pytest.raises(Exception, match="inner_solver not implemented"):
        skp.GBBSolver(inner_solver="bad")
    with pytest.raises(Exception, match="xtol must be a float"):
        skp.GBBSolver(inner_solver="gbb", xtol="nope")


def test_gbbsolver_solve_wraps_solver_output_into_result(monkeypatch):
    problem = make_problem_with_matrices()

    def fake_gbb_solver(problem_arg, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0.0):
        assert problem_arg is problem
        assert m == 4 and n == 4
        assert X.shape == (4, 2)
        assert A.shape == (4, 4)
        assert B.shape == (4, 2)
        return 0, 12.5, np.ones((4, 2)), 0.25, 7, "fake success"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 100.0)

    solver = skp.GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert result.success is True
    assert result.status == 0
    assert result.message == "fake success"
    assert result.fun == 12.5
    assert result.normcrit == 0.25
    assert result.nbiter == 7
    assert result.nfev == problem.stats["fev"]
    assert np.array_equal(result.solution, np.ones((4, 2)))
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf
    assert result.error >= 0.0
    assert "cpu" in result


def test_gbbsolver_solve_uses_xsol_for_error(monkeypatch):
    problem = make_problem_with_matrices()

    def fake_gbb_solver(*args, **kwargs):
        return 0, 1.0, problem.Xsol.copy(), 0.0, 1, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 0.0)

    solver = skp.GBBSolver(verbose=0)
    result = solver.solve(problem)
    assert result.error == pytest.approx(0.0)


def test_gbbsolver_full_results_requires_stats_keys(monkeypatch):
    problem = make_problem_with_matrices()

    def fake_gbb_solver(*args, **kwargs):
        return 0, 1.0, np.zeros((4, 2)), 0.0, 1, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 0.0)

    solver = skp.GBBSolver(full_results=True, verbose=0)
    with pytest.raises(Exception, match="problem\\.stats\\[\"total_fun\"\\]"):
        solver.solve(problem)


def test_gbbsolver_full_results_populates_lists(monkeypatch):
    problem = make_problem_with_matrices()
    problem.stats["total_fun"] = [9.0, 4.0]
    problem.stats["total_grad"] = [3.0, 2.0]

    def fake_gbb_solver(*args, **kwargs):
        return 0, 1.0, np.zeros((4, 2)), 0.0, 1, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 0.0)

    solver = skp.GBBSolver(full_results=True, verbose=0)
    result = solver.solve(problem)

    assert result.total_fun == [9.0, 4.0]
    assert result.total_crit == [3.0, 2.0]


def test_gbbsolver_open_file_raises_systemexit_on_bad_filename(monkeypatch):
    solver = skp.GBBSolver(filename="bad-file.txt", verbose=0)

    def fake_open(*args, **kwargs):
        raise OSError("cannot open")

    monkeypatch.setattr("builtins.open", fake_open)

    with pytest.raises(SystemExit):
        solver.open_file()


def test_optimize_result_attribute_access_and_repr():
    r = skp.OptimizeResult(success=True, status=0, message="ok", fun=1.5)
    assert r.success is True
    assert r.status == 0
    assert r.message == "ok"
    assert r.fun == 1.5
    assert "success" in repr(r)
    assert "message" in repr(r)
    assert sorted(dir(r)) == sorted(r.keys())
