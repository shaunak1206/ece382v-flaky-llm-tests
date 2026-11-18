import io
import os
import builtins
import numpy as np
import pytest

import skprocrustes.skprocrustes as skp


def make_problem_with_matrices():
    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    B = np.array([[2.0], [0.0]])
    C = np.array([[1.0]])
    Xsol = np.array([[1.0], [0.0]])
    return skp.ProcrustesProblem((2, 2, 1, 1), matrices=(A, B, C, Xsol))


def test_procrustes_problem_default_builds_shapes_and_stats_empty():
    np.random.seed(0)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)

    assert problem.sizes == (4, 4, 2, 2)
    assert problem.A.shape == (4, 4)
    assert problem.B.shape == (4, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (4, 2)
    assert problem.stats == {}


def test_procrustes_problem_user_matrices_are_used():
    problem = make_problem_with_matrices()

    assert np.array_equal(problem.A, np.array([[2.0, 0.0], [0.0, 3.0]]))
    assert np.array_equal(problem.B, np.array([[2.0], [0.0]]))
    assert np.array_equal(problem.C, np.array([[1.0]]))
    assert np.array_equal(problem.Xsol, np.array([[1.0], [0.0]]))


def test_procrustes_problem_rejects_non_square_problems():
    with pytest.raises(Exception, match="square problems"):
        skp.ProcrustesProblem((2, 3, 1, 1), problemnumber=1)


def test_procrustes_problem_rejects_incompatible_dimensions():
    with pytest.raises(Exception, match="divisible by p"):
        skp.ProcrustesProblem((3, 3, 2, 2), problemnumber=1)


def test_procrustes_problem_rejects_missing_problem_number_and_matrices():
    with pytest.raises(Exception, match="no problem number"):
        skp.ProcrustesProblem((2, 2, 1, 1))


def test_procrustes_problem_problemnumber_3_requires_allowed_n():
    with pytest.raises(Exception, match="requires n = 50"):
        skp.ProcrustesProblem((6, 6, 2, 2), problemnumber=3)


def test_procrustes_problem_rejects_bad_matrices_container_type():
    A = np.eye(2)
    B = np.ones((2, 1))
    C = np.ones((1, 1))
    with pytest.raises(Exception, match="list or tuple"):
        skp.ProcrustesProblem((2, 2, 1, 1), matrices=A)  # not a tuple/list


def test_procrustes_problem_rejects_bad_matrix_shapes():
    A = np.eye(2)
    B = np.ones((2, 1))
    C = np.ones((2, 1))  # wrong shape, should be (1,1)
    with pytest.raises(Exception, match="C must be \\(p,q\\)"):
        skp.ProcrustesProblem((2, 2, 1, 1), matrices=(A, B, C))


def test_optimize_result_attribute_access_and_repr():
    res = skp.OptimizeResult(success=True, status=0, fun=1.23)
    assert res.success is True
    assert res.status == 0
    assert res.fun == 1.23
    assert "success" in repr(res)
    assert "fun" in repr(res)
    assert sorted(dir(res)) == ["fun", "status", "success"]


def test_optimize_result_missing_attribute_raises_attribute_error():
    res = skp.OptimizeResult(a=1)
    with pytest.raises(AttributeError):
        _ = res.missing


def test_spgsolver_default_options():
    solver = skp.SPGSolver()
    assert solver.solvername == "spg"
    assert solver.options["full_results"] is False
    assert solver.options["strategy"] == "newfw"
    assert solver.options["gtol"] == 1e-3
    assert solver.options["eta"] == 0.85
    assert solver.options["etavar"] is False
    assert solver.options["maxiter"] == 5000
    assert solver.options["verbose"] == 1
    assert solver.options["changevar"] is False
    assert solver.options["polar"] is None
    assert solver.options["timer"] is False
    assert solver.options["precond"] is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"full_results": "yes"}, "full_results must be a boolean"),
        ({"filename": 123}, "filename must be a string"),
        ({"strategy": "bad"}, "strategy not implemented"),
        ({"gtol": 1}, "gtol must be a float"),
        ({"eta": 1}, "eta must be a float"),
        ({"etavar": 1}, "etavar must be a boolean"),
        ({"maxiter": 1.5}, "maxiter must be an integer"),
        ({"verbose": 9}, "verbose must be 0, 1, 2 or 3"),
        ({"changevar": 1}, "changevar must be True or False"),
        ({"polar": "bad"}, "polar must be ns or None"),
        ({"timer": 1}, "timer must be boolean"),
        ({"precond": "bad"}, "precond must be stupid or None"),
    ],
)
def test_spgsolver_option_validation(kwargs, match):
    with pytest.raises(Exception, match=match):
        skp.SPGSolver(**kwargs)


def test_gkbsolver_default_options():
    solver = skp.GKBSolver()
    assert solver.solvername == "gkb"
    assert solver.options["full_results"] is False
    assert solver.options["strategy"] == "newfw"
    assert solver.options["inner_solver"] == "spg"
    assert solver.options["gtol"] == 1e-3
    assert solver.options["eta"] == 0.85
    assert solver.options["etavar"] is False
    assert solver.options["maxiter"] == 5000
    assert solver.options["verbose"] == 1
    assert solver.options["bloboptest"] is False
    assert solver.options["timer"] is False
    assert solver.options["precond"] is None
    assert solver.options["halfreorth"] is False
    assert solver.options["changevar"] is False
    assert solver.options["polar"] is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"inner_solver": "bad"}, "inner_solver not implemented"),
        ({"projected_gradient": 3, "inner_solver": "gbb"}, "projected_gradient must be 1 or 2"),
        ({"xtol": "x", "inner_solver": "gbb"}, "xtol must be a float"),
        ({"ftol": "x", "inner_solver": "gbb"}, "ftol must be a float"),
        ({"zeta": "x", "inner_solver": "gbb"}, "zeta must be a float"),
        ({"kappa": "x", "inner_solver": "gbb"}, "kappa must be a float"),
        ({"tau": "x", "inner_solver": "gbb"}, "tau must be a float"),
        ({"halfreorth": 1}, "halfreorth must be boolean"),
    ],
)
def test_gkbsolver_option_validation(kwargs, match):
    with pytest.raises(Exception, match=match):
        skp.GKBSolver(**kwargs)


def test_gbbsolver_default_options():
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


def test_spgsolver_solve_uses_backend_and_sets_result(monkeypatch):
    problem = make_problem_with_matrices()
    problem.stats = {"nbiter": 7, "fev": 9, "total_fun": [4.0, 1.0], "total_grad": [2.0, 0.5]}

    def fake_spg_solver(problem_arg, solvername, options, fileobj):
        assert problem_arg is problem
        assert solvername == "spg"
        return np.array([[1.0], [0.0]]), 0.25, 0.01, 0, "ok"

    monkeypatch.setattr(skp, "spg_solver", fake_spg_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 10.0)

    solver = skp.SPGSolver(full_results=True, verbose=0)
    result = solver.solve(problem)

    assert result.success is True
    assert result.status == 0
    assert result.message == "ok"
    assert np.array_equal(result.solution, np.array([[1.0], [0.0]]))
    assert result.fun == 0.25
    assert result.normcrit == 0.01
    assert result.cpu == 0.0
    assert result.nbiter == 7
    assert result.nfev == 9
    assert result.total_fun == [4.0, 1.0]
    assert result.total_crit == [2.0, 0.5]
    assert result.error == 0.0


def test_spgsolver_solve_requires_full_results_stats(monkeypatch):
    problem = make_problem_with_matrices()
    problem.stats = {"nbiter": 1, "fev": 2}

    monkeypatch.setattr(
        skp,
        "spg_solver",
        lambda *args, **kwargs: (np.array([[1.0], [0.0]]), 0.0, 0.0, 0, "ok"),
    )

    solver = skp.SPGSolver(full_results=True, verbose=0)
    with pytest.raises(Exception, match="problem.stats\\[\"total_fun\"\\] and problem.stats\\[\"total_grad\"\\]"):
        solver.solve(problem)


def test_gkbsolver_solve_uses_backend_and_sets_result(monkeypatch):
    problem = make_problem_with_matrices()
    problem.stats = {
        "nbiter": 5,
        "fev": 6,
        "blocksteps": 2,
        "total_fun": [5.0, 2.0],
        "total_grad": [1.5, 0.2],
    }

    def fake_gkb_setup(problem_arg, solvername, options, fileobj):
        assert problem_arg is problem
        assert solvername == "gkb"
        return np.array([[1.0], [0.0]]), 0.75, 0.03, 0, "done"

    monkeypatch.setattr(skp, "gkb_setup", fake_gkb_setup)
    monkeypatch.setattr(skp.time, "time", lambda: 20.0)

    solver = skp.GKBSolver(full_results=True, verbose=0)
    result = solver.solve(problem)

    assert result.success is True
    assert result.status == 0
    assert result.message == "done"
    assert np.array_equal(result.solution, np.array([[1.0], [0.0]]))
    assert result.fun == 0.75
    assert result.normcrit == 0.03
    assert result.cpu == 0.0
    assert result.nbiter == 5
    assert result.nfev == 6
    assert result.blocksteps == 2
    assert result.total_fun == [5.0, 2.0]
    assert result.total_crit == [1.5, 0.2]
    assert result.error == 0.0


def test_gbbsolver_solve_uses_backend_and_sets_result(monkeypatch):
    problem = make_problem_with_matrices()
    problem.stats = {"nbiter": 4, "fev": 8, "total_fun": [3.0], "total_grad": [0.4]}

    def fake_gbb_solver(problem_arg, m, n, X, A, B, options, inner, fileobj):
        assert problem_arg is problem
        assert m == 2 and n == 2
        return 0, 1.25, np.array([[1.0], [0.0]]), 0.05, 4, "gbb-ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 30.0)

    solver = skp.GBBSolver(full_results=True, verbose=0)
    result = solver.solve(problem)

    assert result.success is True
    assert result.status == 0
    assert result.message == "gbb-ok"
    assert np.array_equal(result.solution, np.array([[1.0], [0.0]]))
    assert result.fun == 1.25
    assert result.error == 0.0
    assert result.cpu == 0.0
    assert result.nbiter == 4
    assert result.nfev == 8
    assert result.total_fun == [3.0]
    assert result.total_crit == [0.4]


def test_gbbsolver_solve_requires_full_results_stats(monkeypatch):
    problem = make_problem_with_matrices()
    problem.stats = {"nbiter": 1, "fev": 2}

    monkeypatch.setattr(
        skp,
        "gbb_solver",
        lambda *args, **kwargs: (0, 0.0, np.array([[1.0], [0.0]]), 0.0, 0, "ok"),
    )

    solver = skp.GBBSolver(full_results=True, verbose=0)
    with pytest.raises(Exception, match="problem.stats\\[\"total_fun\"\\] and problem.stats\\[\"total_grad\"\\]"):
        solver.solve(problem)
