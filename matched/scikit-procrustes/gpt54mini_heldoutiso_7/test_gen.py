import io
import os
import types
import numpy as np
import pytest

import skprocrustes.skprocrustes as skp


def make_problem(m=2, n=2, p=1, q=1, with_xsol=True):
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    B = np.array([[5.0],
                  [6.0]])
    C = np.array([[1.0]])
    if with_xsol:
        X = np.array([[1.0],
                      [0.0]])
        return skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C, X))
    return skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C))


def test_problem_rejects_non_square_sizes():
    with pytest.raises(Exception, match="square problems"):
        skp.ProcrustesProblem((2, 3, 1, 1), matrices=(np.zeros((2, 3)), np.zeros((2, 1)), np.zeros((1, 1))))


def test_problem_rejects_incompatible_dimensions():
    with pytest.raises(Exception, match="divisible"):
        skp.ProcrustesProblem((3, 3, 2, 2), matrices=(np.zeros((3, 3)), np.zeros((3, 2)), np.zeros((2, 2))))


def test_problem_rejects_invalid_matrices_container_type():
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        skp.ProcrustesProblem((2, 2, 1, 1), matrices=np.zeros((2, 2)))


def test_problem_rejects_too_few_matrices():
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        skp.ProcrustesProblem((2, 2, 1, 1), matrices=(np.zeros((2, 2)), np.zeros((2, 1))))


def test_problem_rejects_bad_matrix_shapes():
    A = np.zeros((2, 3))
    B = np.zeros((2, 1))
    C = np.zeros((1, 1))
    with pytest.raises(Exception, match="A must be \\(m,n\\)"):
        skp.ProcrustesProblem((2, 2, 1, 1), matrices=(A, B, C))


def test_problem_sets_user_matrices_and_xsol():
    problem = make_problem()
    assert problem.sizes == (2, 2, 1, 1)
    assert np.allclose(problem.A, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(problem.B, np.array([[5.0], [6.0]]))
    assert np.allclose(problem.C, np.array([[1.0]]))
    assert np.allclose(problem.Xsol, np.array([[1.0], [0.0]]))
    assert problem.stats == {}


def test_problem_default_generation_has_expected_shapes_and_stats():
    np.random.seed(0)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    assert problem.A.shape == (4, 4)
    assert problem.B.shape == (4, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (4, 2)
    assert problem.stats == {}


def test_problem_problemnumber_3_invalid_n_raises():
    np.random.seed(0)
    with pytest.raises(Exception, match="Problem 6 requires n = 50"):
        skp.ProcrustesProblem((6, 6, 2, 2), problemnumber=3)


def test_optimize_result_attribute_access_and_repr():
    r = skp.OptimizeResult(success=True, status=0, message="ok", fun=1.23)
    assert r.success is True
    assert r.status == 0
    assert r.message == "ok"
    assert r.fun == 1.23
    rep = repr(r)
    assert "success" in rep and "message" in rep and "fun" in rep
    assert "success" in dir(r)


def test_optimize_result_missing_attribute_raises_attribute_error():
    r = skp.OptimizeResult()
    with pytest.raises(AttributeError):
        _ = r.nope


def test_optimize_result_show_prints_summary(capsys):
    r = skp.OptimizeResult(success=True, status=0, message="ok", fun=1.0, solution=np.zeros((1, 1)))
    r.show()
    out = capsys.readouterr().out
    assert "Summary" in out
    assert "success" in out
    assert "solution" not in out


def test_spgsolver_default_and_invalid_options():
    s = skp.SPGSolver()
    assert s.options["full_results"] is False
    assert s.options["strategy"] == "newfw"
    assert s.options["verbose"] == 1

    with pytest.raises(Exception, match="strategy not implemented"):
        skp.SPGSolver(strategy="bad")

    with pytest.raises(Exception, match="verbose must be 0, 1, 2 or 3"):
        skp.SPGSolver(verbose=9)

    with pytest.raises(Exception, match="gtol must be a float"):
        skp.SPGSolver(gtol=1)


def test_gkbsolver_default_and_invalid_inner_solver_options():
    s = skp.GKBSolver()
    assert s.options["inner_solver"] == "spg"
    assert s.options["polar"] is None

    with pytest.raises(Exception, match="inner_solver not implemented"):
        skp.GKBSolver(inner_solver="zzz")

    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        skp.GKBSolver(inner_solver="gbb", projected_gradient=3)


def test_gbbsolver_default_and_invalid_options():
    s = skp.GBBSolver()
    assert s.options["projected_gradient"] == 1
    assert s.options["tau"] == pytest.approx(1e-3)

    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        skp.GBBSolver(projected_gradient=5)

    with pytest.raises(Exception, match="xtol must be a float"):
        skp.GBBSolver(xtol=1)


def test_solver_open_file_with_filename(monkeypatch, tmp_path):
    filename = tmp_path / "out.txt"
    s = skp.SPGSolver(filename=str(filename))
    s.open_file()
    try:
        assert s.file is not None
        assert not s.file.closed
    finally:
        s.close_file()
    assert s.file.closed


def test_solver_open_file_invalid_filename_exits(monkeypatch):
    s = skp.SPGSolver(filename="bad/name/that/does/not/exist.txt")
    with pytest.raises(SystemExit):
        s.open_file()


def test_spgsolver_solve_uses_stubbed_backend(monkeypatch):
    problem = make_problem()
    problem.stats["nbiter"] = 7
    problem.stats["fev"] = 11
    problem.stats["total_fun"] = [3.0, 2.0]
    problem.stats["total_grad"] = [4.0, 1.0]

    def fake_spg_solver(problem_arg, solvername, options, fileobj):
        assert solvername == "spg"
        assert options["full_results"] is True
        return problem_arg.Xsol.copy(), 0.0, 0.0, 0, "done"

    monkeypatch.setattr(skp, "spg_solver", fake_spg_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 100.0)

    s = skp.SPGSolver(full_results=True)
    result = s.solve(problem)

    assert result.success is True
    assert result.status == 0
    assert result.message == "done"
    assert result.nbiter == 7
    assert result.nfev == 11
    assert result.fun == 0.0
    assert result.error == pytest.approx(0.0)
    assert result.total_fun == [3.0, 2.0]
    assert result.total_crit == [4.0, 1.0]


def test_gkbsolver_solve_uses_stubbed_backend(monkeypatch):
    problem = make_problem()
    problem.stats["nbiter"] = 4
    problem.stats["fev"] = 8
    problem.stats["blocksteps"] = 2
    problem.stats["total_fun"] = [5.0, 1.0]
    problem.stats["total_grad"] = [6.0, 0.5]

    def fake_gkb_setup(problem_arg, solvername, options, fileobj):
        assert solvername == "gkb"
        assert options["full_results"] is True
        return problem_arg.Xsol.copy(), 1.5, 0.25, 0, "ok"

    monkeypatch.setattr(skp, "gkb_setup", fake_gkb_setup)
    monkeypatch.setattr(skp.time, "time", lambda: 50.0)

    s = skp.GKBSolver(full_results=True)
    result = s.solve(problem)

    assert result.success is True
    assert result.status == 0
    assert result.message == "ok"
    assert result.fun == 1.5
    assert result.normcrit == 0.25
    assert result.blocksteps == 2
    assert result.nbiter == 4
    assert result.nfev == 8
    assert result.total_fun == [5.0, 1.0]
    assert result.total_crit == [6.0, 0.5]


def test_gbbsolver_solve_uses_stubbed_backend(monkeypatch):
    problem = make_problem()
    problem.stats["nbiter"] = 3
    problem.stats["fev"] = 9
    problem.stats["total_fun"] = [7.0, 2.0]
    problem.stats["total_crit"] = [8.0, 1.0]

    def fake_gbb_solver(problem_arg, m, n, X, A, B, options, inner, fileobj):
        assert m == problem_arg.sizes[0]
        assert n == problem_arg.sizes[1]
        assert inner is False
        return 0, 2.5, problem_arg.Xsol.copy(), 0.125, 1, "gbb-ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(skp.time, "time", lambda: 10.0)

    s = skp.GBBSolver(full_results=True)
    result = s.solve(problem)

    assert result.success is True
    assert result.status == 0
    assert result.message == "gbb-ok"
    assert result.fun == 2.5
    assert result.nbiter == 3
    assert result.nfev == 9
    assert result.total_fun == [7.0, 2.0]
    assert result.total_crit == [8.0, 1.0]
    assert result.error == pytest.approx(0.0)
