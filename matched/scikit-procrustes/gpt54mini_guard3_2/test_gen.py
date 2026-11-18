import io
import os

import numpy as np
import pytest

import skprocrustes.skprocrustes as skp


def test_procrustes_problem_rejects_non_square_sizes():
    with pytest.raises(Exception, match="square problems"):
        skp.ProcrustesProblem((2, 3, 1, 1), problemnumber=1)


def test_procrustes_problem_rejects_incompatible_dimensions():
    with pytest.raises(Exception, match="divisible by p"):
        skp.ProcrustesProblem((5, 5, 2, 2), problemnumber=1)


def test_procrustes_problem_with_user_matrices_sets_attributes():
    m = n = 4
    p = q = 2
    A = np.arange(m * n, dtype=float).reshape(m, n)
    B = np.ones((m, q))
    C = np.eye(p)

    problem = skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C))

    assert problem.sizes == (m, n, p, q)
    assert problem.problemnumber is None
    assert np.array_equal(problem.A, A)
    assert np.array_equal(problem.B, B)
    assert np.array_equal(problem.C, C)
    assert problem.stats == {}


def test_procrustes_problem_with_user_xsol_sets_optional_solution():
    m = n = 4
    p = q = 2
    A = np.eye(m)
    B = np.ones((m, q))
    C = np.eye(p)
    Xsol = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    problem = skp.ProcrustesProblem((m, n, p, q), matrices=[A, B, C, Xsol])

    assert np.array_equal(problem.Xsol, Xsol)


def test_procrustes_problem_rejects_bad_matrices_container_type():
    m = n = 4
    p = q = 2
    A = np.eye(m)
    B = np.ones((m, q))
    C = np.eye(p)

    with pytest.raises(Exception, match="list or tuple"):
        skp.ProcrustesProblem((m, n, p, q), matrices={0: A, 1: B, 2: C})


def test_procrustes_problem_rejects_too_few_matrices():
    m = n = 4
    p = q = 2
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        skp.ProcrustesProblem((m, n, p, q), matrices=[np.eye(m), np.ones((m, q))])


def test_procrustes_problem_rejects_wrong_A_shape():
    m = n = 4
    p = q = 2
    A = np.eye(m + 1)
    B = np.ones((m, q))
    C = np.eye(p)

    with pytest.raises(Exception, match="A must be \\(m,n\\)"):
        skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C))


def test_procrustes_problem_rejects_problem3_invalid_n():
    np.random.seed(0)
    with pytest.raises(Exception, match="Problem 6 requires n = 50"):
        skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=3)


def test_procrustes_problem_problem1_is_deterministic_with_seed():
    np.random.seed(123)
    p1 = skp.ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    np.random.seed(123)
    p2 = skp.ProcrustesProblem((6, 6, 2, 2), problemnumber=1)

    assert np.allclose(p1.A, p2.A)
    assert np.allclose(p1.B, p2.B)
    assert np.allclose(p1.C, p2.C)
    assert np.allclose(p1.Xsol, p2.Xsol)


def test_procrustes_problem_problem2_is_deterministic_with_seed():
    np.random.seed(321)
    p1 = skp.ProcrustesProblem((6, 6, 2, 2), problemnumber=2)
    np.random.seed(321)
    p2 = skp.ProcrustesProblem((6, 6, 2, 2), problemnumber=2)

    assert np.allclose(p1.A, p2.A)
    assert np.allclose(p1.B, p2.B)
    assert np.allclose(p1.Xsol, p2.Xsol)


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


def test_gbbsolver_rejects_invalid_inner_option():
    with pytest.raises(Exception, match="inner_solver not implemented"):
        skp.GKBSolver(inner_solver="bad")


def test_gbbsolver_rejects_invalid_projected_gradient():
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        skp.GBBSolver(projected_gradient=3)


def test_gbbsolver_open_file_uses_stdout_when_filename_none():
    solver = skp.GBBSolver()
    solver.open_file()
    try:
        assert solver.file is skp.sys.stdout
    finally:
        solver.close_file()


def test_gbbsolver_open_and_close_file_with_filename(tmp_path):
    path = tmp_path / "gbb_output.txt"
    solver = skp.GBBSolver(filename=str(path))
    solver.open_file()
    try:
        assert solver.file is not None
        solver.file.write("hello")
    finally:
        solver.close_file()

    assert path.read_text() == "hello"


def test_gbbsolver_open_file_handles_oserror_and_exits(monkeypatch):
    solver = skp.GBBSolver(filename="bad_name.txt")

    def fake_open(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(skp.sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as excinfo:
        solver.open_file()

    assert excinfo.value.code == -1


def test_gbbsolver_solve_calls_underlying_solver_and_builds_result(monkeypatch):
    np.random.seed(42)

    m = n = 4
    p = q = 2
    A = np.eye(m)
    B = np.zeros((m, q))
    C = np.eye(p)
    Xsol = np.zeros((n, p))
    problem = skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C, Xsol))
    problem.stats = {"nbiter": 7, "fev": 11}

    captured = {}

    def fake_gbb_solver(problem_arg, largedim, smalldim, X, A_arg, B_arg, options, inner, fileobj, B1=None, blobopprod=0.0):
        captured["problem"] = problem_arg
        captured["largedim"] = largedim
        captured["smalldim"] = smalldim
        captured["Xshape"] = X.shape
        captured["A"] = A_arg
        captured["B"] = B_arg
        captured["options"] = dict(options)
        captured["inner"] = inner
        captured["fileobj_is_stdout"] = fileobj is skp.sys.stdout
        return 0, 1.5, np.eye(n, p), 0.25, 3, "ok"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    solver = skp.GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert captured["problem"] is problem
    assert captured["largedim"] == m
    assert captured["smalldim"] == n
    assert captured["Xshape"] == (n, p)
    assert np.array_equal(captured["A"], A)
    assert np.array_equal(captured["B"], B)
    assert captured["inner"] is False
    assert captured["fileobj_is_stdout"] is True

    assert result.success is True
    assert result.status == 0
    assert result.message == "ok"
    assert result.fun == pytest.approx(1.5)
    assert result.normcrit == pytest.approx(0.25)
    assert result.nbiter == 7
    assert result.nfev == 11
    assert result.error == pytest.approx(1.0)
    assert result.blocksteps == 3


def test_gbbsolver_solve_without_xsol_sets_error_to_infinity(monkeypatch):
    np.random.seed(7)

    m = n = 4
    p = q = 2
    A = np.eye(m)
    B = np.zeros((m, q))
    C = np.eye(p)
    problem = skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C))
    problem.stats = {"nbiter": 0, "fev": 0}

    def fake_gbb_solver(problem_arg, largedim, smalldim, X, A_arg, B_arg, options, inner, fileobj, B1=None, blobopprod=0.0):
        return 0, 0.0, np.eye(n, p), 0.0, 0, "done"

    monkeypatch.setattr(skp, "gbb_solver", fake_gbb_solver)

    result = skp.GBBSolver(verbose=0).solve(problem)
    assert result.error == np.inf
    assert result.success is True
