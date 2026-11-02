import io
import os
import numpy as np
import pytest

import skprocrustes.skprocrustes as skp


def test_problem_rejects_non_square_dimensions():
    with pytest.raises(Exception, match="square problems"):
        skp.ProcrustesProblem((4, 5, 2, 2), problemnumber=1)


def test_problem_rejects_incompatible_dimensions():
    with pytest.raises(Exception, match="divisible by p"):
        skp.ProcrustesProblem((5, 5, 2, 2), problemnumber=1)


def test_problem_with_user_matrices_sets_attributes_and_stats():
    m = n = 4
    p = q = 2
    A = np.arange(m * n, dtype=float).reshape(m, n)
    B = np.ones((m, q), dtype=float)
    C = np.eye(p, q)
    X = np.eye(n, p)

    problem = skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C, X))

    assert problem.sizes == (m, n, p, q)
    assert np.array_equal(problem.A, A)
    assert np.array_equal(problem.B, B)
    assert np.array_equal(problem.C, C)
    assert np.array_equal(problem.Xsol, X)
    assert isinstance(problem.stats, dict)
    assert problem.stats == {}


def test_problem_with_user_matrices_validates_length_and_type():
    m = n = 4
    p = q = 2
    A = np.zeros((m, n))
    B = np.zeros((m, q))
    C = np.zeros((p, q))

    with pytest.raises(Exception, match="list or tuple"):
        skp.ProcrustesProblem((m, n, p, q), matrices=A)

    with pytest.raises(Exception, match="list or tuple"):
        skp.ProcrustesProblem((m, n, p, q), matrices=(A, B))


def test_problem_with_user_matrices_validates_shapes():
    m = n = 4
    p = q = 2
    A = np.zeros((m, n))
    B = np.zeros((m, q))
    C = np.zeros((p, q))
    X = np.zeros((n, p))

    with pytest.raises(Exception, match="A must be \\(m,n\\)"):
        skp.ProcrustesProblem((m, n, p, q), matrices=(A[:3], B, C, X))

    with pytest.raises(Exception, match="B must be \\(m,q\\)"):
        skp.ProcrustesProblem((m, n, p, q), matrices=(A, B[:3], C, X))

    with pytest.raises(Exception, match="C must be \\(p,q\\)"):
        skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C[:1], X))

    with pytest.raises(Exception, match="Xsol must be \\(n,p\\)"):
        skp.ProcrustesProblem((m, n, p, q), matrices=(A, B, C, X[:3]))


def test_problemnumber_required_when_matrices_missing():
    with pytest.raises(Exception, match="Problem matrices are empty"):
        skp.ProcrustesProblem((4, 4, 2, 2))


def test_problemnumber_3_invalid_n_raises():
    with pytest.raises(Exception, match="requires n = 50"):
        skp.ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_problem_default_generation_is_reproducible_with_seed():
    np.random.seed(123)
    p1 = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    np.random.seed(123)
    p2 = skp.ProcrustesProblem((4, 4, 2, 2), problemnumber=1)

    assert np.allclose(p1.A, p2.A)
    assert np.allclose(p1.B, p2.B)
    assert np.allclose(p1.C, p2.C)
    assert np.allclose(p1.Xsol, p2.Xsol)


def test_optimize_result_attribute_access_and_repr():
    r = skp.OptimizeResult(success=True, status=0, fun=1.5)
    assert r.success is True
    assert r.status == 0
    assert r.fun == 1.5
    assert "success" in dir(r)
    assert "status" in repr(r)
    with pytest.raises(AttributeError):
        _ = r.missing_key


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
        skp.GBBSolver(full_results="nope")

    with pytest.raises(Exception, match="xtol must be a float"):
        skp.GBBSolver(xtol=1)

    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        skp.GBBSolver(verbose=2)

    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        skp.GBBSolver(projected_gradient=3)


def test_gbbsolver_inner_solver_spg_allows_changevar_and_polar():
    solver = skp.GBBSolver(inner_solver="spg", changevar=True, polar="ns")
    assert solver.options["inner_solver"] == "spg"
    assert solver.options["changevar"] is True
    assert solver.options["polar"] == "ns"


def test_gbbsolver_inner_solver_gbb_sets_related_defaults():
    solver = skp.GBBSolver(inner_solver="gbb")
    assert solver.options["inner_solver"] == "gbb"
    assert solver.options["polar"] is None
    assert solver.options["xtol"] == 1e-6
    assert solver.options["ftol"] == 1e-12
    assert solver.options["zeta"] == 1e-4
    assert solver.options["kappa"] == 0.1
    assert solver.options["tau"] == 1e-3
    assert solver.options["projected_gradient"] == 1


def test_gbbsolver_rejects_bad_filename_type_and_bad_inner_solver():
    with pytest.raises(Exception, match="filename must be string"):
        skp.GBBSolver(filename=123)

    with pytest.raises(Exception, match="inner_solver not implemented"):
        skp.GBBSolver(inner_solver="bad")


def test_gbbsolver_open_file_uses_stdout_when_filename_none():
    solver = skp.GBBSolver()
    solver.open_file()
    try:
        assert solver.file is os.sys.stdout
    finally:
        solver.close_file()


def test_gbbsolver_open_file_creates_and_closes_file(tmp_path):
    path = tmp_path / "out.txt"
    solver = skp.GBBSolver(filename=str(path))
    solver.open_file()
    try:
        assert solver.file.name == str(path)
        assert not solver.file.closed
    finally:
        solver.close_file()
    assert solver.file.closed


def test_gbbsolver_open_file_handles_oserror(monkeypatch):
    solver = skp.GBBSolver(filename="badname.txt")

    def fake_open(*args, **kwargs):
        raise OSError("cannot open")

    monkeypatch.setattr(skp, "open", fake_open, raising=False)
    with pytest.raises(SystemExit):
        solver.open_file()


def test_gbbsolver_solve_wires_result_from_backend(monkeypatch):
    solver = skp.GBBSolver(verbose=0)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(
        np.eye(4),
        np.zeros((4, 2)),
        np.eye(2),
        np.eye(4, 2),
    ))
    problem.stats["nbiter"] = 7
    problem.stats["fev"] = 11
    problem.stats["blocksteps"] = 3

    def fake_gbb_setup(problem_arg, solvername, options, fileobj):
        assert solvername == "gbb"
        assert options["verbose"] == 0
        assert fileobj is os.sys.stdout
        return np.eye(4, 2), 12.5, 0.25, 0, "ok"

    monkeypatch.setattr(skp, "gbb_setup", fake_gbb_setup)

    result = solver.solve(problem)

    assert result.success is True
    assert result.status == 0
    assert result.message == "ok"
    assert np.allclose(result.solution, np.eye(4, 2))
    assert result.fun == 12.5
    assert result.normcrit == 0.25
    assert result.nbiter == 7
    assert result.nfev == 11
    assert result.blocksteps == 3
    assert np.isfinite(result.cpu)
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf


def test_gbbsolver_solve_uses_xsol_error(monkeypatch):
    solver = skp.GBBSolver(verbose=0)
    Xsol = np.eye(4, 2)
    problem = skp.ProcrustesProblem((4, 4, 2, 2), matrices=(
        np.eye(4),
        np.zeros((4, 2)),
        np.eye(2),
        Xsol,
    ))
    problem.stats["nbiter"] = 1
    problem.stats["fev"] = 2
    problem.stats["blocksteps"] = 1

    def fake_gbb_setup(problem_arg, solvername, options, fileobj):
        return Xsol.copy(), 0.0, 0.0, 0, "ok"

    monkeypatch.setattr(skp, "gbb_setup", fake_gbb_setup)

    result = solver.solve(problem)
    assert result.error == 0.0
