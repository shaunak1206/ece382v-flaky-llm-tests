import io
import os
import numpy as np
import pytest

from skprocrustes.skprocrustes import (
    ProcrustesProblem,
    GBBSolver,
    OptimizeResult,
)


def test_procrustesproblem_rejects_non_square_sizes():
    with pytest.raises(Exception, match="square problems"):
        ProcrustesProblem((2, 3, 1, 1), problemnumber=1)


def test_procrustesproblem_rejects_incompatible_dimensions():
    with pytest.raises(Exception, match="divisible by p"):
        ProcrustesProblem((5, 5, 2, 2), problemnumber=1)


def test_procrustesproblem_builds_default_problem_and_solution_is_permutation():
    np.random.seed(0)
    problem = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)

    assert problem.A.shape == (4, 4)
    assert problem.B.shape == (4, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (4, 2)
    assert problem.stats == {}

    # Xsol is built from np.eye(n, p) and permuted, so each column has one 1.
    assert np.allclose(problem.Xsol.sum(axis=0), np.ones(2))
    assert np.allclose(problem.Xsol.sum(axis=1), np.array([1, 1, 0, 0]))
    assert np.all(np.logical_or(problem.Xsol == 0.0, problem.Xsol == 1.0))


def test_procrustesproblem_problemnumber_3_rejects_invalid_n():
    with pytest.raises(Exception, match="requires n = 50"):
        ProcrustesProblem((4, 4, 2, 2), problemnumber=3)


def test_procrustesproblem_accepts_user_matrices_and_solution():
    A = np.arange(16.0).reshape(4, 4)
    B = np.arange(8.0).reshape(4, 2)
    C = np.eye(2)
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    problem = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, X))

    assert np.all(problem.A == A)
    assert np.all(problem.B == B)
    assert np.all(problem.C == C)
    assert np.all(problem.Xsol == X)


def test_procrustesproblem_rejects_bad_matrices_container_type():
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    C = np.eye(2)

    with pytest.raises(Exception, match="must be a list or tuple"):
        ProcrustesProblem((4, 4, 2, 2), matrices=A)  # not a tuple/list


def test_procrustesproblem_rejects_too_few_matrices():
    with pytest.raises(Exception, match="must be a list or tuple"):
        ProcrustesProblem((4, 4, 2, 2), matrices=[np.zeros((4, 4)), np.zeros((4, 2))])


def test_procrustesproblem_rejects_bad_A_shape():
    A = np.zeros((3, 4))
    B = np.zeros((4, 2))
    C = np.eye(2)

    with pytest.raises(Exception, match="A must be \\(m,n\\)"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))


def test_procrustesproblem_rejects_bad_B_shape():
    A = np.zeros((4, 4))
    B = np.zeros((3, 2))
    C = np.eye(2)

    with pytest.raises(Exception, match="B must be \\(m,q\\)"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))


def test_procrustesproblem_rejects_bad_C_shape():
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    C = np.eye(3)

    with pytest.raises(Exception, match="C must be \\(p,q\\)"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))


def test_procrustesproblem_rejects_bad_xsol_shape():
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    C = np.eye(2)
    X = np.zeros((4, 1))

    with pytest.raises(Exception, match="Xsol must be \\(n,p\\)"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, X))


def test_gbbsolver_defaults_are_set():
    solver = GBBSolver()
    opts = solver.options

    assert solver.solvername == "gbb"
    assert opts["full_results"] is False
    assert opts["xtol"] == pytest.approx(1e-6)
    assert opts["gtol"] == pytest.approx(1e-6)
    assert opts["ftol"] == pytest.approx(1e-12)
    assert opts["zeta"] == pytest.approx(1e-4)
    assert opts["kappa"] == pytest.approx(0.1)
    assert opts["eta"] == pytest.approx(0.85)
    assert opts["etavar"] is False
    assert opts["tau"] == pytest.approx(1e-3)
    assert opts["projected_gradient"] == 1
    assert opts["maxiter"] == 5000
    assert opts["verbose"] == 1
    assert opts["filename"] is None
    assert opts["timer"] is False


def test_gbbsolver_rejects_invalid_option_types_and_values():
    with pytest.raises(Exception, match="xtol must be a float"):
        GBBSolver(xtol=1)

    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        GBBSolver(verbose=2)

    with pytest.raises(Exception, match="inner_solver not implemented"):
        GBBSolver(inner_solver="bad")

    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        GBBSolver(projected_gradient=3)


def test_gbbsolver_sets_inner_solver_specific_options_spg():
    solver = GBBSolver(inner_solver="spg", changevar=True, polar="ns")
    assert solver.options["inner_solver"] == "spg"
    assert solver.options["changevar"] is True
    assert solver.options["polar"] == "ns"


def test_gbbsolver_sets_inner_solver_specific_options_gbb_and_overrides_polar():
    solver = GBBSolver(inner_solver="gbb", xtol=1e-5, ftol=1e-10, zeta=1e-3, kappa=0.2, tau=1e-2)
    assert solver.options["inner_solver"] == "gbb"
    assert solver.options["polar"] is None
    assert solver.options["xtol"] == pytest.approx(1e-5)
    assert solver.options["ftol"] == pytest.approx(1e-10)
    assert solver.options["zeta"] == pytest.approx(1e-3)
    assert solver.options["kappa"] == pytest.approx(0.2)
    assert solver.options["tau"] == pytest.approx(1e-2)


def test_gbbsolver_open_file_uses_stdout_when_no_filename():
    solver = GBBSolver()
    solver.open_file()
    try:
        assert solver.file is os.sys.stdout
    finally:
        solver.close_file()


def test_gbbsolver_open_file_and_close_file_with_filename(tmp_path):
    path = tmp_path / "out.txt"
    solver = GBBSolver(filename=str(path))
    solver.open_file()
    try:
        assert solver.file is not os.sys.stdout
        assert not solver.file.closed
    finally:
        solver.close_file()
    assert solver.file.closed


def test_gbbsolver_solve_success_uses_patched_backend(monkeypatch):
    def fake_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj):
        assert problem.sizes == (4, 4, 2, 2)
        assert m == 4 and n == 4
        assert X.shape == (4, 2)
        assert A.shape == (4, 4)
        assert B.shape == (4, 2)
        assert inner is False
        return 0, 12.5, np.ones((4, 2)), 0.25, 3, "done"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape))

    problem = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    problem.stats = {"nbiter": 0, "fev": 0, "blocksteps": 0}

    solver = GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert isinstance(result, OptimizeResult)
    assert result.success is True
    assert result.status == 0
    assert result.message == "done"
    assert result.fun == pytest.approx(12.5)
    assert result.normcrit == pytest.approx(0.25)
    assert result.nbiter == 0
    assert result.nfev == 0
    assert result.blocksteps == 0
    assert result.solution.shape == (4, 2)


def test_gbbsolver_solve_computes_error_when_exact_solution_present(monkeypatch):
    Xsol = np.eye(4, 2)

    def fake_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj):
        return 0, 0.0, Xsol.copy(), 0.0, 0, "ok"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape))

    problem = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    problem.Xsol = Xsol
    problem.stats = {"nbiter": 0, "fev": 0, "blocksteps": 0}

    solver = GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert result.error == pytest.approx(0.0)
    assert result.solution.shape == Xsol.shape


def test_gbbsolver_solve_raises_when_full_results_requested_but_stats_missing(monkeypatch):
    def fake_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj):
        return 0, 0.0, np.zeros((4, 2)), 0.0, 0, "ok"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", fake_gbb_solver)
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape))

    problem = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)
    problem.stats = {"nbiter": 0, "fev": 0, "blocksteps": 0}

    solver = GBBSolver(full_results=True, verbose=0)

    with pytest.raises(Exception, match='problem.stats\\["total_fun"\\] and problem.stats\\["total_grad"\\]'):
        solver.solve(problem)


def test_optimize_result_attribute_access_and_repr():
    res = OptimizeResult(success=True, status=0, message="ok", fun=1.5)
    assert res.success is True
    assert res.status == 0
    assert res.message == "ok"
    assert "success" in dir(res)
    text = repr(res)
    assert "success" in text and "fun" in text

    with pytest.raises(AttributeError):
        _ = res.does_not_exist
