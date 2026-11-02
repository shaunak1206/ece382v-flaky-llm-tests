import os
import sys
import numpy as np
import pytest

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, OptimizeResult


def test_problem_rejects_non_square_problem():
    with pytest.raises(Exception, match="Currently only square problems are supported"):
        ProcrustesProblem((2, 3, 1, 1), problemnumber=1)


def test_problem_rejects_incompatible_dimensions():
    with pytest.raises(Exception, match="Dimensions are incompatible"):
        ProcrustesProblem((5, 5, 2, 2), problemnumber=1)


def test_problem_accepts_custom_matrices_and_sets_attributes():
    m = n = 4
    p = q = 2
    A = np.arange(m * n, dtype=float).reshape(m, n)
    B = np.ones((m, q))
    C = np.eye(p, q)
    X = np.eye(n, p)

    problem = ProcrustesProblem((m, n, p, q), matrices=(A, B, C, X))

    assert problem.sizes == (m, n, p, q)
    assert np.array_equal(problem.A, A)
    assert np.array_equal(problem.B, B)
    assert np.array_equal(problem.C, C)
    assert np.array_equal(problem.Xsol, X)
    assert problem.stats == {}


def test_problem_rejects_matrices_not_tuple_or_list():
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    C = np.eye(2)

    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        ProcrustesProblem((4, 4, 2, 2), matrices=A)  # type: ignore[arg-type]


def test_problem_rejects_too_few_matrices():
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(np.zeros((4, 4)), np.zeros((4, 2))))


def test_problem_rejects_invalid_a_shape():
    A = np.zeros((3, 4))
    B = np.zeros((4, 2))
    C = np.eye(2)
    with pytest.raises(Exception, match="A must be \\(m,n\\)"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))


def test_problem_rejects_invalid_b_shape():
    A = np.zeros((4, 4))
    B = np.zeros((3, 2))
    C = np.eye(2)
    with pytest.raises(Exception, match="B must be \\(m,q\\)"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))


def test_problem_rejects_invalid_c_shape():
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    C = np.eye(3)
    with pytest.raises(Exception, match="C must be \\(p,q\\)"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C))


def test_problem_rejects_invalid_xsol_shape():
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    C = np.eye(2)
    X = np.zeros((3, 2))
    with pytest.raises(Exception, match="Xsol must be \\(n,p\\)"):
        ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, X))


def test_problem_default_construction_with_monkeypatched_random(monkeypatch):
    monkeypatch.setattr(np.random, "rand", lambda *args, **kwargs: np.zeros(args if len(args) > 1 else args[0]))
    monkeypatch.setattr(np.random, "randn", lambda *args, **kwargs: np.ones(args if len(args) > 1 else args[0]))
    monkeypatch.setattr(np.random, "permutation", lambda x: x)

    problem = ProcrustesProblem((4, 4, 2, 2), problemnumber=1)

    assert problem.A.shape == (4, 4)
    assert problem.B.shape == (4, 2)
    assert problem.C.shape == (2, 2)
    assert problem.Xsol.shape == (4, 2)
    assert np.allclose(problem.C, np.eye(2))
    assert np.allclose(problem.Xsol, np.eye(4, 2))


def test_gbbsolver_defaults_are_set():
    solver = GBBSolver()
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
    assert solver.options["verbose"] == 1
    assert solver.options["filename"] is None
    assert solver.options["timer"] is False


def test_gbbsolver_rejects_invalid_option_types_and_values():
    with pytest.raises(Exception, match="full_results must be a boolean"):
        GBBSolver(full_results="nope")  # type: ignore[arg-type]

    with pytest.raises(Exception, match="xtol must be a float"):
        GBBSolver(xtol=1)  # type: ignore[arg-type]

    with pytest.raises(Exception, match="gtol must be a float"):
        GBBSolver(gtol=1)  # type: ignore[arg-type]

    with pytest.raises(Exception, match="ftol must be a float"):
        GBBSolver(ftol=1)  # type: ignore[arg-type]

    with pytest.raises(Exception, match="zeta must be a float"):
        GBBSolver(zeta=1)  # type: ignore[arg-type]

    with pytest.raises(Exception, match="kappa must be a float"):
        GBBSolver(kappa=1)  # type: ignore[arg-type]

    with pytest.raises(Exception, match="eta must be a float"):
        GBBSolver(eta=1)  # type: ignore[arg-type]

    with pytest.raises(Exception, match="etavar must be a boolean"):
        GBBSolver(etavar="nope")  # type: ignore[arg-type]

    with pytest.raises(Exception, match="tau must be a float"):
        GBBSolver(tau=1)  # type: ignore[arg-type]

    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        GBBSolver(projected_gradient=3)

    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        GBBSolver(verbose=2)

    with pytest.raises(Exception, match="timer must be boolean"):
        GBBSolver(timer="nope")  # type: ignore[arg-type]


def test_gbbsolver_write_to_existing_filename_works(tmp_path):
    outfile = tmp_path / "gbb.log"
    solver = GBBSolver(filename=str(outfile))
    assert solver.options["filename"] == str(outfile)

    solver.open_file()
    try:
        assert solver.file is not None
        solver.file.write("hello")
    finally:
        solver.close_file()

    assert outfile.read_text() == "hello"


def test_gbbsolver_open_file_exits_on_invalid_filename(monkeypatch):
    solver = GBBSolver(filename="invalid_dir/subdir/out.log")

    with pytest.raises(SystemExit):
        solver.open_file()


def test_gbbsolver_solve_returns_optimize_result(monkeypatch):
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    problem = ProcrustesProblem((2, 2, 2, 2), matrices=(A, B, C))
    problem.stats["nbiter"] = 7
    problem.stats["fev"] = 11
    problem.stats["blocksteps"] = 3
    problem.stats["total_fun"] = [1.0, 0.5]
    problem.stats["total_crit"] = [0.2, 0.1]

    expected_x = np.eye(2)
    expected_f = 0.0
    expected_opt = 1e-9
    expected_exit = 0
    expected_msg = "mock success"

    def fake_gbb_setup(problem_arg, solvername, options, fileobj):
        assert solvername == "gbb"
        return expected_x, expected_f, expected_opt, expected_exit, expected_msg

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_solver", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wrong solver called")))
    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_setup", fake_gbb_setup)

    solver = GBBSolver(full_results=True, verbose=0)
    result = solver.solve(problem)

    assert isinstance(result, OptimizeResult)
    assert result.success is True
    assert result.status == expected_exit
    assert result.message == expected_msg
    assert np.array_equal(result.solution, expected_x)
    assert result.fun == expected_f
    assert result.normcrit == expected_opt
    assert result.nbiter == 7
    assert result.nfev == 11
    assert result.blocksteps == 3
    assert result.total_fun == [1.0, 0.5]
    assert result.total_crit == [0.2, 0.1]
    assert result.error == 0.0


def test_gbbsolver_solve_uses_inf_error_when_no_xsol(monkeypatch):
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    problem = ProcrustesProblem((2, 2, 2, 2), matrices=(A, B, C))
    problem.stats["nbiter"] = 1
    problem.stats["fev"] = 2
    problem.stats["blocksteps"] = 1

    def fake_gbb_setup(problem_arg, solvername, options, fileobj):
        return np.eye(2), 3.0, 0.0, 0, "ok"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_setup", fake_gbb_setup)

    solver = GBBSolver(verbose=0)
    result = solver.solve(problem)

    assert np.isinf(result.error)
    assert result.success is True


def test_gbbsolver_solve_raises_when_full_results_missing_stats(monkeypatch):
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    problem = ProcrustesProblem((2, 2, 2, 2), matrices=(A, B, C))
    problem.stats["nbiter"] = 1
    problem.stats["fev"] = 2
    problem.stats["blocksteps"] = 1

    def fake_gbb_setup(problem_arg, solvername, options, fileobj):
        return np.eye(2), 0.0, 0.0, 0, "ok"

    monkeypatch.setattr("skprocrustes.skprocrustes.gbb_setup", fake_gbb_setup)

    solver = GBBSolver(full_results=True, verbose=0)

    with pytest.raises(Exception, match='For full results, set problem.stats\\["total_fun"\\] and problem.stats\\["total_grad"\\]'):
        solver.solve(problem)


def test_optimize_result_attribute_access_and_repr():
    res = OptimizeResult(success=True, status=0, message="done")
    assert res.success is True
    assert res.status == 0
    assert res.message == "done"
    assert "success" in repr(res)
    assert "status" in repr(res)
    assert set(dir(res)) == {"success", "status", "message"}
