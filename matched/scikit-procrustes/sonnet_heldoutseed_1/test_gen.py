import numpy as np
import pytest

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(12345)


def _user_problem(n=10, p=2, with_xsol=False):
    A = np.random.randn(n, n) + 5.0 * np.eye(n)
    Q, _ = np.linalg.qr(np.random.randn(n, p))
    C = np.eye(p)
    B = A.dot(Q).dot(C)
    if with_xsol:
        return ProcrustesProblem((n, n, p, p), matrices=(A, B, C, Q))
    return ProcrustesProblem((n, n, p, p), matrices=(A, B, C))


# ---------------------------------------------------------------- Problem

def test_default_problem1_shapes_and_relation():
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    assert prob.A.shape == (10, 10)
    assert prob.B.shape == (10, 2)
    assert prob.C.shape == (2, 2)
    assert prob.Xsol.shape == (10, 2)
    np.testing.assert_allclose(prob.B, prob.A.dot(prob.Xsol).dot(prob.C),
                               atol=1e-10)
    np.testing.assert_allclose(prob.Xsol.T.dot(prob.Xsol), np.eye(2),
                               atol=1e-12)
    assert prob.stats == {}
    assert prob.sizes == (10, 10, 2, 2)


def test_problem1_singular_values_in_range():
    prob = ProcrustesProblem((12, 12, 3, 3), problemnumber=1)
    s = np.linalg.svd(prob.A, compute_uv=False)
    assert s.min() >= 10.0 - 1e-8
    assert s.max() <= 12.0 + 1e-8


def test_problem2_builds():
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=2)
    assert prob.A.shape == (10, 10)
    np.testing.assert_allclose(prob.C, np.eye(2))
    np.testing.assert_allclose(prob.B, prob.A.dot(prob.Xsol).dot(prob.C),
                               atol=1e-8)


def test_problem3_requires_special_n():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_problem3_with_n50():
    prob = ProcrustesProblem((50, 50, 5, 5), problemnumber=3)
    assert prob.A.shape == (50, 50)
    s = np.linalg.svd(prob.A, compute_uv=False)
    assert s.max() > 10.0
    assert s.min() < 0.01


def test_non_square_sizes_raise():
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((10, 8, 2, 2), problemnumber=1)
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_m_not_divisible_by_p_raises():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_no_matrices_no_problemnumber_raises():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2))


def test_user_matrices_without_xsol():
    prob = _user_problem()
    assert prob.A.shape == (10, 10)
    assert prob.B.shape == (10, 2)
    assert prob.C.shape == (2, 2)
    assert not hasattr(prob, "Xsol")


def test_user_matrices_with_xsol():
    prob = _user_problem(with_xsol=True)
    assert prob.Xsol.shape == (10, 2)


def test_user_matrices_wrong_shapes_raise():
    A = np.eye(10)
    B = np.ones((10, 2))
    C = np.eye(2)
    with pytest.raises(Exception, match="A must"):
        ProcrustesProblem((10, 10, 2, 2),
                          matrices=(np.eye(9), B, C))
    with pytest.raises(Exception, match="B must"):
        ProcrustesProblem((10, 10, 2, 2), matrices=(A, np.ones((10, 3)), C))
    with pytest.raises(Exception, match="C must"):
        ProcrustesProblem((10, 10, 2, 2), matrices=(A, B, np.eye(3)))
    with pytest.raises(Exception, match="Xsol must"):
        ProcrustesProblem((10, 10, 2, 2),
                          matrices=(A, B, C, np.ones((10, 3))))


def test_user_matrices_too_few_raise():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2),
                          matrices=(np.eye(10), np.ones((10, 2))))


# ----------------------------------------------------------------- Solver

def test_gbb_default_options():
    solver = GBBSolver()
    o = solver.options
    assert solver.solvername == "gbb"
    assert o["full_results"] is False
    assert o["xtol"] == 1e-6
    assert o["gtol"] == 1e-6
    assert o["ftol"] == 1e-12
    assert o["zeta"] == 1e-4
    assert o["kappa"] == 0.1
    assert o["eta"] == 0.85
    assert o["etavar"] is False
    assert o["tau"] == 1e-3
    assert o["projected_gradient"] == 1
    assert o["maxiter"] == 5000
    assert o["verbose"] == 1
    assert o["filename"] is None
    assert o["timer"] is False


def test_gbb_custom_options_kept():
    solver = GBBSolver(xtol=1e-4, maxiter=10, verbose=0,
                       projected_gradient=2, etavar=True)
    assert solver.options["xtol"] == 1e-4
    assert solver.options["maxiter"] == 10
    assert solver.options["verbose"] == 0
    assert solver.options["projected_gradient"] == 2
    assert solver.options["etavar"] is True


@pytest.mark.parametrize("kwargs", [
    {"full_results": 1},
    {"xtol": 1},
    {"gtol": "a"},
    {"ftol": 1},
    {"zeta": 1},
    {"kappa": 1},
    {"eta": 1},
    {"etavar": 1},
    {"tau": 1},
    {"projected_gradient": 3},
    {"maxiter": 10.0},
    {"verbose": 2},
    {"filename": 5},
    {"timer": "yes"},
])
def test_gbb_invalid_options_raise(kwargs):
    with pytest.raises(Exception):
        GBBSolver(**kwargs)


def test_gbb_solve_result_structure():
    prob = _user_problem(with_xsol=True)
    solver = GBBSolver(verbose=0)
    result = solver.solve(prob)
    for key in ("success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"):
        assert key in result
    assert result.solution.shape == (10, 2)
    np.testing.assert_allclose(result.solution.T.dot(result.solution),
                               np.eye(2), atol=1e-8)
    assert result.status in (0, 1)
    assert result.success == (result.status == 0)
    assert result.nbiter == prob.stats["nbiter"]
    assert result.nfev == prob.stats["fev"]
    assert prob.stats["feasi"] < 1e-8
    assert np.isfinite(result.error)
    assert result.cpu >= 0
    # full_results not requested
    assert result.total_fun == np.inf


def test_gbb_solve_without_xsol_gives_inf_error():
    prob = _user_problem(with_xsol=False)
    result = GBBSolver(verbose=0).solve(prob)
    assert result.error == np.inf


def test_gbb_maxiter_reached():
    prob = _user_problem()
    result = GBBSolver(verbose=0, maxiter=1).solve(prob)
    assert result.status == 1
    assert result.success is False
    assert "Maximum number of iterations" in result.message
    assert prob.stats["nbiter"] == 1


def test_gbb_full_results():
    prob = _user_problem()
    result = GBBSolver(verbose=0, full_results=True, maxiter=20).solve(prob)
    assert isinstance(result.total_fun, list)
    assert isinstance(result.total_crit, list)
    assert len(result.total_fun) == prob.stats["fev"]
    assert len(result.total_crit) == prob.stats["grad"]
    assert all(v >= 0 for v in result.total_fun)


def test_gbb_projected_gradient_2_runs():
    prob = _user_problem()
    result = GBBSolver(verbose=0, projected_gradient=2,
                       maxiter=50).solve(prob)
    assert result.solution.shape == (10, 2)
    assert np.all(np.isfinite(result.solution))


def test_gbb_writes_to_file(tmp_path):
    fname = str(tmp_path / "gbb_out.txt")
    prob = _user_problem()
    solver = GBBSolver(verbose=1, filename=fname, maxiter=5)
    solver.solve(prob)
    with open(fname) as fh:
        content = fh.read()
    assert "Options" in content
    assert "nbiter" in content


def test_gbb_open_close_file_default_stdout():
    import sys
    solver = GBBSolver(verbose=0)
    solver.open_file()
    assert solver.file is sys.stdout
    solver.close_file()
    assert not sys.stdout.closed
