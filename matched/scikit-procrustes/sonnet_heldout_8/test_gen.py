import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys

import numpy as np
import pytest

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver


def _user_problem(m=6, p=2, with_xsol=True, seed=0):
    rng = np.random.RandomState(seed)
    n, q = m, p
    A = rng.randn(m, n)
    Xsol, _ = np.linalg.qr(rng.randn(n, p))
    C = np.eye(p, q)
    B = A.dot(Xsol).dot(C)
    mats = (A, B, C, Xsol) if with_xsol else (A, B, C)
    return (m, n, p, q), mats


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("number,sizes", [(1, (10, 10, 2, 2)),
                                          (2, (12, 12, 3, 3)),
                                          (3, (50, 50, 5, 5))])
def test_default_problems_shapes(number, sizes):
    np.random.seed(1)
    m, n, p, q = sizes
    prob = ProcrustesProblem(sizes, problemnumber=number)
    assert prob.sizes == sizes
    assert prob.problemnumber == number
    assert prob.A.shape == (m, n)
    assert prob.B.shape == (m, q)
    assert prob.C.shape == (p, q)
    assert prob.Xsol.shape == (n, p)


def test_default_problem_consistency_and_orthogonality():
    np.random.seed(2)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    np.testing.assert_allclose(prob.B, prob.A.dot(prob.Xsol).dot(prob.C),
                               atol=1e-12)
    np.testing.assert_allclose(prob.Xsol.T.dot(prob.Xsol), np.eye(2),
                               atol=1e-12)
    np.testing.assert_allclose(prob.C, np.eye(2))


def test_problem1_singular_values_in_range():
    np.random.seed(3)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    s = np.linalg.svd(prob.A, compute_uv=False)
    assert s.min() >= 10.0 - 1e-8
    assert s.max() <= 12.0 + 1e-8


def test_stats_initially_empty_dict():
    prob = ProcrustesProblem((6, 6, 2, 2), problemnumber=1)
    assert prob.stats == {}


def test_non_square_sizes_raise():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 8, 2, 2), problemnumber=1)
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_m_not_divisible_by_p_raises():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_no_problemnumber_nor_matrices_raises():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2))


def test_problem3_invalid_n_raises():
    with pytest.raises(Exception):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_user_matrices_with_xsol():
    sizes, mats = _user_problem()
    prob = ProcrustesProblem(sizes, matrices=mats)
    assert prob.A is mats[0]
    assert prob.B is mats[1]
    assert prob.C is mats[2]
    assert prob.Xsol is mats[3]


def test_user_matrices_without_xsol_and_as_list():
    sizes, mats = _user_problem(with_xsol=False)
    prob = ProcrustesProblem(sizes, matrices=list(mats))
    assert not hasattr(prob, "Xsol")
    assert prob.A.shape == (6, 6)


@pytest.mark.parametrize("which", [0, 1, 2, 3])
def test_user_matrices_wrong_shape_raises(which):
    sizes, mats = _user_problem()
    mats = list(mats)
    mats[which] = np.zeros((mats[which].shape[0] + 1, mats[which].shape[1]))
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=mats)


def test_user_matrices_too_few_raises():
    sizes, mats = _user_problem()
    with pytest.raises(Exception):
        ProcrustesProblem(sizes, matrices=mats[:2])


# ---------------------------------------------------------------------------
# GBBSolver options
# ---------------------------------------------------------------------------

def test_gbb_default_options():
    solver = GBBSolver()
    assert solver.solvername == "gbb"
    opts = solver.options
    assert opts["full_results"] is False
    assert opts["xtol"] == 1e-6
    assert opts["gtol"] == 1e-6
    assert opts["ftol"] == 1e-12
    assert opts["zeta"] == 1e-4
    assert opts["kappa"] == 0.1
    assert opts["eta"] == 0.85
    assert opts["etavar"] is False
    assert opts["tau"] == 1e-3
    assert opts["projected_gradient"] == 1
    assert opts["maxiter"] == 5000
    assert opts["verbose"] == 1
    assert opts["filename"] is None
    assert opts["timer"] is False


def test_gbb_custom_options_kept():
    solver = GBBSolver(xtol=1e-4, gtol=1e-5, maxiter=10, verbose=0,
                       projected_gradient=2, etavar=True, timer=True)
    assert solver.options["xtol"] == 1e-4
    assert solver.options["gtol"] == 1e-5
    assert solver.options["maxiter"] == 10
    assert solver.options["verbose"] == 0
    assert solver.options["projected_gradient"] == 2
    assert solver.options["etavar"] is True
    assert solver.options["timer"] is True


@pytest.mark.parametrize("kwargs", [
    {"full_results": 1},
    {"xtol": 1},
    {"gtol": "small"},
    {"ftol": 1},
    {"zeta": 1},
    {"kappa": 1},
    {"eta": 1},
    {"etavar": "yes"},
    {"tau": 1},
    {"projected_gradient": 3},
    {"maxiter": 10.0},
    {"verbose": 2},
    {"filename": 5},
    {"timer": 1},
])
def test_gbb_invalid_options_raise(kwargs):
    with pytest.raises(Exception):
        GBBSolver(**kwargs)


def test_gbb_open_close_file_stdout_and_file(tmp_path):
    solver = GBBSolver()
    solver.open_file()
    assert solver.file is sys.stdout
    solver.close_file()
    assert not sys.stdout.closed

    fname = str(tmp_path / "out.txt")
    solver = GBBSolver(filename=fname)
    solver.open_file()
    assert os.path.exists(fname)
    solver.close_file()
    assert solver.file.closed


# ---------------------------------------------------------------------------
# GBBSolver.solve
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pg", [1, 2])
def test_gbb_solve_result_structure_and_feasibility(pg):
    np.random.seed(5)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, projected_gradient=pg).solve(prob)
    for key in ("success", "status", "message", "solution", "fun", "error",
                "cpu", "nbiter", "nfev", "total_fun", "total_crit"):
        assert key in result
    X = result.solution
    assert X.shape == (10, 2)
    np.testing.assert_allclose(X.T.dot(X), np.eye(2), atol=1e-6)
    assert np.isfinite(result.fun)
    assert result.fun >= 0
    assert result.cpu >= 0
    assert result.nbiter == prob.stats["nbiter"]
    assert result.nfev == prob.stats["fev"]
    assert result.nfev > 0
    assert result.success == (result.status == 0)
    assert np.isfinite(result.error)


def test_gbb_solve_without_xsol_gives_infinite_error():
    np.random.seed(6)
    sizes, mats = _user_problem(with_xsol=False)
    prob = ProcrustesProblem(sizes, matrices=mats)
    result = GBBSolver(verbose=0).solve(prob)
    assert result.error == np.inf
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf
    np.testing.assert_allclose(result.solution.T.dot(result.solution),
                               np.eye(2), atol=1e-6)


def test_gbb_solve_maxiter_reached_reports_failure():
    np.random.seed(7)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=1).solve(prob)
    assert result.status == 1
    assert result.success is False
    assert result.nbiter == 1
    assert "Maximum number of iterations" in result.message


def test_gbb_solve_full_results_with_stats_prepared():
    np.random.seed(8)
    prob = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    prob.stats["total_crit"] = []
    result = GBBSolver(verbose=0, full_results=True, maxiter=20).solve(prob)
    assert isinstance(result.total_fun, list)
    assert len(result.total_fun) >= 2
    assert all(np.isfinite(v) for v in result.total_fun)


def test_gbb_solve_writes_to_file(tmp_path):
    np.random.seed(9)
    fname = str(tmp_path / "gbb_log.txt")
    prob = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)
    solver = GBBSolver(filename=fname, verbose=1, maxiter=5)
    solver.solve(prob)
    assert os.path.exists(fname)
    with open(fname) as fh:
        content = fh.read()
    assert "GBB Solver" in content
