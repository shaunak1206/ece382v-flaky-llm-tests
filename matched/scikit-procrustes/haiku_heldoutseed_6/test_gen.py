import numpy as np
import pytest

from skprocrustes.skprocrustes import (GBBSolver, OptimizeResult,
                                       ProcrustesProblem, ProcrustesSolver)


def _seeded_problem(sizes, problemnumber=1, seed=0):
    np.random.seed(seed)
    return ProcrustesProblem(sizes, problemnumber=problemnumber)


def _user_data(seed=0):
    """Square 6x6 user problem with p = q = 2 and B = A X C for orthonormal X."""
    rng = np.random.RandomState(seed)
    A = rng.randn(6, 6)
    X = np.linalg.qr(rng.randn(6, 2))[0]
    C = np.eye(2)
    B = A.dot(X).dot(C)
    return A, B, C, X


def _assert_orthonormal(X, p):
    np.testing.assert_allclose(X.T.dot(X), np.eye(p), atol=1e-6)


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_default_problem1_matrix_shapes():
    prob = _seeded_problem((10, 10, 2, 2), problemnumber=1)
    assert prob.A.shape == (10, 10)
    assert prob.B.shape == (10, 2)
    assert prob.C.shape == (2, 2)
    assert prob.Xsol.shape == (10, 2)


def test_default_problem_B_consistent_with_solution():
    prob = _seeded_problem((10, 10, 2, 2), problemnumber=1)
    np.testing.assert_allclose(prob.B, prob.A.dot(prob.Xsol).dot(prob.C),
                               atol=1e-10)
    np.testing.assert_allclose(prob.C, np.eye(2))
    _assert_orthonormal(prob.Xsol, 2)


def test_problem1_singular_values_in_range():
    prob = _seeded_problem((10, 10, 2, 2), problemnumber=1)
    s = np.linalg.svd(prob.A, compute_uv=False)
    assert np.all(s >= 10.0 - 1e-8)
    assert np.all(s <= 12.0 + 1e-8)


def test_problem3_singular_value_groups():
    prob = _seeded_problem((50, 50, 5, 5), problemnumber=3)
    s = np.linalg.svd(prob.A, compute_uv=False)  # descending order
    eps = 1e-8
    # n1=15 values in [10, 11), n2=15 in [5, 6), n3=12 in [2, 3), n4=8 in [0, 1e-3)
    assert np.all((s[0:15] >= 10 - eps) & (s[0:15] <= 11 + eps))
    assert np.all((s[15:30] >= 5 - eps) & (s[15:30] <= 6 + eps))
    assert np.all((s[30:42] >= 2 - eps) & (s[30:42] <= 3 + eps))
    assert np.all((s[42:] >= 0 - eps) & (s[42:] <= 1e-3 + eps))


def test_problem3_unsupported_n_raises():
    with pytest.raises(Exception, match="requires n"):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_missing_matrices_and_problemnumber_raises():
    with pytest.raises(Exception, match="Problem matrices are empty"):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=None)
    with pytest.raises(Exception, match="Problem matrices are empty"):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=4)


def test_non_square_sizes_raise():
    with pytest.raises(Exception, match="only square problems"):
        ProcrustesProblem((10, 12, 2, 2), problemnumber=1)
    with pytest.raises(Exception, match="only square problems"):
        ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_m_not_divisible_by_p_raises():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_user_matrices_list_stored_without_xsol():
    A, B, C, _ = _user_data()
    prob = ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C])
    assert prob.A is A
    assert prob.B is B
    assert prob.C is C
    assert "Xsol" not in prob.__dict__


def test_user_matrices_tuple_with_xsol_stored():
    A, B, C, X = _user_data()
    prob = ProcrustesProblem((6, 6, 2, 2), matrices=(A, B, C, X))
    assert prob.Xsol is X
    assert prob.sizes == (6, 6, 2, 2)


def test_user_matrices_wrong_shapes_raise():
    cases = [
        ([np.zeros((6, 5)), np.zeros((6, 2)), np.eye(2)], "A must be"),
        ([np.zeros((6, 6)), np.zeros((6, 3)), np.eye(2)], "B must be"),
        ([np.zeros((6, 6)), np.zeros((6, 2)), np.zeros((3, 2))], "C must be"),
        ([np.zeros((6, 6)), np.zeros((6, 2)), np.eye(2),
          np.zeros((5, 2))], "Xsol must be"),
    ]
    for matrices, message in cases:
        with pytest.raises(Exception, match=message):
            ProcrustesProblem((6, 6, 2, 2), matrices=matrices)


def test_too_few_user_matrices_raises():
    with pytest.raises(Exception, match="matrices must be a list or tuple"):
        ProcrustesProblem((6, 6, 2, 2),
                          matrices=[np.zeros((6, 6)), np.zeros((6, 2))])


def test_seeded_construction_is_reproducible():
    p1 = _seeded_problem((10, 10, 2, 2), problemnumber=1, seed=7)
    p2 = _seeded_problem((10, 10, 2, 2), problemnumber=1, seed=7)
    np.testing.assert_allclose(p1.A, p2.A)
    np.testing.assert_allclose(p1.B, p2.B)
    np.testing.assert_allclose(p1.Xsol, p2.Xsol)


# ---------------------------------------------------------------------------
# GBBSolver
# ---------------------------------------------------------------------------

def test_gbb_default_options_and_identity():
    solver = GBBSolver()
    assert isinstance(solver, ProcrustesSolver)
    assert solver.solvername == "gbb"
    expected = {
        "full_results": False, "xtol": 1e-6, "gtol": 1e-6, "ftol": 1e-12,
        "zeta": 1e-4, "kappa": 0.1, "eta": 0.85, "etavar": False,
        "tau": 1e-3, "projected_gradient": 1, "maxiter": 5000,
        "verbose": 1, "filename": None, "timer": False,
    }
    assert set(solver.options.keys()) == set(expected.keys())
    for key, value in expected.items():
        assert solver.options[key] == value


def test_gbb_user_options_are_kept():
    solver = GBBSolver(xtol=1e-5, maxiter=10, verbose=0)
    assert solver.options["xtol"] == 1e-5
    assert solver.options["maxiter"] == 10
    assert solver.options["verbose"] == 0


def test_gbb_invalid_options_raise():
    bad_options = [
        {"xtol": 1}, {"gtol": 1}, {"ftol": "x"}, {"zeta": 1},
        {"kappa": 1}, {"eta": 1}, {"etavar": 1}, {"tau": 1},
        {"projected_gradient": 3}, {"maxiter": 1.5}, {"verbose": 2},
        {"full_results": 1}, {"filename": 3}, {"timer": 1},
    ]
    for kwargs in bad_options:
        with pytest.raises(Exception):
            GBBSolver(**kwargs)


def test_gbb_solve_result_structure_small_p():
    # p < n/2, so the low-rank (projected gradient) update path is used.
    problem = _seeded_problem((10, 10, 2, 2), problemnumber=1, seed=1)
    np.random.seed(2)
    result = GBBSolver(maxiter=200, verbose=0).solve(problem)
    assert isinstance(result, OptimizeResult)
    for key in ("success", "status", "message", "solution", "fun",
                "error", "cpu", "nbiter", "nfev"):
        assert key in result.keys()
    assert result.solution.shape == (10, 2)
    assert result.nbiter <= 200
    _assert_orthonormal(result.solution, 2)


def test_gbb_solve_feasible_iterate_large_p():
    # p >= n/2, so the full (I + tau H) update path is used.
    problem = _seeded_problem((10, 10, 5, 5), problemnumber=1, seed=3)
    np.random.seed(4)
    result = GBBSolver(maxiter=100, verbose=0).solve(problem)
    assert result.solution.shape == (10, 5)
    _assert_orthonormal(result.solution, 5)


def test_gbb_solve_is_reproducible_with_seed():
    problem = _seeded_problem((10, 10, 2, 2), problemnumber=1, seed=5)
    np.random.seed(11)
    r1 = GBBSolver(maxiter=30, verbose=0).solve(problem)
    np.random.seed(11)
    r2 = GBBSolver(maxiter=30, verbose=0).solve(problem)
    np.testing.assert_allclose(r1.solution, r2.solution)


def test_gbb_error_is_inf_without_known_solution():
    A, B, C, _ = _user_data()
    problem = ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C])
    np.random.seed(0)
    result = GBBSolver(maxiter=50, verbose=0).solve(problem)
    assert result.error == np.inf


def test_gbb_error_is_finite_with_known_solution():
    problem = _seeded_problem((10, 10, 2, 2), problemnumber=1, seed=6)
    np.random.seed(0)
    result = GBBSolver(maxiter=100, verbose=0).solve(problem)
    assert np.isfinite(result.error)
    assert result.error >= 0.0


def test_gbb_maxiter_gives_failure_status():
    problem = _seeded_problem((10, 10, 2, 2), problemnumber=1, seed=8)
    np.random.seed(0)
    result = GBBSolver(maxiter=1, verbose=0).solve(problem)
    assert result.status == 1
    assert result.success is False
    assert "Maximum number" in result.message


def test_gbb_writes_output_file(tmp_path):
    problem = _seeded_problem((10, 10, 2, 2), problemnumber=1, seed=9)
    log = tmp_path / "gbb.log"
    np.random.seed(0)
    GBBSolver(filename=str(log), verbose=1, maxiter=3).solve(problem)
    assert log.exists()
    assert "Options:" in log.read_text()


def test_gbb_unwritable_filename_exits(tmp_path):
    bad_name = str(tmp_path / "missing_dir" / "gbb.log")
    solver = GBBSolver(filename=bad_name, verbose=0)
    problem = _seeded_problem((10, 10, 2, 2), problemnumber=1, seed=10)
    with pytest.raises(SystemExit):
        solver.solve(problem)
