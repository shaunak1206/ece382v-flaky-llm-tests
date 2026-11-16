import numpy as np
import pytest

from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, ProcrustesSolver


# ---------------------------------------------------------------------------
# ProcrustesProblem
# ---------------------------------------------------------------------------

def test_problem_rejects_nonsquare_sizes():
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((10, 12, 2, 2), problemnumber=1)
    with pytest.raises(Exception, match="square"):
        ProcrustesProblem((10, 10, 2, 3), problemnumber=1)


def test_problem_rejects_m_not_divisible_by_p():
    with pytest.raises(Exception, match="divisible"):
        ProcrustesProblem((10, 10, 3, 3), problemnumber=1)


def test_problem_stores_sizes_problemnumber_and_empty_stats():
    problem = ProcrustesProblem((12, 12, 3, 3), problemnumber=1)
    assert problem.sizes == (12, 12, 3, 3)
    assert problem.problemnumber == 1
    assert problem.stats == {}


def test_problem1_shapes_and_structure():
    np.random.seed(0)
    problem = ProcrustesProblem((12, 12, 3, 3), problemnumber=1)
    assert problem.A.shape == (12, 12)
    assert problem.B.shape == (12, 3)
    assert problem.C.shape == (3, 3)
    assert problem.Xsol.shape == (12, 3)
    np.testing.assert_array_equal(problem.C, np.eye(3, 3))
    # Xsol is a row permutation of eye(n, p), so it has orthonormal columns.
    np.testing.assert_allclose(problem.Xsol.T @ problem.Xsol, np.eye(3))


def test_problem_predefined_B_equals_A_Xsol_C():
    np.random.seed(3)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    np.testing.assert_allclose(problem.B,
                               problem.A @ problem.Xsol @ problem.C,
                               atol=1e-10)


def test_problem1_singular_values_in_range():
    np.random.seed(0)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    sv = np.linalg.svd(problem.A, compute_uv=False)
    assert sv.min() >= 10.0 - 1e-8
    assert sv.max() <= 12.0 + 1e-8


def test_problem2_builds_finite_matrices():
    np.random.seed(0)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=2)
    assert problem.A.shape == (10, 10)
    assert problem.B.shape == (10, 2)
    assert np.all(np.isfinite(problem.A))
    assert np.all(np.isfinite(problem.B))


def test_problem3_singular_value_structure():
    np.random.seed(0)
    problem = ProcrustesProblem((50, 50, 2, 2), problemnumber=3)
    sv_asc = np.sort(np.linalg.svd(problem.A, compute_uv=False))
    # Eight tiny singular values (r/1000) come first, then values >= 2.
    assert np.all(sv_asc[:8] < 1e-3)
    assert sv_asc[8] >= 2.0 - 1e-8
    assert sv_asc[-1] <= 12.0 + 1e-8


def test_problem3_rejects_unsupported_n():
    with pytest.raises(Exception, match="requires n"):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=3)


def test_problem_without_matrices_or_valid_number_raises():
    with pytest.raises(Exception, match="no problem number"):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=None)
    with pytest.raises(Exception, match="no problem number"):
        ProcrustesProblem((10, 10, 2, 2), problemnumber=7)


def test_problem_seeded_generation_is_reproducible():
    np.random.seed(42)
    p1 = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    np.random.seed(42)
    p2 = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    np.testing.assert_array_equal(p1.A, p2.A)
    np.testing.assert_array_equal(p1.B, p2.B)
    np.testing.assert_array_equal(p1.Xsol, p2.Xsol)


def test_problem_user_matrices_stored_without_xsol():
    rng = np.random.RandomState(0)
    A = rng.randn(4, 4)
    B = rng.randn(4, 2)
    C = np.eye(2)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C])
    assert problem.A is A
    assert problem.B is B
    assert problem.C is C
    assert "Xsol" not in problem.__dict__


def test_problem_user_matrices_tuple_with_known_solution():
    rng = np.random.RandomState(1)
    A = rng.randn(4, 4)
    B = rng.randn(4, 2)
    C = np.eye(2)
    X = rng.randn(4, 2)
    problem = ProcrustesProblem((4, 4, 2, 2), matrices=(A, B, C, X))
    assert problem.Xsol is X


def test_problem_user_matrices_wrong_shapes_raise():
    rng = np.random.RandomState(2)
    A = rng.randn(4, 4)
    B = rng.randn(4, 2)
    C = np.eye(2)
    X = rng.randn(4, 2)
    with pytest.raises(Exception, match="A must be"):
        ProcrustesProblem((4, 4, 2, 2), matrices=[rng.randn(4, 3), B, C])
    with pytest.raises(Exception, match="B must be"):
        ProcrustesProblem((4, 4, 2, 2), matrices=[A, rng.randn(4, 3), C])
    with pytest.raises(Exception, match="C must be"):
        ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, rng.randn(3, 2)])
    with pytest.raises(Exception, match="Xsol must be"):
        ProcrustesProblem((4, 4, 2, 2), matrices=[A, B, C, rng.randn(3, 2)])


def test_problem_user_matrices_too_few_raises():
    rng = np.random.RandomState(3)
    with pytest.raises(Exception, match="must be a list or tuple"):
        ProcrustesProblem((4, 4, 2, 2),
                          matrices=[rng.randn(4, 4), rng.randn(4, 2)])


# ---------------------------------------------------------------------------
# GBBSolver
# ---------------------------------------------------------------------------

def test_gbb_default_options_and_solvername():
    solver = GBBSolver()
    assert isinstance(solver, ProcrustesSolver)
    assert solver.solvername == "gbb"
    expected = {
        "full_results": False,
        "xtol": 1e-6,
        "gtol": 1e-6,
        "ftol": 1e-12,
        "zeta": 1e-4,
        "kappa": 0.1,
        "eta": 0.85,
        "etavar": False,
        "tau": 1e-3,
        "projected_gradient": 1,
        "maxiter": 5000,
        "verbose": 1,
        "filename": None,
        "timer": False,
    }
    for key, value in expected.items():
        assert solver.options[key] == value, key


def test_gbb_invalid_option_types_raise():
    cases = [
        ({"xtol": 1}, "xtol must be a float"),
        ({"gtol": 1}, "gtol must be a float"),
        ({"ftol": 1}, "ftol must be a float"),
        ({"zeta": 1}, "zeta must be a float"),
        ({"kappa": 1}, "kappa must be a float"),
        ({"eta": 1}, "eta must be a float"),
        ({"tau": 1}, "tau must be a float"),
        ({"maxiter": 1.0}, "maxiter must be an integer"),
        ({"full_results": 1}, "full_results must be a boolean"),
        ({"etavar": 1}, "etavar must be a boolean"),
        ({"timer": 1}, "timer must be boolean"),
        ({"filename": 3}, "filename must be string"),
    ]
    for kwargs, message in cases:
        with pytest.raises(Exception, match=message):
            GBBSolver(**kwargs)


def test_gbb_invalid_option_values_raise():
    with pytest.raises(Exception, match="verbose must be 0"):
        GBBSolver(verbose=2)
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        GBBSolver(projected_gradient=3)


def test_gbb_solve_returns_feasible_solution():
    np.random.seed(1)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    solver = GBBSolver(verbose=0, maxiter=300)
    result = solver.solve(problem)

    X = result["solution"]
    assert X.shape == (10, 2)
    # Curvilinear search keeps iterates on the Stiefel manifold.
    np.testing.assert_allclose(X.T @ X, np.eye(2), atol=1e-6)
    assert result["status"] in (0, 1)
    assert np.isfinite(result["fun"])
    assert 0 <= result["nbiter"] <= 300


def test_gbb_solve_error_against_known_solution():
    np.random.seed(2)
    problem = ProcrustesProblem((10, 10, 2, 2), problemnumber=1)
    result = GBBSolver(verbose=0, maxiter=300).solve(problem)

    assert np.isfinite(result["error"])
    assert result["error"] >= 0.0


def test_gbb_solve_error_is_inf_without_known_solution():
    rng = np.random.RandomState(4)
    A = rng.randn(6, 6)
    B = rng.randn(6, 2)
    C = np.eye(2)
    problem = ProcrustesProblem((6, 6, 2, 2), matrices=[A, B, C])
    result = GBBSolver(verbose=0, maxiter=50).solve(problem)

    assert result["error"] == np.inf
    assert result["solution"].shape == (6, 2)


def test_gbb_solve_writes_output_to_file(tmp_path):
    np.random.seed(5)
    problem = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)
    out = tmp_path / "gbb_log.txt"
    solver = GBBSolver(verbose=1, maxiter=5, filename=str(out))
    solver.solve(problem)

    assert out.exists()
    content = out.read_text()
    assert "Options:" in content


@pytest.mark.xfail(
    strict=True,
    reason="gbb_solver stores 'total_grad' but GBBSolver.solve checks "
           "for 'total_crit', so full_results=True always raises.")
def test_gbb_solve_full_results_does_not_raise():
    np.random.seed(6)
    problem = ProcrustesProblem((8, 8, 2, 2), problemnumber=1)
    solver = GBBSolver(full_results=True, verbose=0, maxiter=20)
    result = solver.solve(problem)
    assert len(result["total_fun"]) > 0
