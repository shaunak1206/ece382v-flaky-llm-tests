import numpy as np
import pytest
import os
import sys
from skprocrustes.skprocrustes import ProcrustesProblem, GBBSolver, OptimizeResult

# For some tests we will create small sized problems to keep test time reasonable

def test_procrustesproblem_init_and_attributes():
    # Valid problem: m=n=4 divisible by p=q=2
    sizes = (4,4,2,2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    assert problem.sizes == sizes
    assert problem.A.shape == (4,4)
    assert problem.B.shape == (4,2)
    assert problem.C.shape == (2,2)
    assert problem.Xsol.shape == (4,2)

def test_procrustesproblem_invalid_sizes_not_square():
    # Non-square m != n
    with pytest.raises(Exception, match="Currently only square problems are supported."):
        ProcrustesProblem((3,4,2,2))

def test_procrustesproblem_invalid_sizes_not_divisible():
    # m not divisible by p
    with pytest.raises(Exception, match="Dimensions are incompatible: m must be divisible by p."):
        ProcrustesProblem((5,5,3,3), problemnumber=1)

def test_procrustesproblem_matrices_user_defined_valid():
    m,n,p,q = 4,4,2,2
    A = np.eye(m,n)
    B = np.ones((m,q))
    C = np.eye(p,q)
    X = np.eye(n,p)
    problem = ProcrustesProblem((m,n,p,q), matrices=[A,B,C,X])
    np.testing.assert_array_equal(problem.A, A)
    np.testing.assert_array_equal(problem.B, B)
    np.testing.assert_array_equal(problem.C, C)
    np.testing.assert_array_equal(problem.Xsol, X)

def test_procrustesproblem_matrices_user_defined_invalid_shapes():
    m,n,p,q = 4,4,2,2
    A = np.eye(m,n)
    B = np.ones((m,q))
    C = np.eye(p,q)
    X = np.eye(n,p)
    # Wrong shape for A
    with pytest.raises(Exception, match="A must be \\(m,n\\)"):
        ProcrustesProblem((m,n,p,q), matrices=[np.eye(3), B, C, X])
    # Wrong shape for B
    with pytest.raises(Exception, match="B must be \\(m,q\\)"):
        ProcrustesProblem((m,n,p,q), matrices=[A, np.eye(3), C, X])
    # Wrong shape for C
    with pytest.raises(Exception, match="C must be \\(p,q\\)"):
        ProcrustesProblem((m,n,p,q), matrices=[A, B, np.eye(3), X])
    # Wrong shape for X
    with pytest.raises(Exception, match="Xsol must be \\(n,p\\)"):
        ProcrustesProblem((m,n,p,q), matrices=[A, B, C, np.eye(3)])

def test_procrustesproblem_matrices_less_than_three_raises():
    with pytest.raises(Exception, match="matrices must be a list or tuple containing matrices A, B and C"):
        ProcrustesProblem((4,4,2,2), matrices=[np.eye(4), np.eye(4)])

def test_gbbs_overall_solve_output_attrs():
    sizes = (6,6,3,3)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    solver = GBBSolver(full_results=False)
    result = solver.solve(problem)
    assert isinstance(result, OptimizeResult)
    # Check keys
    assert hasattr(result, "success")
    assert hasattr(result, "status")
    assert hasattr(result, "message")
    assert hasattr(result, "solution")
    assert hasattr(result, "fun")
    assert hasattr(result, "error")
    assert hasattr(result, "cpu")
    assert hasattr(result, "nbiter")
    assert hasattr(result, "nfev")
    # total_fun and total_crit should be np.inf because full_results=False
    assert result.total_fun == np.inf
    assert result.total_crit == np.inf

def test_gbbsolver_setoptions_defaults_and_custom():
    # default options
    solver = GBBSolver()
    opts = solver.options
    # Check presence and defaults
    assert opts["full_results"] is False
    assert isinstance(opts["xtol"], float)
    assert isinstance(opts["gtol"], float)
    assert opts["projected_gradient"] in (1, 2)
    # test setting valid options
    solver = GBBSolver(full_results=True, xtol=1e-5, gtol=1e-5,
                       ftol=1e-10, zeta=0.5, kappa=0.2, eta=0.9,
                       etavar=True, tau=1e-2, projected_gradient=2,
                       maxiter=10, verbose=0, filename=None, timer=False)
    opts = solver.options
    assert opts["full_results"] is True
    assert opts["xtol"] == 1e-5
    assert opts["gtol"] == 1e-5
    assert opts["ftol"] == 1e-10
    assert opts["projected_gradient"] == 2
    assert opts["verbose"] == 0

def test_gbbsolver_setoptions_invalid_values():
    with pytest.raises(Exception, match="full_results must be a boolean"):
        GBBSolver(full_results="nope")
    with pytest.raises(Exception, match="xtol must be a float"):
        GBBSolver(xtol="bad")
    with pytest.raises(Exception, match="gtol must be a float"):
        GBBSolver(gtol="bad")
    with pytest.raises(Exception, match="ftol must be a float"):
        GBBSolver(ftol="bad")
    with pytest.raises(Exception, match="kappa must be a float"):
        GBBSolver(kappa="bad")
    with pytest.raises(Exception, match="eta must be a float"):
        GBBSolver(eta="bad")
    with pytest.raises(Exception, match="etavar must be a boolean"):
        GBBSolver(etavar="bad")
    with pytest.raises(Exception, match="projected_gradient must be 1 or 2"):
        GBBSolver(projected_gradient=3)
    with pytest.raises(Exception, match="verbose must be 0, or 1"):
        GBBSolver(verbose=10)
    with pytest.raises(Exception, match="filename must be string"):
        GBBSolver(filename=123)
    with pytest.raises(Exception, match="timer must be boolean"):
        GBBSolver(timer="yes")

def test_gbbsolver_open_close_file_stdout(monkeypatch):
    solver = GBBSolver(filename=None)
    solver.open_file()
    assert solver.file == sys.stdout
    # Test close_file on stdout does nothing bad
    solver.close_file()

def test_gbbsolver_open_file_invalid_filename(monkeypatch):
    import builtins
    solver = GBBSolver(filename="///bad_filename///")
    # monkeypatch sys.exit to raise instead for test
    called = {}
    def fake_exit(code):
        called['code'] = code
        raise SystemExit(code)
    monkeypatch.setattr(sys, "exit", fake_exit)
    # monkeypatch print to capture output
    output = {}
    def fake_print(*args, **kwargs):
        output['text'] = args[0] if args else ""
    monkeypatch.setattr(builtins, "print", fake_print)
    with pytest.raises(SystemExit):
        solver.open_file()
    assert "not a valid filename" in output.get('text', '')

def test_gbbsolver_solve_runs_and_returns(monkeypatch):
    # Patch gbb_solver to simple dummy function returning controlled output
    from skprocrustes import skprocrustes

    def dummy_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0):
        return 0, 1.23, X, 0.01, 5, "Success message"

    monkeypatch.setattr(sys.modules['skprocrustes.skprocrustes'], "gbb_solver", dummy_gbb_solver)
    sizes = (4,4,2,2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    solver = GBBSolver(full_results=False)
    result = solver.solve(problem)
    # Check result properties
    assert result.success
    assert result.status == 0
    assert isinstance(result.solution, np.ndarray)
    assert np.allclose(result.fun, 1.23)
    assert np.allclose(result.error, np.inf)  # because problem.Xsol changed randomly, very unlikely to be close
    assert result.nbiter == problem.stats["nbiter"]
    # nfev and others should be present
    assert hasattr(result, "nfev")

def test_gbbsolver_solve_with_full_results(monkeypatch):
    # Patch gbb_solver again for full_results=True
    def dummy_gbb_solver(problem, m, n, X, A, B, options, inner, fileobj, B1=None, blobopprod=0):
        problem.stats["total_fun"] = [0.1,0.05]
        problem.stats["total_crit"] = [0.01,0.005]
        return 0, 0.05, X, 0.001, 5, "Success message"

    monkeypatch.setattr(sys.modules['skprocrustes.skprocrustes'], "gbb_solver", dummy_gbb_solver)
    sizes = (4,4,2,2)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    problem.stats["nbiter"] = 2
    problem.stats["fev"] = 3
    problem.stats["total_fun"] = [0.1,0.05]
    problem.stats["total_crit"] = [0.01,0.005]
    solver = GBBSolver(full_results=True)
    result = solver.solve(problem)
    # total_fun and total_crit must be lists from problem.stats
    assert isinstance(result.total_fun, list)
    assert isinstance(result.total_crit, list)
    assert len(result.total_fun) == 2
    assert len(result.total_crit) == 2

def test_optimize_result_dict_attr_access():
    # Initialize with some dict
    d = {"success": True, "value": 10}
    result = OptimizeResult(d)
    assert result.success is True
    assert result["success"] is True
    result.new_attr = 5
    assert result["new_attr"] == 5
    del result.new_attr
    with pytest.raises(AttributeError):
        _ = result.new_attr

def test_optimize_result_repr_and_dir():
    d = {"a":1, "longkey":"text"}
    result = OptimizeResult(d)
    s = repr(result)
    assert "a" in s
    assert "longkey" in s
    dirlist = dir(result)
    assert "a" in dirlist
    assert "longkey" in dirlist

def test_procrustesproblem_problemnumber_3_nvalues(monkeypatch):
    # problemnumber=3 cases require n in {50,95,500,1000}
    for n in [50,95,500,1000]:
        sizes = (n,n, n//10, n//10)
        # monkeypatch np.random.rand to always return 0.5 for reproducibility
        monkeypatch.setattr("numpy.random.rand", lambda *a, **k: 0.5*np.ones(a))
        problem = ProcrustesProblem(sizes, problemnumber=3)
        assert problem.A.shape == (n,n)
    # Invalid n triggers exception
    with pytest.raises(Exception, match="Problem 6 requires n = 50,"):
        ProcrustesProblem((10,10,5,5), problemnumber=3)

def test_procrustesproblem_random_xsol_shape(monkeypatch):
    # Check that Xsol is always shape (n,p)
    sizes = (6,6,3,3)
    problem = ProcrustesProblem(sizes, problemnumber=1)
    assert problem.Xsol.shape == (6,3)

def test_optimize_result_show(capsys):
    res = OptimizeResult({"success": True, "message": "done", "solution": np.arange(4)})
    res.show()
    captured = capsys.readouterr()
    assert "success" in captured.out
    assert "done" in captured.out
    assert "solution" not in captured.out  # solution excluded from print output
