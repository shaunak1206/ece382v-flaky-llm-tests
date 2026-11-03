import math
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from jmcm import JointModel  # noqa: E402
from jmcm.core import ReadData  # noqa: E402

FORMULA = "y|id|t~x1+x2|x3"


def _make_df(n_subjects=8, n_times=4, noise=0.2, seed=0):
    """Balanced longitudinal data: y = 1 + 2t + 0.5*x1 - 1.0*x2 + noise."""
    rng = np.random.RandomState(seed)
    rows = []
    for sid in range(1, n_subjects + 1):
        t = np.linspace(0.0, 1.0, n_times)
        x1 = rng.normal(size=n_times)
        x2 = rng.normal(size=n_times)
        x3 = rng.normal(size=n_times)
        y = 1.0 + 2.0 * t + 0.5 * x1 - 1.0 * x2 + rng.normal(scale=noise, size=n_times)
        for k in range(n_times):
            rows.append((y[k], sid, t[k], x1[k], x2[k], x3[k]))
    return pd.DataFrame(rows, columns=["y", "id", "t", "x1", "x2", "x3"])


def _patsy_only_model(df):
    """A bare instance holding only the ReadData state (no polynomial columns)."""
    model = object.__new__(JointModel)
    ReadData.__init__(model, df, FORMULA, ())
    return model


@pytest.fixture(autouse=True)
def _isolate_warning_filters():
    # ModelFit._get_step switches warnings to errors globally; keep that local.
    with warnings.catch_warnings():
        yield


def test_init_stores_formula_poly_orders_and_optim():
    model = JointModel(_make_df(), FORMULA, (1, 1, 1))
    assert isinstance(model, JointModel)
    assert model.formula == FORMULA
    assert model.poly_orders == (1, 1, 1)
    assert model.optim_meth == "default"


def test_parameter_vector_sizes_follow_poly_orders():
    model = JointModel(_make_df(), FORMULA, (2, 1, 0))
    # mean: 3 polynomial columns + 2 covariates; innovation: 2 + 1; gamma: 1
    assert model._num_bta == 5
    assert model._num_lmd == 3
    assert model._num_gma == 1
    assert model._bta.shape == (5,)
    assert model._lmd.shape == (3,)
    assert model._gma.shape == (1,)


def test_counts_subjects_and_measurements():
    model = JointModel(_make_df(n_subjects=8, n_times=4), FORMULA, (1, 0, 0))
    assert model.m == 8
    assert list(model.vec_n) == [4] * 8
    assert len(model.vec_y) == 32
    assert model._num_theta == model._num_bta + model._num_lmd + model._num_gma


def test_unsorted_input_is_sorted_by_id_then_time():
    df = _make_df()
    shuffled = df.sample(frac=1.0, random_state=3)
    model = JointModel(shuffled, FORMULA, (1, 0, 0))
    expected = df.sort_values(["id", "t"])
    assert list(model.df.index) == list(expected.index)
    assert np.allclose(model.vec_y, expected["y"].values)
    assert np.allclose(model.vec_t, expected["t"].values)


def test_default_fit_gives_finite_estimates():
    model = JointModel(_make_df(), FORMULA, (1, 0, 0))
    assert np.isfinite(model.max_log_lik)
    assert np.isfinite(model.bic)
    assert np.all(np.isfinite(model._theta))


def test_bic_matches_closed_form():
    model = JointModel(_make_df(), FORMULA, (1, 0, 0))
    expected = (model._num_theta * math.log(model.m)
                - 2 * model.max_log_lik) / model.m
    assert model.bic == pytest.approx(expected, rel=1e-9)


def test_bfgs_fit_gives_finite_estimates():
    model = JointModel(_make_df(), FORMULA, (1, 0, 0), optim_meth="BFGS")
    assert model.optim_meth == "BFGS"
    assert np.isfinite(model.bic)
    assert np.all(np.isfinite(model._theta))
    assert model._bta.shape == (4,)


def test_unknown_optim_meth_raises_name_error():
    with pytest.raises(NameError):
        JointModel(_make_df(), FORMULA, (1, 0, 0), optim_meth="newton")


def test_short_poly_tuple_triggers_model_selection():
    model = JointModel(_make_df(n_times=4), FORMULA, poly_orders=(1, 1))
    assert isinstance(model.poly_orders, tuple)
    assert len(model.poly_orders) == 3
    n_max = int(np.max(model.vec_n))
    assert all(isinstance(v, int) and 0 <= v <= n_max - 1
               for v in model.poly_orders)
    assert model._num_bta == model.poly_orders[0] + 1 + 2


def test_empty_poly_orders_triggers_model_selection():
    model = JointModel(_make_df(n_times=4), FORMULA, poly_orders=())
    assert isinstance(model.poly_orders, tuple)
    assert len(model.poly_orders) == 3
    n_max = int(np.max(model.vec_n))
    assert all(0 <= v <= n_max - 1 for v in model.poly_orders)
    assert model._num_gma == model.poly_orders[2] + 1


def test_model_select_flag_keeps_orders_within_bounds(capsys):
    model = JointModel(_make_df(), FORMULA, (2, 1, 1), model_select=True)
    p, d, q = model.poly_orders
    assert 0 <= p <= 2
    assert 0 <= d <= 1
    assert 0 <= q <= 1
    # the fitted design must agree with whichever orders were chosen
    assert model._num_bta == p + 1 + 2
    assert model._num_lmd == d + 1 + 1
    assert model._num_gma == q + 1


def test_model_select_false_keeps_given_orders():
    model = JointModel(_make_df(), FORMULA, (2, 1, 0))
    assert model.poly_orders == (2, 1, 0)


def test_new_design_block_shapes():
    model = _patsy_only_model(_make_df(n_subjects=5, n_times=4))
    mat_X, mat_Z, mat_W = model._new_design((2, 1, 0))
    n_rows = 5 * 4
    assert mat_X.shape == (n_rows, 3 + 2)
    assert mat_Z.shape == (n_rows, 2 + 1)
    # one W row per within-subject pair: 5 subjects * C(4, 2) = 30
    assert mat_W.shape == (5 * 6, 1)


def test_new_design_mat_w_has_unit_first_column():
    model = _patsy_only_model(_make_df(n_subjects=5, n_times=4))
    _, _, mat_W = model._new_design((0, 0, 2))
    assert mat_W.shape == (30, 3)
    assert np.allclose(mat_W[:, 0], 1.0)


def test_malformed_formula_raises_value_error():
    with pytest.raises(ValueError):
        JointModel(_make_df(), "y~x1+x2", poly_orders=(1, 0, 0))


def test_missing_column_raises_key_error():
    with pytest.raises(KeyError):
        JointModel(_make_df(), "y|nope|t~x1+x2|x3", poly_orders=(1, 0, 0))


def test_summary_prints_all_sections(capsys):
    model = JointModel(_make_df(), FORMULA, (1, 0, 0))
    model.summary()
    out = capsys.readouterr().out
    for label in ["Model:", "Mean Parameters:",
                  "Innovation Variance Parameters:",
                  "Autoregressive Parameters:",
                  "Maximum of log-likelihood:", "BIC:"]:
        assert label in out


def test_wald_test_prints_one_row_per_parameter(capsys):
    model = JointModel(_make_df(), FORMULA, (1, 0, 0))
    model.wald_test()
    out = capsys.readouterr().out
    assert "Wald Test" in out
    assert out.count("beta") == model._num_bta
    assert out.count("lambda") == model._num_lmd
    assert out.count("gamma") == model._num_gma


def test_bootstrap_data_has_resampled_shapes():
    model = JointModel(_make_df(n_subjects=8, n_times=4), FORMULA, (1, 0, 0))
    np.random.seed(0)
    x_b, z_b, w_b, y_b, n_b = model._bootstrap_data()
    m, n = model.m, 4
    assert x_b.shape == (m * n, model._num_bta)
    assert z_b.shape == (m * n, model._num_lmd)
    assert w_b.shape == (m * n * (n - 1) // 2, model._num_gma)
    assert y_b.shape == (m * n,)
    assert n_b.shape == (m,)
    assert np.all(n_b == n)


def test_bootstrap_rows_are_copies_of_original_subjects():
    model = JointModel(_make_df(n_subjects=8, n_times=4), FORMULA, (1, 0, 0))
    np.random.seed(0)
    _, _, _, y_b, _ = model._bootstrap_data()
    original = model.vec_y.reshape(model.m, 4)
    resampled = y_b.reshape(model.m, 4)
    for row in resampled:
        assert any(np.array_equal(row, orig) for orig in original)


def test_boot_curve_draws_three_panels(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    model = JointModel(_make_df(), FORMULA, (1, 0, 0))
    np.random.seed(1)
    model.boot_curve(2)
    fig = plt.gcf()
    try:
        assert len(fig.axes) == 3
    finally:
        plt.close("all")


def test_mean_coefficients_recovered_on_linear_data():
    df = _make_df(n_subjects=10, n_times=5, noise=0.2, seed=1)
    model = JointModel(df, FORMULA, (1, 0, 0))
    intercept, slope_t, beta_x1, beta_x2 = model._bta
    assert abs(intercept - 1.0) < 0.35
    assert abs(slope_t - 2.0) < 0.35
    assert abs(beta_x1 - 0.5) < 0.35
    assert abs(beta_x2 + 1.0) < 0.35


def test_single_measurement_subject_fits():
    base = _make_df(n_subjects=6, n_times=4)
    extra = pd.DataFrame([{"y": 0.5, "id": 99, "t": 0.0,
                           "x1": 0.1, "x2": -0.2, "x3": 0.3}])
    df = pd.concat([base, extra], ignore_index=True)
    model = JointModel(df, FORMULA, (1, 0, 0))
    assert model.m == 7
    assert list(model.vec_n) == [4, 4, 4, 4, 4, 4, 1]
    assert np.isfinite(model.bic)
