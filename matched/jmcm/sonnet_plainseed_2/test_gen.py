import math
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import jmcm  # noqa: E402
from jmcm import JointModel  # noqa: E402
import jmcm.joint_model as joint_model_module  # noqa: E402


FORMULA = "y|id|t~x1|x1"


def make_df(m=12, times=(0.0, 1.0, 2.0, 3.0), seed=0, id_offset=0):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(m):
        subject_effect = rng.normal(0, 0.3)
        for t in times:
            x1 = rng.normal(0, 1)
            y = 1.0 + 0.5 * t + 0.8 * x1 + subject_effect + rng.normal(0, 0.3)
            rows.append((y, i + id_offset, t, x1))
    return pd.DataFrame(rows, columns=["y", "id", "t", "x1"])


def fit(df, formula=FORMULA, poly_orders=(1, 1, 1), **kwargs):
    # the library turns warnings into errors globally, so isolate each fit
    with warnings.catch_warnings():
        return JointModel(df, formula, poly_orders, **kwargs)


@pytest.fixture(scope="module")
def df():
    return make_df()


@pytest.fixture(scope="module")
def model(df):
    return fit(df)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_package_exports_joint_model():
    assert jmcm.name == "jmcm"
    assert jmcm.JointModel is JointModel


def test_constructor_stores_arguments(model):
    assert model.formula == FORMULA
    assert model.poly_orders == (1, 1, 1)
    assert model.optim_meth == "default"


def test_design_matrix_shapes(model, df):
    n_rows = len(df)
    assert model.mat_X.shape == (n_rows, 3)  # intercept, t, x1
    assert model.mat_Z.shape == (n_rows, 3)
    # 4 measurements per subject -> 6 pairs per subject, 2 polynomial columns
    assert model.mat_W.shape == (12 * 6, 2)
    assert len(model.vec_y) == n_rows
    assert len(model.vec_t) == n_rows


def test_subject_counts(model):
    assert model.m == 12
    assert list(model.vec_n) == [4] * 12


def test_polynomial_columns_of_mean_design(model):
    assert np.allclose(model.mat_X[:, 0], 1.0)
    assert np.allclose(model.mat_X[:, 1], model.vec_t)
    assert np.allclose(model.mat_Z[:, 0], 1.0)
    assert np.allclose(model.mat_Z[:, 1], model.vec_t)
    # W holds powers of within-subject time lags, first column is all ones
    assert np.allclose(model.mat_W[:, 0], 1.0)
    assert np.all(model.mat_W[:, 1] > 0)


def test_estimated_parameters_have_expected_sizes_and_are_finite(model):
    assert model._bta.shape == (3,)
    assert model._lmd.shape == (3,)
    assert model._gma.shape == (2,)
    assert np.all(np.isfinite(model._theta))
    assert np.shares_memory(model._theta, model._bta)


def test_mean_parameters_recover_true_signal(model):
    # data generated with intercept 1, time slope 0.5 and covariate effect 0.8
    assert abs(model._bta[1] - 0.5) < 0.4
    assert abs(model._bta[2] - 0.8) < 0.4


def test_bic_matches_definition(model):
    assert math.isfinite(model.max_log_lik)
    assert math.isfinite(model.bic)
    expected = (len(model._theta) * math.log(model.m) - 2 * model.max_log_lik) / model.m
    assert model.bic == pytest.approx(expected)


def test_max_log_lik_is_half_negative_objective(model):
    value = model._obj_fun(model._theta)
    assert model.max_log_lik == pytest.approx(-value / 2)


def test_data_is_sorted_by_id_and_time_regardless_of_input_order():
    base = make_df(m=8, seed=3, id_offset=100)
    shuffled = base.sample(frac=1.0, random_state=1).reset_index(drop=True)
    m_sorted = fit(base)
    m_shuffled = fit(shuffled)
    expected = base.sort_values(by=["id", "t"])
    assert np.allclose(m_shuffled.vec_y, expected["y"].values)
    assert np.allclose(m_shuffled.vec_y, m_sorted.vec_y)
    assert np.allclose(m_shuffled._bta, m_sorted._bta, atol=1e-6)


def test_unbalanced_subjects_are_supported():
    base = make_df(m=10, seed=5)
    # remove one measurement from two of the subjects
    drop = base[(base["id"] == 2) & (base["t"] == 3.0)].index.tolist()
    drop += base[(base["id"] == 7) & (base["t"] == 0.0)].index.tolist()
    unbalanced = base.drop(index=drop)
    m = fit(unbalanced)
    assert m.m == 10
    assert list(m.vec_n) == [4, 4, 3, 4, 4, 4, 4, 3, 4, 4]
    assert m.mat_W.shape[0] == 8 * 6 + 2 * 3
    assert np.all(np.isfinite(m._theta))
    assert math.isfinite(m.bic)


def test_bfgs_optimiser_fits_model(df):
    m = fit(df, optim_meth="BFGS")
    assert m.optim_meth == "BFGS"
    assert np.all(np.isfinite(m._theta))
    assert math.isfinite(m.bic)
    assert m._bta.shape == (3,)


def test_invalid_optimiser_raises_name_error(df):
    with pytest.raises(NameError):
        fit(df, optim_meth="newton")


def test_summary_prints_estimates(model, capsys):
    model.summary()
    out = capsys.readouterr().out
    assert "Model:" in out
    assert FORMULA in out
    assert "Mean Parameters:" in out
    assert "Innovation Variance Parameters:" in out
    assert "Autoregressive Parameters:" in out
    assert "Maximum of log-likelihood:" in out
    assert "BIC:" in out


def test_wald_test_prints_table_for_every_parameter():
    m = fit(make_df(m=12, seed=7), poly_orders=(1, 1, 0))
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.wald_test()
    out = buf.getvalue()
    assert "Wald Test" in out
    assert "chi-square" in out
    assert "p-value" in out
    for name in ("beta0", "beta1", "beta2", "lambda0", "lambda1", "lambda2", "gamma0"):
        assert name in out
    assert "gamma1" not in out


def test_new_design_adds_polynomial_columns(model):
    mat_X, mat_Z, mat_W = model._new_design((2, 1, 1))
    n_rows = model.mat_X.shape[0]
    assert mat_X.shape == (n_rows, 3 + model.mat_X.shape[1])
    assert mat_Z.shape == (n_rows, 2 + model.mat_Z.shape[1])
    assert mat_W.shape == (model.mat_W.shape[0], 2)
    assert np.allclose(mat_X[:, 0], 1.0)
    assert np.allclose(mat_X[:, 2], model.vec_t ** 2)


def test_bootstrap_data_shapes_are_consistent(model):
    np.random.seed(123)
    x_boot, z_boot, w_boot, y_boot, n_boot = model._bootstrap_data()
    total = int(np.sum(n_boot))
    assert len(n_boot) == model.m
    assert x_boot.shape == (total, model.mat_X.shape[1])
    assert z_boot.shape == (total, model.mat_Z.shape[1])
    assert len(y_boot) == total
    assert w_boot.shape == (int(np.sum(n_boot * (n_boot - 1) // 2)), model.mat_W.shape[1])
    # every resampled subject size is one of the original sizes
    assert set(n_boot.tolist()) <= set(model.vec_n.tolist())


def test_boot_curve_draws_three_panels(model, monkeypatch):
    shown = []
    monkeypatch.setattr(joint_model_module.plt, "show", lambda *a, **k: shown.append(True))
    np.random.seed(2024)
    with warnings.catch_warnings():
        model.boot_curve(3)
    assert shown == [True]
    fig = plt.gcf()
    assert len(fig.axes) == 3
    labels = [ax.get_ylabel() for ax in fig.axes]
    assert labels == ["Response", "Log-Variance", "Autoregres. Coeffic."]
    # each panel shows lower band, estimate and upper band
    assert all(len(ax.lines) == 3 for ax in fig.axes)


def test_model_selection_with_given_maximum_orders(capsys):
    base = make_df(m=10, seed=11)
    m = fit(base, poly_orders=(1, 1, 1), model_select=True)
    out = capsys.readouterr().out
    assert "0 0 0" in out and "1 1 1" in out
    assert len(m.poly_orders) == 3
    assert all(isinstance(o, int) and 0 <= o <= 1 for o in m.poly_orders)
    p, d, q = m.poly_orders
    # polynomial columns plus the single covariate column
    assert m._num_bta == p + 2
    assert m._num_lmd == d + 2
    assert m._num_gma == q + 1
    assert math.isfinite(m.bic)


def test_automatic_model_selection_when_orders_omitted():
    base = make_df(m=15, times=(0.0, 1.0, 3.0), seed=21)
    m = fit(base, poly_orders=())
    assert isinstance(m.poly_orders, tuple)
    assert len(m.poly_orders) == 3
    # the maximal order is n_max - 1 = 2
    assert all(0 <= o <= 2 for o in m.poly_orders)
    assert m._num_gma == m.poly_orders[2] + 1
    assert np.all(np.isfinite(m._theta))
    assert math.isfinite(m.bic)
