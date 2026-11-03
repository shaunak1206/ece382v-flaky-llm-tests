import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jmcm
from jmcm import JointModel
from jmcm.core import BaseFunc, ModelFit, ReadData


def make_data(m=15, n=4, seed=0, with_x=False):
    """Balanced longitudinal data with jittered measurement times."""
    rng = np.random.RandomState(seed)
    rows = []
    for subject in range(1, m + 1):
        intercept = rng.normal(0.0, 0.5)
        times = np.arange(n) + rng.uniform(-0.2, 0.2, size=n)
        for t in times:
            row = {
                "id": subject,
                "t": float(t),
                "y": 1.0 + 2.0 * t + intercept + rng.normal(0.0, 0.3),
            }
            if with_x:
                row["x"] = rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def df():
    return make_data()


@pytest.fixture(scope="module")
def fitted(df):
    return JointModel(df, "y|id|t~1|1", (1, 1, 1))


def test_package_exports_joint_model():
    assert jmcm.name == "jmcm"
    assert jmcm.JointModel is JointModel


def test_joint_model_inherits_from_core_classes():
    assert issubclass(JointModel, ReadData)
    assert issubclass(JointModel, ModelFit)
    assert issubclass(JointModel, BaseFunc)


def test_constructor_stores_arguments(fitted):
    assert fitted.formula == "y|id|t~1|1"
    assert fitted.poly_orders == (1, 1, 1)
    assert fitted.optim_meth == "default"


def test_subject_counts_and_sizes(fitted):
    assert fitted.m == 15
    assert len(fitted.vec_n) == 15
    assert np.all(fitted.vec_n == 4)
    assert len(fitted.vec_y) == 60
    assert len(fitted.vec_t) == 60


def test_design_matrix_shapes(fitted):
    # polynomial orders (1, 1, 1) give two columns for each of beta, lambda, gamma
    assert fitted.mat_X.shape == (60, 2)
    assert fitted.mat_Z.shape == (60, 2)
    # each subject with 4 measurements contributes 6 rows to W
    assert fitted.mat_W.shape == (15 * 6, 2)
    assert np.allclose(fitted.mat_X[:, 0], 1.0)
    assert np.allclose(fitted.mat_X[:, 1], fitted.vec_t)
    assert np.allclose(fitted.mat_W[:, 0], 1.0)
    assert np.all(fitted.mat_W[:, 1] > 0)


def test_parameter_dimensions(fitted):
    assert fitted._num_bta == 2
    assert fitted._num_lmd == 2
    assert fitted._num_gma == 2
    assert len(fitted._theta) == 6
    assert len(fitted._bta) == 2
    assert len(fitted._lmd) == 2
    assert len(fitted._gma) == 2


def test_mean_parameters_recover_true_trend(fitted):
    assert np.all(np.isfinite(fitted._theta))
    assert 0.0 < fitted._bta[0] < 2.0
    assert 1.5 < fitted._bta[1] < 2.5


def test_log_likelihood_and_bic_are_consistent(fitted):
    assert np.isfinite(fitted.max_log_lik)
    assert np.isfinite(fitted.bic)
    expected = (len(fitted._theta) * math.log(fitted.m) - 2 * fitted.max_log_lik) / fitted.m
    assert fitted.bic == pytest.approx(expected)
    assert fitted.max_log_lik == pytest.approx(-fitted._obj_fun(fitted._theta) / 2)


def test_data_is_sorted_by_id_and_time(df):
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    model = JointModel(shuffled, "y|id|t~1|1", (1, 1, 1))
    ids = model.df["id"].values
    assert np.all(np.diff(ids) >= 0)
    for subject in np.unique(ids):
        times = model.df.loc[model.df["id"] == subject, "t"].values
        assert np.all(np.diff(times) > 0)


def test_row_order_of_input_does_not_change_estimates(df, fitted):
    shuffled = df.sample(frac=1.0, random_state=2).reset_index(drop=True)
    model = JointModel(shuffled, "y|id|t~1|1", (1, 1, 1))
    assert np.allclose(model._theta, fitted._theta, rtol=1e-6, atol=1e-8)
    assert model.bic == pytest.approx(fitted.bic)


def test_bfgs_optimiser_fits_model(df):
    model = JointModel(df, "y|id|t~1|1", (1, 1, 0), optim_meth="BFGS")
    assert model.optim_meth == "BFGS"
    assert len(model._theta) == 2 + 2 + 1
    assert np.all(np.isfinite(model._theta))
    assert np.isfinite(model.bic)


def test_invalid_optimiser_raises_name_error(df):
    with pytest.raises(NameError):
        JointModel(df, "y|id|t~1|1", (1, 1, 1), optim_meth="newton")


def test_covariate_adds_column_to_design_matrix():
    data = make_data(m=12, n=4, seed=3, with_x=True)
    model = JointModel(data, "y|id|t~x|1", (1, 1, 1))
    assert model.mat_X.shape == (48, 3)
    assert model.mat_Z.shape == (48, 2)
    assert model._num_bta == 3
    assert np.all(np.isfinite(model._bta))


def test_different_polynomial_orders_change_parameter_counts(df):
    model = JointModel(df, "y|id|t~1|1", (2, 1, 0))
    assert model._num_bta == 3
    assert model._num_lmd == 2
    assert model._num_gma == 1
    assert model.mat_W.shape == (15 * 6, 1)
    assert np.all(np.isfinite(model._theta))


def test_summary_prints_estimates(capsys, fitted):
    fitted.summary()
    out = capsys.readouterr().out
    assert "Model:" in out
    assert "y|id|t~1|1" in out
    assert "Mean Parameters:" in out
    assert "Innovation Variance Parameters:" in out
    assert "Autoregressive Parameters:" in out
    assert "Maximum of log-likelihood:" in out
    assert "BIC:" in out


def test_wald_test_prints_table_for_each_parameter_block(capsys, df):
    model = JointModel(df, "y|id|t~1|1", (1, 1, 0))
    model.wald_test()
    out = capsys.readouterr().out
    assert "Wald Test" in out
    assert "beta0" in out and "beta1" in out
    assert "lambda0" in out and "lambda1" in out
    assert "gamma0" in out
    assert "gamma1" not in out
    assert "p-value" in out


def test_new_design_builds_matrices_for_requested_orders(fitted):
    mat_X, mat_Z, mat_W = fitted._new_design((2, 0, 1))
    # new polynomial columns are placed in front of the existing covariate columns
    assert mat_X.shape[0] == 60
    assert mat_X.shape[1] == 3 + fitted.mat_X.shape[1]
    assert mat_Z.shape[1] == 1 + fitted.mat_Z.shape[1]
    assert mat_W.shape == (15 * 6, 2)
    assert np.allclose(mat_X[:, 0], 1.0)
    assert np.allclose(mat_X[:, 2], fitted.vec_t ** 2)


def test_model_selection_with_given_orders_stays_within_bounds(df, capsys):
    model = JointModel(df, "y|id|t~1|1", (1, 1, 0), model_select=True)
    capsys.readouterr()
    assert isinstance(model.poly_orders, tuple)
    assert len(model.poly_orders) == 3
    for chosen, upper in zip(model.poly_orders, (1, 1, 0)):
        assert 0 <= chosen <= upper
    assert model._num_bta == model.poly_orders[0] + 1
    assert model._num_lmd == model.poly_orders[1] + 1
    assert model._num_gma == model.poly_orders[2] + 1
    assert np.isfinite(model.bic)


def test_automatic_model_selection_when_orders_missing():
    data = make_data(m=20, n=3, seed=5)
    model = JointModel(data, "y|id|t~1|1")
    assert isinstance(model.poly_orders, tuple)
    assert len(model.poly_orders) == 3
    assert all(isinstance(order, (int, np.integer)) for order in model.poly_orders)
    # the largest possible order is n_max - 1 = 2
    assert all(0 <= order <= 2 for order in model.poly_orders)
    assert model._num_bta == model.poly_orders[0] + 1
    assert model._num_lmd == model.poly_orders[1] + 1
    assert model._num_gma == model.poly_orders[2] + 1
    assert np.all(np.isfinite(model._theta))


def test_bootstrap_data_resamples_subjects(fitted):
    np.random.seed(0)
    x_boot, z_boot, w_boot, y_boot, n_boot = fitted._bootstrap_data()
    assert len(n_boot) == fitted.m
    total = int(np.sum(n_boot))
    assert x_boot.shape == (total, fitted.mat_X.shape[1])
    assert z_boot.shape == (total, fitted.mat_Z.shape[1])
    assert len(y_boot) == total
    assert w_boot.shape == (int(np.sum(n_boot * (n_boot - 1) // 2)), fitted.mat_W.shape[1])
    # every resampled response must come from the original data
    assert set(np.round(y_boot, 10)).issubset(set(np.round(fitted.vec_y, 10)))


def test_boot_curve_draws_figure_without_error(monkeypatch, df):
    model = JointModel(df, "y|id|t~1|1", (1, 1, 0))
    shown = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(True))
    np.random.seed(1)
    model.boot_curve(2)
    assert shown == [True]
    assert len(plt.get_fignums()) >= 1
    plt.close("all")
