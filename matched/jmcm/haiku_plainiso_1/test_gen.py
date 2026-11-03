"""Unit tests for jmcm.JointModel (fixed orders, model selection, outputs)."""

import numpy as np
import pandas as pd
import pytest

from jmcm import JointModel

FORMULA = 'y|id|t~x1+x2|x1'
N_SUBJECTS = 20
N_OBS = 4
N_PAIRS = N_OBS * (N_OBS - 1) // 2
MEAN_COVARIATES = 2  # x1, x2
VAR_COVARIATES = 1   # x1


def make_df(seed=0):
    rng = np.random.RandomState(seed)
    ids = np.repeat(np.arange(1, N_SUBJECTS + 1), N_OBS)
    t = np.tile(np.arange(N_OBS, dtype=float), N_SUBJECTS)
    x1 = rng.normal(size=ids.size)
    x2 = rng.normal(size=ids.size)
    noise = rng.normal(scale=0.1, size=ids.size)
    y = 1.0 + 0.5 * t + 0.3 * x1 + noise
    return pd.DataFrame({'y': y, 'id': ids, 't': t, 'x1': x1, 'x2': x2})


def check_selected_orders(model, max_order):
    orders = model.poly_orders
    assert isinstance(orders, tuple) and len(orders) == 3
    assert all(isinstance(p, int) for p in orders)
    assert all(0 <= p <= max_order for p in orders)
    assert model.mat_X.shape[1] == orders[0] + 1 + MEAN_COVARIATES
    assert model.mat_Z.shape[1] == orders[1] + 1 + VAR_COVARIATES
    assert model.mat_W.shape[1] == orders[2] + 1


def test_stores_formula_and_orders():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    assert model.formula == FORMULA
    assert model.poly_orders == (1, 1, 1)
    assert model.optim_meth == 'default'


def test_design_shapes_follow_orders():
    model = JointModel(make_df(), FORMULA, poly_orders=(2, 1, 0))
    n_rows = N_SUBJECTS * N_OBS
    assert model.mat_X.shape == (n_rows, 3 + MEAN_COVARIATES)
    assert model.mat_Z.shape == (n_rows, 2 + VAR_COVARIATES)
    assert model.mat_W.shape == (N_SUBJECTS * N_PAIRS, 1)


def test_mat_W_has_one_row_per_within_subject_pair():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    assert model.mat_W.shape == (N_SUBJECTS * N_PAIRS, 2)


def test_mat_W_columns_hold_time_lag_powers():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    np.testing.assert_allclose(model.mat_W[:, 0], 1.0)
    # Subject 1 has times 0,1,2,3, so lags for pairs (1,0),(2,0),(2,1),(3,0),(3,1),(3,2)
    np.testing.assert_allclose(model.mat_W[:N_PAIRS, 1], [1, 2, 1, 3, 2, 1])


def test_zero_orders_give_intercept_only_design():
    model = JointModel(make_df(), FORMULA, poly_orders=(0, 0, 0))
    assert model.mat_X.shape[1] == 1 + MEAN_COVARIATES
    np.testing.assert_allclose(model.mat_X[:, 0], 1.0)
    np.testing.assert_allclose(model.mat_Z[:, 0], 1.0)
    np.testing.assert_allclose(model.mat_W, 1.0)
    assert model.mat_W.shape == (N_SUBJECTS * N_PAIRS, 1)


def test_subject_and_measurement_counts():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    assert model.m == N_SUBJECTS
    np.testing.assert_array_equal(model.vec_n, np.full(N_SUBJECTS, N_OBS))
    assert model.vec_y.shape == (N_SUBJECTS * N_OBS,)


def test_rows_sorted_by_id_then_time():
    shuffled = make_df().sample(frac=1.0, random_state=5)
    model = JointModel(shuffled, FORMULA, poly_orders=(1, 1, 1))
    ids = model.df['id'].values
    t = model.df['t'].values
    assert np.all(np.diff(ids) >= 0)
    assert np.all(np.diff(t.reshape(N_SUBJECTS, N_OBS), axis=1) > 0)
    np.testing.assert_allclose(model.vec_t, t)
    np.testing.assert_allclose(model.vec_y, model.df['y'].values)


def test_input_dataframe_not_mutated():
    df = make_df().sample(frac=1.0, random_state=5)
    before = df.copy()
    JointModel(df, FORMULA, poly_orders=(1, 1, 1))
    pd.testing.assert_frame_equal(df, before)


def test_parameter_vectors_have_expected_lengths():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    assert model._bta.shape == (2 + MEAN_COVARIATES,)
    assert model._lmd.shape == (2 + VAR_COVARIATES,)
    assert model._gma.shape == (2,)
    assert model._theta.shape == (9,)
    assert model._num_theta == 9


def test_default_fit_parameters_are_finite():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    assert np.all(np.isfinite(model._theta))
    assert np.isfinite(model.max_log_lik)
    assert np.isfinite(model.bic)


def test_bfgs_fit_parameters_are_finite():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1),
                       optim_meth='BFGS')
    assert model.optim_meth == 'BFGS'
    assert np.all(np.isfinite(model._theta))
    assert np.isfinite(model.bic)


def test_default_fit_recovers_mean_coefficients():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    # Mean columns are [intercept, t, x1, x2]; truth is [1.0, 0.5, 0.3, 0.0]
    np.testing.assert_allclose(model._bta[:2], [1.0, 0.5], atol=0.1)
    assert abs(model._bta[2] - 0.3) < 0.1


def test_default_fit_recovers_log_variance_intercept():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    # Innovation variance is 0.01 everywhere, so the intercept is log(0.01)
    assert abs(model._lmd[0] - np.log(0.01)) < 0.6


def test_fit_is_deterministic():
    first = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    second = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    np.testing.assert_allclose(first._theta, second._theta)
    assert np.isclose(first.bic, second.bic)


def test_bic_matches_formula():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    expected = (model._num_theta * np.log(model.m)
                - 2 * model.max_log_lik) / model.m
    assert np.isclose(model.bic, expected, rtol=1e-10)


def test_invalid_optim_method_raises_name_error():
    with pytest.raises(NameError):
        JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1),
                   optim_meth='newton')


def test_empty_tuple_triggers_model_selection():
    model = JointModel(make_df(), FORMULA)
    check_selected_orders(model, max_order=N_OBS - 1)


def test_short_tuple_triggers_model_selection():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 2))
    check_selected_orders(model, max_order=N_OBS - 1)


def test_model_select_true_searches_grid():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1),
                       model_select=True)
    check_selected_orders(model, max_order=1)


def test_summary_prints_estimates(capsys):
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    capsys.readouterr()
    model.summary()
    out = capsys.readouterr().out
    for header in ['Model:', 'Mean Parameters:',
                   'Innovation Variance Parameters:',
                   'Autoregressive Parameters:', 'BIC:']:
        assert header in out


def test_wald_test_prints_every_parameter(capsys):
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    capsys.readouterr()
    model.wald_test()
    out = capsys.readouterr().out
    for name in ['Wald Test', 'beta0', 'beta3', 'lambda0', 'lambda2',
                 'gamma0', 'gamma1', 'p-value']:
        assert name in out
    assert 'beta4' not in out


def test_info_matrices_have_parameter_dimensions():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    i_11, i_22, i_33 = model._info_mat()
    assert i_11.shape == (4, 4)
    assert i_22.shape == (3, 3)
    assert i_33.shape == (2, 2)
    assert all(np.all(np.isfinite(m)) for m in (i_11, i_22, i_33))


def test_bootstrap_resample_shapes():
    model = JointModel(make_df(), FORMULA, poly_orders=(1, 1, 1))
    np.random.seed(0)
    x_b, z_b, w_b, y_b, n_b = model._bootstrap_data()
    n_rows = N_SUBJECTS * N_OBS
    assert x_b.shape == (n_rows, 4)
    assert z_b.shape == (n_rows, 3)
    assert w_b.shape == (N_SUBJECTS * N_PAIRS, 2)
    assert y_b.shape == (n_rows,)
    np.testing.assert_array_equal(n_b, np.full(N_SUBJECTS, N_OBS))
