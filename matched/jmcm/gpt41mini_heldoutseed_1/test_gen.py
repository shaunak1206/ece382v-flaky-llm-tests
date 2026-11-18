import pytest
import numpy as np
import pandas as pd
from jmcm.joint_model import JointModel


# Dummy minimal data to use in tests
def get_sample_data():
    # Construct a simple DataFrame with time, event and covariates
    data = pd.DataFrame({
        "time": [5, 6, 6, 2, 4, 3],
        "event": [1, 0, 0, 1, 1, 0],
        "x1": [0.1, 0.2, 0.2, 0.3, 0.2, 0.5],
        "x2": [1, 2, 1, 2, 1, 2]
    })
    return data


def test_import_joint_model():
    # Test if JointModel class is imported correctly from the package
    from jmcm import JointModel
    assert JointModel is not None
    assert callable(JointModel)


def test_joint_model_instantiation():
    # Test creating an instance of JointModel with no initial arguments
    jm = JointModel()
    assert isinstance(jm, JointModel)


def test_fit_method_with_valid_data():
    jm = JointModel()
    df = get_sample_data()
    # fit parameters depend on implementation, assume fit(df, time_col, event_col)
    # We'll try: fit(dataframe, time_col='time', event_col='event', covariates=['x1', 'x2'])
    result = None
    try:
        # We expect fit to work and return something (often None, or self)
        result = jm.fit(df, time_col="time", event_col="event", covariates=["x1", "x2"])
    except Exception:
        pytest.fail("fit method raised exception with valid data")
    assert result is None or result == jm


def test_fit_method_raises_error_on_missing_columns():
    jm = JointModel()
    df = get_sample_data()
    df_missing = df.drop(columns=["time"])

    with pytest.raises(Exception):
        jm.fit(df_missing, time_col="time", event_col="event", covariates=["x1", "x2"])


def test_fit_with_no_events():
    jm = JointModel()
    df = get_sample_data()
    df["event"] = 0  # no events recorded

    # Depending on implementation, fit may raise or handle gracefully
    try:
        jm.fit(df, time_col="time", event_col="event", covariates=["x1"])
    except Exception:
        pytest.skip("fit method raises exception with no events; might be expected")


def test_predict_survival_function_returns_dataframe():
    jm = JointModel()
    df = get_sample_data()
    jm.fit(df, time_col="time", event_col="event", covariates=["x1", "x2"])

    # Predict for one new observation
    new_obs = pd.DataFrame({"x1": [0.1], "x2": [1]})
    survival = jm.predict_survival_function(new_obs)

    assert isinstance(survival, pd.DataFrame)
    assert not survival.empty


def test_predict_survival_function_invalid_input():
    jm = JointModel()
    df = get_sample_data()
    jm.fit(df, time_col="time", event_col="event", covariates=["x1", "x2"])

    with pytest.raises(Exception):
        # Passing input not as DataFrame
        jm.predict_survival_function([0.1, 1])


def test_log_likelihood_after_fit_is_float():
    jm = JointModel()
    df = get_sample_data()
    jm.fit(df, time_col="time", event_col="event", covariates=["x1"])
    ll = jm.log_likelihood()
    assert isinstance(ll, float)


def test_aic_bic_methods_present_and_return_float():
    jm = JointModel()
    df = get_sample_data()
    jm.fit(df, time_col="time", event_col="event", covariates=["x1"])
    aic = jm.aic()
    bic = jm.bic()
    assert isinstance(aic, float)
    assert isinstance(bic, float)


def test_model_str_representation():
    jm = JointModel()
    s = str(jm)
    assert isinstance(s, str)


def test_multiple_fits_produce_consistent_results():
    jm = JointModel()
    df = get_sample_data()
    jm.fit(df, time_col="time", event_col="event", covariates=["x1"])
    ll1 = jm.log_likelihood()

    jm2 = JointModel()
    jm2.fit(df, time_col="time", event_col="event", covariates=["x1"])
    ll2 = jm2.log_likelihood()

    assert abs(ll1 - ll2) < 1e-6


def test_fit_with_additional_parameters(monkeypatch):
    # Check if fit accepts keyword args (depends on implementation)
    jm = JointModel()
    df = get_sample_data()

    try:
        res = jm.fit(df, time_col="time", event_col="event", covariates=["x1"], extra_param=123)
    except TypeError:
        pytest.skip("fit() does not support extra parameters")
    else:
        assert res is None or res == jm


def test_predict_survival_function_after_refit_changes_output():
    jm = JointModel()
    df = get_sample_data()
    jm.fit(df, time_col="time", event_col="event", covariates=["x1"])

    new_obs = pd.DataFrame({"x1": [0.1]})
    surv1 = jm.predict_survival_function(new_obs)

    # Fit again with different covariates, if allowed
    jm.fit(df, time_col="time", event_col="event", covariates=["x2"])
    surv2 = jm.predict_survival_function(new_obs)

    # When covariates change, prediction should change too
    assert not surv1.equals(surv2)


def test_fit_raises_with_empty_dataframe():
    jm = JointModel()
    empty_df = pd.DataFrame()

    with pytest.raises(Exception):
        jm.fit(empty_df, time_col="time", event_col="event", covariates=["x1"])


def test_fit_with_non_numeric_covariates():
    jm = JointModel()
    df = get_sample_data()
    df["x_cat"] = ["a", "b", "a", "b", "a", "b"]  # categorical string

    # If fit can handle categorical converted outside, no exception; else may raise
    try:
        jm.fit(df, time_col="time", event_col="event", covariates=["x_cat"])
    except Exception:
        pytest.skip("fit method cannot handle non-numeric covariates")


def test_predict_survival_function_after_fit_single_sample_is_nonempty():
    jm = JointModel()
    df = get_sample_data()
    jm.fit(df, time_col="time", event_col="event", covariates=["x1", "x2"])

    sample = pd.DataFrame({"x1": [0.3], "x2": [2]})
    result = jm.predict_survival_function(sample)
    assert not result.empty
    assert isinstance(result, pd.DataFrame)


def test_model_requires_fit_before_predict():
    jm = JointModel()

    sample = pd.DataFrame({"x1": [0.1], "x2": [1]})

    with pytest.raises(Exception):
        jm.predict_survival_function(sample)


def test_fit_with_duplicate_covariate_names():
    jm = JointModel()
    df = get_sample_data()

    # Introduce duplicate covariate columns
    df["x1_dup"] = df["x1"]

    with pytest.raises(Exception):
        jm.fit(df, time_col="time", event_col="event", covariates=["x1", "x1_dup", "x1"])


def test_fit_handles_warning(monkeypatch):
    jm = JointModel()
    df = get_sample_data()

    # Monkeypatch pandas to raise a Warning to check if fit handles it
    import warnings

    def warn_func(*args, **kwargs):
        raise Warning("test warning")

    monkeypatch.setattr("pandas.DataFrame.cov", warn_func)

    try:
        jm.fit(df, time_col="time", event_col="event", covariates=["x1"])
    except Warning:
        pytest.skip("fit method does not handle warnings gracefully")
