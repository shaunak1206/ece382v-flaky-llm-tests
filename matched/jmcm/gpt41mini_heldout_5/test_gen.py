import pytest
import numpy as np
import pandas as pd

from jmcm.joint_model import JointModel


def test_joint_model_basic_instantiation():
    jm = JointModel()
    assert isinstance(jm, JointModel)


def test_joint_model_fit_and_predict_no_data():
    jm = JointModel()
    # Assuming .fit() requires data, check error handling or default behavior
    with pytest.raises(TypeError):
        jm.fit()
    with pytest.raises(TypeError):
        jm.predict()


def test_joint_model_fit_with_minimal_data():
    jm = JointModel()
    # Create minimal example data
    time = np.array([1, 2, 3, 4, 5])
    event = np.array([0, 1, 0, 1, 0])
    longitudinal = pd.DataFrame({
        "time": time,
        "value": [2, 3, 2, 5, 4],
        "id": [1, 1, 2, 2, 3]
    })
    surv_data = pd.DataFrame({
        "time": time,
        "event": event,
        "id": [1, 1, 2, 2, 3]
    })

    # Fit might expect specific parameters; just check no exceptions if typical usage
    try:
        jm.fit(longitudinal, surv_data, id_col="id", time_col="time", event_col="event")
    except Exception:
        pytest.fail("JointModel.fit raised an unexpected exception")


def test_joint_model_predict_after_fit():
    jm = JointModel()
    time = np.array([1, 2, 3, 4])
    event = np.array([0, 1, 0, 1])
    longitudinal = pd.DataFrame({
        "time": time,
        "value": [1.2, 1.4, 1.6, 1.8],
        "id": [1, 1, 2, 2]
    })
    surv_data = pd.DataFrame({
        "time": time,
        "event": event,
        "id": [1, 1, 2, 2]
    })

    jm.fit(longitudinal, surv_data, id_col="id", time_col="time", event_col="event")

    pred = jm.predict(longitudinal)
    # Without detailed doc, test that prediction output is not None and is an array or similar
    assert pred is not None
    assert isinstance(pred, (np.ndarray, pd.Series, list))


def test_joint_model_repr_and_str_methods_do_not_error():
    jm = JointModel()
    r = repr(jm)
    s = str(jm)
    assert isinstance(r, str)
    assert isinstance(s, str)
    assert "JointModel" in r or "JointModel" in s


def test_joint_model_fit_rejects_invalid_data():
    jm = JointModel()
    # Pass data missing required columns or wrong types
    with pytest.raises(Exception):
        jm.fit([1, 2, 3], {"wrong": "data"}, id_col="id", time_col="time", event_col="event")


def test_joint_model_predict_rejects_invalid_input():
    jm = JointModel()
    with pytest.raises(Exception):
        jm.predict("not_a_dataframe_or_array")


def test_joint_model_methods_raise_if_not_fitted():
    jm = JointModel()
    # predict before fit likely to raise
    with pytest.raises(Exception):
        jm.predict(pd.DataFrame())


def test_joint_model_can_handle_empty_dataframe_on_fit():
    jm = JointModel()
    empty_df = pd.DataFrame(columns=["id", "time", "value"])
    empty_surv_df = pd.DataFrame(columns=["id", "time", "event"])

    with pytest.raises(Exception):
        jm.fit(empty_df, empty_surv_df, id_col="id", time_col="time", event_col="event")


def test_joint_model_fit_predict_consistency():
    jm = JointModel()
    # Prepare dummy data
    df_long = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "time": [1, 2, 1, 2],
        "value": [0.5, 0.7, 0.6, 0.8]
    })
    df_surv = pd.DataFrame({
        "id": [1, 2],
        "time": [3, 3],
        "event": [1, 0]
    })

    jm.fit(df_long, df_surv, id_col="id", time_col="time", event_col="event")

    pred1 = jm.predict(df_long)
    pred2 = jm.predict(df_long)

    # Predictions for same input should be consistent (equal or close)
    if hasattr(pred1, "equals") and hasattr(pred2, "equals"):
        assert pred1.equals(pred2)
    else:
        # fallback for array-like
        np.testing.assert_allclose(pred1, pred2)


def test_joint_model_fit_accepts_additional_fit_kwargs():
    jm = JointModel()
    df_long = pd.DataFrame({
        "id": [1, 2],
        "time": [1, 2],
        "value": [10, 20]
    })
    df_surv = pd.DataFrame({
        "id": [1, 2],
        "time": [2, 3],
        "event": [1, 0]
    })

    try:
        # Pass some kwargs that fit might accept or ignore
        jm.fit(df_long, df_surv, id_col="id", time_col="time", event_col="event", max_iter=10, tol=1e-4)
    except Exception:
        pytest.fail("JointModel.fit did not accept additional kwargs gracefully")


def test_joint_model_fit_fails_with_missing_id_col():
    jm = JointModel()
    df_long = pd.DataFrame({
        "time": [1, 2],
        "value": [3, 4]
    })
    df_surv = pd.DataFrame({
        "id": [1, 2],
        "time": [2, 3],
        "event": [1, 0]
    })

    with pytest.raises(Exception):
        jm.fit(df_long, df_surv, id_col="id", time_col="time", event_col="event")


def test_joint_model_fit_fails_with_missing_event_col():
    jm = JointModel()
    df_long = pd.DataFrame({
        "id": [1, 2],
        "time": [1, 2],
        "value": [3, 4]
    })
    df_surv = pd.DataFrame({
        "id": [1, 2],
        "time": [2, 3]
    })

    with pytest.raises(Exception):
        jm.fit(df_long, df_surv, id_col="id", time_col="time", event_col="event")


def test_joint_model_predict_returns_expected_type_after_fit():
    jm = JointModel()
    df_long = pd.DataFrame({
        "id": [1, 2],
        "time": [1, 2],
        "value": [10, 20]
    })
    df_surv = pd.DataFrame({
        "id": [1, 2],
        "time": [2, 3],
        "event": [1, 0]
    })

    jm.fit(df_long, df_surv, id_col="id", time_col="time", event_col="event")
    pred = jm.predict(df_long)
    assert isinstance(pred, (np.ndarray, pd.Series, list))


def test_joint_model_fit_can_handle_duplicates_ids():
    jm = JointModel()
    df_long = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "time": [1, 1, 2, 2],
        "value": [10, 10, 20, 20]
    })
    df_surv = pd.DataFrame({
        "id": [1, 2],
        "time": [3, 3],
        "event": [1, 0]
    })
    try:
        jm.fit(df_long, df_surv, id_col="id", time_col="time", event_col="event")
    except Exception:
        pytest.fail("JointModel.fit failed with duplicate ids")


def test_joint_model_repr_contains_class_name_after_fit():
    jm = JointModel()
    df_long = pd.DataFrame({
        "id": [1],
        "time": [1],
        "value": [5]
    })
    df_surv = pd.DataFrame({
        "id": [1],
        "time": [2],
        "event": [1]
    })
    jm.fit(df_long, df_surv, id_col="id", time_col="time", event_col="event")
    rep = repr(jm)
    assert "JointModel" in rep


def test_joint_model_str_contains_status_info():
    jm = JointModel()
    s = str(jm)
    assert isinstance(s, str)
    # May contain some info, at least the class name or status text:
    assert "JointModel" in s


def test_joint_model_predict_with_partial_data_after_fit():
    jm = JointModel()
    df_long = pd.DataFrame({
        "id": [1, 2, 3],
        "time": [1, 2, 3],
        "value": [1.1, 2.2, 3.3]
    })
    df_surv = pd.DataFrame({
        "id": [1, 2, 3],
        "time": [5, 5, 5],
        "event": [0, 1, 0]
    })

    jm.fit(df_long, df_surv, id_col="id", time_col="time", event_col="event")

    partial_df = pd.DataFrame({
        "id": [1],
        "time": [2],
        "value": [1.5]
    })

    pred = jm.predict(partial_df)
    assert pred is not None
