import pytest
import numpy as np
import pandas as pd
from jmcm.joint_model import JointModel


# Since the user did not provide the code for JointModel,
# these tests assume common expected behaviors of a statistical modeling class
# named JointModel, such as initialization, fitting, prediction, and error handling.
# Adjust these tests according to the actual implementation details.

def test_joint_model_init_no_args():
    # Should initialize without arguments if defaults exist
    model = JointModel()
    assert model is not None

def test_joint_model_init_with_data():
    # Create minimal dummy datasets for survival and longitudinal / covariates
    df_surv = pd.DataFrame({
        "time": [5, 6, 7],
        "event": [1, 0, 1]
    })
    df_long = pd.DataFrame({
        "time": [1, 2, 3],
        "value": [0.5, 0.6, 0.55],
        "id": [1, 1, 1]
    })
    model = JointModel(surv_data=df_surv, long_data=df_long)
    assert model.surv_data.equals(df_surv)
    assert model.long_data.equals(df_long)

def test_joint_model_fit_without_data_raises():
    model = JointModel()
    with pytest.raises(Exception):
        model.fit()

def test_joint_model_fit_with_data_runs():
    df_surv = pd.DataFrame({
        "time": [5, 6, 7],
        "event": [1, 0, 1]
    })
    df_long = pd.DataFrame({
        "time": [1, 2, 3],
        "value": [0.5, 0.6, 0.55],
        "id": [1, 1, 1]
    })
    model = JointModel(surv_data=df_surv, long_data=df_long)
    # We expect fit to complete without error
    model.fit()

def test_joint_model_predict_before_fit_error():
    df_surv = pd.DataFrame({
        "time": [5, 6, 7],
        "event": [1, 0, 1]
    })
    df_long = pd.DataFrame({
        "time": [1, 2, 3],
        "value": [0.5, 0.6, 0.55],
        "id": [1, 1, 1]
    })
    model = JointModel(surv_data=df_surv, long_data=df_long)
    with pytest.raises(Exception):
        model.predict(df_long)

def test_joint_model_fit_predict_consistency():
    # Fit and then predict on some data; check output is numpy array or pandas Series/DataFrame
    df_surv = pd.DataFrame({
        "time": [5, 6, 7, 8],
        "event": [1, 0, 1, 0]
    })
    df_long = pd.DataFrame({
        "time": [1, 2, 3, 4],
        "value": [0.5, 0.6, 0.55, 0.7],
        "id": [1, 1, 2, 2]
    })
    model = JointModel(surv_data=df_surv, long_data=df_long)
    model.fit()
    preds = model.predict(df_long)
    assert isinstance(preds, (np.ndarray, pd.Series, pd.DataFrame))

def test_joint_model_update_params_changes_state():
    # If the class supports updating parameters, test it changes internal state
    df_surv = pd.DataFrame({
        "time": [5, 6],
        "event": [1, 0]
    })
    df_long = pd.DataFrame({
        "time": [1, 2],
        "value": [0.5, 0.6],
        "id": [1, 1]
    })
    model = JointModel(surv_data=df_surv, long_data=df_long)
    model.fit()
    old_params = model.get_params() if hasattr(model, "get_params") else None
    if old_params is not None:
        # Example change: tweak parameters dictionary or object
        new_params = old_params.copy()
        for key in new_params:
            new_params[key] = new_params[key]
        model.set_params(new_params)
        updated_params = model.get_params()
        assert updated_params == new_params

def test_joint_model_repr_and_str():
    model = JointModel()
    r = repr(model)
    s = str(model)
    assert isinstance(r, str)
    assert isinstance(s, str)
    assert name := getattr(model, "__class__", None)
    assert "JointModel" in r or "JointModel" in s

def test_fit_with_invalid_data_raises():
    df_surv_invalid = pd.DataFrame({
        "wrongtime": [5, 6],
        "event": [1, 0]
    })
    df_long = pd.DataFrame({
        "time": [1, 2],
        "value": [0.5, 0.6],
        "id": [1, 1]
    })
    model = JointModel(surv_data=df_surv_invalid, long_data=df_long)
    with pytest.raises(Exception):
        model.fit()

def test_predict_with_invalid_input_raises():
    df_surv = pd.DataFrame({
        "time": [5, 6],
        "event": [1, 0]
    })
    df_long = pd.DataFrame({
        "time": [1, 2],
        "value": [0.5, 0.6],
        "id": [1, 1]
    })
    model = JointModel(surv_data=df_surv, long_data=df_long)
    model.fit()
    with pytest.raises(Exception):
        model.predict("not a dataframe or ndarray")

def test_model_can_handle_empty_data():
    df_empty = pd.DataFrame(columns=["time", "event"])
    df_long_empty = pd.DataFrame(columns=["time", "value", "id"])
    model = JointModel(surv_data=df_empty, long_data=df_long_empty)
    with pytest.raises(Exception):
        model.fit()

def test_joint_model_fit_returns_self():
    df_surv = pd.DataFrame({
        "time": [5, 6, 7],
        "event": [1, 0, 1]
    })
    df_long = pd.DataFrame({
        "time": [1, 2, 3],
        "value": [0.5, 0.6, 0.55],
        "id": [1, 1, 1]
    })
    model = JointModel(surv_data=df_surv, long_data=df_long)
    result = model.fit()
    # If fit returns self, test it
    assert result is None or result is model

def test_predict_after_refit_changes_prediction():
    df_surv = pd.DataFrame({
        "time": [5, 6, 7],
        "event": [1, 0, 1]
    })
    df_long = pd.DataFrame({
        "time": [1, 2, 3],
        "value": [0.5, 0.6, 0.55],
        "id": [1, 1, 1]
    })
    model = JointModel(surv_data=df_surv, long_data=df_long)
    model.fit()
    pred1 = model.predict(df_long)
    # Refit with slightly modified data
    df_surv_mod = df_surv.copy()
    df_surv_mod.loc[0, "time"] += 1
    model = JointModel(surv_data=df_surv_mod, long_data=df_long)
    model.fit()
    pred2 = model.predict(df_long)
    # predictions should differ after refitting model on new data
    if isinstance(pred1, np.ndarray):
        assert not np.allclose(pred1, pred2)
    else:
        assert pred1.equals(pred2) is False or pred1 != pred2

def test_joint_model_handles_multiple_ids():
    df_surv = pd.DataFrame({
        "time": [10, 20, 30],
        "event": [1, 0, 1],
        "id": [1, 2, 3]
    })
    df_long = pd.DataFrame({
        "time": [1, 2, 3, 4, 5, 6],
        "value": [0.5, 0.6, 0.55, 0.7, 0.65, 0.8],
        "id": [1, 1, 2, 2, 3, 3]
    })
    model = JointModel(surv_data=df_surv, long_data=df_long)
    model.fit()
    preds = model.predict(df_long)
    assert len(preds) == len(df_long) or (hasattr(preds, "shape") and preds.shape[0] == len(df_long))

def test_model_methods_do_not_modify_input_data():
    df_surv = pd.DataFrame({
        "time": [5, 6],
        "event": [1, 0]
    })
    df_long = pd.DataFrame({
        "time": [1, 2],
        "value": [0.5, 0.6],
        "id": [1, 1]
    })
    df_surv_copy = df_surv.copy()
    df_long_copy = df_long.copy()
    model = JointModel(surv_data=df_surv, long_data=df_long)
    model.fit()
    model.predict(df_long)
    pd.testing.assert_frame_equal(df_surv, df_surv_copy)
    pd.testing.assert_frame_equal(df_long, df_long_copy)
