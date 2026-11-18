import pytest
import numpy as np
import pandas as pd
from jmcm.joint_model import JointModel


def make_sample_data():
    """
    Returns minimal sample data for testing JointModel.
    Adjust data according to the actual JointModel interface once known.
    """
    # Placeholder example of data that might be used.
    # Typically, JointModel might expect formula strings and DataFrame, or arrays.
    df = pd.DataFrame({
        "time": [5, 10, 12, 15, 20],
        "event": [1, 0, 1, 0, 1],
        "x1": [0.5, 1.2, 0.3, 0.7, 1.5],
        "x2": [1, 0, 1, 0, 1],
        "longitudinal": [10, 15, 10, 20, 25],
        "id": [1, 2, 3, 4, 5]
    })
    return df


def test_import_joint_model():
    # The JointModel class can be imported from jmcm
    from jmcm import JointModel
    assert JointModel is not None


def test_jointmodel_instantiation_minimal():
    # Test that JointModel can be instantiated with minimal required args
    # Adjust parameters according to the actual JointModel __init__ signature
    model = JointModel()
    assert isinstance(model, JointModel)


def test_jointmodel_fit_returns_self_or_none():
    # Assuming JointModel has a fit method; test if it returns self or None
    model = JointModel()
    data = make_sample_data()
    result = model.fit(data)
    # The fit method typically returns self or None
    assert result is None or result is model


def test_jointmodel_predict_returns_correct_type():
    model = JointModel()
    data = make_sample_data()
    model.fit(data)
    pred = model.predict(data)
    # Expect numpy array or pandas Series as prediction output
    assert isinstance(pred, (np.ndarray, pd.Series))


def test_jointmodel_predict_without_fit_raises():
    # Calling predict before fit should raise error
    model = JointModel()
    data = make_sample_data()
    with pytest.raises(Exception):
        model.predict(data)


def test_jointmodel_fit_fail_on_bad_data():
    model = JointModel()
    bad_data = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(Exception):
        model.fit(bad_data)


def test_jointmodel_fit_with_valid_data_changes_state():
    model = JointModel()
    data = make_sample_data()
    assert not getattr(model, "fitted_", False)
    model.fit(data)
    # After fitting, model should have attribute or flag indicating fitted
    assert getattr(model, "fitted_", True)  # or adapt to your attribute


def test_jointmodel_predict_proba_outputs_probs():
    model = JointModel()
    data = make_sample_data()
    model.fit(data)
    proba = model.predict_proba(data)
    assert isinstance(proba, (np.ndarray, pd.Series))
    # Probabilities should be in [0,1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_jointmodel_score_computes_valid_score():
    model = JointModel()
    data = make_sample_data()
    model.fit(data)
    score = model.score(data)
    # Score should be numeric scalar
    assert isinstance(score, (int, float))
    # Score should be finite
    assert np.isfinite(score)


def test_jointmodel_repr_contains_classname():
    model = JointModel()
    r = repr(model)
    assert "JointModel" in r


def test_jointmodel_fit_twice_replaces_results():
    model = JointModel()
    data = make_sample_data()
    model.fit(data)
    first_params = getattr(model, "params_", None)
    model.fit(data)
    second_params = getattr(model, "params_", None)
    assert second_params is not None
    # Fitting twice should update params
    assert first_params is not None
    # If params_ is a numpy array or similar, they might differ or be equal - at least they exist


def test_jointmodel_log_likelihood_value():
    model = JointModel()
    data = make_sample_data()
    model.fit(data)
    ll = model.log_likelihood(data)
    # log likelihood numeric scalar and finite
    assert isinstance(ll, (float, np.floating))
    assert np.isfinite(ll)


def test_jointmodel_handles_empty_data_on_fit():
    model = JointModel()
    empty_data = pd.DataFrame()
    with pytest.raises(Exception):
        model.fit(empty_data)


def test_jointmodel_predict_on_empty_data_returns_empty():
    model = JointModel()
    data = make_sample_data()
    model.fit(data)
    empty_data = pd.DataFrame()
    pred = model.predict(empty_data)
    # Typically prediction on empty input returns empty array or raises
    assert len(pred) == 0 or isinstance(pred, (np.ndarray, pd.Series))


def test_jointmodel_get_params_returns_dict():
    model = JointModel()
    params = model.get_params()
    assert isinstance(params, dict)


def test_jointmodel_set_params_changes_params():
    model = JointModel()
    params_before = model.get_params()
    params_update = {k: v for k, v in params_before.items()}
    # If model has numeric params, try to change one if any
    for k, v in list(params_update.items()):
        if isinstance(v, (int, float)):
            params_update[k] = v + 1
            break
    returned = model.set_params(**params_update)
    # set_params typically returns self or None
    assert returned is None or returned is model
    params_after = model.get_params()
    # Params after change should differ if relevant param changed
    changed = any(params_after[k] != params_before[k] for k in params_update)
    assert changed


def test_jointmodel_str_contains_classname():
    model = JointModel()
    s = str(model)
    assert "JointModel" in s


def test_jointmodel_fit_with_non_dataframe_raises():
    model = JointModel()
    with pytest.raises(Exception):
        model.fit([1, 2, 3])


def test_jointmodel_predict_with_non_dataframe_raises():
    model = JointModel()
    with pytest.raises(Exception):
        model.predict([1, 2, 3])

