import pytest
import numpy as np
import pandas as pd
from jmcm.joint_model import JointModel

# Since the user only provided jmcm/__init__.py and it only imports JointModel from jmcm/joint_model.py,
# we must assume JointModel is implemented there.
# These tests assume the structure and interface of JointModel based on common joint model usage.
# Adjust test names and logic if your JointModel differs.

@pytest.fixture
def sample_longitudinal_data():
    # Example DataFrame representing longitudinal data
    return pd.DataFrame({
        'subject': [1,1,2,2,3,3],
        'time': [0,1,0,1,0,1],
        'measurement': [10, 12, 9, 11, 15, 14]
    })

@pytest.fixture
def sample_survival_data():
    # Example DataFrame representing survival data
    return pd.DataFrame({
        'subject': [1,2,3],
        'event_time': [5, 4, 6],
        'event_observed': [1, 0, 1]
    })

@pytest.fixture
def basic_model(sample_longitudinal_data, sample_survival_data):
    # Instantiate a basic JointModel with mock or minimal arguments
    # Since we don't know actual constructor signatures, assume something typical
    # e.g. JointModel(longitudinal_data, survival_data, ...)

    # If JointModel constructor differs, adapt accordingly.
    model = JointModel(longitudinal_data=sample_longitudinal_data, survival_data=sample_survival_data)
    return model

def test_import_joint_model():
    from jmcm import JointModel
    assert JointModel is not None

def test_model_initialization_with_data(sample_longitudinal_data, sample_survival_data):
    model = JointModel(longitudinal_data=sample_longitudinal_data, survival_data=sample_survival_data)
    assert hasattr(model, "longitudinal_data")
    assert hasattr(model, "survival_data")

def test_model_initialization_without_data_raises():
    with pytest.raises(TypeError):
        JointModel()

def test_fit_method_runs(basic_model):
    # Assume JointModel has a fit() method
    result = basic_model.fit()
    # Should return some indicator of success, often self or None
    assert result is None or result == basic_model

def test_predict_longitudinal_returns_array(basic_model):
    # Assume predict_longitudinal returns np.ndarray or list
    basic_model.fit()
    preds = basic_model.predict_longitudinal()
    assert isinstance(preds, (np.ndarray, list))

def test_predict_survival_returns_array(basic_model):
    # Assume predict_survival returns np.ndarray or list
    basic_model.fit()
    preds = basic_model.predict_survival()
    assert isinstance(preds, (np.ndarray, list))

def test_predict_longitudinal_without_fit_raises(basic_model):
    with pytest.raises(RuntimeError):
        basic_model.predict_longitudinal()

def test_predict_survival_without_fit_raises(basic_model):
    with pytest.raises(RuntimeError):
        basic_model.predict_survival()

def test_update_data_method(basic_model, sample_longitudinal_data):
    # Assume the model has an update_data method to set new longitudinal data
    new_data = sample_longitudinal_data.copy()
    new_data['measurement'] += 1
    basic_model.update_data(new_longitudinal=new_data)
    assert (basic_model.longitudinal_data['measurement'] == new_data['measurement']).all()

def test_handling_missing_values(sample_longitudinal_data, sample_survival_data):
    # Add missing values to data and ensure model can handle or raises properly
    data_with_nan = sample_longitudinal_data.copy()
    data_with_nan.loc[0, 'measurement'] = np.nan

    model = JointModel(longitudinal_data=data_with_nan, survival_data=sample_survival_data)
    with pytest.raises(ValueError):
        model.fit()

def test_model_str_repr(basic_model):
    s = str(basic_model)
    r = repr(basic_model)
    assert isinstance(s, str)
    assert isinstance(r, str)
    assert len(s) > 0
    assert len(r) > 0

def test_model_contains_longitudinal_and_survival_attributes(basic_model):
    assert hasattr(basic_model, "longitudinal_data")
    assert isinstance(basic_model.longitudinal_data, pd.DataFrame)
    assert hasattr(basic_model, "survival_data")
    assert isinstance(basic_model.survival_data, pd.DataFrame)

def test_passing_non_dataframe_to_constructor_raises(sample_survival_data):
    with pytest.raises(TypeError):
        JointModel(longitudinal_data=[1,2,3], survival_data=sample_survival_data)
    with pytest.raises(TypeError):
        JointModel(longitudinal_data=pd.DataFrame(), survival_data="not a dataframe")

def test_fit_without_survival_data_raises(sample_longitudinal_data):
    with pytest.raises(ValueError):
        JointModel(longitudinal_data=sample_longitudinal_data, survival_data=None).fit()

def test_fit_without_longitudinal_data_raises(sample_survival_data):
    with pytest.raises(ValueError):
        JointModel(longitudinal_data=None, survival_data=sample_survival_data).fit()

def test_model_can_reset_and_refit(basic_model):
    basic_model.fit()
    basic_model.reset()
    assert not basic_model.is_fitted()
    basic_model.fit()
    assert basic_model.is_fitted()

def test_multiple_calls_to_fit_do_not_crash(basic_model):
    for _ in range(3):
        basic_model.fit()
    assert basic_model.is_fitted()

def test_score_method_after_fit_returns_float(basic_model):
    basic_model.fit()
    score = basic_model.score()
    assert isinstance(score, float)

def test_score_without_fit_raises(basic_model):
    with pytest.raises(RuntimeError):
        basic_model.score()
