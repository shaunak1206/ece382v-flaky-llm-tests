import pytest
import numpy as np
import pandas as pd
from jmcm.joint_model import JointModel


# Since no implementation details of JointModel are provided,
# these tests assume some common typical interfaces and behavior
# expected from a generic "JointModel" class:
# - It can be instantiated,
# - It has fit and predict methods,
# - It may raise errors on bad input,
# - It handles numpy/pandas data inputs,
# - It has attributes storing fit results,
# - It supports resetting or re-fitting.

# Adjust the test details based on actual JointModel methods and behavior.


def test_joint_model_import():
    # Confirm the import works correctly
    jm = JointModel()
    assert jm is not None


def test_joint_model_fit_accepts_numpy_arrays():
    jm = JointModel()
    # example data: two features, 10 samples
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    try:
        jm.fit(X, y)
    except Exception as e:
        pytest.fail(f"fit raised an unexpected exception with numpy arrays: {e}")


def test_joint_model_fit_accepts_pandas_dataframes():
    jm = JointModel()
    df_X = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
    df_y = pd.Series(np.random.rand(10))
    try:
        jm.fit(df_X, df_y)
    except Exception as e:
        pytest.fail(f"fit raised an unexpected exception with pandas inputs: {e}")


def test_joint_model_predict_returns_array_after_fit():
    jm = JointModel()
    X_train = np.random.rand(10, 2)
    y_train = np.random.rand(10)
    jm.fit(X_train, y_train)
    X_test = np.random.rand(5, 2)
    preds = jm.predict(X_test)
    assert isinstance(preds, np.ndarray) or isinstance(preds, list)
    assert len(preds) == X_test.shape[0]


def test_predict_without_fit_raises():
    jm = JointModel()
    X_test = np.random.rand(5, 2)
    with pytest.raises(Exception):
        jm.predict(X_test)


def test_fit_with_mismatched_input_lengths_raises():
    jm = JointModel()
    X = np.random.rand(10, 2)
    y = np.random.rand(9)  # length mismatch
    with pytest.raises(ValueError):
        jm.fit(X, y)


def test_fit_input_type_validation():
    jm = JointModel()
    X = "not an array"
    y = np.random.rand(10)
    with pytest.raises(TypeError):
        jm.fit(X, y)


def test_predict_input_type_validation():
    jm = JointModel()
    jm.fit(np.random.rand(10, 2), np.random.rand(10))
    X_test = "not an array"
    with pytest.raises(TypeError):
        jm.predict(X_test)


def test_model_attributes_after_fit():
    jm = JointModel()
    X = np.random.rand(8, 3)
    y = np.random.rand(8)
    jm.fit(X, y)
    # Assuming the model sets some attribute like coef_ or similar
    # If none exist, this test should be adapted
    assert hasattr(jm, "coef_") or hasattr(jm, "params") or hasattr(jm, "fitted")


def test_fit_twice_overwrites_previous_fit():
    jm = JointModel()
    X1 = np.random.rand(10, 2)
    y1 = np.random.rand(10)
    jm.fit(X1, y1)
    first_fit_attrs = jm.__dict__.copy()
    X2 = np.random.rand(15, 2)
    y2 = np.random.rand(15)
    jm.fit(X2, y2)
    second_fit_attrs = jm.__dict__.copy()
    assert first_fit_attrs != second_fit_attrs


def test_fit_handles_missing_values():
    jm = JointModel()
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        jm.fit(X, y)


def test_predict_shape_matches_expected():
    jm = JointModel()
    X_train = np.random.rand(12, 4)
    y_train = np.random.rand(12)
    jm.fit(X_train, y_train)
    X_pred = np.random.rand(7, 4)
    preds = jm.predict(X_pred)
    assert len(preds) == 7


def test_fit_with_empty_data_raises():
    jm = JointModel()
    X = np.empty((0, 3))
    y = np.empty((0,))
    with pytest.raises(ValueError):
        jm.fit(X, y)


def test_predict_with_empty_data_raises():
    jm = JointModel()
    X_train = np.random.rand(10, 2)
    y_train = np.random.rand(10)
    jm.fit(X_train, y_train)
    X_empty = np.empty((0, 2))
    with pytest.raises(ValueError):
        jm.predict(X_empty)


def test_fit_and_predict_with_single_sample():
    jm = JointModel()
    X = np.random.rand(1, 3)
    y = np.random.rand(1)
    jm.fit(X, y)
    preds = jm.predict(X)
    assert len(preds) == 1


def test_fit_with_non_numeric_data_raises():
    jm = JointModel()
    X = np.array([["a", "b"], ["c", "d"]])
    y = np.array([1, 2])
    with pytest.raises(ValueError):
        jm.fit(X, y)


def test_predict_with_incorrect_feature_dim_raises():
    jm = JointModel()
    X_train = np.random.rand(10, 3)
    y_train = np.random.rand(10)
    jm.fit(X_train, y_train)
    X_test = np.random.rand(5, 2)  # fewer features than training
    with pytest.raises(ValueError):
        jm.predict(X_test)


def test_model_str_and_repr_do_not_error():
    jm = JointModel()
    s = str(jm)
    r = repr(jm)
    assert isinstance(s, str)
    assert isinstance(r, str)


def test_fit_accepts_list_inputs_as_data():
    jm = JointModel()
    X = [[1,2],[3,4],[5,6],[7,8],[9,10]]
    y = [1,2,3,4,5]
    try:
        jm.fit(X, y)
    except Exception as e:
        pytest.fail(f"fit raised exception with list inputs: {e}")


def test_predict_accepts_list_input():
    jm = JointModel()
    X_train = np.random.rand(10, 2)
    y_train = np.random.rand(10)
    jm.fit(X_train, y_train)
    X_test = [[0.1, 0.2], [0.3, 0.4]]
    preds = jm.predict(X_test)
    assert len(preds) == len(X_test)


def test_fit_predict_consistency():
    jm = JointModel()
    np.random.seed(0)
    X = np.random.rand(20, 3)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2] * 0.1
    jm.fit(X, y)
    preds = jm.predict(X)
    # We expect predictions to be close to y (not necessarily exact)
    assert np.mean((preds - y) ** 2) < 1.0  # arbitrary reasonable threshold


def test_predict_multiple_calls_consistent():
    jm = JointModel()
    X_train = np.random.rand(30, 2)
    y_train = np.random.rand(30)
    jm.fit(X_train, y_train)
    X_test = np.random.rand(5, 2)
    p1 = jm.predict(X_test)
    p2 = jm.predict(X_test)
    assert np.allclose(p1, p2)

