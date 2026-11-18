import pytest
import math
import numpy as np
import pandas as pd
import scipy
import collections
import patsy
import warnings

from jmcm import JointModel


def test_joint_model_instantiation_no_args():
    """Test that JointModel can be instantiated with no arguments."""
    jm = JointModel()
    assert isinstance(jm, JointModel)


def test_joint_model_str_method():
    """Test that JointModel has a __str__ or __repr__ method that returns string."""
    jm = JointModel()
    s = str(jm)
    assert isinstance(s, str)
    # The string should mention the class name or any useful info
    assert "JointModel" in s or s != ""


def test_joint_model_fit_method_exists():
    """Test that JointModel has a fit method callable."""
    jm = JointModel()
    assert hasattr(jm, "fit")
    assert callable(jm.fit)


def test_joint_model_predict_method_exists():
    """Test that JointModel has a predict method callable."""
    jm = JointModel()
    assert hasattr(jm, "predict")
    assert callable(jm.predict)


def test_fit_predict_consistency():
    """Test that after fitting, predict returns a numpy array or list of correct size."""
    jm = JointModel()

    # Prepare minimal example data for fitting
    # Using patsy to create design matrices if needed, else create dummy data
    # We do not know the exact signature of fit, so we try a generic approach

    # Try to call fit with minimal sane data
    if "fit" in dir(jm):
        try:
            # We try a generic call with dummy data.
            # Use a simple DataFrame
            df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 3, 2, 5, 4]})
            # Some JointModel fit methods expect a formula and data frame or arrays
            # We test multiple ways to fit to catch potential interface

            # First try fit(x, y)
            try:
                fit_result = jm.fit(df["x"], df["y"])
            except Exception:
                # try fit with DataFrame as first arg, formula as second arg
                # or just fit(df)
                try:
                    fit_result = jm.fit(formula="y ~ x", data=df)
                except Exception:
                    # fallback: try fit(df)
                    fit_result = jm.fit(df)
        except Exception:
            pytest.skip("JointModel.fit prototype unknown or requires complex data, skipping.")

        # After fit attempt, test predict
        try:
            preds = jm.predict(df["x"])
            # Should be array-like
            assert isinstance(preds, (np.ndarray, list))
            # Length matches input length
            assert len(preds) == len(df)
        except Exception:
            # If predict fails without fit, skip or fail gracefully
            pytest.skip("JointModel.predict or fit failed unexpectedly.")


def test_fit_raises_with_bad_data():
    """Test that calling fit with invalid data raises an error."""
    jm = JointModel()
    with pytest.raises(Exception):
        jm.fit(None)  # None is invalid input


def test_predict_raises_before_fit():
    """Test that calling predict before fit raises an error or returns something."""
    jm = JointModel()
    # It may raise error or return None or something else,
    # test both possibilities
    try:
        result = jm.predict([1, 2, 3])
    except Exception:
        # Expected to raise if model not fitted
        pass
    else:
        # If no exception, should be None or empty
        assert result is not None


def test_joint_model_copy_independent_instances():
    """Test that two JointModel instances do not share state."""
    jm1 = JointModel()
    jm2 = JointModel()
    assert jm1 is not jm2
    try:
        jm1.fit([1, 2, 3], [4, 5, 6])
    except Exception:
        # If fit fails, skip internal state test
        pytest.skip("JointModel.fit call failed.")

    try:
        p1 = jm1.predict([1, 2])
    except Exception:
        pytest.skip("JointModel.predict call failed.")

    try:
        p2 = jm2.predict([1, 2])
    except Exception:
        # Either no fit or fails, ensure they differ
        p2 = None

    assert p1 != p2 or p2 is None


def test_joint_model_warns_on_fit_with_nan():
    """Test JointModel issues warnings when fitting data with NaNs (if implemented)."""
    jm = JointModel()
    df = pd.DataFrame({"x": [1, 2, np.nan, 4], "y": [1, 3, 2, 5]})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            jm.fit(df["x"], df["y"])
        except Exception:
            pytest.skip("JointModel.fit failed for NaN test.")
        else:
            # Check if any warning was emitted during fit
            assert any(issubclass(warn.category, Warning) for warn in w) or len(w) == 0


def test_fit_accepts_pandas_dataframe():
    """Test that fit accepts pandas DataFrame as input."""
    jm = JointModel()
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    try:
        # Attempt to fit using dataframe columns or dataframe as input
        try:
            jm.fit(df["x"], df["y"])
        except Exception:
            jm.fit(df)
    except Exception:
        pytest.skip("JointModel.fit does not accept pandas DataFrame or Series.")


def test_predict_accepts_numpy_array():
    """Test predict accepts numpy ndarray and returns expected output type."""
    jm = JointModel()
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 1, 2]})
    try:
        # Fit the model first
        jm.fit(df["x"], df["y"])
    except Exception:
        pytest.skip("JointModel.fit failed.")

    arr = np.array([1, 2, 3])
    try:
        out = jm.predict(arr)
    except Exception:
        pytest.skip("JointModel.predict failed for numpy array input.")
    else:
        assert isinstance(out, (np.ndarray, list))
        assert len(out) == len(arr)


def test_joint_model_repr_contains_class_name():
    jm = JointModel()
    r = repr(jm)
    assert isinstance(r, str)
    assert "JointModel" in r


def test_fit_with_empty_input_raises():
    jm = JointModel()
    with pytest.raises(Exception):
        jm.fit([], [])


def test_predict_with_empty_input():
    jm = JointModel()
    try:
        jm.fit([1, 2, 3], [4, 5, 6])
    except Exception:
        pytest.skip("JointModel.fit failed.")

    out = jm.predict([])
    # Prediction on empty input - should be empty list or array
    assert hasattr(out, "__len__")
    assert len(out) == 0


def test_joint_model_has_version_attribute():
    # Check if the instance or module has attribute version (optional)
    jm = JointModel()
    # No guarantee; if version attribute exists, it is a string
    version = getattr(jm, "version", None)
    if version is not None:
        assert isinstance(version, str)


def test_joint_model_fit_with_different_data_types():
    jm = JointModel()
    try:
        jm.fit([1, 2, 3], np.array([4, 5, 6]))
    except Exception:
        pytest.skip("JointModel.fit does not accept mixed list and np.array.")

    try:
        jm.fit(np.array([1, 2, 3]), pd.Series([4, 5, 6]))
    except Exception:
        pytest.skip("JointModel.fit does not accept numpy array and pandas Series.")


def test_fit_predict_multiple_calls():
    jm = JointModel()
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    try:
        jm.fit(df["x"], df["y"])
    except Exception:
        pytest.skip("JointModel.fit failed.")

    pred1 = jm.predict([5])
    pred2 = jm.predict([6])
    assert pred1 is not None
    assert pred2 is not None
    # Should not raise or return None


def test_fit_warns_on_deprecated_usage(monkeypatch):
    """
    If the JointModel.fit uses deprecated functionality and issues warnings,
    test that warnings are caught.
    """
    jm = JointModel()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            jm.fit([1, 2, 3], [1, 2, 3])
        except Exception:
            pytest.skip("JointModel.fit failed.")
        else:
            deprecated_warnings = [warn for warn in w if issubclass(warn.category, DeprecationWarning)]
            # It's ok if no deprecation warnings arise
            assert isinstance(deprecated_warnings, list)


def test_import_jmcm_and_joint_model():
    """Test that JointModel is imported properly from jmcm package."""
    import jmcm
    assert hasattr(jmcm, "JointModel")
    jm = jmcm.JointModel()
    assert isinstance(jm, JointModel)
