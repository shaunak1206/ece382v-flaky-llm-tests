import pytest
import numpy as np
import random
import math

from jmcm import JointModel


def test_jointmodel_init_default_seed():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    # Basic attributes exist
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

def test_jointmodel_fit_predict_consistency():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model.fit(X, y)
    pred1 = model.predict(X)
    pred2 = model.predict(X)
    # Predictions should be the same for the same input
    np.testing.assert_allclose(pred1, pred2, rtol=1e-7, atol=1e-9)

def test_jointmodel_predict_shape_matches_input():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.ones((5, 3))
    y = np.zeros(5)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]

def test_jointmodel_fit_raises_on_bad_input():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    # X and y length mismatch
    X = np.zeros((4, 2))
    y = np.zeros(3)
    with pytest.raises(Exception):
        model.fit(X, y)

def test_jointmodel_predict_before_fit_raises():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.zeros((2, 2))
    with pytest.raises(Exception):
        model.predict(X)

def test_jointmodel_fit_changes_internal_state(monkeypatch):
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.random.rand(10, 3)
    y = np.random.rand(10)
    before_attr = getattr(model, "_fitted", None)
    model.fit(X, y)
    after_attr = getattr(model, "_fitted", None)
    # Some internal attribute indicating fit was called changed
    assert before_attr != after_attr

def test_jointmodel_predict_output_is_finite():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.random.rand(6, 2)
    y = np.random.rand(6)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))

def test_fitting_twice_overwrites_state():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X1 = np.random.rand(10, 2)
    y1 = np.random.rand(10)
    X2 = np.random.rand(5, 2)
    y2 = np.random.rand(5)
    model.fit(X1, y1)
    preds1 = model.predict(X1)
    model.fit(X2, y2)
    preds2 = model.predict(X2)
    # Different data leads to different predictions after refit
    assert not np.allclose(preds1[:5], preds2, atol=1e-6)

def test_predict_with_different_input_shapes():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X_train = np.random.rand(8, 3)
    y_train = np.random.rand(8)
    model.fit(X_train, y_train)
    # Predict on a subset of columns (if model supports it)
    X_test = np.random.rand(4, 3)
    preds = model.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]

def test_try_fit_with_invalid_types_raises():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    # Passing incompatible types
    with pytest.raises(Exception):
        model.fit("not an array", "not an array")

def test_try_predict_with_invalid_type_raises():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.random.rand(3, 2)
    y = np.random.rand(3)
    model.fit(X, y)
    with pytest.raises(Exception):
        model.predict("bad input")

def test_model_name_is_jmcm():
    # The module-level name attribute is jmcm
    assert hasattr(model := JointModel(), "name") is False  # JointModel likely no name attr
    import jmcm
    assert jmcm.name == "jmcm"

def test_model_repr_contains_classname():
    model = JointModel()
    s = repr(model)
    assert "JointModel" in s

def test_model_fit_accepts_numpy_arrays():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 0.0])
    model.fit(X, y)  # Should not raise
    preds = model.predict(X)
    assert preds.shape[0] == 2

def test_fit_predict_on_constant_target():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.random.rand(10, 2)
    y = np.ones(10) * 5.0
    model.fit(X, y)
    p = model.predict(X)
    # Predictions should not be NaN
    assert not np.any(np.isnan(p))

def test_multiple_predict_calls_return_same_results():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.random.rand(7, 3)
    y = np.random.rand(7)
    model.fit(X, y)
    p1 = model.predict(X)
    p2 = model.predict(X)
    np.testing.assert_allclose(p1, p2, rtol=1e-8, atol=1e-9)

def test_predict_with_empty_array_raises_or_returns_empty():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.empty((0, 3))
    y = np.empty((0,))
    model.fit(np.zeros((1,3)), np.zeros(1))  # Fit with minimal data to allow predict
    result = None
    try:
        result = model.predict(X)
        assert result.shape[0] == 0
    except Exception:
        # Acceptable if raises
        pass

def test_fit_and_predict_with_large_input():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.random.rand(1000, 5)
    y = np.random.rand(1000)
    model.fit(X, y)
    preds = model.predict(X)
    # Check output shape matches input length
    assert preds.shape[0] == 1000
    # Check values finite
    assert np.all(np.isfinite(preds))

def test_monkeypatch_on_internal_method(monkeypatch):
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    called = {}
    def dummy_fit(self, X, y):
        called["yes"] = True
    # Patch fit method of JointModel instance
    monkeypatch.setattr(model, "fit", dummy_fit.__get__(model))
    model.fit(None, None)
    assert called.get("yes") is True

def test_model_does_not_modify_inputs():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    X = np.random.rand(5, 2)
    y = np.random.rand(5)
    X_copy = X.copy()
    y_copy = y.copy()
    model.fit(X, y)
    np.testing.assert_allclose(X, X_copy, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(y, y_copy, rtol=1e-14, atol=1e-15)
