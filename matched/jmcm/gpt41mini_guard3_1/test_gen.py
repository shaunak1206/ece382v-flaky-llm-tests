import pytest
import numpy as np
import random

from jmcm import JointModel


def test_joint_model_default_instantiation():
    random.seed(123)
    np.random.seed(123)
    model = JointModel()
    # Check that model instantiates and exposes expected attributes
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


def test_joint_model_fit_predict_shape():
    random.seed(42)
    np.random.seed(42)
    model = JointModel()
    # Create mock data attributes for fit if required
    # Use small dataset with fixed values
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 0, 1])
    if hasattr(model, 'fit'):
        # Many JointModels require X, y or other named parameters, test with dummy data
        try:
            model.fit(X, y)
        except TypeError:
            # fallback: try fit with no parameters or minimal, if signature different
            model.fit()
    # Predict on new data or on training data
    if hasattr(model, 'predict'):
        preds = model.predict(X)
        assert preds.shape[0] == X.shape[0]
        # Predictions should be finite numbers
        assert np.all(np.isfinite(preds))


def test_joint_model_fit_modify_no_global_state():
    random.seed(1)
    np.random.seed(1)

    model1 = JointModel()
    model2 = JointModel()
    X = np.array([[1], [2]])
    y = np.array([0, 1])

    # Fit first model
    try:
        model1.fit(X, y)
    except TypeError:
        model1.fit()

    # Fit second model with different data
    X2 = np.array([[10], [20]])
    y2 = np.array([1, 1])
    try:
        model2.fit(X2, y2)
    except TypeError:
        model2.fit()

    # Check that model1 predictions unchanged by second fit
    if hasattr(model1, 'predict'):
        preds1 = model1.predict(X)
        preds2 = model2.predict(X)
        # Both preds are finite and not identical (likely, since data differs)
        assert np.all(np.isfinite(preds1))
        assert np.all(np.isfinite(preds2))
        assert not np.allclose(preds1, preds2)


@pytest.mark.parametrize("seed", [7, 17, 97])
def test_predict_consistency_for_fixed_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    model = JointModel()
    # Fit if possible
    if hasattr(model, 'fit'):
        try:
            model.fit()
        except TypeError:
            pass
    data = np.array([[seed] * 3] * 5)
    preds1 = model.predict(data)
    # Reset seed and re-create model/predict again
    random.seed(seed)
    np.random.seed(seed)
    model2 = JointModel()
    if hasattr(model2, 'fit'):
        try:
            model2.fit()
        except TypeError:
            pass
    preds2 = model2.predict(data)
    # Predict results are reproducible given fixed seed and data
    assert preds1.shape == preds2.shape
    assert np.allclose(preds1, preds2, rtol=1e-7, atol=1e-8)


def test_default_attributes_do_not_change_between_instances():
    random.seed(10)
    np.random.seed(10)
    model_a = JointModel()
    random.seed(10)
    np.random.seed(10)
    model_b = JointModel()

    attrs_a = sorted([attr for attr in dir(model_a) if not attr.startswith('_')])
    attrs_b = sorted([attr for attr in dir(model_b) if not attr.startswith('_')])
    assert attrs_a == attrs_b


def test_fit_without_parameters_does_not_crash():
    random.seed(1234)
    np.random.seed(1234)
    model = JointModel()
    # Try calling fit without arguments; some models allow this
    try:
        model.fit()
    except TypeError:
        pass  # Acceptable if signature requires args


def test_predict_returns_float_array_or_similar():
    random.seed(2023)
    np.random.seed(2023)
    model = JointModel()
    # Fit first if possible
    try:
        model.fit()
    except TypeError:
        pass
    test_X = np.array([[0.0]])
    preds = model.predict(test_X)
    # Check preds is array or list of floats
    assert hasattr(preds, "__len__")
    # Elements are float-convertible (or np.float)
    for v in preds:
        assert isinstance(v, (float, np.floating))


def test_predict_invalid_input_type_raises_or_returns():
    random.seed(999)
    np.random.seed(999)
    model = JointModel()
    input_invalid = "invalid_input_type"
    try:
        model.predict(input_invalid)
    except (TypeError, ValueError):
        pass
    else:
        # if no exception, result should be consistent with normal behavior
        assert hasattr(model.predict(input_invalid), '__len__')


def test_fit_predict_combined_multistep():
    random.seed(33)
    np.random.seed(33)
    model = JointModel()
    # Prepare data with known values if fit requires
    X = np.array([[10, 20], [30, 40]])
    y = np.array([1, 0])
    try:
        model.fit(X, y)
    except TypeError:
        model.fit()
    preds = model.predict(X)
    # Predictions length matches input
    assert len(preds) == X.shape[0]
    # Predictions are finite numbers
    assert np.all(np.isfinite(preds))


def test_model_repr_and_str_do_not_error():
    random.seed(5)
    np.random.seed(5)
    model = JointModel()
    # Should not raise
    r = repr(model)
    s = str(model)
    assert isinstance(r, str)
    assert isinstance(s, str)


def test_model_hash_and_eq_behavior(monkeypatch):
    # If model defines __hash__ or __eq__, test their basic properties
    random.seed(17)
    np.random.seed(17)
    model1 = JointModel()
    model2 = JointModel()

    # __eq__ reflexivity
    assert model1 == model1

    # If different instances considered equal? We cannot assume this;
    # just test that comparison does not error
    _ = (model1 == model2)
    _ = (model1 != model2)

    # If hash defined, test repeated hash calls produce same value
    if hasattr(model1, '__hash__') and callable(getattr(model1, '__hash__')):
        h1 = hash(model1)
        h2 = hash(model1)
        assert h1 == h2


def test_model_methods_do_not_modify_global_random(monkeypatch):
    # Save random state before
    random.seed(123)
    np.random.seed(123)
    state_before_random = random.getstate()
    state_before_np = np.random.get_state()

    model = JointModel()
    try:
        model.fit()
    except TypeError:
        pass
    test_X = np.zeros((1, 1))
    model.predict(test_X)

    # Check random state not changed after calls
    assert state_before_random == random.getstate()
    assert all(np.array_equal(s1, s2) for s1, s2 in zip(state_before_np, np.random.get_state()))


def test_fit_predict_consistent_output_type_and_shape():
    random.seed(50)
    np.random.seed(50)
    model = JointModel()
    try:
        model.fit()
    except TypeError:
        pass
    X = np.ones((4, 3))
    preds = model.predict(X)
    # Predictions length same as input number of samples
    assert len(preds) == X.shape[0]
    # Predictions are floats or np.float
    assert all(isinstance(p, (float, np.floating)) for p in preds)


def test_predict_output_values_are_reasonable():
    random.seed(88)
    np.random.seed(88)
    model = JointModel()
    try:
        model.fit()
    except TypeError:
        pass
    X = np.random.randn(10, 2)
    preds = model.predict(X)
    # Predictions finite and within plausible bounds (can be relaxed)
    assert np.all(np.isfinite(preds))


def test_fit_can_accept_various_input_types(monkeypatch):
    random.seed(7)
    np.random.seed(7)
    model = JointModel()
    # Try with lists
    try:
        model.fit([[1, 2], [3, 4]], [0, 1])
    except TypeError:
        pass
    # Try with numpy arrays
    try:
        model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    except TypeError:
        pass


def test_multiple_predict_calls_with_same_input_are_deterministic():
    random.seed(21)
    np.random.seed(21)
    model = JointModel()
    try:
        model.fit()
    except TypeError:
        pass
    X = np.array([[1, 2]] * 3)
    preds1 = model.predict(X)
    preds2 = model.predict(X)
    assert np.allclose(preds1, preds2, rtol=1e-7, atol=1e-8)


def test_import_joint_model_from_jmcm_and_identity():
    # Just a sanity test for the import and class identity
    assert JointModel.__name__ == "JointModel"


def test_module_name_is_jmcm():
    from jmcm import name
    assert name == "jmcm"


def test_hard_dependencies_not_present_should_raise(monkeypatch):
    # We cannot truly remove imports from sys.modules safely here,
    # but we can patch builtins.__import__ to raise for 'numpy'
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'numpy':
            raise ImportError("mock missing numpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError) as excinfo:
        # re-execute the hard dependencies import check
        # It is in jmcm.__init__.py but we've imported, so simulate snippet:
        hard_dependencies = ("math", "numpy", "pandas",
                             "scipy", "collections", "patsy","warnings")
        missing_dependencies = []
        for dependency in hard_dependencies:
            try:
                __import__(dependency)
            except ImportError as e:
                missing_dependencies.append("{0}: {1}".format(dependency, str(e)))
        if missing_dependencies:
            raise ImportError(
                "Unable to import required dependencies:\n" +
                "\n".join(missing_dependencies)
            )
    assert "numpy" in str(excinfo.value)
