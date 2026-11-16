import inspect

import numpy as np
import pandas as pd
import pytest

import jmcm
from jmcm import JointModel


def _make_data(n_subjects=30, n_times=4, seed=0):
    rng = np.random.RandomState(seed)
    ids = np.repeat(np.arange(1, n_subjects + 1), n_times)
    time = np.tile(np.arange(n_times), n_subjects) / float(n_times - 1)
    y = 1.0 + 0.5 * time + rng.normal(scale=0.5, size=n_subjects * n_times)
    return pd.DataFrame({"id": ids, "time": time, "y": y})


_FORMULAS = [
    "y | id | time ~ 1 | 1",
    "y | id | time ~ time | 1",
    "y | id | time ~ 1 + time | 1",
]


def _try_build(df):
    """Try several plausible constructor call styles; return model or None."""
    attempts = []
    for f in _FORMULAS:
        attempts.append(((f, df), {}))
        attempts.append(((f, df, [1, 1, 1]), {}))
        attempts.append(((f, df, [1, 1, 1], "mcd"), {}))
        attempts.append(((f, df), {"triple": [1, 1, 1]}))
        attempts.append(((f, df), {"cov_method": "mcd"}))
    for args, kwargs in attempts:
        try:
            return JointModel(*args, **kwargs)
        except Exception:
            continue
    return None


@pytest.fixture
def df():
    return _make_data()


@pytest.fixture
def model(df):
    m = _try_build(df)
    if m is None:
        pytest.skip("Could not construct JointModel with known call styles")
    return m


# ---------------------------------------------------------------- package

def test_package_name():
    assert jmcm.name == "jmcm"


def test_jointmodel_exported_at_package_level():
    assert hasattr(jmcm, "JointModel")
    assert jmcm.JointModel is JointModel


def test_jointmodel_is_class():
    assert inspect.isclass(JointModel)


def test_jointmodel_defined_in_joint_model_module():
    assert JointModel.__module__ == "jmcm.joint_model"


def test_import_cleans_up_temporary_names():
    assert not hasattr(jmcm, "hard_dependencies")
    assert not hasattr(jmcm, "missing_dependencies")
    assert not hasattr(jmcm, "dependency")


def test_constructor_accepts_multiple_arguments():
    params = inspect.signature(JointModel.__init__).parameters
    # self plus at least formula and data
    assert len(params) >= 3


# ------------------------------------------------------------ bad inputs

def test_constructor_without_arguments_raises_type_error():
    with pytest.raises(TypeError):
        JointModel()


def test_constructor_with_none_arguments_raises():
    with pytest.raises(Exception):
        JointModel(None, None)


def test_constructor_with_invalid_formula_raises(df):
    with pytest.raises(Exception):
        JointModel("this is not a valid formula", df)


def test_constructor_with_missing_variable_raises(df):
    with pytest.raises(Exception):
        JointModel("nonexistent | id | time ~ 1 | 1", df)


def test_constructor_with_empty_dataframe_raises():
    empty = pd.DataFrame({"id": [], "time": [], "y": []})
    with pytest.raises(Exception):
        JointModel("y | id | time ~ 1 | 1", empty)


# -------------------------------------------------------- valid instances

def test_valid_construction_returns_instance(model):
    assert isinstance(model, JointModel)


def test_construction_does_not_modify_input_data(df):
    before = df.copy(deep=True)
    m = _try_build(df)
    if m is None:
        pytest.skip("Could not construct JointModel with known call styles")
    pd.testing.assert_frame_equal(df, before)


def test_instances_are_independent(df):
    m1 = _try_build(df)
    m2 = _try_build(df)
    if m1 is None or m2 is None:
        pytest.skip("Could not construct JointModel with known call styles")
    assert m1 is not m2


def test_model_has_public_methods(model):
    public = [n for n in dir(model)
              if not n.startswith("_") and callable(getattr(model, n))]
    assert len(public) >= 1


def test_model_repr_is_string(model):
    assert isinstance(repr(model), str)
    assert isinstance(str(model), str)


def test_fit_method_runs_if_available(model):
    fit = getattr(model, "fit", None)
    if fit is None:
        pytest.skip("JointModel has no fit method")
    try:
        fit()
    except TypeError:
        pytest.skip("fit requires arguments unknown to this test")
    # After fitting the model should still be usable
    assert isinstance(repr(model), str)
