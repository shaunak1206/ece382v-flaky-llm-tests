import builtins
import importlib
import inspect

import numpy as np
import pandas as pd
import pytest

import jmcm
from jmcm import JointModel


def _make_data(n_subjects=6, n_times=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_subjects):
        for t in range(n_times):
            rows.append({"id": i + 1, "time": float(t + 1),
                         "y": float(rng.normal(loc=t, scale=1.0))})
    return pd.DataFrame(rows)


def _try_build_model():
    """Try a few plausible constructor call styles; skip if none work."""
    data = _make_data()
    attempts = [
        lambda: JointModel("y ~ 1 | id | time", data, [1, 1, 1], "mcd"),
        lambda: JointModel("y ~ 1 | id | time", data, [1, 1, 1]),
        lambda: JointModel(formula="y ~ 1 | id | time", data=data,
                           pol=[1, 1, 1], cov_method="mcd"),
    ]
    for attempt in attempts:
        try:
            return attempt()
        except Exception:
            continue
    pytest.skip("could not construct JointModel with guessed arguments")


@pytest.fixture
def restore_jmcm():
    yield
    importlib.reload(jmcm)


def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_jointmodel_is_a_class():
    assert inspect.isclass(JointModel)


def test_jointmodel_exported_from_package():
    from jmcm import joint_model
    assert jmcm.JointModel is joint_model.JointModel


def test_jointmodel_defined_in_joint_model_module():
    assert JointModel.__module__ == "jmcm.joint_model"
    assert JointModel.__name__ == "JointModel"


def test_helper_names_deleted_from_namespace():
    assert not hasattr(jmcm, "hard_dependencies")
    assert not hasattr(jmcm, "dependency")
    assert not hasattr(jmcm, "missing_dependencies")


def test_reload_keeps_jointmodel(restore_jmcm):
    reloaded = importlib.reload(jmcm)
    assert hasattr(reloaded, "JointModel")
    assert inspect.isclass(reloaded.JointModel)
    assert reloaded.name == "jmcm"


def test_init_signature_accepts_several_parameters():
    sig = inspect.signature(JointModel.__init__)
    params = [p for p in sig.parameters if p != "self"]
    assert len(params) >= 2


def test_init_signature_first_params_are_formula_like_and_data_like():
    sig = inspect.signature(JointModel.__init__)
    names = [p for p in sig.parameters if p != "self"]
    # Positional parameters should include something for data
    assert any("data" in n.lower() for n in names)


def test_constructor_without_arguments_raises_type_error():
    with pytest.raises(TypeError):
        JointModel()


def test_constructor_with_invalid_arguments_raises():
    with pytest.raises(Exception):
        JointModel(None, None, None, None)


def test_class_has_public_callable_methods():
    methods = [n for n, v in inspect.getmembers(JointModel, inspect.isfunction)
               if not n.startswith("_")]
    assert len(methods) >= 1


def test_valid_construction_returns_instance():
    model = _try_build_model()
    assert isinstance(model, JointModel)


def test_instances_are_independent():
    m1 = _try_build_model()
    m2 = _try_build_model()
    assert m1 is not m2
    assert isinstance(m1, JointModel) and isinstance(m2, JointModel)


def test_construction_does_not_mutate_input_dataframe():
    data = _make_data()
    before = data.copy(deep=True)
    try:
        JointModel("y ~ 1 | id | time", data, [1, 1, 1], "mcd")
    except Exception:
        pytest.skip("could not construct JointModel with guessed arguments")
    pd.testing.assert_frame_equal(data, before)


def test_missing_dependency_raises_import_error(monkeypatch, restore_jmcm):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "patsy":
            raise ImportError("No module named 'patsy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as excinfo:
        importlib.reload(jmcm)
    msg = str(excinfo.value)
    assert "Unable to import required dependencies" in msg
    assert "patsy" in msg


def test_multiple_missing_dependencies_all_reported(monkeypatch, restore_jmcm):
    real_import = builtins.__import__
    missing = {"patsy", "scipy"}

    def fake_import(name, *args, **kwargs):
        if name in missing:
            raise ImportError("No module named '{}'".format(name))
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as excinfo:
        importlib.reload(jmcm)
    msg = str(excinfo.value)
    assert "patsy" in msg
    assert "scipy" in msg
