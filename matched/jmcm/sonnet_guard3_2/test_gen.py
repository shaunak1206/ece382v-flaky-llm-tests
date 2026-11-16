import builtins
import importlib
import inspect
import sys

import pytest

import jmcm
from jmcm import JointModel


def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_jointmodel_is_exported_class():
    assert inspect.isclass(JointModel)
    assert JointModel.__name__ == "JointModel"


def test_jointmodel_in_package_namespace():
    assert "JointModel" in dir(jmcm)
    assert jmcm.JointModel is JointModel


def test_jointmodel_defined_in_joint_model_submodule():
    submodule = importlib.import_module("jmcm.joint_model")
    assert submodule.JointModel is JointModel
    assert JointModel.__module__ == "jmcm.joint_model"


def test_package_registered_in_sys_modules():
    assert sys.modules["jmcm"] is jmcm
    assert "jmcm.joint_model" in sys.modules


def test_temporary_names_are_cleaned_up():
    assert not hasattr(jmcm, "hard_dependencies")
    assert not hasattr(jmcm, "missing_dependencies")
    assert not hasattr(jmcm, "dependency")


def test_hard_dependencies_are_importable():
    for dep in ("math", "numpy", "pandas", "scipy", "collections",
                "patsy", "warnings"):
        assert importlib.import_module(dep) is not None


def test_jointmodel_init_is_callable_with_self():
    assert callable(JointModel)
    params = list(inspect.signature(JointModel.__init__).parameters)
    assert params[0] == "self"


def test_jointmodel_instance_without_init():
    obj = object.__new__(JointModel)
    assert isinstance(obj, JointModel)
    assert type(obj) is JointModel


def test_jointmodel_can_be_subclassed():
    class Sub(JointModel):
        pass

    assert issubclass(Sub, JointModel)
    assert Sub.__mro__[1] is JointModel


def test_jointmodel_methods_are_callable_attributes():
    public = [n for n, v in vars(JointModel).items()
              if not n.startswith("_") and callable(v)]
    for n in public:
        assert callable(getattr(JointModel, n))


def test_reimport_missing_dependency_raises(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "patsy":
            raise ImportError("No module named 'patsy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
    finally:
        monkeypatch.undo()
        importlib.reload(jmcm)
    msg = str(excinfo.value)
    assert "Unable to import required dependencies" in msg
    assert "patsy" in msg
    # state restored after the failed reload
    assert not hasattr(jmcm, "hard_dependencies")
    assert jmcm.JointModel is JointModel


def test_reimport_reports_all_missing_dependencies(monkeypatch):
    real_import = builtins.__import__
    missing = ("patsy", "scipy")

    def fake_import(name, *args, **kwargs):
        if name in missing:
            raise ImportError("cannot import " + name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
    finally:
        monkeypatch.undo()
        importlib.reload(jmcm)
    msg = str(excinfo.value)
    for m in missing:
        assert "{0}: cannot import {0}".format(m) in msg


def test_reload_succeeds_and_keeps_same_class():
    reloaded = importlib.reload(jmcm)
    assert reloaded is jmcm
    assert jmcm.JointModel is JointModel
    assert jmcm.name == "jmcm"
