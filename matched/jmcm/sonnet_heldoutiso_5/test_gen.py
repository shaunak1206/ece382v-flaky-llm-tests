import builtins
import importlib
import inspect
import sys

import pytest

import jmcm
from jmcm import JointModel


def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_jointmodel_is_exposed_in_package():
    assert hasattr(jmcm, "JointModel")


def test_jointmodel_is_a_class():
    assert inspect.isclass(JointModel)


def test_jointmodel_is_same_object_as_in_submodule():
    submodule = importlib.import_module("jmcm.joint_model")
    assert jmcm.JointModel is submodule.JointModel


def test_jointmodel_class_name():
    assert JointModel.__name__ == "JointModel"


def test_jointmodel_defined_in_joint_model_module():
    assert JointModel.__module__ == "jmcm.joint_model"


def test_jointmodel_submodule_registered_in_sys_modules():
    assert "jmcm.joint_model" in sys.modules


def test_temporary_dependency_names_are_cleaned_up():
    for attr in ("hard_dependencies", "missing_dependencies", "dependency"):
        assert not hasattr(jmcm, attr)


def test_hard_dependencies_are_importable():
    for dep in ("math", "numpy", "pandas", "scipy", "collections",
                "patsy", "warnings"):
        assert importlib.import_module(dep) is not None


def test_jointmodel_signature_is_introspectable():
    sig = inspect.signature(JointModel)
    assert isinstance(sig, inspect.Signature)


def test_jointmodel_has_callable_init():
    assert callable(JointModel.__init__)


def test_jointmodel_defines_public_members():
    public = [n for n in vars(JointModel) if not n.startswith("_")]
    assert len(public) > 0


def test_jointmodel_can_be_subclassed():
    class Sub(JointModel):
        pass

    assert issubclass(Sub, JointModel)


def test_reload_keeps_jointmodel_available():
    reloaded = importlib.reload(jmcm)
    assert reloaded.JointModel is JointModel or inspect.isclass(reloaded.JointModel)
    assert reloaded.name == "jmcm"


def test_missing_dependency_raises_importerror(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "patsy":
            raise ImportError("simulated missing patsy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
        message = str(excinfo.value)
        assert "Unable to import required dependencies" in message
        assert "patsy" in message
        assert "simulated missing patsy" in message
    finally:
        monkeypatch.undo()
        importlib.reload(jmcm)

    assert not hasattr(jmcm, "missing_dependencies")
    assert inspect.isclass(jmcm.JointModel)
