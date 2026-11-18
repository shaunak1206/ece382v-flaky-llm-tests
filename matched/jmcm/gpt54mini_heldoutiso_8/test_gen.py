import importlib
import inspect

import pytest

import jmcm


def test_package_exports_jointmodel():
    assert hasattr(jmcm, "JointModel")
    assert jmcm.JointModel is not None


def test_jointmodel_is_a_class():
    assert inspect.isclass(jmcm.JointModel)


def test_jointmodel_module_is_joint_model_module():
    assert jmcm.JointModel.__module__ == "jmcm.joint_model"


def test_jointmodel_name_matches_class_name():
    assert jmcm.JointModel.__name__ == "JointModel"


def test_jointmodel_can_be_imported_from_package():
    from jmcm import JointModel

    assert JointModel is jmcm.JointModel


def test_jointmodel_matches_module_attribute():
    joint_model_module = importlib.import_module("jmcm.joint_model")
    assert getattr(joint_model_module, "JointModel") is jmcm.JointModel


def test_jointmodel_is_documented_or_has_docstring_attribute():
    # The class may or may not have a meaningful docstring, but it should expose the attribute.
    assert hasattr(jmcm.JointModel, "__doc__")


def test_jointmodel_signature_is_introspectable():
    sig = inspect.signature(jmcm.JointModel)
    assert isinstance(sig, inspect.Signature)


def test_jointmodel_signature_parameters_are_named():
    sig = inspect.signature(jmcm.JointModel)
    for name, param in sig.parameters.items():
        assert isinstance(name, str)
        assert isinstance(param, inspect.Parameter)


def test_jointmodel_class_has_a_mro():
    mro = jmcm.JointModel.mro()
    assert jmcm.JointModel in mro
    assert object in mro


def test_jointmodel_repr_contains_class_name():
    rep = repr(jmcm.JointModel)
    assert "JointModel" in rep


def test_jointmodel_accessible_through_sys_modules_reload():
    # Re-importing the top-level package should preserve the same class object.
    reloaded = importlib.reload(jmcm)
    assert reloaded.JointModel is jmcm.JointModel
