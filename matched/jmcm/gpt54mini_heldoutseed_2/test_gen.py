import inspect

import jmcm
from jmcm import JointModel
from jmcm.joint_model import JointModel as JointModelFromSubmodule


def test_package_exports_joint_model():
    assert hasattr(jmcm, "JointModel")
    assert jmcm.JointModel is JointModel


def test_joint_model_matches_submodule_export():
    assert JointModel is JointModelFromSubmodule


def test_joint_model_is_a_class():
    assert inspect.isclass(JointModel)


def test_joint_model_is_type_object():
    assert issubclass(JointModel, object)


def test_joint_model_name():
    assert JointModel.__name__ == "JointModel"


def test_joint_model_module():
    assert JointModel.__module__ == "jmcm.joint_model"


def test_joint_model_qualname_contains_class_name():
    assert "JointModel" in JointModel.__qualname__


def test_joint_model_has_init():
    assert hasattr(JointModel, "__init__")
    assert callable(JointModel.__init__)


def test_joint_model_init_signature_is_introspectable():
    sig = inspect.signature(JointModel.__init__)
    params = list(sig.parameters.values())
    assert params, "Expected __init__ to have at least one parameter"
    assert params[0].name == "self"


def test_joint_model_repr_contains_class_name_when_uninitialized():
    instance = object.__new__(JointModel)
    assert "JointModel" in repr(instance)


def test_joint_model_str_contains_class_name_when_uninitialized():
    instance = object.__new__(JointModel)
    assert "JointModel" in str(instance)


def test_joint_model_docstring_is_either_none_or_string():
    assert JointModel.__doc__ is None or isinstance(JointModel.__doc__, str)
