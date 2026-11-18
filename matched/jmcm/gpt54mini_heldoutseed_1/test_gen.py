import importlib
import inspect
import sys

import pytest

import jmcm


def test_package_name_constant():
    assert jmcm.name == "jmcm"


def test_jointmodel_is_exported_from_package():
    assert hasattr(jmcm, "JointModel")
    from jmcm import JointModel

    assert JointModel is jmcm.JointModel


def test_jointmodel_is_a_class():
    assert inspect.isclass(jmcm.JointModel)


def test_jointmodel_module_attribute_points_to_joint_model_module():
    assert jmcm.JointModel.__module__ == "jmcm.joint_model"


def test_jointmodel_qualname_contains_class_name():
    assert jmcm.JointModel.__qualname__.split(".")[-1] == "JointModel"


def test_jointmodel_mro_starts_with_class_and_object():
    mro = jmcm.JointModel.mro()
    assert mro[0] is jmcm.JointModel
    assert object in mro


def test_jointmodel_can_be_imported_via_import_module():
    mod = importlib.import_module("jmcm")
    assert mod.JointModel is jmcm.JointModel


def test_jointmodel_is_same_object_on_reload():
    before = jmcm.JointModel
    reloaded = importlib.reload(jmcm)
    assert reloaded.JointModel is before


def test_jointmodel_has_a_docstring_or_none():
    assert hasattr(jmcm.JointModel, "__doc__")


def test_jointmodel_repr_mentions_class_name():
    assert "JointModel" in repr(jmcm.JointModel)


def _can_instantiate_without_required_args(cls):
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return False

    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is inspect._empty:
            return False
    return True


def _instantiate_if_possible():
    cls = jmcm.JointModel
    if not _can_instantiate_without_required_args(cls):
        pytest.skip("JointModel requires mandatory constructor arguments")
    return cls()


def test_jointmodel_instance_type_when_default_constructible():
    obj = _instantiate_if_possible()
    assert isinstance(obj, jmcm.JointModel)


def test_jointmodel_instance_class_matches_export():
    obj = _instantiate_if_possible()
    assert obj.__class__ is jmcm.JointModel


def test_jointmodel_instance_repr_mentions_class_name():
    obj = _instantiate_if_possible()
    assert "JointModel" in repr(obj)


def test_jointmodel_instance_has_no_unexpected_module_attribute():
    obj = _instantiate_if_possible()
    assert obj.__class__.__module__ == "jmcm.joint_model"


def test_jointmodel_public_constructor_signature_has_no_required_args_if_default_constructible():
    cls = jmcm.JointModel
    sig = inspect.signature(cls)
    required = [
        p for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and p.default is inspect._empty
    ]
    if required:
        pytest.skip("JointModel constructor requires arguments")
    assert len(required) == 0
