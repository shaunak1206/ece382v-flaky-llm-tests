import builtins
import importlib
import importlib.util
import inspect
import sys

import pytest

import jmcm
from jmcm.joint_model import JointModel as DirectJointModel


HARD_DEPENDENCIES = ("math", "numpy", "pandas", "scipy",
                     "collections", "patsy", "warnings")


def _import_jmcm_with_blocked(monkeypatch, blocked):
    """Re-import the jmcm package while making the named top-level
    modules fail to import. All changes are undone by monkeypatch."""
    for key in list(sys.modules):
        if key == "jmcm" or key.startswith("jmcm."):
            monkeypatch.delitem(sys.modules, key)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in blocked:
            raise ImportError("blocked for test: {}".format(name))
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return importlib.import_module("jmcm")


def test_package_exposes_joint_model():
    assert hasattr(jmcm, "JointModel")


def test_joint_model_is_a_class():
    assert inspect.isclass(jmcm.JointModel)


def test_joint_model_class_name():
    assert jmcm.JointModel.__name__ == "JointModel"


def test_package_joint_model_is_same_object_as_submodule_class():
    assert jmcm.JointModel is DirectJointModel


def test_joint_model_defined_in_joint_model_module():
    assert jmcm.JointModel.__module__.endswith("joint_model")


def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_package_does_not_leak_import_helpers():
    for leaked in ("hard_dependencies", "missing_dependencies", "dependency"):
        assert not hasattr(jmcm, leaked)


def test_joint_model_is_type_object():
    assert isinstance(jmcm.JointModel, type)
    assert issubclass(jmcm.JointModel, object)


def test_joint_model_mro_ends_with_object():
    mro = jmcm.JointModel.__mro__
    assert mro[0] is jmcm.JointModel
    assert mro[-1] is object


def test_joint_model_has_init_with_self_parameter():
    params = list(inspect.signature(jmcm.JointModel.__init__).parameters)
    assert params[0] == "self"


def test_joint_model_class_namespace_is_accessible():
    namespace = vars(jmcm.JointModel)
    assert "__init__" in namespace
    assert namespace["__module__"] == jmcm.JointModel.__module__


def test_joint_model_dir_is_list_of_strings():
    names = dir(jmcm.JointModel)
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)
    assert "__init__" in names


def test_joint_model_is_hashable_and_identity_stable():
    assert hash(jmcm.JointModel) == hash(jmcm.JointModel)
    assert importlib.import_module("jmcm").JointModel is jmcm.JointModel


def test_all_hard_dependencies_are_installed():
    missing = [dep for dep in HARD_DEPENDENCIES
               if importlib.util.find_spec(dep) is None]
    assert missing == []


def test_import_raises_import_error_when_one_dependency_missing(monkeypatch):
    with pytest.raises(ImportError) as excinfo:
        _import_jmcm_with_blocked(monkeypatch, {"patsy"})
    assert "patsy" in str(excinfo.value)


def test_import_error_message_has_expected_header(monkeypatch):
    with pytest.raises(ImportError) as excinfo:
        _import_jmcm_with_blocked(monkeypatch, {"scipy"})
    assert str(excinfo.value).startswith(
        "Unable to import required dependencies:")


def test_import_error_lists_all_missing_dependencies(monkeypatch):
    with pytest.raises(ImportError) as excinfo:
        _import_jmcm_with_blocked(monkeypatch, {"scipy", "pandas"})
    message = str(excinfo.value)
    assert "scipy" in message
    assert "pandas" in message


def test_import_error_message_has_one_line_per_missing_dependency(monkeypatch):
    with pytest.raises(ImportError) as excinfo:
        _import_jmcm_with_blocked(monkeypatch, {"numpy", "patsy"})
    lines = str(excinfo.value).split("\n")
    dependency_lines = [line for line in lines if line.startswith(
        ("numpy:", "patsy:"))]
    assert len(dependency_lines) == 2
</br>
