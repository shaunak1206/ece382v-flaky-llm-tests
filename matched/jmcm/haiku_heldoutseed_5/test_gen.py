import importlib
import inspect
import sys

import pytest

import jmcm
from jmcm import JointModel
import jmcm.joint_model as joint_model_module


def _clear_jmcm_modules(monkeypatch):
    """Remove jmcm and its submodules from sys.modules so they can be re-imported.

    monkeypatch restores the original entries on teardown.
    """
    for key in list(sys.modules):
        if key == "jmcm" or key.startswith("jmcm."):
            monkeypatch.delitem(sys.modules, key)


def _fresh_import_jmcm(monkeypatch, blocked=()):
    """Re-import the jmcm package, optionally making some dependencies unimportable."""
    _clear_jmcm_modules(monkeypatch)
    for name in blocked:
        # Setting a sys.modules entry to None makes a later import raise ImportError.
        monkeypatch.setitem(sys.modules, name, None)
    return importlib.import_module("jmcm")


# ---------------------------------------------------------------------------
# Package-level behaviour of jmcm/__init__.py
# ---------------------------------------------------------------------------

def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_package_imports_cleanly_with_all_dependencies(monkeypatch):
    fresh = _fresh_import_jmcm(monkeypatch)
    assert fresh.name == "jmcm"
    assert hasattr(fresh, "JointModel")


def test_dependency_bookkeeping_names_are_deleted(monkeypatch):
    fresh = _fresh_import_jmcm(monkeypatch)
    for leftover in ("hard_dependencies", "dependency", "missing_dependencies"):
        assert not hasattr(fresh, leftover)


@pytest.mark.parametrize("blocked", ["numpy", "pandas", "scipy", "patsy"])
def test_missing_dependency_raises_import_error(monkeypatch, blocked):
    with pytest.raises(ImportError, match=blocked):
        _fresh_import_jmcm(monkeypatch, blocked=(blocked,))


def test_missing_multiple_dependencies_all_reported(monkeypatch):
    with pytest.raises(ImportError) as excinfo:
        _fresh_import_jmcm(monkeypatch, blocked=("numpy", "patsy"))
    message = str(excinfo.value)
    assert "Unable to import required dependencies" in message
    assert "numpy" in message
    assert "patsy" in message


def test_missing_dependency_error_message_header(monkeypatch):
    with pytest.raises(ImportError, match="Unable to import required dependencies"):
        _fresh_import_jmcm(monkeypatch, blocked=("scipy",))


# ---------------------------------------------------------------------------
# JointModel exposure and class-level properties
# ---------------------------------------------------------------------------

def test_joint_model_is_exported_from_package():
    assert hasattr(jmcm, "JointModel")
    assert jmcm.JointModel is JointModel


def test_joint_model_same_object_as_submodule_definition():
    assert JointModel is joint_model_module.JointModel


def test_joint_model_is_a_class():
    assert inspect.isclass(JointModel)
    assert isinstance(JointModel, type)


def test_joint_model_name_is_correct():
    assert JointModel.__name__ == "JointModel"


def test_joint_model_defined_inside_jmcm_package():
    assert JointModel.__module__.startswith("jmcm")


def test_joint_model_has_callable_init():
    assert callable(JointModel.__init__)


def test_joint_model_can_be_subclassed():
    class _Subclass(JointModel):
        pass

    assert issubclass(_Subclass, JointModel)
    assert _Subclass.__name__ == "_Subclass"


def test_joint_model_mro_ends_with_object():
    assert JointModel.__mro__[-1] is object
    assert JointModel in JointModel.__mro__


def test_joint_model_is_hashable_and_comparable_to_itself():
    assert hash(JointModel) == hash(JointModel)
    assert JointModel == jmcm.JointModel
    assert JointModel != int
