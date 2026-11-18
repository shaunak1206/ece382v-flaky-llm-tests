import builtins
import importlib
import sys

import pytest


MODULE_NAME = "jmcm"
DEPENDENCIES = ("math", "numpy", "pandas", "scipy", "collections", "patsy", "warnings")


def _import_fresh_jmcm():
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def _fake_import_raising_for(target_name):
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == target_name:
            raise ImportError(f"mocked missing dependency: {name}")
        return real_import(name, globals, locals, fromlist, level)

    return _import


def test_import_jmcm_exports_jointmodel():
    jmcm = _import_fresh_jmcm()
    assert hasattr(jmcm, "JointModel")
    assert jmcm.JointModel is not None


def test_import_jmcm_sets_name_attribute():
    jmcm = _import_fresh_jmcm()
    assert jmcm.name == "jmcm"


def test_import_jmcm_does_not_leave_dependency_list_defined():
    jmcm = _import_fresh_jmcm()
    assert not hasattr(jmcm, "hard_dependencies")
    assert not hasattr(jmcm, "missing_dependencies")


def test_import_jmcm_jointmodel_is_importable_from_submodule():
    jmcm = _import_fresh_jmcm()
    submodule = importlib.import_module("jmcm.joint_model")
    assert jmcm.JointModel is submodule.JointModel


@pytest.mark.parametrize("missing_dependency", DEPENDENCIES)
def test_import_jmcm_raises_importerror_when_dependency_missing(missing_dependency, monkeypatch):
    sys.modules.pop(MODULE_NAME, None)
    monkeypatch.setattr(builtins, "__import__", _fake_import_raising_for(missing_dependency))

    with pytest.raises(ImportError) as excinfo:
        importlib.import_module(MODULE_NAME)

    message = str(excinfo.value)
    assert "Unable to import required dependencies:" in message
    assert missing_dependency in message


@pytest.mark.parametrize("missing_dependency", DEPENDENCIES)
def test_import_jmcm_cleanup_after_failed_import(missing_dependency, monkeypatch):
    sys.modules.pop(MODULE_NAME, None)
    monkeypatch.setattr(builtins, "__import__", _fake_import_raising_for(missing_dependency))

    with pytest.raises(ImportError):
        importlib.import_module(MODULE_NAME)

    assert MODULE_NAME not in sys.modules


def test_import_jmcm_can_be_reimported_after_successful_import():
    jmcm1 = _import_fresh_jmcm()
    sys.modules.pop(MODULE_NAME, None)
    jmcm2 = importlib.import_module(MODULE_NAME)
    assert jmcm1.JointModel is jmcm2.JointModel


def test_import_jmcm_after_failure_then_success(monkeypatch):
    sys.modules.pop(MODULE_NAME, None)
    monkeypatch.setattr(builtins, "__import__", _fake_import_raising_for("numpy"))
    with pytest.raises(ImportError):
        importlib.import_module(MODULE_NAME)

    monkeypatch.setattr(builtins, "__import__", builtins.__import__)
    sys.modules.pop(MODULE_NAME, None)
    jmcm = importlib.import_module(MODULE_NAME)
    assert hasattr(jmcm, "JointModel")


def test_import_jmcm_jointmodel_is_a_class():
    jmcm = _import_fresh_jmcm()
    assert isinstance(jmcm.JointModel, type)


def test_import_jmcm_jointmodel_module_name_starts_with_package():
    jmcm = _import_fresh_jmcm()
    assert jmcm.JointModel.__module__.startswith("jmcm")
