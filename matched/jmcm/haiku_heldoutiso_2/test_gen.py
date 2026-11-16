import builtins
import importlib
import inspect
import sys

import pytest

import jmcm
from jmcm import JointModel


HARD_DEPENDENCIES = ("math", "numpy", "pandas", "scipy",
                     "collections", "patsy", "warnings")


def _purge_jmcm_modules(monkeypatch):
    """Remove jmcm and its submodules from sys.modules (restored on teardown)."""
    for mod_name in list(sys.modules):
        if mod_name == "jmcm" or mod_name.startswith("jmcm."):
            monkeypatch.delitem(sys.modules, mod_name)


def _import_jmcm_with_blocked(monkeypatch, blocked):
    """Fresh-import jmcm while making the given top-level module names unimportable."""
    _purge_jmcm_modules(monkeypatch)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError("blocked for test: {0}".format(name))
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return importlib.import_module("jmcm")


# ---------------------------------------------------------------------------
# Package-level exposure of JointModel
# ---------------------------------------------------------------------------

def test_jointmodel_is_importable_from_package():
    assert JointModel is not None


def test_jointmodel_matches_submodule_definition():
    from jmcm.joint_model import JointModel as SubmoduleJointModel
    assert jmcm.JointModel is SubmoduleJointModel


def test_jointmodel_is_a_class():
    assert isinstance(JointModel, type)


def test_jointmodel_defined_in_jmcm_namespace():
    assert JointModel.__module__.startswith("jmcm")


def test_jointmodel_listed_in_package_attributes():
    assert "JointModel" in dir(jmcm)


def test_jointmodel_signature_is_introspectable():
    sig = inspect.signature(JointModel)
    assert isinstance(sig.parameters, dict) or hasattr(sig, "parameters")


def test_jointmodel_is_stable_across_repeated_imports():
    first = jmcm.JointModel
    second = importlib.import_module("jmcm").JointModel
    assert first is second


# ---------------------------------------------------------------------------
# Package metadata and namespace hygiene
# ---------------------------------------------------------------------------

def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_package_exposes_joint_model_submodule():
    submodule = importlib.import_module("jmcm.joint_model")
    assert hasattr(submodule, "JointModel")


def test_helper_names_are_cleaned_from_namespace():
    for leaked in ("hard_dependencies", "missing_dependencies", "dependency"):
        assert not hasattr(jmcm, leaked), "{0} leaked into jmcm namespace".format(leaked)


def test_hard_dependencies_are_importable_in_environment():
    for dep in HARD_DEPENDENCIES:
        importlib.import_module(dep)


# ---------------------------------------------------------------------------
# Missing-dependency behaviour of the package __init__
# ---------------------------------------------------------------------------

def test_missing_dependency_raises_import_error(monkeypatch):
    with pytest.raises(ImportError) as excinfo:
        _import_jmcm_with_blocked(monkeypatch, {"scipy"})
    assert "Unable to import required dependencies" in str(excinfo.value)


def test_missing_dependency_message_names_the_module(monkeypatch):
    with pytest.raises(ImportError) as excinfo:
        _import_jmcm_with_blocked(monkeypatch, {"patsy"})
    message = str(excinfo.value)
    assert "patsy" in message
    assert "blocked for test: patsy" in message


def test_all_missing_dependencies_are_reported(monkeypatch):
    blocked = {"numpy", "pandas"}
    with pytest.raises(ImportError) as excinfo:
        _import_jmcm_with_blocked(monkeypatch, blocked)
    message = str(excinfo.value)
    for name in blocked:
        assert name in message


def test_missing_dependency_prevents_joint_model_import(monkeypatch):
    with pytest.raises(ImportError):
        _import_jmcm_with_blocked(monkeypatch, {"math"})
    assert "jmcm" not in sys.modules


def test_no_error_when_nothing_is_blocked(monkeypatch):
    module = _import_jmcm_with_blocked(monkeypatch, set())
    assert module.JointModel is not None
    assert module.name == "jmcm"
