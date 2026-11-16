import builtins
import importlib
import inspect
import sys

import pytest

import jmcm
from jmcm import JointModel
from jmcm import joint_model as joint_model_module

# The jmcm package uses no random number generators, so no seeding is needed.

HARD_DEPENDENCIES = ("math", "numpy", "pandas",
                     "scipy", "collections", "patsy", "warnings")

# Captured once at import time; reading the builtin is not a modification.
_REAL_IMPORT = builtins.__import__


def _make_failing_import(blocked):
    """Return an __import__ replacement that fails for the blocked names."""
    def fake_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError("No module named '{}'".format(name))
        return _REAL_IMPORT(name, *args, **kwargs)
    return fake_import


def _reload_with_blocked(monkeypatch, blocked):
    """Reload jmcm with some dependencies blocked and return the ImportError info.

    The builtin __import__ is replaced only through monkeypatch. The patch is
    undone and the package is reloaded normally in a finally block, so the
    module is restored even if the test fails.
    """
    monkeypatch.setattr(builtins, "__import__", _make_failing_import(blocked))
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
    finally:
        monkeypatch.undo()
        importlib.reload(jmcm)
    return excinfo


def test_package_name_constant():
    assert jmcm.name == "jmcm"


def test_joint_model_is_exported_as_class():
    assert inspect.isclass(JointModel)


def test_joint_model_is_callable():
    assert callable(JointModel)


def test_joint_model_name():
    assert JointModel.__name__ == "JointModel"


def test_joint_model_is_same_object_as_submodule_attribute():
    assert jmcm.JointModel is joint_model_module.JointModel


def test_joint_model_defined_in_joint_model_submodule():
    assert JointModel.__module__ == "jmcm.joint_model"
    assert "jmcm.joint_model" in sys.modules


def test_from_import_matches_package_attribute():
    from jmcm import JointModel as imported_again
    assert imported_again is jmcm.JointModel


def test_joint_model_signature_is_inspectable():
    # Constructor details are not known here; only check introspection works.
    inspect.signature(JointModel)


def test_helper_names_removed_from_package_namespace():
    for leftover in ("hard_dependencies", "missing_dependencies", "dependency"):
        assert not hasattr(jmcm, leftover)


def test_reload_with_all_dependencies_present_keeps_joint_model():
    before = jmcm.JointModel
    importlib.reload(jmcm)
    assert jmcm.JointModel is before
    assert jmcm.JointModel is joint_model_module.JointModel


def test_hard_dependencies_are_installed_in_environment():
    for dep in HARD_DEPENDENCIES:
        importlib.import_module(dep)


def test_single_missing_dependency_raises_import_error(monkeypatch):
    excinfo = _reload_with_blocked(monkeypatch, {"scipy"})
    message = str(excinfo.value)
    assert "scipy: No module named 'scipy'" in message
    assert "numpy:" not in message


def test_error_message_has_expected_header(monkeypatch):
    excinfo = _reload_with_blocked(monkeypatch, {"patsy"})
    assert str(excinfo.value).startswith(
        "Unable to import required dependencies:\n")


def test_all_missing_dependencies_are_listed_in_declared_order(monkeypatch):
    excinfo = _reload_with_blocked(monkeypatch, set(HARD_DEPENDENCIES))
    lines = str(excinfo.value).split("\n")[1:]
    names = [line.split(": ", 1)[0] for line in lines]
    assert names == list(HARD_DEPENDENCIES)


def test_missing_dependency_message_includes_original_error_text(monkeypatch):
    excinfo = _reload_with_blocked(monkeypatch, {"pandas", "math"})
    message = str(excinfo.value)
    assert "pandas: No module named 'pandas'" in message
    assert "math: No module named 'math'" in message


def test_only_blocked_dependencies_are_reported(monkeypatch):
    excinfo = _reload_with_blocked(monkeypatch, {"collections"})
    lines = str(excinfo.value).split("\n")[1:]
    assert len(lines) == 1
    assert lines[0].startswith("collections: ")


def test_builtin_import_restored_after_failed_reload(monkeypatch):
    _reload_with_blocked(monkeypatch, {"scipy"})
    assert builtins.__import__ is _REAL_IMPORT


def test_package_usable_after_failed_reload(monkeypatch):
    _reload_with_blocked(monkeypatch, {"numpy"})
    assert jmcm.name == "jmcm"
    assert jmcm.JointModel is joint_model_module.JointModel
    assert not hasattr(jmcm, "hard_dependencies")
