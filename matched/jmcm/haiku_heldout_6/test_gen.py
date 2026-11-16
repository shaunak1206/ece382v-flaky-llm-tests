import builtins
import importlib
import inspect

import pytest

import jmcm
from jmcm import JointModel


HARD_DEPENDENCIES = ("math", "numpy", "pandas", "scipy",
                     "collections", "patsy", "warnings")


def _blocking_import(blocked):
    """Return an __import__ replacement that fails for the blocked top-level names."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in blocked:
            raise ImportError("No module named '%s'" % name)
        return real_import(name, globals, locals, fromlist, level)

    return fake_import


def _reload_with_blocked(monkeypatch, blocked):
    """Reload jmcm with some dependencies blocked and return the ImportError raised."""
    monkeypatch.setattr(builtins, "__import__", _blocking_import(blocked))
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
    finally:
        monkeypatch.undo()
        importlib.reload(jmcm)
    return excinfo


def test_package_name_is_jmcm():
    assert jmcm.name == "jmcm"


def test_joint_model_is_exported_from_package():
    assert hasattr(jmcm, "JointModel")


def test_joint_model_is_a_class():
    assert isinstance(JointModel, type)


def test_joint_model_is_same_object_as_submodule_attribute():
    from jmcm.joint_model import JointModel as SubmoduleJointModel

    assert jmcm.JointModel is SubmoduleJointModel


def test_joint_model_defined_inside_jmcm_package():
    assert JointModel.__module__.startswith("jmcm")


def test_joint_model_signature_is_inspectable():
    # Should not raise for a well-formed class.
    sig = inspect.signature(JointModel)
    assert sig is not None


def test_joint_model_constructor_rejects_unknown_keyword():
    sig = inspect.signature(JointModel.__init__)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kw:
        pytest.skip("constructor accepts arbitrary keyword arguments")
    with pytest.raises(TypeError):
        JointModel(_this_parameter_does_not_exist=1)


@pytest.mark.parametrize("dependency", HARD_DEPENDENCIES)
def test_hard_dependencies_are_importable(dependency):
    assert __import__(dependency) is not None


def test_helper_variables_removed_from_namespace():
    for leftover in ("hard_dependencies", "dependency", "missing_dependencies"):
        assert not hasattr(jmcm, leftover)


def test_reload_succeeds_when_all_dependencies_present():
    importlib.reload(jmcm)
    assert jmcm.JointModel is JointModel


def test_missing_dependency_raises_import_error(monkeypatch):
    excinfo = _reload_with_blocked(monkeypatch, {"patsy"})
    assert type(excinfo.value) is ImportError


def test_missing_dependency_error_header_message(monkeypatch):
    excinfo = _reload_with_blocked(monkeypatch, {"patsy"})
    assert "Unable to import required dependencies:" in str(excinfo.value)


def test_missing_dependency_error_names_the_module(monkeypatch):
    excinfo = _reload_with_blocked(monkeypatch, {"scipy"})
    assert "scipy" in str(excinfo.value)


def test_missing_dependency_error_lists_all_missing_modules(monkeypatch):
    excinfo = _reload_with_blocked(monkeypatch, {"numpy", "patsy"})
    message = str(excinfo.value)
    assert "numpy" in message
    assert "patsy" in message


def test_module_usable_after_failed_import_is_restored(monkeypatch):
    _reload_with_blocked(monkeypatch, {"pandas"})
    importlib.reload(jmcm)
    assert jmcm.name == "jmcm"
    assert jmcm.JointModel is JointModel
</br>
