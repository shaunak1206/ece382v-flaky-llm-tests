import builtins
import importlib
import inspect
import os

import pytest

import jmcm
import jmcm.joint_model as joint_model_module


_INIT_PATH = os.path.join(os.path.dirname(jmcm.__file__), "__init__.py")

_HARD_DEPENDENCIES = ("math", "numpy", "pandas",
                      "scipy", "collections", "patsy", "warnings")


def _read_init_source():
    with open(_INIT_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def _exec_init_in_fresh_namespace():
    """Run a copy of jmcm/__init__.py in a new namespace.

    The package's own module object is left untouched. The relative import
    ``from .joint_model import JointModel`` resolves through ``__package__``,
    and ``jmcm.joint_model`` is already cached in ``sys.modules``.
    """
    namespace = {
        "__name__": "jmcm_init_copy",
        "__package__": "jmcm",
        "__file__": _INIT_PATH,
    }
    code = compile(_read_init_source(), _INIT_PATH, "exec")
    exec(code, namespace)
    return namespace


def _make_import_blocker(blocked_names):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in blocked_names:
            raise ImportError("No module named '%s'" % name)
        return real_import(name, *args, **kwargs)

    return fake_import


def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_joint_model_exposed_on_package():
    assert hasattr(jmcm, "JointModel")


def test_joint_model_is_same_object_as_submodule_class():
    assert jmcm.JointModel is joint_model_module.JointModel


def test_joint_model_is_a_class():
    assert inspect.isclass(jmcm.JointModel)


def test_joint_model_defined_in_joint_model_module():
    assert jmcm.JointModel.__module__ == "jmcm.joint_model"


def test_joint_model_listed_in_package_namespace():
    assert "JointModel" in dir(jmcm)


def test_helper_names_removed_from_package_namespace():
    for leftover in ("hard_dependencies", "missing_dependencies", "dependency"):
        assert leftover not in vars(jmcm)


def test_init_source_executes_cleanly_in_fresh_namespace():
    namespace = _exec_init_in_fresh_namespace()
    assert namespace["name"] == "jmcm"
    assert "JointModel" in namespace


def test_fresh_exec_exposes_same_joint_model_class():
    namespace = _exec_init_in_fresh_namespace()
    assert namespace["JointModel"] is joint_model_module.JointModel


def test_fresh_exec_removes_helper_names():
    namespace = _exec_init_in_fresh_namespace()
    for leftover in ("hard_dependencies", "missing_dependencies", "dependency"):
        assert leftover not in namespace


def test_fresh_exec_does_not_modify_package_namespace():
    before = set(vars(jmcm).keys())
    _exec_init_in_fresh_namespace()
    after = set(vars(jmcm).keys())
    assert before == after


@pytest.mark.parametrize("dependency_name", _HARD_DEPENDENCIES)
def test_each_hard_dependency_is_importable(dependency_name):
    module = importlib.import_module(dependency_name)
    assert module.__name__ == dependency_name


def test_missing_dependency_raises_import_error(monkeypatch):
    monkeypatch.setattr(builtins, "__import__",
                        _make_import_blocker({"scipy"}))
    with pytest.raises(ImportError):
        _exec_init_in_fresh_namespace()


def test_missing_dependency_error_has_header(monkeypatch):
    monkeypatch.setattr(builtins, "__import__",
                        _make_import_blocker({"scipy"}))
    with pytest.raises(ImportError) as excinfo:
        _exec_init_in_fresh_namespace()
    assert str(excinfo.value).startswith(
        "Unable to import required dependencies:\n")


def test_missing_dependency_message_names_module(monkeypatch):
    monkeypatch.setattr(builtins, "__import__",
                        _make_import_blocker({"scipy"}))
    with pytest.raises(ImportError) as excinfo:
        _exec_init_in_fresh_namespace()
    assert "scipy: No module named 'scipy'" in str(excinfo.value)


def test_multiple_missing_dependencies_all_reported_in_declared_order(monkeypatch):
    monkeypatch.setattr(builtins, "__import__",
                        _make_import_blocker({"scipy", "patsy"}))
    with pytest.raises(ImportError) as excinfo:
        _exec_init_in_fresh_namespace()
    message = str(excinfo.value)
    scipy_position = message.find("scipy:")
    patsy_position = message.find("patsy:")
    assert scipy_position != -1
    assert patsy_position != -1
    assert scipy_position < patsy_position


def test_unrelated_blocked_module_does_not_raise(monkeypatch):
    monkeypatch.setattr(builtins, "__import__",
                        _make_import_blocker({"not_a_real_dependency"}))
    namespace = _exec_init_in_fresh_namespace()
    assert namespace["JointModel"] is joint_model_module.JointModel
