import builtins
import importlib
import inspect
import sys

import pytest


def _drop_jmcm_modules():
    for name in list(sys.modules):
        if name == "jmcm" or name.startswith("jmcm."):
            sys.modules.pop(name, None)


def _import_fresh_jmcm():
    _drop_jmcm_modules()
    return importlib.import_module("jmcm")


def test_import_jmcm_succeeds():
    mod = _import_fresh_jmcm()
    assert mod is sys.modules["jmcm"]


def test_module_name_variable_is_set():
    mod = _import_fresh_jmcm()
    assert mod.name == "jmcm"


def test_jointmodel_is_exported_from_package():
    mod = _import_fresh_jmcm()
    assert hasattr(mod, "JointModel")
    assert mod.JointModel is not None


def test_jointmodel_is_a_class():
    mod = _import_fresh_jmcm()
    assert inspect.isclass(mod.JointModel)


def test_from_import_matches_package_attribute():
    mod = _import_fresh_jmcm()
    from jmcm import JointModel  # noqa: WPS433

    assert JointModel is mod.JointModel


def test_jointmodel_origin_is_jmcm_joint_model_module():
    mod = _import_fresh_jmcm()
    assert mod.JointModel.__module__ == "jmcm.joint_model"


def test_imported_jointmodel_has_a_name_attribute():
    mod = _import_fresh_jmcm()
    assert mod.JointModel.__name__ == "JointModel"


def test_package_does_not_leave_dependency_list_globals_around():
    mod = _import_fresh_jmcm()
    assert not hasattr(mod, "hard_dependencies")
    assert not hasattr(mod, "missing_dependencies")
    assert not hasattr(mod, "dependency")


def test_reimport_after_removal_gives_access_to_jointmodel_again():
    mod1 = _import_fresh_jmcm()
    _drop_jmcm_modules()
    mod2 = importlib.import_module("jmcm")
    assert mod1.JointModel.__name__ == mod2.JointModel.__name__
    assert mod2.JointModel is not None


def test_import_raises_if_a_hard_dependency_is_missing(monkeypatch):
    _drop_jmcm_modules()
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            raise ImportError("mock missing numpy")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("jmcm")

    message = str(excinfo.value)
    assert "Unable to import required dependencies" in message
    assert "numpy: mock missing numpy" in message


def test_import_raises_if_math_is_missing(monkeypatch):
    _drop_jmcm_modules()
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "math":
            raise ImportError("no math available")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("jmcm")

    assert "math: no math available" in str(excinfo.value)


def test_import_raises_if_patsy_is_missing(monkeypatch):
    _drop_jmcm_modules()
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "patsy":
            raise ImportError("patsy unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("jmcm")

    assert "patsy: patsy unavailable" in str(excinfo.value)
