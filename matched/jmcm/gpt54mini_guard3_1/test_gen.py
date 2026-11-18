import builtins
import importlib
import sys

import pytest

import jmcm


def _reload_jmcm_with_missing_dependency(monkeypatch, missing_name):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == missing_name:
            raise ImportError(f"forced missing dependency: {missing_name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as excinfo:
        importlib.reload(jmcm)
    # Restore the real import machinery and reload again so later tests see a healthy module.
    monkeypatch.setattr(builtins, "__import__", original_import)
    importlib.reload(jmcm)
    return str(excinfo.value)


def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_jointmodel_is_exported_from_package():
    assert hasattr(jmcm, "JointModel")


def test_jointmodel_is_class():
    assert isinstance(jmcm.JointModel, type)


def test_jointmodel_reexport_matches_submodule():
    from jmcm.joint_model import JointModel as submodule_joint_model

    assert jmcm.JointModel is submodule_joint_model


def test_jointmodel_qualname_looks_correct():
    assert jmcm.JointModel.__name__ == "JointModel"
    assert jmcm.JointModel.__module__ == "jmcm.joint_model"


def test_package_dir_contains_jointmodel():
    assert "JointModel" in dir(jmcm)


def test_internal_dependency_names_not_left_in_namespace():
    assert not hasattr(jmcm, "hard_dependencies")
    assert not hasattr(jmcm, "missing_dependencies")
    assert not hasattr(jmcm, "dependency")


def test_import_jmcm_from_package_namespace():
    module = importlib.import_module("jmcm")
    assert module is jmcm
    assert module.JointModel is jmcm.JointModel


def test_missing_math_dependency_raises_importerror(monkeypatch):
    msg = _reload_jmcm_with_missing_dependency(monkeypatch, "math")
    assert "math" in msg
    assert "Unable to import required dependencies" in msg


def test_missing_numpy_dependency_raises_importerror(monkeypatch):
    msg = _reload_jmcm_with_missing_dependency(monkeypatch, "numpy")
    assert "numpy" in msg
    assert "Unable to import required dependencies" in msg


def test_missing_pandas_dependency_raises_importerror(monkeypatch):
    msg = _reload_jmcm_with_missing_dependency(monkeypatch, "pandas")
    assert "pandas" in msg
    assert "Unable to import required dependencies" in msg


def test_missing_scipy_dependency_raises_importerror(monkeypatch):
    msg = _reload_jmcm_with_missing_dependency(monkeypatch, "scipy")
    assert "scipy" in msg
    assert "Unable to import required dependencies" in msg


def test_missing_collections_dependency_raises_importerror(monkeypatch):
    msg = _reload_jmcm_with_missing_dependency(monkeypatch, "collections")
    assert "collections" in msg
    assert "Unable to import required dependencies" in msg


def test_missing_patsy_dependency_raises_importerror(monkeypatch):
    msg = _reload_jmcm_with_missing_dependency(monkeypatch, "patsy")
    assert "patsy" in msg
    assert "Unable to import required dependencies" in msg


def test_missing_warnings_dependency_raises_importerror(monkeypatch):
    msg = _reload_jmcm_with_missing_dependency(monkeypatch, "warnings")
    assert "warnings" in msg
    assert "Unable to import required dependencies" in msg
