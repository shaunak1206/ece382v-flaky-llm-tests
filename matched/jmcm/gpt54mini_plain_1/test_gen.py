import builtins
import importlib
import sys
import types

import pytest


HARD_DEPENDENCIES = ("math", "numpy", "pandas", "scipy", "collections", "patsy", "warnings")


def _fresh_import_jmcm(monkeypatch, fail_on=None):
    """
    Import jmcm from a clean state. Optionally make importing one dependency fail.
    """
    for mod in list(sys.modules):
        if mod == "jmcm" or mod.startswith("jmcm."):
            sys.modules.pop(mod, None)

    if fail_on is None:
        return importlib.import_module("jmcm")

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == fail_on:
            raise ImportError(f"mocked missing dependency: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("jmcm")
    return str(excinfo.value)


def test_package_import_exposes_jointmodel():
    jmcm = importlib.import_module("jmcm")
    assert hasattr(jmcm, "JointModel")


def test_jointmodel_is_a_class():
    jmcm = importlib.import_module("jmcm")
    assert isinstance(jmcm.JointModel, type)


def test_jointmodel_reexport_matches_submodule_class():
    jmcm = importlib.import_module("jmcm")
    submodule = importlib.import_module("jmcm.joint_model")
    assert jmcm.JointModel is submodule.JointModel


def test_jointmodel_class_name_is_correct():
    jmcm = importlib.import_module("jmcm")
    assert jmcm.JointModel.__name__ == "JointModel"


def test_jointmodel_module_path_is_expected():
    jmcm = importlib.import_module("jmcm")
    assert jmcm.JointModel.__module__ == "jmcm.joint_model"


def test_jmcm_package_name_attribute_is_set():
    jmcm = importlib.import_module("jmcm")
    assert jmcm.name == "jmcm"


@pytest.mark.parametrize("dep", HARD_DEPENDENCIES)
def test_hard_dependencies_are_importable_via_jmcm_module(monkeypatch, dep):
    # Importing jmcm should succeed when all dependencies are available.
    jmcm = _fresh_import_jmcm(monkeypatch)
    assert hasattr(jmcm, "JointModel")


def test_import_raises_clear_error_for_missing_numpy(monkeypatch):
    msg = _fresh_import_jmcm(monkeypatch, fail_on="numpy")
    assert "Unable to import required dependencies" in msg
    assert "numpy: mocked missing dependency: numpy" in msg


def test_import_raises_clear_error_for_missing_pandas(monkeypatch):
    msg = _fresh_import_jmcm(monkeypatch, fail_on="pandas")
    assert "Unable to import required dependencies" in msg
    assert "pandas: mocked missing dependency: pandas" in msg


def test_import_raises_clear_error_for_missing_scipy(monkeypatch):
    msg = _fresh_import_jmcm(monkeypatch, fail_on="scipy")
    assert "Unable to import required dependencies" in msg
    assert "scipy: mocked missing dependency: scipy" in msg


def test_import_raises_clear_error_for_missing_patsy(monkeypatch):
    msg = _fresh_import_jmcm(monkeypatch, fail_on="patsy")
    assert "Unable to import required dependencies" in msg
    assert "patsy: mocked missing dependency: patsy" in msg


def test_import_raises_clear_error_for_missing_math(monkeypatch):
    msg = _fresh_import_jmcm(monkeypatch, fail_on="math")
    assert "Unable to import required dependencies" in msg
    assert "math: mocked missing dependency: math" in msg
