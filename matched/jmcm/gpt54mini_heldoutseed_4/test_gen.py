import builtins
import importlib
import sys

import pytest


def _import_jmcm(monkeypatch, missing=None):
    """
    Import jmcm fresh, optionally simulating a missing hard dependency.
    """
    missing = set(missing or ())
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in missing:
            raise ImportError(f"No module named '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "jmcm", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    return importlib.import_module("jmcm")


def test_import_exposes_module_name(monkeypatch):
    jmcm = _import_jmcm(monkeypatch)
    assert jmcm.name == "jmcm"


def test_import_exports_jointmodel(monkeypatch):
    jmcm = _import_jmcm(monkeypatch)
    assert hasattr(jmcm, "JointModel")


def test_jointmodel_matches_submodule_class(monkeypatch):
    jmcm = _import_jmcm(monkeypatch)
    joint_model_module = importlib.import_module("jmcm.joint_model")
    assert jmcm.JointModel is joint_model_module.JointModel


def test_jointmodel_is_class_like(monkeypatch):
    jmcm = _import_jmcm(monkeypatch)
    assert isinstance(jmcm.JointModel, type)


def test_module_allows_reimport(monkeypatch):
    jmcm1 = _import_jmcm(monkeypatch)
    monkeypatch.delitem(sys.modules, "jmcm", raising=False)
    jmcm2 = importlib.import_module("jmcm")
    assert jmcm1.JointModel is jmcm2.JointModel
    assert jmcm1.name == jmcm2.name == "jmcm"


def test_private_helper_names_not_kept(monkeypatch):
    jmcm = _import_jmcm(monkeypatch)
    assert not hasattr(jmcm, "hard_dependencies")
    assert not hasattr(jmcm, "missing_dependencies")


@pytest.mark.parametrize(
    "missing_dep",
    [
        "math",
        "numpy",
        "pandas",
        "scipy",
        "collections",
        "patsy",
        "warnings",
    ],
)
def test_missing_dependency_raises_importerror(monkeypatch, missing_dep):
    with pytest.raises(ImportError) as excinfo:
        _import_jmcm(monkeypatch, missing=[missing_dep])

    message = str(excinfo.value)
    assert "Unable to import required dependencies" in message
    assert missing_dep in message


def test_missing_multiple_dependencies_reports_all(monkeypatch):
    with pytest.raises(ImportError) as excinfo:
        _import_jmcm(monkeypatch, missing=["numpy", "patsy"])

    message = str(excinfo.value)
    assert "numpy" in message
    assert "patsy" in message
    assert message.count(":") >= 2


def test_missing_dependency_does_not_leak_partial_module(monkeypatch):
    with pytest.raises(ImportError):
        _import_jmcm(monkeypatch, missing=["math"])

    assert "jmcm" not in sys.modules


def test_import_after_failure_still_works(monkeypatch):
    with pytest.raises(ImportError):
        _import_jmcm(monkeypatch, missing=["warnings"])

    monkeypatch.setattr(builtins, "__import__", builtins.__import__)
    monkeypatch.delitem(sys.modules, "jmcm", raising=False)
    jmcm = importlib.import_module("jmcm")
    assert jmcm.name == "jmcm"
    assert hasattr(jmcm, "JointModel")
