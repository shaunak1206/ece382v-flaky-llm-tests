import builtins
import importlib
import inspect
import sys

import pytest

import jmcm
from jmcm import JointModel


def test_package_name_attribute():
    assert jmcm.name == "jmcm"
    assert jmcm.__name__ == "jmcm"


def test_jointmodel_is_exported_class():
    assert inspect.isclass(JointModel)
    assert jmcm.JointModel is JointModel
    assert "JointModel" in dir(jmcm)


def test_jointmodel_defined_in_submodule():
    assert JointModel.__module__ == "jmcm.joint_model"
    submodule = sys.modules["jmcm.joint_model"]
    assert submodule.JointModel is JointModel


def test_jointmodel_class_name():
    assert JointModel.__name__ == "JointModel"


def test_jointmodel_defines_own_init():
    assert "__init__" in vars(JointModel)
    assert JointModel.__init__ is not object.__init__


def test_init_signature_starts_with_self():
    params = list(inspect.signature(JointModel.__init__).parameters)
    assert params[0] == "self"


def test_calling_without_required_args_raises_type_error():
    params = list(inspect.signature(JointModel.__init__).parameters.values())[1:]
    required = [
        p for p in params
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if not required:
        pytest.skip("JointModel has no required constructor arguments")
    with pytest.raises(TypeError):
        JointModel()


def test_temporary_names_removed_from_namespace():
    for attr in ("hard_dependencies", "dependency", "missing_dependencies"):
        assert not hasattr(jmcm, attr)


def test_hard_dependencies_are_importable():
    for dep in ("math", "numpy", "pandas", "scipy", "collections",
                "patsy", "warnings"):
        assert importlib.import_module(dep) is not None


def test_reload_with_missing_dependency_raises_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "patsy":
            raise ImportError("no patsy here")
        return real_import(name, *args, **kwargs)

    try:
        with monkeypatch.context() as m:
            m.setattr(builtins, "__import__", fake_import)
            with pytest.raises(ImportError) as excinfo:
                importlib.reload(jmcm)
        message = str(excinfo.value)
        assert "Unable to import required dependencies" in message
        assert "patsy: no patsy here" in message
    finally:
        importlib.reload(jmcm)
    assert jmcm.JointModel is JointModel
    assert not hasattr(jmcm, "missing_dependencies")


def test_reload_reports_all_missing_dependencies(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("patsy", "scipy"):
            raise ImportError("missing " + name)
        return real_import(name, *args, **kwargs)

    try:
        with monkeypatch.context() as m:
            m.setattr(builtins, "__import__", fake_import)
            with pytest.raises(ImportError) as excinfo:
                importlib.reload(jmcm)
        message = str(excinfo.value)
        assert "scipy: missing scipy" in message
        assert "patsy: missing patsy" in message
    finally:
        importlib.reload(jmcm)
    assert jmcm.name == "jmcm"


def test_reload_without_problems_keeps_class_identity():
    importlib.reload(jmcm)
    assert jmcm.JointModel is JointModel
    assert jmcm.name == "jmcm"
