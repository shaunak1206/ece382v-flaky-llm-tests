import builtins
import importlib
import inspect

import pytest

import jmcm
from jmcm import JointModel


def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_joint_model_is_exported_class():
    assert inspect.isclass(JointModel)
    assert jmcm.JointModel is JointModel


def test_joint_model_defined_in_submodule():
    assert JointModel.__module__ == "jmcm.joint_model"
    assert JointModel.__name__ == "JointModel"


def test_joint_model_importable_from_submodule():
    mod = importlib.import_module("jmcm.joint_model")
    assert mod.JointModel is JointModel


def test_temporary_names_are_cleaned_from_namespace():
    for attr in ("hard_dependencies", "dependency", "missing_dependencies"):
        assert not hasattr(jmcm, attr)


def test_joint_model_has_init_with_signature():
    sig = inspect.signature(JointModel.__init__)
    assert "self" in sig.parameters


def test_joint_model_requires_arguments():
    params = list(inspect.signature(JointModel.__init__).parameters.values())[1:]
    required = [
        p for p in params
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if required:
        with pytest.raises(TypeError):
            JointModel()
    else:
        pytest.skip("constructor has no required arguments")


def test_joint_model_can_be_subclassed():
    class Child(JointModel):
        pass

    assert issubclass(Child, JointModel)
    assert Child.__mro__[1] is JointModel


def test_joint_model_is_not_an_instance_of_itself_as_class():
    assert isinstance(JointModel, type)
    assert not isinstance(JointModel, JointModel)


def test_joint_model_public_methods_are_callable():
    members = inspect.getmembers(JointModel, predicate=inspect.isfunction)
    assert members  # at least __init__
    for _, func in members:
        assert callable(func)


def test_hard_dependencies_are_importable():
    for dep in ("math", "numpy", "pandas", "scipy", "collections",
                "patsy", "warnings"):
        assert importlib.import_module(dep) is not None


def _reload_with_blocked(monkeypatch, blocked):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError("No module named '{}'".format(name))
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
    finally:
        monkeypatch.undo()
        importlib.reload(jmcm)
    return str(excinfo.value)


def test_missing_dependency_raises_import_error(monkeypatch):
    msg = _reload_with_blocked(monkeypatch, {"scipy"})
    assert "Unable to import required dependencies" in msg
    assert "scipy" in msg


def test_multiple_missing_dependencies_all_reported(monkeypatch):
    msg = _reload_with_blocked(monkeypatch, {"patsy", "pandas"})
    assert "patsy" in msg
    assert "pandas" in msg


def test_package_usable_after_reload():
    importlib.reload(jmcm)
    assert jmcm.name == "jmcm"
    assert inspect.isclass(jmcm.JointModel)
