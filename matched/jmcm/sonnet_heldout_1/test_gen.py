import builtins
import importlib
import inspect
import sys

import pytest

import jmcm
from jmcm import JointModel


def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_joint_model_is_exported_class():
    assert inspect.isclass(JointModel)
    assert jmcm.JointModel is JointModel


def test_joint_model_defined_in_submodule():
    assert JointModel.__name__ == "JointModel"
    assert JointModel.__module__ == "jmcm.joint_model"
    assert "jmcm.joint_model" in sys.modules
    assert sys.modules["jmcm.joint_model"].JointModel is JointModel


def test_temporary_names_are_cleaned_up():
    for attr in ("hard_dependencies", "dependency", "missing_dependencies"):
        assert not hasattr(jmcm, attr)


def test_hard_dependencies_are_importable():
    for dep in ("math", "numpy", "pandas", "scipy", "collections",
                "patsy", "warnings"):
        assert importlib.import_module(dep) is not None


def test_joint_model_has_init_and_signature():
    sig = inspect.signature(JointModel)
    assert isinstance(sig, inspect.Signature)
    assert callable(JointModel.__init__)


def test_joint_model_is_new_style_class():
    assert issubclass(JointModel, object)
    assert isinstance(JointModel, type)


def test_joint_model_can_be_subclassed():
    class Child(JointModel):
        pass

    assert issubclass(Child, JointModel)
    obj = object.__new__(Child)
    assert isinstance(obj, JointModel)


def test_joint_model_instantiation_without_required_args_raises():
    sig = inspect.signature(JointModel)
    required = [
        p for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                       p.KEYWORD_ONLY)
    ]
    if not required:
        pytest.skip("JointModel has no required constructor arguments")
    with pytest.raises(TypeError):
        JointModel()


def test_joint_model_rejects_unknown_keyword():
    sig = inspect.signature(JointModel)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        pytest.skip("constructor accepts arbitrary keywords")
    with pytest.raises(TypeError):
        JointModel(this_keyword_does_not_exist_xyz=1)


def test_reload_preserves_joint_model():
    before = jmcm.JointModel
    reloaded = importlib.reload(jmcm)
    assert reloaded.JointModel is before
    assert reloaded.name == "jmcm"
    assert not hasattr(reloaded, "hard_dependencies")


def test_missing_dependency_raises_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy":
            raise ImportError("simulated missing scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(jmcm)
    msg = str(excinfo.value)
    assert "Unable to import required dependencies" in msg
    assert "scipy" in msg
    assert "simulated missing scipy" in msg
    assert jmcm.JointModel is JointModel


def test_multiple_missing_dependencies_are_all_reported(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("patsy", "pandas"):
            raise ImportError("simulated missing " + name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(jmcm)
    msg = str(excinfo.value)
    assert "patsy: simulated missing patsy" in msg
    assert "pandas: simulated missing pandas" in msg
