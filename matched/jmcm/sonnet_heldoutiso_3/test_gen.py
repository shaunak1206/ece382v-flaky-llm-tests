import builtins
import importlib
import inspect
import sys

import pytest

import jmcm
from jmcm import JointModel


def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_jointmodel_is_exported_from_package():
    assert hasattr(jmcm, "JointModel")
    assert jmcm.JointModel is JointModel


def test_jointmodel_is_a_class():
    assert inspect.isclass(JointModel)


def test_jointmodel_defined_in_joint_model_submodule():
    assert JointModel.__module__ == "jmcm.joint_model"
    submodule = importlib.import_module("jmcm.joint_model")
    assert submodule.JointModel is JointModel


def test_submodule_registered_in_sys_modules():
    assert "jmcm" in sys.modules
    assert "jmcm.joint_model" in sys.modules


def test_cleanup_of_temporary_names():
    for tmp in ("hard_dependencies", "dependency", "missing_dependencies"):
        assert not hasattr(jmcm, tmp)


@pytest.mark.parametrize(
    "dep", ["math", "numpy", "pandas", "scipy", "collections", "patsy", "warnings"]
)
def test_hard_dependencies_are_importable(dep):
    assert importlib.import_module(dep) is not None
    assert dep in sys.modules


def test_jointmodel_signature_is_introspectable():
    sig = inspect.signature(JointModel)
    assert isinstance(sig, inspect.Signature)


def test_jointmodel_call_without_required_args_raises_type_error():
    sig = inspect.signature(JointModel)
    required = [
        p for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if not required:
        pytest.skip("JointModel has no required constructor arguments")
    with pytest.raises(TypeError):
        JointModel()


def test_jointmodel_can_be_subclassed():
    class Sub(JointModel):
        pass

    assert issubclass(Sub, JointModel)
    assert Sub.__mro__[1] is JointModel


def test_jointmodel_has_init_and_public_callable_members():
    assert callable(getattr(JointModel, "__init__"))
    public = [
        n for n, v in inspect.getmembers(JointModel)
        if not n.startswith("_") and callable(v)
    ]
    assert len(public) >= 0  # introspection works without errors


def _blocking_import(blocked):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError("No module named '{0}'".format(name))
        return real_import(name, *args, **kwargs)

    return fake_import


def test_missing_dependency_raises_import_error_on_reload(monkeypatch):
    monkeypatch.setattr(builtins, "__import__", _blocking_import({"patsy"}))
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
        msg = str(excinfo.value)
        assert "Unable to import required dependencies" in msg
        assert "patsy" in msg
    finally:
        monkeypatch.undo()
        importlib.reload(jmcm)


def test_all_missing_dependencies_are_reported(monkeypatch):
    monkeypatch.setattr(
        builtins, "__import__", _blocking_import({"scipy", "patsy"})
    )
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
        msg = str(excinfo.value)
        assert "scipy" in msg
        assert "patsy" in msg
    finally:
        monkeypatch.undo()
        importlib.reload(jmcm)


def test_reload_restores_normal_state():
    importlib.reload(jmcm)
    assert jmcm.name == "jmcm"
    assert inspect.isclass(jmcm.JointModel)
    assert not hasattr(jmcm, "missing_dependencies")
