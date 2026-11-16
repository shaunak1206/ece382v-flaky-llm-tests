import builtins
import importlib
import inspect

import pytest

import jmcm
from jmcm import JointModel


def _params(obj):
    return list(inspect.signature(obj).parameters.values())


def _required_params(obj):
    return [
        p for p in _params(obj)
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]


def test_package_name():
    assert jmcm.name == "jmcm"


def test_joint_model_is_a_class():
    assert inspect.isclass(JointModel)


def test_joint_model_metadata():
    assert JointModel.__name__ == "JointModel"
    assert JointModel.__module__.startswith("jmcm")


def test_joint_model_is_same_object_as_submodule_attribute():
    from jmcm import joint_model

    assert joint_model.JointModel is JointModel


def test_package_namespace_has_no_leftover_loop_variables():
    for leftover in ("hard_dependencies", "missing_dependencies", "dependency"):
        assert not hasattr(jmcm, leftover)


def test_hard_dependencies_are_importable():
    for dep in ("math", "numpy", "pandas", "scipy", "collections", "patsy", "warnings"):
        assert importlib.import_module(dep) is not None


def test_constructor_signature_is_inspectable():
    params = _params(JointModel)
    for p in params:
        assert isinstance(p.name, str)
        assert p.name != "self"


def test_constructor_requires_its_required_arguments():
    if not _required_params(JointModel):
        pytest.skip("JointModel has no required constructor arguments")
    with pytest.raises(TypeError):
        JointModel()


def test_constructor_rejects_unexpected_keyword():
    if any(p.kind == p.VAR_KEYWORD for p in _params(JointModel)):
        pytest.skip("constructor accepts arbitrary keyword arguments")
    with pytest.raises(TypeError):
        JointModel(__definitely_not_a_parameter__=1)


def test_constructor_rejects_too_many_positional_args():
    params = _params(JointModel)
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        pytest.skip("constructor accepts arbitrary positional arguments")
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    with pytest.raises(TypeError):
        JointModel(*([None] * (len(positional) + 1)))


def test_public_functions_on_class_are_callable():
    public = [
        name for name, value in vars(JointModel).items()
        if not name.startswith("_") and inspect.isfunction(value)
    ]
    for name in public:
        assert callable(getattr(JointModel, name))


def test_fit_method_takes_self_when_present():
    fit = getattr(JointModel, "fit", None)
    if fit is None:
        pytest.skip("JointModel has no fit method")
    params = _params(fit)
    assert params and params[0].name == "self"


def test_reload_keeps_same_joint_model_class():
    reloaded = importlib.reload(jmcm)
    try:
        assert reloaded.JointModel is JointModel
        assert inspect.isclass(reloaded.JointModel)
    finally:
        importlib.reload(jmcm)


def test_missing_single_dependency_raises_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "patsy":
            raise ImportError("simulated missing patsy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(ImportError, match="patsy"):
            importlib.reload(jmcm)
    finally:
        monkeypatch.undo()
        importlib.reload(jmcm)


def test_missing_multiple_dependencies_all_reported(monkeypatch):
    real_import = builtins.__import__
    missing = {"patsy", "scipy"}

    def fake_import(name, *args, **kwargs):
        if name in missing:
            raise ImportError("simulated missing {}".format(name))
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(jmcm)
        message = str(excinfo.value)
        assert "Unable to import required dependencies" in message
        for dep in missing:
            assert dep in message
    finally:
        monkeypatch.undo()
        importlib.reload(jmcm)
