import builtins
import importlib
import inspect
import sys

import pytest


PACKAGE = "jmcm"


def _fresh_import_jmcm(monkeypatch, blocked=()):
    """Import the jmcm package from scratch, optionally making some
    top-level modules fail to import. All changes are undone by monkeypatch."""
    for key in [k for k in sys.modules if k == PACKAGE or k.startswith(PACKAGE + ".")]:
        monkeypatch.delitem(sys.modules, key)

    real_import = builtins.__import__
    blocked = set(blocked)

    def fake_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError("No module named '{}'".format(name))
        return real_import(name, *args, **kwargs)

    if blocked:
        monkeypatch.setattr(builtins, "__import__", fake_import)

    return importlib.import_module(PACKAGE)


def test_package_name_is_jmcm():
    import jmcm

    assert jmcm.name == "jmcm"


def test_joint_model_is_importable_from_package():
    from jmcm import JointModel

    assert JointModel is not None


def test_joint_model_is_a_class():
    from jmcm import JointModel

    assert inspect.isclass(JointModel)


def test_joint_model_is_reexported_from_submodule():
    import jmcm
    joint_model_module = importlib.import_module("jmcm.joint_model")

    assert jmcm.JointModel is joint_model_module.JointModel


def test_joint_model_module_belongs_to_package():
    from jmcm import JointModel

    assert JointModel.__module__.startswith(PACKAGE)


def test_joint_model_is_new_style_class():
    from jmcm import JointModel

    assert type(JointModel) is type


def test_joint_model_has_object_in_mro():
    from jmcm import JointModel

    assert object in JointModel.__mro__


def test_joint_model_listed_in_package_namespace():
    import jmcm

    assert "JointModel" in dir(jmcm)


def test_import_helper_globals_are_removed():
    import jmcm

    for leftover in ("hard_dependencies", "missing_dependencies", "dependency"):
        assert not hasattr(jmcm, leftover), leftover


def test_import_succeeds_when_dependencies_present(monkeypatch):
    jmcm = _fresh_import_jmcm(monkeypatch, blocked=())

    assert hasattr(jmcm, "JointModel")


@pytest.mark.parametrize("missing", ["math", "numpy", "pandas", "scipy", "patsy", "warnings"])
def test_missing_dependency_raises_import_error(monkeypatch, missing):
    with pytest.raises(ImportError) as excinfo:
        _fresh_import_jmcm(monkeypatch, blocked=(missing,))

    assert missing in str(excinfo.value)


def test_missing_dependency_error_has_header(monkeypatch):
    with pytest.raises(ImportError) as excinfo:
        _fresh_import_jmcm(monkeypatch, blocked=("scipy",))

    assert str(excinfo.value).startswith("Unable to import required dependencies:")


def test_missing_dependency_error_uses_name_prefix_format(monkeypatch):
    with pytest.raises(ImportError) as excinfo:
        _fresh_import_jmcm(monkeypatch, blocked=("patsy",))

    assert "patsy: " in str(excinfo.value)


def test_multiple_missing_dependencies_are_all_reported(monkeypatch):
    with pytest.raises(ImportError) as excinfo:
        _fresh_import_jmcm(monkeypatch, blocked=("numpy", "pandas"))

    message = str(excinfo.value)
    assert "numpy" in message
    assert "pandas" in message
