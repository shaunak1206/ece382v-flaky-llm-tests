import builtins
import importlib
import inspect
import os
import subprocess
import sys
from unittest import mock

import pytest

import jmcm
from jmcm import JointModel


def _reload_with_blocked(blocked, message="blocked"):
    """Reload jmcm while the given top-level modules fail to import."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError("{0} {1}".format(message, name))
        return real_import(name, *args, **kwargs)

    try:
        with mock.patch.object(builtins, "__import__", fake_import):
            importlib.reload(jmcm)
    finally:
        pass


@pytest.fixture
def restore_jmcm():
    yield
    # A failed reload leaves helper names in the module namespace; reload cleanly.
    importlib.reload(jmcm)


def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_joint_model_is_exported():
    assert hasattr(jmcm, "JointModel")


def test_joint_model_is_a_class():
    assert inspect.isclass(JointModel)


def test_joint_model_defined_in_submodule():
    assert JointModel.__module__ == "jmcm.joint_model"
    assert JointModel.__name__ == "JointModel"


def test_joint_model_same_object_as_in_submodule():
    from jmcm import joint_model
    assert joint_model.JointModel is JointModel
    assert jmcm.JointModel is JointModel


@pytest.mark.parametrize("attr", ["hard_dependencies", "dependency",
                                  "missing_dependencies"])
def test_helper_names_are_cleaned_up(attr):
    assert not hasattr(jmcm, attr)


@pytest.mark.parametrize("dep", ["math", "numpy", "pandas", "scipy",
                                 "collections", "patsy", "warnings"])
def test_hard_dependencies_are_importable(dep):
    assert importlib.import_module(dep) is not None


def test_signature_is_inspectable():
    sig = inspect.signature(JointModel)
    assert isinstance(sig, inspect.Signature)


def test_joint_model_can_be_subclassed():
    class Child(JointModel):
        pass

    assert issubclass(Child, JointModel)
    assert Child.__mro__[1] is JointModel


def test_reload_succeeds_and_keeps_joint_model(restore_jmcm):
    importlib.reload(jmcm)
    assert jmcm.JointModel is JointModel
    assert jmcm.name == "jmcm"
    assert not hasattr(jmcm, "dependency")


def test_missing_dependency_raises_import_error(restore_jmcm):
    with pytest.raises(ImportError) as excinfo:
        _reload_with_blocked({"patsy"})
    msg = str(excinfo.value)
    assert "Unable to import required dependencies" in msg
    assert "patsy" in msg


def test_missing_dependency_message_includes_original_error(restore_jmcm):
    with pytest.raises(ImportError) as excinfo:
        _reload_with_blocked({"scipy"}, message="custom failure for")
    assert "scipy: custom failure for scipy" in str(excinfo.value)


def test_multiple_missing_dependencies_all_reported(restore_jmcm):
    with pytest.raises(ImportError) as excinfo:
        _reload_with_blocked({"numpy", "pandas"})
    msg = str(excinfo.value)
    assert "numpy" in msg
    assert "pandas" in msg
    assert "scipy:" not in msg
    # one line for header plus one line per missing dependency
    assert len(msg.splitlines()) == 3


def test_import_in_fresh_interpreter():
    root = os.path.dirname(os.path.dirname(os.path.abspath(jmcm.__file__)))
    code = ("import jmcm; "
            "print(jmcm.name, jmcm.JointModel.__name__)")
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["jmcm", "JointModel"]
