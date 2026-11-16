import builtins
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import jmcm
from jmcm import JointModel


PROJECT_ROOT = str(Path(jmcm.__file__).resolve().parent.parent)
HARD_DEPENDENCIES = ("math", "numpy", "pandas", "scipy",
                     "collections", "patsy", "warnings")


def _run_import_with_blocked(blocked):
    """Import jmcm in a fresh interpreter where the given top-level
    modules raise ImportError. Returns the CompletedProcess."""
    code = (
        "import builtins\n"
        "_orig = builtins.__import__\n"
        "_blocked = set({blocked!r})\n"
        "def _fake(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in _blocked:\n"
        "        raise ImportError('No module named ' + name)\n"
        "    return _orig(name, *args, **kwargs)\n"
        "builtins.__import__ = _fake\n"
        "import jmcm\n"
    ).format(blocked=sorted(blocked))
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def test_package_name_is_jmcm():
    assert jmcm.name == "jmcm"


def test_joint_model_is_exported_from_package():
    assert hasattr(jmcm, "JointModel")


def test_joint_model_is_a_class():
    assert inspect.isclass(JointModel)
    assert isinstance(JointModel, type)


def test_joint_model_class_name():
    assert JointModel.__name__ == "JointModel"


def test_joint_model_is_new_style_class():
    assert issubclass(JointModel, object)


def test_joint_model_defined_in_joint_model_submodule():
    submodule = importlib.import_module("jmcm.joint_model")
    assert hasattr(submodule, "JointModel")
    assert submodule.JointModel is JointModel


def test_joint_model_module_belongs_to_jmcm_package():
    assert JointModel.__module__.split(".")[0] == "jmcm"


def test_joint_model_signature_is_introspectable():
    sig = inspect.signature(JointModel)
    assert sig is not None


def test_joint_model_has_callable_constructor():
    assert callable(JointModel)
    assert callable(JointModel.__init__)


def test_repeated_import_returns_same_class():
    again = importlib.import_module("jmcm")
    assert again is jmcm
    assert again.JointModel is JointModel


def test_all_hard_dependencies_are_importable():
    for dependency in HARD_DEPENDENCIES:
        assert importlib.import_module(dependency) is not None


def test_loop_helper_names_removed_from_namespace():
    for leftover in ("hard_dependencies", "dependency",
                     "missing_dependencies"):
        assert not hasattr(jmcm, leftover)


def test_import_succeeds_when_all_dependencies_present():
    result = _run_import_with_blocked(set())
    assert result.returncode == 0, result.stderr


def test_missing_dependency_raises_import_error():
    result = _run_import_with_blocked({"patsy"})
    assert result.returncode != 0
    assert "ImportError" in result.stderr


def test_missing_dependency_error_names_the_module():
    result = _run_import_with_blocked({"scipy"})
    assert "Unable to import required dependencies" in result.stderr
    assert "scipy" in result.stderr


def test_all_missing_dependencies_are_reported_together():
    result = _run_import_with_blocked({"numpy", "pandas"})
    assert result.returncode != 0
    assert "numpy" in result.stderr
    assert "pandas" in result.stderr
