import importlib
import inspect
import sys

import pytest

import jmcm
from jmcm import JointModel


HARD_DEPENDENCIES = ("math", "numpy", "pandas", "scipy",
                     "collections", "patsy", "warnings")


def _jmcm_module_names():
    return [name for name in sys.modules
            if name == "jmcm" or name.startswith("jmcm.")]


@pytest.fixture
def jmcm_sandbox(monkeypatch):
    """Remove all jmcm modules from sys.modules so the package can be
    re-imported under altered conditions. monkeypatch restores them after
    the test."""
    for name in _jmcm_module_names():
        monkeypatch.delitem(sys.modules, name)
    return monkeypatch


# ---------------------------------------------------------------------------
# Package-level surface
# ---------------------------------------------------------------------------

def test_package_name_is_jmcm():
    assert jmcm.name == "jmcm"


def test_joint_model_is_exported_and_is_a_class():
    assert hasattr(jmcm, "JointModel")
    assert inspect.isclass(JointModel)


def test_exported_joint_model_is_same_object_as_submodule_attribute():
    joint_model_module = importlib.import_module("jmcm.joint_model")
    assert jmcm.JointModel is joint_model_module.JointModel


def test_joint_model_is_defined_in_joint_model_submodule():
    assert JointModel.__module__ == "jmcm.joint_model"


def test_joint_model_inherits_from_object():
    assert JointModel.__mro__[-1] is object


def test_joint_model_init_is_callable():
    assert callable(JointModel.__init__)


def test_joint_model_signature_is_introspectable():
    signature = inspect.signature(JointModel)
    assert len(signature.parameters) >= 0


def test_constructor_rejects_unknown_keyword_argument():
    # Either the signature rejects the unknown keyword, or a required
    # argument is missing; both surface as TypeError.
    with pytest.raises(TypeError):
        JointModel(__definitely_not_a_real_parameter__=1)


def test_reload_preserves_joint_model_identity():
    reloaded = importlib.reload(jmcm)
    assert reloaded.JointModel is JointModel


# ---------------------------------------------------------------------------
# Hard dependency check in jmcm/__init__.py
# ---------------------------------------------------------------------------

def test_dependency_loop_variables_are_removed_from_namespace():
    assert not hasattr(jmcm, "hard_dependencies")
    assert not hasattr(jmcm, "missing_dependencies")
    assert not hasattr(jmcm, "dependency")


def test_all_hard_dependencies_are_importable():
    for dependency in HARD_DEPENDENCIES:
        assert importlib.import_module(dependency) is not None


def test_fresh_import_succeeds_when_dependencies_present(jmcm_sandbox):
    module = importlib.import_module("jmcm")
    assert module.name == "jmcm"
    assert inspect.isclass(module.JointModel)


def test_import_fails_when_scipy_is_unavailable(jmcm_sandbox):
    jmcm_sandbox.setitem(sys.modules, "scipy", None)
    with pytest.raises(ImportError, match="scipy"):
        importlib.import_module("jmcm")


def test_missing_dependency_error_has_expected_header(jmcm_sandbox):
    jmcm_sandbox.setitem(sys.modules, "patsy", None)
    with pytest.raises(ImportError) as exc_info:
        importlib.import_module("jmcm")
    assert str(exc_info.value).startswith(
        "Unable to import required dependencies:"
    )


def test_error_reports_every_missing_dependency(jmcm_sandbox):
    jmcm_sandbox.setitem(sys.modules, "scipy", None)
    jmcm_sandbox.setitem(sys.modules, "patsy", None)
    with pytest.raises(ImportError) as exc_info:
        importlib.import_module("jmcm")
    message = str(exc_info.value)
    assert "scipy" in message
    assert "patsy" in message


def test_joint_model_submodule_not_loaded_when_dependency_missing(jmcm_sandbox):
    jmcm_sandbox.setitem(sys.modules, "scipy", None)
    with pytest.raises(ImportError):
        importlib.import_module("jmcm")
    assert "jmcm.joint_model" not in sys.modules
