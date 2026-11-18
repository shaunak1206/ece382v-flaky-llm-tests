import builtins
import importlib
import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

import jmcm


PACKAGE_DIR = pathlib.Path(jmcm.__file__).resolve().parent
INIT_FILE = PACKAGE_DIR / "__init__.py"


def _load_package_as(name, import_hook=None):
    """
    Load jmcm/__init__.py under a temporary package name.

    This allows testing the import-time dependency check without disturbing the
    real jmcm import already present in sys.modules.
    """
    original_import = builtins.__import__

    def wrapped_import(module_name, *args, **kwargs):
        if import_hook is not None:
            import_hook(module_name)
        return original_import(module_name, *args, **kwargs)

    if import_hook is not None:
        builtins.__import__ = wrapped_import

    try:
        spec = importlib.util.spec_from_file_location(
            name,
            str(INIT_FILE),
            submodule_search_locations=[str(PACKAGE_DIR)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if import_hook is not None:
            builtins.__import__ = original_import
        sys.modules.pop(name, None)


def test_package_has_jointmodel_attribute():
    assert hasattr(jmcm, "JointModel")


def test_jointmodel_is_class():
    assert isinstance(jmcm.JointModel, type)


def test_jointmodel_import_from_package_matches_attribute():
    from jmcm import JointModel

    assert JointModel is jmcm.JointModel


def test_jointmodel_module_is_joint_model_submodule():
    assert jmcm.JointModel.__module__ == "jmcm.joint_model"


def test_jointmodel_name_is_jointmodel():
    assert jmcm.JointModel.__name__ == "JointModel"


def test_package_allows_direct_class_instantiation_without_arguments():
    # This verifies the exported symbol is the actual class object and can be constructed.
    obj = jmcm.JointModel()
    assert isinstance(obj, jmcm.JointModel)


def test_package_module_name():
    assert jmcm.name == "jmcm"


def test_dependency_names_are_not_left_in_namespace():
    for attr in ("hard_dependencies", "missing_dependencies", "dependency"):
        assert not hasattr(jmcm, attr)


def test_importing_real_package_twice_keeps_same_class_object():
    first = jmcm.JointModel
    reloaded = importlib.reload(jmcm)
    assert reloaded.JointModel is first


def test_imported_class_is_same_as_submodule_class():
    from jmcm.joint_model import JointModel as SubmoduleJointModel

    assert SubmoduleJointModel is jmcm.JointModel


def test_import_error_message_includes_missing_dependency_name():
    def fail_on_numpy(module_name):
        if module_name == "numpy":
            raise ImportError("No module named numpy")

    with pytest.raises(ImportError) as excinfo:
        _load_package_as("jmcm_missing_numpy", import_hook=fail_on_numpy)

    message = str(excinfo.value)
    assert "Unable to import required dependencies" in message
    assert "numpy: No module named numpy" in message


def test_import_error_message_can_report_multiple_missing_dependencies():
    missing = {"numpy", "pandas"}

    def fail_on_selected(module_name):
        if module_name in missing:
            raise ImportError(f"simulated missing {module_name}")

    with pytest.raises(ImportError) as excinfo:
        _load_package_as("jmcm_missing_two", import_hook=fail_on_selected)

    message = str(excinfo.value)
    assert "numpy: simulated missing numpy" in message
    assert "pandas: simulated missing pandas" in message


def test_temporary_import_does_not_pollute_real_package():
    temp_name = "jmcm_temp_import"
    mod = _load_package_as(temp_name)
    assert isinstance(mod, ModuleType)
    assert temp_name not in sys.modules
    assert jmcm.JointModel is not None
