import builtins
import importlib.util
import pathlib
import sys
import types

import pytest

import jmcm
from jmcm import JointModel


def _jmcm_package_path():
    return pathlib.Path(jmcm.__file__).resolve().parent


def _load_package_under_name(module_name, import_hook=None):
    """
    Load jmcm/__init__.py as a fresh package under a temporary module name.
    This lets us test package import-time behavior without disturbing the real import.
    """
    pkg_dir = _jmcm_package_path()
    init_py = pkg_dir / "__init__.py"

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(init_py),
        submodule_search_locations=[str(pkg_dir)],
    )
    module = importlib.util.module_from_spec(spec)

    original_import = builtins.__import__
    if import_hook is not None:
        builtins.__import__ = import_hook
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        builtins.__import__ = original_import
        sys.modules.pop(module_name, None)


def test_package_name_attribute():
    assert jmcm.name == "jmcm"


def test_jointmodel_is_exported_from_package():
    assert hasattr(jmcm, "JointModel")
    assert jmcm.JointModel is JointModel


def test_jointmodel_is_a_class():
    assert isinstance(JointModel, type)


def test_jointmodel_class_name():
    assert JointModel.__name__ == "JointModel"


def test_jointmodel_module_points_to_joint_model_submodule():
    assert JointModel.__module__ == "jmcm.joint_model"


def test_jointmodel_is_in_package_namespace():
    assert "JointModel" in jmcm.__dict__


def test_package_does_not_keep_dependency_list_variables():
    assert not hasattr(jmcm, "hard_dependencies")
    assert not hasattr(jmcm, "missing_dependencies")
    assert not hasattr(jmcm, "dependency")


def test_package_has_expected_module_attributes():
    assert isinstance(jmcm.__doc__, (str, type(None)))
    assert jmcm.__name__ == "jmcm"
    assert isinstance(jmcm.__package__, str)


def test_jointmodel_has_object_as_base_or_in_mro():
    assert object in JointModel.__mro__


def test_fresh_import_succeeds_with_normal_imports():
    fresh_name = "jmcm_test_fresh_import_ok"
    module = _load_package_under_name(fresh_name)
    try:
        assert module.name == "jmcm"
        assert hasattr(module, "JointModel")
        assert isinstance(module.JointModel, type)
    finally:
        sys.modules.pop(fresh_name, None)


def test_import_raises_helpful_message_when_numpy_is_missing():
    fresh_name = "jmcm_test_missing_numpy"

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            raise ImportError("No module named 'numpy'")
        return real_import(name, globals, locals, fromlist, level)

    with pytest.raises(ImportError) as excinfo:
        _load_package_under_name(fresh_name, import_hook=fake_import)

    message = str(excinfo.value)
    assert "Unable to import required dependencies:" in message
    assert "numpy:" in message
    assert "No module named 'numpy'" in message


def test_import_raises_helpful_message_when_multiple_dependencies_are_missing():
    fresh_name = "jmcm_test_missing_multiple"

    real_import = builtins.__import__
    missing = {"numpy", "pandas", "scipy"}

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in missing:
            raise ImportError(f"No module named '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    with pytest.raises(ImportError) as excinfo:
        _load_package_under_name(fresh_name, import_hook=fake_import)

    message = str(excinfo.value)
    assert "Unable to import required dependencies:" in message
    for dep in sorted(missing):
        assert dep + ":" in message
        assert f"No module named '{dep}'" in message


def test_import_message_includes_all_missing_dependencies_in_order():
    fresh_name = "jmcm_test_missing_order"

    real_import = builtins.__import__
    missing = {"numpy", "pandas"}

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in missing:
            raise ImportError(f"No module named '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    with pytest.raises(ImportError) as excinfo:
        _load_package_under_name(fresh_name, import_hook=fake_import)

    message = str(excinfo.value)
    numpy_index = message.index("numpy:")
    pandas_index = message.index("pandas:")
    assert numpy_index < pandas_index


def test_reimporting_real_package_keeps_same_jointmodel_object():
    import importlib

    reloaded = importlib.reload(jmcm)
    assert reloaded.JointModel is JointModel
    assert reloaded.name == "jmcm"
