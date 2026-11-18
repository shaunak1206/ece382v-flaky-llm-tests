import importlib
import sys
import types

import torch_layer_normalization


def test_package_exports_layer_normalization_name():
    assert hasattr(torch_layer_normalization, "LayerNormalization")
    assert isinstance(torch_layer_normalization.LayerNormalization, type)


def test_package_version_string():
    assert torch_layer_normalization.__version__ == "0.11.0"


def test_version_is_string():
    assert isinstance(torch_layer_normalization.__version__, str)


def test_layer_normalization_is_imported_from_submodule():
    submodule = importlib.import_module("torch_layer_normalization.layer_normalization")
    assert torch_layer_normalization.LayerNormalization is submodule.LayerNormalization


def test_star_import_exports_layer_normalization_and_version():
    ns = {}
    exec("from torch_layer_normalization import *", ns, ns)
    assert "LayerNormalization" in ns
    assert "torch_layer_normalization" not in ns
    assert ns["LayerNormalization"] is torch_layer_normalization.LayerNormalization


def test_package_allows_direct_access_to_reexported_symbol():
    cls = torch_layer_normalization.LayerNormalization
    assert cls.__name__ == "LayerNormalization"


def test_module_has_expected_dunder_file():
    assert hasattr(torch_layer_normalization, "__file__")
    assert torch_layer_normalization.__file__.endswith("__init__.py")


def test_module_has_expected_dunder_name():
    assert torch_layer_normalization.__name__ == "torch_layer_normalization"


def test_reload_preserves_version_and_symbol_identity():
    original_cls = torch_layer_normalization.LayerNormalization
    original_version = torch_layer_normalization.__version__

    reloaded = importlib.reload(torch_layer_normalization)

    assert reloaded.__version__ == original_version
    assert reloaded.LayerNormalization is original_cls


def test_import_module_returns_same_object_as_sys_modules_entry():
    mod = importlib.import_module("torch_layer_normalization")
    assert mod is sys.modules["torch_layer_normalization"]


def test_submodule_can_be_imported_independently():
    submodule = importlib.import_module("torch_layer_normalization.layer_normalization")
    assert isinstance(submodule, types.ModuleType)
    assert hasattr(submodule, "LayerNormalization")


def test_package_dir_contains_reexported_symbol_name():
    names = dir(torch_layer_normalization)
    assert "LayerNormalization" in names
    assert "__version__" in names


def test_symbol_is_same_after_from_import():
    from torch_layer_normalization import LayerNormalization

    assert LayerNormalization is torch_layer_normalization.LayerNormalization


def test_no_unexpected_breakage_when_accessing_attribute_multiple_times():
    a = torch_layer_normalization.LayerNormalization
    b = torch_layer_normalization.LayerNormalization
    assert a is b


def test_package_metadata_is_nonempty():
    assert torch_layer_normalization.__version__.strip() != ""
    assert torch_layer_normalization.__name__.strip() != ""
