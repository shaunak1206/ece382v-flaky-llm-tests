import importlib
import inspect
import sys

import pypfm
import pypfm.pypfm as pypfm_module


def test_package_exports_pfml_loader():
    assert hasattr(pypfm, "PFMLoader")
    assert pypfm.PFMLoader is pypfm_module.PFMLoader


def test_from_import_pfml_loader_matches_package_attribute():
    from pypfm import PFMLoader

    assert PFMLoader is pypfm.PFMLoader


def test_version_attribute_exists():
    assert hasattr(pypfm, "__version__")


def test_version_attribute_value():
    assert pypfm.__version__ == "1.0.0"


def test_version_is_string():
    assert isinstance(pypfm.__version__, str)


def test_pfml_loader_is_class():
    assert inspect.isclass(pypfm.PFMLoader)


def test_pfml_loader_class_name():
    assert pypfm.PFMLoader.__name__ == "PFMLoader"


def test_pfml_loader_module_name():
    assert pypfm.PFMLoader.__module__ == "pypfm.pypfm"


def test_pfml_loader_referenced_in_package_namespace_only_once():
    # The package should expose the same class object as the implementation module.
    assert getattr(pypfm, "PFMLoader") is getattr(pypfm_module, "PFMLoader")


def test_package_import_does_not_replace_module_object():
    before = sys.modules["pypfm"]
    reloaded = importlib.reload(pypfm)
    assert reloaded is before
    assert reloaded.PFMLoader is pypfm_module.PFMLoader


def test_reloading_package_keeps_version_constant():
    reloaded = importlib.reload(pypfm)
    assert reloaded.__version__ == "1.0.0"


def test_reloading_package_keeps_pfml_loader_available():
    reloaded = importlib.reload(pypfm)
    assert hasattr(reloaded, "PFMLoader")
    assert reloaded.PFMLoader is pypfm_module.PFMLoader
