import importlib
import inspect
import sys

import pypfm
from pypfm import PFMLoader


def test_package_version_exact():
    assert pypfm.__version__ == "1.0.0"


def test_package_exposes_pfml_loader():
    assert hasattr(pypfm, "PFMLoader")
    assert pypfm.PFMLoader is PFMLoader


def test_from_import_pfml_loader_matches_package_attribute():
    from pypfm import PFMLoader as imported_loader

    assert imported_loader is pypfm.PFMLoader


def test_pfml_loader_is_class():
    assert inspect.isclass(PFMLoader)


def test_pfml_loader_module_is_pypfm_submodule():
    assert PFMLoader.__module__ == "pypfm.pypfm"


def test_package_and_submodule_share_same_class_object():
    submodule = importlib.import_module("pypfm.pypfm")
    assert submodule.PFMLoader is pypfm.PFMLoader
    assert submodule.PFMLoader is PFMLoader


def test_package_import_does_not_replace_sys_modules_entry():
    assert sys.modules["pypfm"] is pypfm


def test_reimporting_package_keeps_public_api_stable():
    reloaded = importlib.reload(pypfm)
    assert reloaded is pypfm
    assert reloaded.PFMLoader is PFMLoader
    assert reloaded.__version__ == "1.0.0"


def test_reimporting_submodule_keeps_pfml_loader_stable():
    submodule = importlib.import_module("pypfm.pypfm")
    reloaded = importlib.reload(submodule)
    assert reloaded.PFMLoader is PFMLoader
    assert reloaded.PFMLoader is pypfm.PFMLoader


def test_package_allows_attribute_access_to_version_and_loader():
    assert getattr(pypfm, "__version__") == "1.0.0"
    assert getattr(pypfm, "PFMLoader") is PFMLoader


def test_package_has_expected_public_names():
    public_names = {name for name in dir(pypfm) if not name.startswith("_")}
    assert "PFMLoader" in public_names
    assert "__version__" in pypfm.__dict__
    assert "PFMLoader" in pypfm.__dict__


def test_version_is_safely_parseable():
    major, minor, patch = pypfm.__version__.split(".")
    assert (major, minor, patch) == ("1", "0", "0")
    assert all(part.isdigit() for part in (major, minor, patch))
