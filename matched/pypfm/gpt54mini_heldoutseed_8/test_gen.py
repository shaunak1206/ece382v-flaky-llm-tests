import importlib
import types

import pypfm
from pypfm import PFMLoader


def test_package_exposes_version_string():
    assert isinstance(pypfm.__version__, str)


def test_package_version_matches_expected():
    assert pypfm.__version__ == "1.0.0"


def test_package_exports_pfmloder_name():
    assert hasattr(pypfm, "PFMLoader")


def test_from_import_pfmloder_is_same_object_as_package_attribute():
    assert PFMLoader is pypfm.PFMLoader


def test_pfmloder_is_class_like():
    assert isinstance(PFMLoader, type)


def test_pfmloder_originates_from_submodule():
    assert PFMLoader.__module__ == "pypfm.pypfm"


def test_submodule_import_has_same_pfmloder_object():
    submodule = importlib.import_module("pypfm.pypfm")
    assert submodule.PFMLoader is PFMLoader


def test_package_import_reexports_submodule_pfmloder():
    submodule = importlib.import_module("pypfm.pypfm")
    assert pypfm.PFMLoader is submodule.PFMLoader


def test_package_reload_preserves_version_and_export():
    reloaded = importlib.reload(pypfm)
    assert reloaded.__version__ == "1.0.0"
    assert reloaded.PFMLoader is PFMLoader


def test_dir_contains_pfmloder():
    assert "PFMLoader" in dir(pypfm)


def test_import_star_like_access_works_via_getattr():
    assert getattr(pypfm, "PFMLoader") is PFMLoader


def test_package_is_a_module():
    assert isinstance(pypfm, types.ModuleType)
