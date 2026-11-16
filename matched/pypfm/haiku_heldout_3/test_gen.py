"""Unit tests for the pypfm package root (pypfm/__init__.py) and PFMLoader.

Note: only pypfm/__init__.py was provided, so these tests cover the package
surface (version string, re-exports, and PFMLoader's class-level identity).
Behavioral tests of PFMLoader's file-loading methods need pypfm/pypfm.py.
"""
import importlib
import re

import pypfm
import pypfm.pypfm as pypfm_module
from pypfm import PFMLoader


SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def test_version_is_defined():
    assert hasattr(pypfm, "__version__")


def test_version_is_string():
    assert isinstance(pypfm.__version__, str)


def test_version_is_not_empty():
    assert pypfm.__version__.strip() != ""


def test_version_exact_value():
    assert pypfm.__version__ == "1.0.0"


def test_version_is_semver_format():
    assert SEMVER_PATTERN.match(pypfm.__version__) is not None


def test_version_components_are_integers():
    major, minor, patch = pypfm.__version__.split(".")
    assert int(major) >= 0
    assert int(minor) >= 0
    assert int(patch) >= 0


def test_pfmloader_exported_from_package():
    assert hasattr(pypfm, "PFMLoader")


def test_pfmloader_importable_directly_from_package():
    from pypfm import PFMLoader as ImportedLoader

    assert ImportedLoader is PFMLoader


def test_pfmloader_is_same_object_as_submodule_attribute():
    assert pypfm.PFMLoader is pypfm_module.PFMLoader


def test_pfmloader_is_a_class():
    assert isinstance(PFMLoader, type)


def test_pfmloader_class_name():
    assert PFMLoader.__name__ == "PFMLoader"


def test_pfmloader_defined_in_pypfm_module():
    assert PFMLoader.__module__.startswith("pypfm")


def test_star_import_exposes_pfmloader():
    namespace = {}
    exec("from pypfm import *", namespace)
    assert "PFMLoader" in namespace
    assert namespace["PFMLoader"] is PFMLoader


def test_reload_keeps_pfmloader_and_version():
    reloaded = importlib.reload(pypfm)
    assert hasattr(reloaded, "PFMLoader")
    assert reloaded.__version__ == "1.0.0"
